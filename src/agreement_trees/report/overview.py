"""The dataset overview and the per-condition indexes of the published pages.

  build_stats      aggregate samples / minimal-pair counts across every condition into one JSON blob
                   (output/overview/stats.json). The source is html/decision_trees/: each condition's index.html
                   LANGUAGES blob and the per-language diagnostics dict of
                   agreement_trees.pairs.diagnostics.diagnostics_row_to_json. Conditions and NP subgroups are
                   discovered from what is on disk (_discover_conditions), so a rebuilt condition appears
                   automatically.
  refresh_indexes  rebuild every built condition's diagnostics CSV and deprel index.html (VP and NP) in parallel,
                   then the cross-pipeline decision_trees/index.html, purely from the on-disk
                   output/decision_trees and output/minimal_pairs artifacts (no refitting). A condition is
                   skipped unless something that feeds its index is newer than the index.

Command lines: scripts/overview/generate_html_indexes.py and build_stats.py (rendering the stats into
html/index.html is agreement_trees.report.html.html_dataset).
"""
import contextlib
import glob
import io
import json
import math
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd

from agreement_trees.npa.agreement import refresh_deprel_index as refresh_npa_index
from agreement_trees.pairs.diagnostics import BUCKET_JSON_KEYS
from agreement_trees.pipeline_utils import available_system_memory_bytes, guard_process_memory
from agreement_trees.report.viz_overview import generate_html_overview_index
from agreement_trees.taxonomy import FLAT_CONDITION_META, classify_flat_condition, NPA_FEATURE_NAMES, npa_subgroup_label, GROUP_PREFIXES
from agreement_trees.vp.conditions import CONDITIONS
from agreement_trees.vp.pipeline import refresh_deprel_index as refresh_sva_index
from multiblimp.config import OUTPUT_OVERVIEW_DIR, HTML_DECISION_TREES_DIR, OUTPUT_DECISION_TREES_DIR, RESOURCE_DIR, OUTPUT_MINIMAL_PAIRS_DIR, OUTPUT_DIAGNOSTICS_DIR, REPO_ROOT
from multiblimp.languages import get_lang_treebanks


# ---------------------------------------------------------------- dataset stats

STATS_PATH = os.path.join(OUTPUT_OVERVIEW_DIR, "stats.json")


HTML_ROOT = HTML_DECISION_TREES_DIR


PARQUET_ROOT = OUTPUT_DECISION_TREES_DIR


NAME_ALIASES = {
    "Norwegian Bokmaal": "Norwegian Bokmål",
}


BUCKET_LABELS = {
    "no_candidates": "No candidate",
    "no_inflections": "No inflection",
    "ambiguous_readings": "Ambiguous reading",
    "same_forms": "Same inflection",
    "same_features": "Same feature",
    "undefined_features": "Undefined feature",
    "ambiguous_subjects": "Ambiguous subject",
    "conflicting_features": "Conflicting features",
}


CONDITION_META = {
    **FLAT_CONDITION_META,
    "npa": ("Noun Phrase", "Agreement (all role pairs)"),
}


def _flat_meta(cond: str) -> tuple[str, str]:
    """(group, label) for a fixed or ad-hoc candidate flat condition id."""
    if cond in FLAT_CONDITION_META:
        return FLAT_CONDITION_META[cond]
    flat = classify_flat_condition(cond)
    return (flat[0], flat[1]) if flat else ("Other", cond)


PUBLISHED_BUCKET_KEYS = {v: k for k, v in BUCKET_JSON_KEYS.items()}


LANGUAGES_RE = re.compile(r"const LANGUAGES = (\[.*?\]);\s*const plotData\b", re.S)


def _load_languages_six(index_html_path: str) -> list[dict] | None:
    if not os.path.exists(index_html_path):
        return None
    html = open(index_html_path, encoding="utf-8").read()
    match = LANGUAGES_RE.search(html)
    if not match:
        return None
    return json.loads(match.group(1))


def _nan_to_none(x):
    # viz_deprel emits NaN (not None) for a language's accuracy without a tree; "is not None" checks pass it and it
    # would poison weighted averages and render as "NaN%", so it is normalised to None once, here
    return None if isinstance(x, float) and math.isnan(x) else x


def _record(entry: dict, cond: str, subgroup: str, dir_rel: str) -> dict:
    diag = entry.get("diag") or {}
    buckets = {
        PUBLISHED_BUCKET_KEYS[k]: v[0]
        for k, v in (diag.get("buckets") or {}).items()
        if k in PUBLISHED_BUCKET_KEYS
    }
    lang_url = entry.get("langUrl")
    return dict(
        cond=cond, subgroup=subgroup, lang=NAME_ALIASES.get(entry["name"], entry["name"]),
        samples=entry.get("nRaw") or 0, pairs=entry.get("nPairs") or 0,
        # tree accuracy (None, not 0, when absent, so weighted averages skip it)
        acc=_nan_to_none(entry.get("acc")),
        # langUrl is the bare filename relative to the condition's index.html; dir_rel is that condition's
        # directory under HTML_ROOT: joined into a path from html/index.html
        url=(f"decision_trees/{dir_rel}/{lang_url}" if lang_url else None),
        # distinct lemmas / forms: a rough treebank-richness signal
        n_lemma=diag.get("nLemma") or 0, n_form=diag.get("nForm") or 0,
        # rows dropped before the tree fit because the feature was missing on the head, the subject or both (an
        # earlier population than the outcome buckets, which only cover attempted swaps)
        head_unk=diag.get("headUnk") or 0, nsubj_unk=diag.get("nsubjUnk") or 0,
        both_unk=diag.get("bothUnk") or 0,
        buckets=buckets,
    )


def _tried_languages(dir_path: str) -> set[str]:
    """Every language attempted for one condition directory: one <Language>.html per run, including the
    placeholder page of a language without candidates (which has no LANGUAGES entry; svNa: 186 tried, 131 in
    the blob)."""
    if not os.path.isdir(dir_path):
        return set()
    return {
        NAME_ALIASES.get(name, name)
        for name in (
            fn[:-len(".html")].replace("_", " ")
            for fn in os.listdir(dir_path)
            if fn.endswith(".html") and fn != "index.html"
            and not fn.endswith("__unk.html")  # per-treebank incl.-unk view of a language
        )
    }


def _discover_conditions() -> tuple[list[str], list[str]]:
    """(flat_condition_ids, npa_subgroup_ids) that currently have an index.html under HTML_ROOT, discovered each build."""
    if not os.path.isdir(HTML_ROOT):
        return [], []

    flat = sorted(
        entry for entry in os.listdir(HTML_ROOT)
        if entry != "npa"
        and os.path.exists(os.path.join(HTML_ROOT, entry, "index.html"))
    )
    npa_root = os.path.join(HTML_ROOT, "npa")
    npa_subs = sorted(
        entry for entry in os.listdir(npa_root)
        if os.path.exists(os.path.join(npa_root, entry, "index.html"))
    ) if os.path.isdir(npa_root) else []
    return flat, npa_subs


def collect_records() -> tuple[list[dict], dict[str, set[str]]]:
    flat_conditions, npa_subgroups = _discover_conditions()
    if not flat_conditions and not npa_subgroups:
        raise FileNotFoundError(
            f"No condition index.html pages found under {HTML_ROOT} -- "
            "run scripts/overview/generate_html_indexes.py (or a pipeline's own "
            "fit script) first."
        )

    records = []
    tried = {}
    for cond in flat_conditions:
        cond_dir = os.path.join(HTML_ROOT, cond)
        tried[cond] = _tried_languages(cond_dir)
        six = _load_languages_six(os.path.join(cond_dir, "index.html"))
        if six:
            records += [_record(e, cond, cond, cond) for e in six]

    for sub in npa_subgroups:
        sub_dir = os.path.join(HTML_ROOT, "npa", sub)
        tried[sub] = _tried_languages(sub_dir)
        six = _load_languages_six(os.path.join(sub_dir, "index.html"))
        if six:
            records += [_record(e, "npa", sub, f"npa/{sub}") for e in six]

    return records, tried


def _parquet_paths(cond: str, is_npa_sub: bool) -> list[str]:
    # Flat conditions nest one level deeper than npa's own convention --
    # decision_trees/{cond}/{cond}_{deprel}/*.parquet vs.
    # decision_trees/npa/{subgroup}/*.parquet -- both observed directly on
    # disk (see PARQUET_ROOT's comment). The nested deprel varies by
    # condition family (svNa -> nsubj, ovNa -> obj, iovNa -> iobj, ...), so
    # it's read off disk rather than assumed; a condition dir with more than
    # one subdirectory (shouldn't happen -- one deprel per target_id by
    # convention) uses whichever sorts first rather than guessing further.
    if is_npa_sub:
        pattern = os.path.join(PARQUET_ROOT, "npa", cond, "*.parquet")
        return sorted(glob.glob(pattern))

    cond_dir = os.path.join(PARQUET_ROOT, cond)
    subdirs = sorted(
        d for d in glob.glob(os.path.join(cond_dir, "*")) if os.path.isdir(d)
    )
    if not subdirs:
        return []
    return sorted(glob.glob(os.path.join(subdirs[0], "*.parquet")))


def treebank_samples() -> list[dict]:
    """Per-sample treebank attribution from this repo's own decision-tree
    cache (PARQUET_ROOT) -- the one thing LANGUAGES.six never carries. Real
    counts, not estimates: each row in one of these parquet files is one raw
    candidate sample, tagged with the exact UD treebank it came from.

    Only covers whichever language x condition combinations already have a
    cached parquet on disk -- not necessarily every combination this page's
    other sections report on, since that depends on how far the pipeline's
    own fitting run has gotten for each condition.

    Deliberately reports *samples* by treebank, not *pairs* by treebank --
    there's no per-row flag distinguishing "became an accepted minimal pair"
    from "did not" in these files (that determination happens later in the
    pipeline, working from bucket-membership logic this function doesn't
    re-run), so treating a sample count as a pair count would fabricate
    precision that isn't there. Samples are still the right proxy: every
    pair for a language/condition is drawn from exactly this sample pool, so
    a treebank's share of samples is a treebank's share of where the pairs
    *could* have come from.
    """
    flat_conditions, npa_subgroups = _discover_conditions()
    out = []
    for cond in flat_conditions:
        group, label = _flat_meta(cond)
        for path in _parquet_paths(cond, is_npa_sub=False):
            lang = NAME_ALIASES.get(
                os.path.splitext(os.path.basename(path))[0].replace("_", " "),
                os.path.splitext(os.path.basename(path))[0].replace("_", " "),
            )
            counts = pd.read_parquet(path, columns=["treebank"])["treebank"].value_counts()
            out.append({
                "language": lang, "condition_id": cond, "condition_label": f"{group} ({label})",
                "group": group, "samples_by_treebank": counts.to_dict(), "total_samples": int(counts.sum()),
            })
    for sub in npa_subgroups:
        for path in _parquet_paths(sub, is_npa_sub=True):
            lang = NAME_ALIASES.get(
                os.path.splitext(os.path.basename(path))[0].replace("_", " "),
                os.path.splitext(os.path.basename(path))[0].replace("_", " "),
            )
            counts = pd.read_parquet(path, columns=["treebank"])["treebank"].value_counts()
            out.append({
                "language": lang, "condition_id": f"npa/{sub}", "condition_label": npa_subgroup_label(sub),
                "group": "Noun Phrase", "samples_by_treebank": counts.to_dict(), "total_samples": int(counts.sum()),
            })
    return out


def build_stats(out_path: str = STATS_PATH) -> dict:
    """Aggregate every condition's published page into the stats blob, write it to out_path and return it."""
    records, tried_by_key = collect_records()

    languages = sorted(set(r["lang"] for r in records))
    conditions = sorted(CONDITION_META, key=lambda c: (CONDITION_META[c][0], CONDITION_META[c][1]))
    # Any condition not yet in CONDITION_META still shows up (grouped as
    # "Other") rather than silently dropping real data.
    unknown = sorted(set(r["cond"] for r in records) - set(CONDITION_META))
    for cond in unknown:
        CONDITION_META[cond] = _flat_meta(cond)
    conditions += unknown

    total_samples = sum(r["samples"] for r in records)
    total_pairs = sum(r["pairs"] for r in records)

    def ratio(pairs, samples):
        # Pairs aren't a subset of samples -- one candidate instance can
        # yield several minimal pairs (e.g. multiple valid contrastive
        # forms), so this can legitimately exceed 1.0. None (not 0) when
        # there were no samples at all, so the UI can render "—" instead of
        # a misleading 0.0.
        return round(pairs / samples, 2) if samples else None

    def weighted_avg(num, den):
        # Same None-not-0 convention as ratio() above, for averaging a
        # per-record metric (e.g. decision-tree accuracy) across records --
        # weighting by samples means a record with 0 samples (nothing to
        # score) can never skew the average even when the site still
        # reports some placeholder accuracy for it.
        return round(num / den, 3) if den else None

    # per top-level condition (summed across its subgroups, e.g. NPA's role pairs)
    cond_agg = defaultdict(lambda: {
        "samples": 0, "pairs": 0, "langs": set(), "acc_num": 0.0, "acc_den": 0,
        "head_unk": 0, "nsubj_unk": 0, "both_unk": 0,
    })
    # per language (summed across every condition/subgroup)
    lang_agg = defaultdict(lambda: {
        "samples": 0, "pairs": 0, "conds": set(), "acc_num": 0.0, "acc_den": 0,
        "n_lemma": 0, "n_form": 0, "top_pairs": -1, "url": None,
    })
    # language x condition matrix -- pairs/samples/lexical-diversity sum
    # across every record for that cell (for the "npa" column, across all 4
    # role-pair subgroups, same aggregation "samples" itself already gets);
    # acc is a samples-weighted average since accuracy isn't additive;
    # url is the single highest-pairs contributing record's link, the
    # closest thing to a "representative" deep link for an aggregate cell
    # (exact for the 9 flat conditions, which only ever have one record).
    matrix_pairs = defaultdict(int)
    matrix_samples = defaultdict(int)
    matrix_acc_num = defaultdict(float)
    matrix_acc_den = defaultdict(int)
    matrix_n_lemma = defaultdict(int)
    matrix_n_form = defaultdict(int)
    matrix_head_unk = defaultdict(int)
    matrix_nsubj_unk = defaultdict(int)
    matrix_both_unk = defaultdict(int)
    matrix_url = defaultdict(lambda: None)
    matrix_top_pairs = defaultdict(lambda: -1)
    # language x NPA-subgroup matrix -- the main matrix only goes down to
    # top-level condition ("npa", summed over all role pairs), so a category
    # drill-down into one specific role pair (e.g. Head-Determiner Number)
    # needs its own per-language breakdown. Same field set as the main
    # matrix, minus the aggregation ambiguity -- exactly one record per
    # (language, subgroup) cell, so acc/url reduce to that record's own value.
    npa_matrix_pairs = defaultdict(int)
    npa_matrix_samples = defaultdict(int)
    npa_matrix_acc_num = defaultdict(float)
    npa_matrix_acc_den = defaultdict(int)
    npa_matrix_n_lemma = defaultdict(int)
    npa_matrix_n_form = defaultdict(int)
    npa_matrix_head_unk = defaultdict(int)
    npa_matrix_nsubj_unk = defaultdict(int)
    npa_matrix_both_unk = defaultdict(int)
    npa_matrix_url = defaultdict(lambda: None)
    npa_matrix_top_pairs = defaultdict(lambda: -1)
    # global funnel across every bucket-labeled record
    funnel = defaultdict(int)
    # NPA subgroup-level detail (its own table, since it's several role
    # pairs x ~50-90 languages each)
    npa_sub_agg = defaultdict(lambda: {
        "samples": 0, "pairs": 0, "langs": set(), "acc_num": 0.0, "acc_den": 0,
        "head_unk": 0, "nsubj_unk": 0, "both_unk": 0,
    })

    for r in records:
        lang, cond = r["lang"], r["cond"]

        c = cond_agg[cond]
        c["samples"] += r["samples"]; c["pairs"] += r["pairs"]; c["langs"].add(lang)

        l = lang_agg[lang]
        l["samples"] += r["samples"]; l["pairs"] += r["pairs"]; l["conds"].add(cond)
        l["n_lemma"] += r["n_lemma"]; l["n_form"] += r["n_form"]

        matrix_pairs[(lang, cond)] += r["pairs"]
        matrix_samples[(lang, cond)] += r["samples"]
        matrix_n_lemma[(lang, cond)] += r["n_lemma"]
        matrix_n_form[(lang, cond)] += r["n_form"]
        matrix_head_unk[(lang, cond)] += r["head_unk"]
        matrix_nsubj_unk[(lang, cond)] += r["nsubj_unk"]
        matrix_both_unk[(lang, cond)] += r["both_unk"]

        if r["acc"] is not None:
            weight = r["samples"]
            c["acc_num"] += r["acc"] * weight; c["acc_den"] += weight
            l["acc_num"] += r["acc"] * weight; l["acc_den"] += weight
            matrix_acc_num[(lang, cond)] += r["acc"] * weight
            matrix_acc_den[(lang, cond)] += weight

        if r["url"] and r["pairs"] > l["top_pairs"]:
            l["top_pairs"] = r["pairs"]; l["url"] = r["url"]
        if r["url"] and r["pairs"] > matrix_top_pairs[(lang, cond)]:
            matrix_top_pairs[(lang, cond)] = r["pairs"]; matrix_url[(lang, cond)] = r["url"]

        for bucket, n in r["buckets"].items():
            funnel[BUCKET_LABELS.get(bucket, bucket)] += n

        c["head_unk"] += r["head_unk"]; c["nsubj_unk"] += r["nsubj_unk"]; c["both_unk"] += r["both_unk"]

        if cond == "npa":
            sub = r["subgroup"]
            s = npa_sub_agg[sub]
            s["samples"] += r["samples"]; s["pairs"] += r["pairs"]; s["langs"].add(lang)
            s["head_unk"] += r["head_unk"]; s["nsubj_unk"] += r["nsubj_unk"]; s["both_unk"] += r["both_unk"]
            npa_matrix_pairs[(lang, sub)] += r["pairs"]
            npa_matrix_samples[(lang, sub)] += r["samples"]
            npa_matrix_n_lemma[(lang, sub)] += r["n_lemma"]
            npa_matrix_n_form[(lang, sub)] += r["n_form"]
            npa_matrix_head_unk[(lang, sub)] += r["head_unk"]
            npa_matrix_nsubj_unk[(lang, sub)] += r["nsubj_unk"]
            npa_matrix_both_unk[(lang, sub)] += r["both_unk"]
            if r["acc"] is not None:
                weight = r["samples"]
                s["acc_num"] += r["acc"] * weight; s["acc_den"] += weight
                npa_matrix_acc_num[(lang, sub)] += r["acc"] * weight
                npa_matrix_acc_den[(lang, sub)] += weight
            if r["url"] and r["pairs"] > npa_matrix_top_pairs[(lang, sub)]:
                npa_matrix_top_pairs[(lang, sub)] = r["pairs"]; npa_matrix_url[(lang, sub)] = r["url"]

    languages_sorted = sorted(languages, key=lambda l: -lang_agg[l]["pairs"])

    npa_subgroups_sorted = sorted(npa_sub_agg)
    npa_languages_sorted = sorted(
        {lang for lang, _ in npa_matrix_pairs},
        key=lambda lang: -sum(npa_matrix_pairs.get((lang, sub), 0) for sub in npa_subgroups_sorted),
    )

    # "npa" (the aggregate condition, summed over all role pairs) never
    # appears as its own key in tried_by_key -- only its subgroups do, same
    # as everywhere else in this script -- so its tried set is the union of
    # theirs.
    npa_tried = set().union(*(tried_by_key.get(sub, set()) for sub in npa_subgroups_sorted))

    COVERAGE_THRESHOLDS = (10, 30, 50, 100, 500)

    def coverage_stats(key: str, is_npa_sub: bool) -> dict:
        """How many of this condition's attempted languages produced any
        samples at all, and how many cleared each minimal-pair threshold --
        the two matrices already carry per-language pairs/samples for
        exactly this key, so no extra bookkeeping needed beyond the tried set.
        """
        tried_langs = tried_by_key.get(key, set()) if key != "npa" else npa_tried
        pairs_lookup = npa_matrix_pairs if is_npa_sub else matrix_pairs
        samples_lookup = npa_matrix_samples if is_npa_sub else matrix_samples
        with_samples = sum(1 for lang in tried_langs if samples_lookup.get((lang, key), 0) > 0)
        ge = {
            t: sum(1 for lang in tried_langs if pairs_lookup.get((lang, key), 0) >= t)
            for t in COVERAGE_THRESHOLDS
        }
        zero_langs = sorted(
            lang for lang in tried_langs if samples_lookup.get((lang, key), 0) == 0
        )
        return {
            "tried": len(tried_langs),
            "with_samples": with_samples,
            **{f"ge{t}": ge[t] for t in COVERAGE_THRESHOLDS},
            "tried_languages": sorted(tried_langs),
            "zero_languages": zero_langs,
        }

    def coverage_union(keys: list[tuple[str, bool]]) -> dict:
        """Same shape as coverage_stats, but unioned across several
        conditions/subgroups at once (e.g. every condition in one category
        tab, or the whole corpus) -- a language's samples/pairs get summed
        across all of `keys` before checking the thresholds, so a language
        only clears "ge100" here if its combined total across the group
        does, not any single condition alone.
        """
        tried_langs = set()
        lang_samples = defaultdict(int)
        lang_pairs = defaultdict(int)
        for key, is_npa_sub in keys:
            pairs_lookup = npa_matrix_pairs if is_npa_sub else matrix_pairs
            samples_lookup = npa_matrix_samples if is_npa_sub else matrix_samples
            key_tried = tried_by_key.get(key, set()) if key != "npa" else npa_tried
            tried_langs |= key_tried
            for lang in key_tried:
                lang_samples[lang] += samples_lookup.get((lang, key), 0)
                lang_pairs[lang] += pairs_lookup.get((lang, key), 0)
        with_samples = sum(1 for lang in tried_langs if lang_samples[lang] > 0)
        ge = {t: sum(1 for lang in tried_langs if lang_pairs[lang] >= t) for t in COVERAGE_THRESHOLDS}
        zero_langs = sorted(lang for lang in tried_langs if lang_samples[lang] == 0)
        return {
            "tried": len(tried_langs),
            "with_samples": with_samples,
            **{f"ge{t}": ge[t] for t in COVERAGE_THRESHOLDS},
            "zero_languages": zero_langs,
        }

    # Global coverage: every condition except the "npa" aggregate (replaced
    # by its own role-pair subgroups, so a language covered by two NPA role
    # pairs isn't unioned in twice under two different keys).
    global_coverage_keys = [(cond, False) for cond in conditions if cond != "npa"]
    global_coverage_keys += [(sub, True) for sub in npa_subgroups_sorted]

    # Which UD treebank(s) each attempted language's samples could come from
    # -- an inventory (which treebanks the pipeline is *eligible* to draw
    # from for this language), not a per-sample/per-pair count. LANGUAGES.six
    # never carries a per-sample treebank field (it only shows up in a
    # handful of illustrative example rows, not a full accounting).
    all_tried_langs = set().union(*tried_by_key.values()) if tried_by_key else set()
    lang_treebanks_raw = get_lang_treebanks(RESOURCE_DIR)

    # Cross-referenced against this same page's own totals for that exact
    # language x condition, so the UI can show "these N sampled rows are
    # part of the M samples/P pairs already shown for this combo" instead of
    # this breakdown's own figures floating unanchored from the rest of the
    # page's numbers.
    treebank_pairs = treebank_samples()
    for entry in treebank_pairs:
        is_sub = entry["condition_id"].startswith("npa/")
        key = entry["condition_id"].split("/", 1)[1] if is_sub else entry["condition_id"]
        pairs_lookup = npa_matrix_pairs if is_sub else matrix_pairs
        samples_lookup = npa_matrix_samples if is_sub else matrix_samples
        entry["published_pairs"] = pairs_lookup.get((entry["language"], key), 0)
        entry["published_samples"] = samples_lookup.get((entry["language"], key), 0)

    def dropped_dict(agg: dict) -> dict:
        # Rows dropped before the decision tree was even fit (missing
        # feature annotation on the head, the subject, or both) -- a
        # different, earlier population than `funnel`, which is scoped to
        # attempted swaps only. See _record()'s head_unk/nsubj_unk/both_unk
        # comment. Scoped to one condition/subgroup's own agg dict, for that
        # category tab's own breakdown.
        return {
            "Head unknown": agg["head_unk"],
            "Subject unknown": agg["nsubj_unk"],
            "Both unknown": agg["both_unk"],
        }

    out = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "totals": {
            "languages": len(languages),
            "conditions": len(conditions),
            "samples": total_samples,
            "pairs": total_pairs,
            "ratio": ratio(total_pairs, total_samples),
            "coverage": coverage_union(global_coverage_keys),
        },
        "conditions": [
            {
                "id": cond,
                "group": CONDITION_META[cond][0],
                "label": CONDITION_META[cond][1],
                "languages": len(cond_agg[cond]["langs"]),
                "samples": cond_agg[cond]["samples"],
                "pairs": cond_agg[cond]["pairs"],
                "ratio": ratio(cond_agg[cond]["pairs"], cond_agg[cond]["samples"]),
                "acc": weighted_avg(cond_agg[cond]["acc_num"], cond_agg[cond]["acc_den"]),
                "dropped_before_fitting": dropped_dict(cond_agg[cond]),
                **coverage_stats(cond, is_npa_sub=False),
            }
            for cond in conditions
        ],
        "languages": [
            {
                "name": lang,
                "samples": lang_agg[lang]["samples"],
                "pairs": lang_agg[lang]["pairs"],
                "n_conditions": len(lang_agg[lang]["conds"]),
                "ratio": ratio(lang_agg[lang]["pairs"], lang_agg[lang]["samples"]),
                "acc": weighted_avg(lang_agg[lang]["acc_num"], lang_agg[lang]["acc_den"]),
                "n_lemma": lang_agg[lang]["n_lemma"],
                "n_form": lang_agg[lang]["n_form"],
                # Deep link into the single highest-pairs condition/subgroup
                # this language appears in -- see the matrix_url comment
                # above for why "top contributing record" is the closest
                # honest stand-in for a per-language link.
                "url": lang_agg[lang]["url"],
            }
            for lang in languages_sorted
        ],
        # Every attempted language (with data or not -- e.g. a zero-yield
        # language's treebank list can be part of *why* it's zero-yield),
        # not just languages_sorted.
        "language_treebanks": [
            {
                "name": lang,
                "treebanks": lang_treebanks_raw.get(lang.replace(" ", "_"), []),
            }
            for lang in sorted(all_tried_langs)
        ],
        # Only the language x condition combinations that already have a
        # cached parquet on disk appear here at all -- see
        # treebank_samples()'s own docstring for why that can be a subset of
        # this page's full coverage.
        "treebank_pairs": treebank_pairs,
        "matrix": {
            "languages": languages_sorted,
            "conditions": conditions,
            "pairs": [[matrix_pairs.get((lang, cond), 0) for cond in conditions] for lang in languages_sorted],
            "samples": [[matrix_samples.get((lang, cond), 0) for cond in conditions] for lang in languages_sorted],
            "acc": [
                [weighted_avg(matrix_acc_num.get((lang, cond), 0.0), matrix_acc_den.get((lang, cond), 0))
                 for cond in conditions]
                for lang in languages_sorted
            ],
            "n_lemma": [[matrix_n_lemma.get((lang, cond), 0) for cond in conditions] for lang in languages_sorted],
            "n_form": [[matrix_n_form.get((lang, cond), 0) for cond in conditions] for lang in languages_sorted],
            "head_unk": [[matrix_head_unk.get((lang, cond), 0) for cond in conditions] for lang in languages_sorted],
            "nsubj_unk": [[matrix_nsubj_unk.get((lang, cond), 0) for cond in conditions] for lang in languages_sorted],
            "both_unk": [[matrix_both_unk.get((lang, cond), 0) for cond in conditions] for lang in languages_sorted],
            "urls": [[matrix_url.get((lang, cond)) for cond in conditions] for lang in languages_sorted],
        },
        "npa_matrix": {
            "languages": npa_languages_sorted,
            "subgroups": npa_subgroups_sorted,
            "pairs": [
                [npa_matrix_pairs.get((lang, sub), 0) for sub in npa_subgroups_sorted]
                for lang in npa_languages_sorted
            ],
            "samples": [
                [npa_matrix_samples.get((lang, sub), 0) for sub in npa_subgroups_sorted]
                for lang in npa_languages_sorted
            ],
            "acc": [
                [weighted_avg(npa_matrix_acc_num.get((lang, sub), 0.0), npa_matrix_acc_den.get((lang, sub), 0))
                 for sub in npa_subgroups_sorted]
                for lang in npa_languages_sorted
            ],
            "n_lemma": [
                [npa_matrix_n_lemma.get((lang, sub), 0) for sub in npa_subgroups_sorted]
                for lang in npa_languages_sorted
            ],
            "n_form": [
                [npa_matrix_n_form.get((lang, sub), 0) for sub in npa_subgroups_sorted]
                for lang in npa_languages_sorted
            ],
            "head_unk": [
                [npa_matrix_head_unk.get((lang, sub), 0) for sub in npa_subgroups_sorted]
                for lang in npa_languages_sorted
            ],
            "nsubj_unk": [
                [npa_matrix_nsubj_unk.get((lang, sub), 0) for sub in npa_subgroups_sorted]
                for lang in npa_languages_sorted
            ],
            "both_unk": [
                [npa_matrix_both_unk.get((lang, sub), 0) for sub in npa_subgroups_sorted]
                for lang in npa_languages_sorted
            ],
            "urls": [
                [npa_matrix_url.get((lang, sub)) for sub in npa_subgroups_sorted]
                for lang in npa_languages_sorted
            ],
        },
        "funnel": dict(sorted(funnel.items(), key=lambda kv: -kv[1])),
        "npa_subgroups": [
            {
                "id": sub,
                "label": npa_subgroup_label(sub),
                "languages": len(npa_sub_agg[sub]["langs"]),
                "samples": npa_sub_agg[sub]["samples"],
                "pairs": npa_sub_agg[sub]["pairs"],
                "ratio": ratio(npa_sub_agg[sub]["pairs"], npa_sub_agg[sub]["samples"]),
                "acc": weighted_avg(npa_sub_agg[sub]["acc_num"], npa_sub_agg[sub]["acc_den"]),
                "dropped_before_fitting": dropped_dict(npa_sub_agg[sub]),
                **coverage_stats(sub, is_npa_sub=True),
            }
            for sub in sorted(npa_sub_agg)
        ],
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")
    print(f"  {out['totals']}")
    return out


# ---------------------------------------------------------------- condition indexes

MIN_WORKER_MEM_GB = 6.0


_CODE_FILES = [
    os.path.join(REPO_ROOT, "src", "agreement_trees", "report", "viz_deprel.py"),
    os.path.join(REPO_ROOT, "src", "agreement_trees", "report", "html", "html_deprel.py"),
    os.path.join(REPO_ROOT, "src", "agreement_trees", "pairs", "diagnostics.py"),
]


def _npa_identifier_to_target_col(npa_identifier: str) -> str:
    """Inverse of agreement_trees.npa.agreement.npa_id: "HEAD-DET_N" -> "HEAD-DET_Number" (the directory holds the abbreviation)."""
    roles, abbr = npa_identifier.rsplit("_", 1)
    return f"{roles}_{NPA_FEATURE_NAMES.get(abbr, abbr)}"


def _sva_spec(target_id: str, deprel: str):
    """(predictor_var, head_label) for an SVA target_id/deprel directory, or
    None if it maps to no known condition. A fixed id (svNa, saPa_single, ...)
    comes from agreement_trees.vp.conditions; anything else must be an ad-hoc
    <family><Feature> candidate (e.g. svCase), whose predictor_var follows the
    same head_{deprel}_{feature}_agreement convention."""
    cid = target_id.removesuffix("_single")
    if cid in CONDITIONS:
        cond, feature = CONDITIONS[cid], CONDITIONS[cid].swap_feat
    else:
        family = next((p for p in sorted(GROUP_PREFIXES, key=len, reverse=True)
                       if cid.startswith(p) and len(cid) > len(p)), None)
        if family is None or f"{family}Na" not in CONDITIONS:
            return None
        cond, feature = CONDITIONS[f"{family}Na"], cid[len(family):]
    return f"head_{deprel}_{feature}_agreement", cond.head_label


def built_sva_tasks() -> tuple[list[dict], list[str]]:
    """(tasks, unrecognized dirs) for every SVA condition x deprel with at
    least one fitted language on disk."""
    tasks, unknown = [], []
    for tid_dir in sorted(glob.glob(os.path.join(glob.escape(OUTPUT_DECISION_TREES_DIR), "*"))):
        target_id = os.path.basename(tid_dir)
        if target_id == "npa" or not os.path.isdir(tid_dir):
            continue
        for save_dir in sorted(glob.glob(os.path.join(glob.escape(tid_dir), f"{target_id}_*"))):
            if not (glob.glob(os.path.join(glob.escape(save_dir), "*.joblib"))
                    or glob.glob(os.path.join(glob.escape(save_dir), "*_label_distribution.json"))):
                continue
            deprel = os.path.basename(save_dir)[len(target_id) + 1:]
            spec = _sva_spec(target_id, deprel)
            if spec is None:
                unknown.append(os.path.relpath(save_dir, OUTPUT_DECISION_TREES_DIR))
                continue
            predictor_var, head_label = spec
            tasks.append(dict(
                kind="sva", name=f"{target_id}/{deprel}", aliases=[target_id],
                save_dir=save_dir,
                pairs_dir=os.path.join(OUTPUT_MINIMAL_PAIRS_DIR, target_id, f"{target_id}_{deprel}"),
                html_index=os.path.join(HTML_DECISION_TREES_DIR, target_id, "index.html"),
                diagnostics_csv=os.path.join(OUTPUT_DIAGNOSTICS_DIR, target_id, f"{target_id}_{deprel}.csv"),
                kwargs=dict(target_id=target_id, deprel=deprel, predictor_var=predictor_var,
                            head_label=head_label),
            ))
    return tasks, unknown


def built_npa_tasks() -> list[dict]:
    """One task per NPA target_col with at least one fitted language on disk
    (read off output/decision_trees/npa/*/*.joblib, not the scan results,
    which lists CANDIDATE languages, most not yet run)."""
    tasks = []
    for cond_dir in sorted(glob.glob(os.path.join(glob.escape(OUTPUT_DECISION_TREES_DIR), "npa", "*"))):
        if not os.path.isdir(cond_dir) or not (glob.glob(os.path.join(glob.escape(cond_dir), "*.joblib"))
                                               or glob.glob(os.path.join(glob.escape(cond_dir), "*_label_distribution.json"))):
            continue
        identifier = os.path.basename(cond_dir)
        target_col = _npa_identifier_to_target_col(identifier)
        tasks.append(dict(
            kind="npa", name=target_col, aliases=[identifier],
            save_dir=cond_dir,
            pairs_dir=os.path.join(OUTPUT_MINIMAL_PAIRS_DIR, "npa", identifier),
            html_index=os.path.join(HTML_DECISION_TREES_DIR, "npa", identifier, "index.html"),
            diagnostics_csv=os.path.join(OUTPUT_DIAGNOSTICS_DIR, "npa", f"{identifier}.csv"),
            kwargs=dict(target_col=target_col),
        ))
    return tasks


def _latest_mtime(*patterns: str) -> float:
    """Max mtime across every file matched by the given glob patterns, or
    -1 if none match."""
    mtimes = [
        os.path.getmtime(p)
        for pattern in patterns
        for p in glob.glob(pattern)
        if os.path.isfile(p)
    ]
    return max(mtimes) if mtimes else -1.0


def needs_refresh(task: dict) -> bool:
    """True unless the condition's index.html (and diagnostics CSV) is newer than every data input AND the index-generating code --
    i.e. unless nothing that feeds it has changed since the last build."""
    outputs = [task["html_index"], task["diagnostics_csv"]]
    if not all(os.path.exists(p) for p in outputs):
        return True
    output_mtime = min(os.path.getmtime(p) for p in outputs)
    input_mtime = _latest_mtime(
        os.path.join(task["save_dir"], "*.joblib"),
        os.path.join(task["save_dir"], "*_label_distribution.json"),
        os.path.join(task["save_dir"], "*.parquet"),
        os.path.join(task["pairs_dir"], "*", "*.parquet"),
        os.path.join(task["pairs_dir"], "*", "meta.json"),
        *_CODE_FILES,
    )
    return input_mtime > output_mtime


def _refresh_one(task: dict) -> tuple[str, bool, str, float]:
    """Worker body: (name, ok, captured log, seconds). Output is captured so
    parallel conditions' logs print as contiguous blocks, not interleaved."""
    start, log = time.time(), io.StringIO()
    ok = True
    with contextlib.redirect_stdout(log):
        try:
            if task["kind"] == "sva":
                refresh_sva_index(**task["kwargs"])
            else:
                refresh_npa_index(**task["kwargs"])
        except Exception:
            ok = False
            traceback.print_exc(file=log)
    return task["name"], ok, log.getvalue(), time.time() - start


def _report(result: tuple[str, bool, str, float], done: int, total: int, verbose: bool) -> bool:
    name, ok, log, seconds = result
    print(f"[{done}/{total}] {'ok  ' if ok else 'FAIL'} {name} ({seconds:.0f}s)", flush=True)
    if verbose or not ok:
        print("".join(f"    {line}\n" for line in log.strip().splitlines()), end="", flush=True)
    return ok


def refresh_indexes(args) -> int:
    """Rebuild the condition indexes and the cross-pipeline overview as the parsed command line of
    scripts/overview/generate_html_indexes.py asks for. Returns the process exit status."""
    sva_tasks, unknown = built_sva_tasks()
    tasks = sva_tasks + built_npa_tasks()
    if unknown:
        print(f"Skipping {len(unknown)} unrecognized SVA dir(s) (no matching condition): {unknown}")

    if args.conditions:
        wanted = set(args.conditions)
        unmatched = sorted(w for w in wanted
                           if not any(w == t["name"] or w in t["aliases"] for t in tasks))
        if unmatched:
            sys.exit(f"Not built on disk, so nothing to refresh: {unmatched}")
        tasks = [t for t in tasks if t["name"] in wanted or wanted & set(t["aliases"])]

    n_sva = sum(t["kind"] == "sva" for t in tasks)
    print(f"{len(tasks)} condition(s) selected ({n_sva} sva, {len(tasks) - n_sva} npa)")

    to_refresh = [t for t in tasks if args.force or needs_refresh(t)]
    if len(to_refresh) < len(tasks):
        print(f"Already up to date, skipping {len(tasks) - len(to_refresh)}")

    if args.list:
        print(f"Would refresh {len(to_refresh)}: {[t['name'] for t in to_refresh]}")
        return 0

    failed = []
    if to_refresh:
        jobs = max(1, min(args.jobs or min(os.cpu_count() or 1, 8), len(to_refresh)))
        available = available_system_memory_bytes()
        if available:
            per_worker = (args.max_worker_mem_gb or MIN_WORKER_MEM_GB) * 1024**3
            jobs = max(1, min(jobs, int(available * 0.8 / per_worker)))
        print(f"Refreshing {len(to_refresh)} condition(s) with {jobs} worker(s)", flush=True)
        total = len(to_refresh)
        if jobs == 1:
            for done, task in enumerate(to_refresh, 1):
                if not _report(_refresh_one(task), done, total, args.verbose):
                    failed.append(task["name"])
        else:
            if args.max_worker_mem_gb is not None:
                mem_bytes = int(args.max_worker_mem_gb * 1024**3)
            else:
                mem_bytes = int(available * 0.8 / jobs) if available else None
            initializer, initargs = (guard_process_memory, (mem_bytes,)) if mem_bytes else (None, ())
            if mem_bytes:
                print(f"Capping each worker to {mem_bytes / 1024**3:.1f} GB RAM")
            # Largest conditions first, so the pool's tail isn't one slow straggler.
            order = sorted(to_refresh, key=lambda t: -len(glob.glob(os.path.join(glob.escape(t["save_dir"]), "*.joblib"))))
            with ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=1,
                                     initializer=initializer, initargs=initargs) as pool:
                futures = {pool.submit(_refresh_one, t): t for t in order}
                for done, future in enumerate(as_completed(futures), 1):
                    task = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:  # worker died (e.g. hit the memory cap)
                        result = (task["name"], False, f"{type(e).__name__}: {e}", 0.0)
                    if not _report(result, done, total, args.verbose):
                        failed.append(task["name"])

    if not args.skip_overview:
        print("\nGenerating cross-pipeline overview index")
        generate_html_overview_index(html_directory=HTML_DECISION_TREES_DIR)

    if failed:
        print(f"\nFailed to refresh {len(failed)}: {failed}")
        return 1
    return 0
