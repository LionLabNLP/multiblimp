"""Aggregate samples/minimal-pair counts across every condition into a
single JSON blob, consumed by overview.html.

Source: the deployed MultiBLiMP v2 site (jshrdt.github.io/multiblimp),
scraped from each condition's index.html LANGUAGES.six blob (the same
per-language diagnostics dict sva_trees.diagnostics.diagnostics_row_to_json
produces -- nRaw/nPairs/diag.buckets map directly onto this script's
samples/pairs/bucket fields). See EXTERNAL_FLAT_CONDITIONS/
EXTERNAL_NPA_SUBGROUPS below for exactly which conditions it covers; this
repo's own in-progress minimal_pairs/ runs are intentionally not pulled in.

Run from the repo root: python scripts/overview/build_stats.py
"""
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from glob import glob

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Absolute (not the other scripts' relative "../../src/") since this
# script's own docstring has it run from the repo root, not from
# scripts/overview/ -- a relative sys.path entry would resolve against
# whatever the caller's cwd happens to be instead.
sys.path.append(os.path.join(REPO_ROOT, "src"))
from multiblimp.condition_taxonomy import (  # noqa: E402
    FLAT_CONDITION_META, NPA_ROLE_PROSE, NPA_FEATURE_NAMES, npa_subgroup_label,
)
from multiblimp.languages import get_lang_treebanks  # noqa: E402
from multiblimp.config import OUTPUT_DECISION_TREES_DIR, OUTPUT_OVERVIEW_DIR  # noqa: E402

OUT_PATH = os.path.join(OUTPUT_OVERVIEW_DIR, "stats.json")

RESOURCE_DIR = os.path.join(REPO_ROOT, "resources")

# Sibling checkout of the deployed site -- github.io repos have no fixed
# location relative to this one, so this is just where it happens to live
# on this machine.
EXTERNAL_SITE_ORIGIN = "https://jshrdt.github.io"
EXTERNAL_ROOT = "/Users/jacobleesuchardt/projects/jshrdt.github.io/multiblimp"
EXTERNAL_FLAT_CONDITIONS = ["svGa", "svNa", "svPa", "spGa", "spNa", "spPa", "saGa", "saNa", "saPa"]
EXTERNAL_NPA_SUBGROUPS = ["DET-ADJ_G", "HEAD-DET_C", "HEAD-DET_G", "HEAD-DET_N"]

# This repo's own local, in-progress pipeline output -- unlike everything
# else in this script, which deliberately reads only the published site
# (see this module's docstring). Used for exactly one thing: the raw
# per-sample "treebank" column the published site's LANGUAGES.six never
# carries (see local_treebank_samples() below). Coverage here is real but
# partial -- comprehensive for npa, sparse for the flat sv/sp/sa conditions
# (as of writing: svNa 7 languages, saGa/spPa 0) -- so anything built from
# it must say so, not imply the same completeness as the rest of this page.
LOCAL_ROOT = OUTPUT_DECISION_TREES_DIR

# The published site itself is inconsistent about this one language's name
# across condition dumps (ASCII "aa" digraph vs. the real "å"), which
# otherwise splits Norwegian Bokmål's counts into two separate rows
# everywhere downstream (language table, heatmap, scatter). Checked every
# other language name for the same aa/oe/ae-digraph collision -- this is the
# only one that occurs in the current data.
NAME_ALIASES = {
    "Norwegian Bokmaal": "Norwegian Bokmål",
}

BUCKET_LABELS = {
    "no_candidates": "No candidate",
    "no_inflections": "No inflection",
    "same_forms": "Same inflection",
    "same_features": "Same feature",
    "undefined_features": "Undefined feature",
    "ambiguous_subjects": "Ambiguous subject",
}

# (group, label) per condition the published site covers -- the flat
# sv/sp/sa x Na/Ga/Pa conditions from the shared taxonomy, plus NPA's own
# aggregate condition (no single flat id there; see FLAT_CONDITION_META's
# own docstring). Mirrors src/word_order/viz_overview.py's _classify_deprel,
# which classifies the same condition ids from the same shared module.
CONDITION_META = {
    **FLAT_CONDITION_META,
    "npa": ("Noun Phrase", "Agreement (all role pairs)"),
}


# diag.buckets keys in the published site's embedded JSON (see
# sva_trees.diagnostics._BUCKET_JSON_KEYS, which produced them) -> this
# script's own bucket keys.
PUBLISHED_BUCKET_KEYS = {
    "no_match": "no_candidates",
    "no_inflection": "no_inflections",
    "same_inflection": "same_forms",
    "same_feature": "same_features",
    "undefined_feature": "undefined_features",
    "ambiguous_subject": "ambiguous_subjects",
}

LANGUAGES_RE = re.compile(r"const LANGUAGES = \{\s*six:\s*(\[.*?\]),\s*binary:\s*(\[.*?\])\s*\};", re.S)


def _load_languages_six(index_html_path: str) -> list[dict] | None:
    if not os.path.exists(index_html_path):
        return None
    html = open(index_html_path, encoding="utf-8").read()
    match = LANGUAGES_RE.search(html)
    if not match:
        return None
    return json.loads(match.group(1))


def _record(entry: dict, cond: str, subgroup: str) -> dict:
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
        # Decision-tree accuracy for this language x condition -- from the
        # same deprel/"Language Overview" pages this script's sibling
        # index-overview page links out to (see viz_deprel.py/html_deprel.py).
        # None (not 0) when the site never reports one, so weighted averages
        # below can skip it rather than silently treating "no accuracy" as
        # "zero accuracy".
        acc=entry.get("acc"),
        url=(EXTERNAL_SITE_ORIGIN + lang_url) if lang_url else None,
        # Distinct lemmas/surface forms seen for this language x condition --
        # a rough lexical-diversity/treebank-richness signal, independent of
        # how many of those forms actually became usable minimal pairs.
        n_lemma=diag.get("nLemma") or 0, n_form=diag.get("nForm") or 0,
        # Rows dropped *before* the decision tree was even fit, because the
        # target feature was missing on the head, the subject, or both (see
        # sva_trees.create_pairs's head_unk/nsubj_unk/both_unk) -- a
        # different, earlier population than the six outcome buckets above,
        # which only cover rows that made it to an attempted swap.
        head_unk=diag.get("headUnk") or 0, nsubj_unk=diag.get("nsubjUnk") or 0,
        both_unk=diag.get("bothUnk") or 0,
        buckets=buckets,
    )


def _tried_languages(dir_path: str) -> set[str]:
    """Every language the pipeline actually attempted for one condition/
    NPA-subgroup directory: one <Language>.html per language it ran
    (index.html aside), whether or not that run produced anything -- a
    language with zero raw candidates still gets a "placeholder" page (see
    word_order.viz_deprel), just without an entry in LANGUAGES.six. That
    page-per-language count is the only way to recover the true
    attempted-but-empty set; LANGUAGES.six alone undercounts (e.g. svNa: 186
    pages tried, only 131 made it into the blob).
    """
    if not os.path.isdir(dir_path):
        return set()
    return {
        NAME_ALIASES.get(name, name)
        for name in (
            fn[:-len(".html")].replace("_", " ")
            for fn in os.listdir(dir_path)
            if fn.endswith(".html") and fn != "index.html"
        )
    }


def collect_records() -> tuple[list[dict], dict[str, set[str]]]:
    if not os.path.isdir(EXTERNAL_ROOT):
        raise FileNotFoundError(
            f"Published site not found at {EXTERNAL_ROOT} -- this script only reads from it now."
        )

    records = []
    tried = {}
    for cond in EXTERNAL_FLAT_CONDITIONS:
        cond_dir = os.path.join(EXTERNAL_ROOT, cond)
        tried[cond] = _tried_languages(cond_dir)
        six = _load_languages_six(os.path.join(cond_dir, "index.html"))
        if six:
            records += [_record(e, cond, cond) for e in six]

    for sub in EXTERNAL_NPA_SUBGROUPS:
        sub_dir = os.path.join(EXTERNAL_ROOT, "npa", sub)
        tried[sub] = _tried_languages(sub_dir)
        six = _load_languages_six(os.path.join(sub_dir, "index.html"))
        if six:
            records += [_record(e, "npa", sub) for e in six]

    return records, tried


def _local_parquet_paths(cond: str, is_npa_sub: bool) -> list[str]:
    # Flat sv/sp/sa conditions nest one level deeper than npa's own
    # convention -- decision_trees/{cond}/{cond}_nsubj/*.parquet vs.
    # decision_trees/npa/{subgroup}/*.parquet -- both observed directly on
    # disk (see LOCAL_ROOT's comment); {cond}_keepunk variants and anything
    # outside this exact pattern are deliberately not globbed for, so an
    # experimental/trial run never gets silently folded into a real
    # condition's counts.
    if is_npa_sub:
        pattern = os.path.join(LOCAL_ROOT, "npa", cond, "*.parquet")
    else:
        pattern = os.path.join(LOCAL_ROOT, cond, f"{cond}_nsubj", "*.parquet")
    return sorted(glob(pattern))


def local_treebank_samples() -> list[dict]:
    """Per-sample treebank attribution from this repo's own local pipeline
    output -- the one thing the published site's data never carries (see
    LOCAL_ROOT's comment on why this is the sole exception to "only read the
    published site"). Real counts, not estimates: each row in one of these
    parquet files is one raw candidate sample, tagged with the exact UD
    treebank it came from.

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
    out = []
    for cond, (group, label) in FLAT_CONDITION_META.items():
        for path in _local_parquet_paths(cond, is_npa_sub=False):
            lang = NAME_ALIASES.get(
                os.path.splitext(os.path.basename(path))[0].replace("_", " "),
                os.path.splitext(os.path.basename(path))[0].replace("_", " "),
            )
            counts = pd.read_parquet(path, columns=["treebank"])["treebank"].value_counts()
            out.append({
                "language": lang, "condition_id": cond, "condition_label": f"{label} ({group})",
                "group": group, "samples_by_treebank": counts.to_dict(), "total_samples": int(counts.sum()),
            })
    for sub in EXTERNAL_NPA_SUBGROUPS:
        for path in _local_parquet_paths(sub, is_npa_sub=True):
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


def build():
    records, tried_by_key = collect_records()

    languages = sorted(set(r["lang"] for r in records))
    conditions = sorted(CONDITION_META, key=lambda c: (CONDITION_META[c][0], CONDITION_META[c][1]))
    # Any condition not yet in CONDITION_META still shows up (grouped as
    # "Other") rather than silently dropping real data.
    unknown = sorted(set(r["cond"] for r in records) - set(CONDITION_META))
    for cond in unknown:
        CONDITION_META[cond] = ("Other", cond)
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
    # Rows dropped before the decision tree was even fit (missing feature
    # annotation on the head, the subject, or both) -- a different, earlier
    # population than `funnel` above, which is scoped to attempted swaps
    # only. See _record()'s head_unk/nsubj_unk/both_unk comment.
    dropped_before_fitting = defaultdict(int)
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

        dropped_before_fitting["head_unk"] += r["head_unk"]
        dropped_before_fitting["nsubj_unk"] += r["nsubj_unk"]
        dropped_before_fitting["both_unk"] += r["both_unk"]
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
    # from for this language), not a per-sample/per-pair count. The
    # published-site source this whole script otherwise reads never carries
    # a per-sample treebank field (it only shows up in a handful of
    # illustrative example rows, not a full accounting); local_treebank_
    # samples() below fills that gap from this repo's own local runs instead
    # -- see LOCAL_ROOT's comment on why that's the one exception to
    # "published site only" and how partial its coverage still is.
    all_tried_langs = set().union(*tried_by_key.values()) if tried_by_key else set()
    lang_treebanks_raw = get_lang_treebanks(RESOURCE_DIR)

    # Cross-referenced against this same run's own published pairs/samples
    # for that exact language x condition, so the UI can show "these N
    # locally-sampled rows are part of the M samples/P pairs already
    # published for this combo" instead of the local figure floating
    # unanchored from the rest of the page's numbers.
    local_treebank_pairs = local_treebank_samples()
    for entry in local_treebank_pairs:
        is_sub = entry["condition_id"].startswith("npa/")
        key = entry["condition_id"].split("/", 1)[1] if is_sub else entry["condition_id"]
        pairs_lookup = npa_matrix_pairs if is_sub else matrix_pairs
        samples_lookup = npa_matrix_samples if is_sub else matrix_samples
        entry["published_pairs"] = pairs_lookup.get((entry["language"], key), 0)
        entry["published_samples"] = samples_lookup.get((entry["language"], key), 0)

    def dropped_dict(agg: dict) -> dict:
        # Same shape/labels as the top-level "dropped_before_fitting" below,
        # scoped to one condition/subgroup's own agg dict instead of the
        # whole corpus -- lets a category tab show its own breakdown rather
        # than only ever the all-10-conditions-combined global one.
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
        # Preview, not a full breakdown -- see local_treebank_samples()'s
        # docstring. Only the language x condition combinations this repo's
        # own local pipeline has already processed appear here at all.
        "local_treebank_pairs": local_treebank_pairs,
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
        # Separate from `funnel` -- see _record()'s comment. All-10-
        # conditions-combined; each condition/subgroup below also carries
        # its own "dropped_before_fitting" at that narrower scope.
        "dropped_before_fitting": {
            "Head unknown": dropped_before_fitting["head_unk"],
            "Subject unknown": dropped_before_fitting["nsubj_unk"],
            "Both unknown": dropped_before_fitting["both_unk"],
        },
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

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {OUT_PATH}")
    print(f"  {out['totals']}")


if __name__ == "__main__":
    build()
