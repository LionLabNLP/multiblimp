import os
from glob import escape as glob_escape, glob
from pathlib import Path
from urllib.parse import quote
from typing import Dict, Tuple
import html as html_lib
import re
import json

import joblib
import pandas as pd
import pyarrow.parquet as pq
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from agreement_trees.fit.decision_tree import UNK_LABELS
from agreement_trees.config import load_agreement_config, load_scan_results
from agreement_trees.fit.scan import (
    cache_file, cache_lang, cache_state, same_thresholds, scan_thresholds, unit_key,
)
from agreement_trees.pipeline_utils import read_label_distribution

from agreement_trees.fit.entropy import (
    chunked_leaf_ids,
    calculate_base_accuracy, pooled_leaf_accuracy, smoothed_accuracy, DEFAULT_LEAF_MIN_ACCURACY,
)
from agreement_trees.report.html.html_deprel import create_html
from agreement_trees.utils import is_agreement_predictor
from multiblimp.config import RESOURCE_DIR
from multiblimp.excluded_treebanks import excluded_treebanks
from multiblimp.languages import get_ud_langs

def _fitted_labels(dist: dict) -> set[str]:
    """The labels of a {label: count} distribution a tree fit would see (non-zero, unk labels dropped)."""
    return {str(k).lower() for k, v in dist.items() if v and str(k).lower() not in UNK_LABELS}


def _yes_count(dist: dict) -> int:
    return int(sum(v for k, v in dist.items() if str(k).lower() == "yes"))


# the chevron glyph of html_deprel's JS chevronSvg(), inlined for this server-side note
_CHEVRON_SVG = (
    '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M6 4l4 4-4 4" stroke="currentColor" stroke-width="1.75" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)


def _notes_disclosure(summary: str, body: str) -> str:
    """<details> collapse for a note that can get long (an omitted-languages list), with a chevron like the
    per-language one."""
    return (
        '<details class="notes-disclosure"><summary>'
        f'<span class="chevron">{_CHEVRON_SVG}</span>{html_lib.escape(summary)}'
        f'</summary><div class="notes-disclosure-body">{body}</div></details>'
    )


# order of the reasons in the "not in the table" list
LEFT_OUT_REASONS = (
    "too few agreeing", "too few agreeing after cleaning", "too few to fit a tree",
    "no agreement annotation", "no data",
)


def _label_counts(dist: dict) -> tuple[int, int, int]:
    """(agree, disagree, unknown) of a {label: count} distribution."""
    yes = no = 0
    for k, v in dist.items():
        if str(k).lower() == "yes":
            yes += int(v)
        elif str(k).lower() == "no":
            no += int(v)
    return yes, no, int(sum(dist.values())) - yes - no


def _scan_failed_langs(langs: set, scan: dict, scan_key: str | None, scan_cfg: dict) -> set:
    """Of `langs`, those a current saved scan covered without passing them for this target: an older fit of such a
    language stays on disk. Nothing, if the scan is missing or was made with other settings."""
    if not scan_key or not same_thresholds(scan.get("thresholds"), scan_thresholds(scan_cfg)):
        return set()
    mode, family = ("vp", scan_key.split(":", 1)[0]) if ":" in scan_key else ("np", None)
    passed = scan.get("targets", {}).get(scan_key, {})
    return {lang for lang in langs if lang not in passed and unit_key(mode, family, lang) in scan.get("scanned", {})}


# accounting groups in display order: {label: what the group means}, "in the table" being the page's own table
ACCOUNTING_GROUPS = {
    "in the table": None,
    "weak evidence": "enough agreeing rows, but too small a share of the rows tagged on both sides to count as agreement",
    "not enough annotation": "too few rows tagged on both sides, or too few agreeing",
    "too few after cleaning": "passed the agreement check, too little left after filtering for the fit",
    "no samples": "no rows for this condition in the treebanks used",
    "not processed": "no current cache, scan or fit; this group should be empty",
}
WEAK_EVIDENCE_REASONS = {"agreement rate too low"}


def _accounting(table_langs: set, fit_left_out: set, no_data_langs: dict, scan: dict, scan_key: str | None,
                config: dict) -> dict[str, list[str]] | None:
    """{group: languages}, each language of the pipeline in exactly one group. None without a current scan."""
    if not scan_key or not same_thresholds(scan.get("thresholds"), scan_thresholds(config["scan"])):
        return None
    mode, family = ("vp", scan_key.split(":", 1)[0]) if ":" in scan_key else ("np", None)
    passed = scan.get("targets", {}).get(scan_key, {})
    failed = scan.get("failed", {}).get(scan_key, {})
    scanned = scan.get("scanned", {})
    ud_langs = {cache_lang(lang) for lang in get_ud_langs(RESOURCE_DIR)}
    # get_ud_langs leaves out a language with no treebank left
    all_excluded = {cache_lang(lang) for lang in excluded_treebanks} - ud_langs
    named = {cache_lang(lang) for lang in config["langs"]}
    universe = (named or ud_langs | all_excluded) | table_langs | fit_left_out | set(passed) | set(failed)

    def group(lang):
        if lang in table_langs:
            return "in the table"
        if lang in passed:
            return "too few after cleaning" if lang in fit_left_out else "not processed"
        if lang in failed:
            return "weak evidence" if failed[lang]["reason"] in WEAK_EVIDENCE_REASONS else "not enough annotation"
        if lang in all_excluded or lang in no_data_langs:
            return "no samples"
        state = cache_state(mode, family, lang)
        if state == "empty":
            return "no samples"
        if state != "cached" or scanned.get(unit_key(mode, family, lang), -1) < os.path.getmtime(cache_file(mode, family, lang)):
            return "not processed"
        return "not enough annotation"

    groups = {label: [] for label in ACCOUNTING_GROUPS}
    for lang in sorted(universe):
        groups[group(lang)].append(lang)
    return groups


def _left_out_rows(omit_langs: dict, no_data_langs: dict, data_dir: str, scan: dict,
                   scan_key: str | None, shown: set, min_yes: int, scan_failed: set) -> list[dict]:
    """One row per language kept out of the table: {lang, yes, no, unk, reason}. Counts come from the
    language's full label distribution file where there is one; scan failures from the scan, for languages
    that are neither listed nor in the table."""
    passed = set(scan.get("targets", {}).get(scan_key, {}))
    failed = scan.get("failed", {}).get(scan_key, {})

    def failed_reason(lang):
        return f"failed agreement check: {failed[lang]['reason']}" if lang in failed else "failed agreement check"

    rows = []
    for lang, dist in omit_langs.items():
        yes, no, unk = _label_counts(read_label_distribution(data_dir, lang) or dist)
        if not yes + no:
            reason = "no agreement annotation"
        elif lang in scan_failed:
            reason = failed_reason(lang)
        elif yes < min_yes:
            reason = "too few agreeing after cleaning" if lang in passed else "too few agreeing"
        else:
            reason = "too few to fit a tree"
        rows.append({"lang": lang, "yes": yes, "no": no, "unk": unk, "reason": reason})
    for lang in no_data_langs:
        rows.append({"lang": lang, "yes": None, "no": None, "unk": None, "reason": "no data"})
    listed = {r["lang"] for r in rows} | shown
    for lang, c in failed.items():
        if lang not in listed:
            rows.append({"lang": lang, "yes": c["yes"], "no": c["no"], "unk": c["total"] - c["yes"] - c["no"],
                         "reason": failed_reason(lang)})
    order = {r: i for i, r in enumerate(LEFT_OUT_REASONS)}
    return sorted(rows, key=lambda r: (order.get(r["reason"], len(order)), r["reason"], r["lang"]))


def _left_out_export(rows: list[dict], html_files: dict, pairs_dir: str | None, diagnostics_by_lang: dict,
                     scan_failed: set) -> list[dict]:
    """The rows the dataset overview counts (agreeing rows or pairs, fit under the current scan), as LANGUAGES
    entries without a tree."""
    entries = []
    for r in rows:
        lang, yes = r["lang"], r["yes"] or 0
        pairs_fn = os.path.join(pairs_dir, lang, "correct_swaps.parquet") if pairs_dir else None
        n_pairs = pq.ParquetFile(pairs_fn).metadata.num_rows if pairs_fn and os.path.exists(pairs_fn) else 0
        attempted = lang in html_files and lang not in scan_failed
        page = html_files.get(lang)
        entries.append({
            "name": lang.replace("_", " "), "reason": r["reason"], "attempted": attempted,
            "langUrl": quote(page.name) if page else None, "nRaw": yes, "nPairs": n_pairs,
            "labelDistribution": {"yes": yes, "no": r["no"] or 0, "unk": r["unk"] or 0},
            "diag": {k: v for k, v in (diagnostics_by_lang.get(lang) or {}).items() if k != "examples"},
        })
    return entries


def _left_out_disclosure(rows: list[dict], html_files: dict, accounting: dict | None = None) -> str:
    """The collapsed list of languages not in the table, one compact table with a language linked to its page
    where it has one. With `accounting`, the summary is the accounting line and the table is grouped by it."""
    if not rows and not accounting:
        return ""

    def num(n):
        return "–" if n is None else f"{n:,}"

    def lang_cell(lang):
        name = html_lib.escape(lang.replace("_", " "))
        page = html_files.get(lang)
        return f'<a href="{quote(page.name)}">{name}</a>' if page else name

    def row_html(r):
        return (
            f"<tr><td>{lang_cell(r['lang'])}</td><td class=\"num\">{num(r['yes'])}</td><td class=\"num\">{num(r['no'])}</td>"
            f"<td class=\"num\">{num(r['unk'])}</td><td>{html_lib.escape(r['reason'])}</td></tr>"
        )

    if accounting is None:
        body = [row_html(r) for r in rows]
    else:
        body = []
        by_lang = {r["lang"]: r for r in rows}
        for label, meaning in ACCOUNTING_GROUPS.items():
            langs = accounting[label]
            if not meaning or not langs:
                continue
            body.append(f'<tr><th colspan="5">{len(langs)} {label}: {html_lib.escape(meaning)}</th></tr>')
            body += [row_html(by_lang[lang]) for lang in langs if lang in by_lang]
            names = [lang_cell(lang) for lang in langs if lang not in by_lang]
            if names:
                body.append(f'<tr><td colspan="5">{", ".join(names)}</td></tr>')
    table = (
        '<table class="left-out-table"><thead><tr><th>Language</th><th class="num">Agree</th>'
        '<th class="num">Disagree</th><th class="num">Unknown</th><th>Reason</th></tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>'
    )
    if accounting is not None:
        return _notes_disclosure(" · ".join(f"{len(accounting[label])} {label}" for label in ACCOUNTING_GROUPS), table)
    n = len(rows)
    return _notes_disclosure(f"{n} language{'s' if n != 1 else ''} not in the table — click to expand", table)


def _dtype_coerced_metrics(dt, df, target_col, smoothing, eval_cache=None, cache_key=None):
    """Coerce df's feature columns to the fitted pipeline's dtypes (in place), then (base, reduced, delta,
    accuracy): smoothed accuracies, each smoothed once over the whole dataset (base = the root's, reduced =
    the tree's, delta = tree - base, never negative); `accuracy` is the tree's raw training accuracy.
    eval_cache: dict shared across calls so leaf ids and accuracy are computed once per (cache_key, df).
    """
    cached = eval_cache.get(cache_key) if eval_cache is not None else None
    if cached is not None and cached[0] is df:
        _, leaf_ids, accuracy = cached
        base_v = calculate_base_accuracy(df, target_col, smoothing=smoothing)
        reduced_v = pooled_leaf_accuracy(leaf_ids, df, target_col, smoothing=smoothing)
        return base_v, reduced_v, reduced_v - base_v, accuracy

    # ensure dtype matching of loaded data and classifier
    cat_cols = [
        col
        for name, _, cols in dt.named_steps["preprocessor"].transformers_
        if name == "cat"
        for col in cols
        if col in df.columns
    ]
    num_cols = [
        col
        for name, _, cols in dt.named_steps["preprocessor"].transformers_
        if name == "num"
        for col in cols
        if col in df.columns
    ]
    df[cat_cols] = df[cat_cols].astype(str)
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    base_v = calculate_base_accuracy(df, target_col, smoothing=smoothing)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    if [name for name, _ in dt.steps] == ["preprocessor", "clf"]:
        # one transform serves both the leaf assignment and the accuracy
        leaf_ids, hits = chunked_leaf_ids(dt, X, y)
        accuracy = hits / len(X) if len(X) else float("nan")
        if eval_cache is not None:
            eval_cache[cache_key] = (df, leaf_ids, accuracy)
    else:
        leaf_ids, _ = chunked_leaf_ids(dt, X)
        accuracy = dt.score(X, y)
    reduced_v = pooled_leaf_accuracy(leaf_ids, df, target_col, smoothing=smoothing)

    return base_v, reduced_v, reduced_v - base_v, accuracy


def _needed_columns(model, parquet_fn, target_col):
    """The parquet columns the index reads: the model's input features, the target, and the stats columns
    (the fit frames are very wide and mostly unused here)."""
    names = pq.ParquetFile(parquet_fn).schema_arrow.names
    wanted = {target_col, "leaf_id", "leaf_top1_acc", "leaf_decision"}
    wanted.update(getattr(model.named_steps["preprocessor"], "feature_names_in_", names))
    return [n for n in names if n in wanted]


def _metrics_df(metrics: list, columns: list) -> pd.DataFrame:
    """pd.DataFrame(metrics).sort_values("language"), safe for an empty list (every language trivial, e.g.
    French participles for Person)."""
    if not metrics:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(metrics).sort_values("language")


def _agreement_row_stats(df, target_col, leaf_threshold, pairs_dir, lang_name, data_dir=None):
    """(n_raw, n_keep, n_pairs, retried) for one language's agreement dataframe.

    n_raw: rows with target_col's positive label (candidates for a feature
        swap), labelled "yes" in both SVA and NPA, over the language's whole data (its label
        distribution file in data_dir, before the train/test split); df's own count without one.
    n_keep: of those, how many pass the leaf_top1_acc > leaf_threshold and leaf_decision filter
        create_pairs uses to pick which rows to re-inflect (held-out rows included): the pairs'
        meta.json num_rows_keep, else recounted on df (pairs written before it existed). Trivial languages (no fitted
        tree, hence no leaf_top1_acc/leaf_decision columns) are single-class by
        construction, so every raw row counts as kept.
    n_pairs: rows actually turned into a re-inflected minimal pair by create_pairs,
        read from "<pairs_dir>/<language>/correct_swaps.parquet". 0 if pairs_dir is
        None or that file doesn't exist yet (create_pairs not run, or n_keep was 0).
    retried: whether agreement_trees.pairs.second_chance attempted a laxer retry for this
        language, regardless of whether it produced any pairs (meta.json's
        "second_pass", set only on that attempt -- see agreement_trees.vp.pipeline).
    """
    is_yes = df[target_col] == "yes"
    dist = read_label_distribution(data_dir, lang_name) if data_dir is not None else {}
    n_raw = _yes_count(dist) if dist else int(is_yes.sum())

    # A language's own meta.json threshold wins over the pipeline-wide one, so
    # a language that got a second, laxer pass is counted at that threshold.
    retried = False
    meta = {}
    if pairs_dir is not None:
        meta_fn = os.path.join(pairs_dir, lang_name, "meta.json")
        if os.path.exists(meta_fn):
            with open(meta_fn) as f:
                meta = json.load(f)
            leaf_threshold = meta.get("leaf_min_acc", leaf_threshold)
            retried = bool(meta.get("second_pass"))

    if "num_rows_keep" in meta:
        n_keep = int(meta["num_rows_keep"])
    elif "leaf_top1_acc" in df.columns and "leaf_decision" in df.columns:
        # An older cache's meta.json can store {leaf_id: that
        # leaf's own cutoff} instead of one number; JSON round-trips the keys
        # as strings, so normalize back to int before mapping onto leaf_id.
        row_threshold = (
            df["leaf_id"].map({int(k): v for k, v in leaf_threshold.items()})
            if isinstance(leaf_threshold, dict) else leaf_threshold
        )
        keep = (df["leaf_top1_acc"] > row_threshold) & df["leaf_decision"]
        n_keep = int((keep & is_yes).sum())
    else:
        n_keep = n_raw

    n_pairs = 0
    if pairs_dir is not None:
        pairs_fn = os.path.join(pairs_dir, lang_name, "correct_swaps.parquet")
        if os.path.exists(pairs_fn):
            n_pairs = pq.ParquetFile(pairs_fn).metadata.num_rows

    return n_raw, n_keep, n_pairs, retried


def calculate_agreement_metrics(
    language_data: Dict[str, Tuple[Pipeline, pd.DataFrame]],
    target_col: str,
    leaf_threshold: float = DEFAULT_LEAF_MIN_ACCURACY,
    pairs_dir: str | None = None,
    smoothing: float = 0.5,
    eval_cache: dict | None = None,
    data_dir: str | None = None,
) -> pd.DataFrame:
    """Per-language metrics of the agreement pages (n_raw/n_keep/n_pairs: see _agreement_row_stats).

    Returns:
        DataFrame with columns: language, base_acc, tree_acc, gain_acc, accuracy,
                                test_acc, test_n, n_raw, n_keep, n_pairs, retried
        (base_acc/tree_acc are smoothed once -- see _dtype_coerced_metrics;
        accuracy is the raw training accuracy, test_acc the raw held-out one.)
    """
    metrics = []

    for lang_name, (dt, df) in tqdm(language_data.items()):
        base_acc, tree_acc, gain_acc, accuracy = _dtype_coerced_metrics(
            dt, df, target_col, smoothing, eval_cache=eval_cache, cache_key=lang_name,
        )
        # Held-out accuracy from the split the tree was fit with (fit_dt
        # attaches test_eval_); NaN for a model without one.
        test_eval = getattr(dt, "test_eval_", None)
        if test_eval is not None and len(test_eval):
            test_acc, test_n = float((test_eval["y"] == test_eval["pred"]).mean()), int(len(test_eval))
        else:
            test_acc, test_n = float("nan"), 0

        n_raw, n_keep, n_pairs, retried = _agreement_row_stats(
            df, target_col, leaf_threshold, pairs_dir, lang_name, data_dir=data_dir,
        )

        metrics.append(
            {
                "language": lang_name,
                "base_acc": base_acc,
                "tree_acc": tree_acc,
                "gain_acc": gain_acc,
                "accuracy": accuracy,
                "test_acc": test_acc,
                "test_n": test_n,
                "n_raw": n_raw,
                "n_keep": n_keep,
                "n_pairs": n_pairs,
                "retried": retried,
            }
        )

    return _metrics_df(metrics, [
        "language", "base_acc", "tree_acc", "gain_acc", "accuracy", "test_acc", "test_n",
        "n_raw", "n_keep", "n_pairs", "retried",
    ])


def scatter_color(metrics_row, language_data, target_col, exclude_labels=None):
    """Yellow if all labels are in exclude_labels, blue otherwise."""
    if exclude_labels:
        lang_name = metrics_row["language"]
        if lang_name not in language_data:
            return "#2563eb"
        _, df = language_data[lang_name]
        labels = set(df[target_col].unique())
        if labels <= set(exclude_labels):
            return "#e5c64d"
    return "#2563eb"


def extract_trivial_distribution(html_path: Path) -> dict[str, int] | None:
    """Extract the {value: count} label distribution write_placeholder_html embeds
    as a <script id="label-distribution"> JSON island, or None if this isn't a
    placeholder page (e.g. it's a real fitted-tree page).

    A trivial/too-small language never gets a fitted tree, so it has no
    data_dir/{lang}.parquet for row-count fallbacks to read (fit_dt is what
    writes that file) — this is the only place that distribution survives.
    """
    content = html_path.read_text(encoding="utf-8")
    if "No decision tree to display" not in content:
        return None
    match = re.search(
        r'<script type="application/json" id="label-distribution">(.*?)</script>',
        content, re.DOTALL,
    )
    if not match:
        return None
    return {str(k): int(v) for k, v in json.loads(match.group(1)).items()}



def generate_html_deprel_index(
    data_dir: str,
    html_directory: str,
    language_data: Dict[str, Tuple[Pipeline, pd.DataFrame]] | None = None,
    target_col: str = "deprel_order",
    smoothing: float = 0.5,
    exclude_labels: set | None = None,
    leaf_threshold: float = DEFAULT_LEAF_MIN_ACCURACY,
    pairs_dir: str | None = None,
    diagnostics_by_lang: dict | None = None,
    agreement_label: str = "Subject-Verb",
    head_role_label: str = "head",
    subject_label: str = "subject",
    nsubj_label: str = "nsubj",
    scan_key: str | None = None,
) -> None:
    """Write the deprel index.html: per-language metrics, scatter and, with diagnostics, bucket breakdowns.

    data_dir: where the fitted models and frames were saved (language_data overrides). html_directory: output
    directory. target_col: the agreement predictor. leaf_threshold and pairs_dir (create_pairs' save_to parent) give N KEEP / N PAIRS.
    exclude_labels: label sets that mark a language uninformative (omitted from the table).
    diagnostics_by_lang: {language: diagnostics_row_to_json dict}.
    agreement_label, head_role_label, subject_label, nsubj_label: prose for the subtitle, the head role
    ("Verb", "Aux") and the fixed role in the "ambiguous subject" bucket and "# nsubj unk" stat (cosmetic;
    NPA passes its own roles).
    scan_key: the target's key in the agreement scan ("sv:Person", "HEAD-DET_Number"): languages a current scan does
    not pass leave the table for the list under it, and its failed languages are listed there too. None skips both.
    """
    if not is_agreement_predictor(target_col):
        raise ValueError(f"{target_col!r} is not an agreement predictor")
    html_path = Path(html_directory)
    # the scan's own bar: fewer "yes" rows than min_yes leaves the table
    config = load_agreement_config()
    scan_cfg = config["scan"]
    min_yes = scan_cfg["min_yes"]

    # "../" reaches decision_trees/index.html from a page one directory below it; NPA nests an extra "npa/" level
    if "decision_trees" in html_path.parts:
        depth = len(html_path.parts) - html_path.parts.index("decision_trees") - 1
    else:
        depth = 1
    overview_href = "../" * depth
    # decision_trees/ sits directly under the site root, where the stats overview lives
    stats_overview_href = "../" * (depth + 1) + "data.html"

    # trivial / too-small languages from the placeholder pages (also "too few samples to fit a tree")
    trivial_langs = {}
    for html_file in html_path.glob("*.html"):
        if html_file.name.lower() == "index.html":
            continue
        dist = extract_trivial_distribution(html_file)
        if dist is not None:
            trivial_langs[html_file.stem] = dist  # ← stem string, not Path

    # Languages that produced literally zero raw instances (never got a
    # tree, pairs, or HTML page at all -- e.g. agreement_trees.vp.pipeline.Pipeline's
    # "Skipping {lang}, raw_df has no entries" console-only skip) -- see
    # Pipeline._write_no_data_markers. Scanned from data_dir (where the
    # marker is written, alongside the .joblib/.parquet files), not
    # html_path, since a no-data language has no HTML page to glob for.
    no_data_langs = {}
    for marker_fn in glob(os.path.join(glob_escape(data_dir), "*.no_data.json")):
        lang_name = Path(marker_fn).stem.removesuffix(".no_data")
        try:
            with open(marker_fn) as f:
                no_data_langs[lang_name] = json.load(f).get("reason", "no reason recorded")
        except (OSError, json.JSONDecodeError):
            continue

    # languages without a fitted tree (single label, too few rows) get a root-card page and no saved
    # .joblib/.parquet, so neither check above sees them: their label distribution file is the only trace
    saved_models = {Path(fn).stem for fn in glob(os.path.join(glob_escape(data_dir), "*.joblib"))}
    for dist_fn in glob(os.path.join(glob_escape(data_dir), "*_label_distribution.json")):
        lang_name = Path(dist_fn).name.removesuffix("_label_distribution.json")
        if lang_name in trivial_langs or lang_name in no_data_langs or lang_name in saved_models:
            continue
        trivial_langs[lang_name] = read_label_distribution(data_dir, lang_name)

    if language_data is None:
        language_data = {}

        plain_models = {Path(fn).stem: fn for fn in glob(os.path.join(glob_escape(data_dir), "*.joblib"))}

        for lang_name in tqdm(sorted(plain_models)):
            if lang_name in trivial_langs:
                continue

            model_fn = plain_models[lang_name]
            parquet_fn = os.path.join(data_dir, lang_name + ".parquet")

            if not os.path.exists(parquet_fn):
                continue

            model = joblib.load(model_fn)
            df = pd.read_parquet(parquet_fn, columns=_needed_columns(model, parquet_fn, target_col))
            language_data[lang_name] = (model, df)

    eval_cache = {}
    metrics_six = calculate_agreement_metrics(
        language_data, target_col, leaf_threshold=leaf_threshold,
        pairs_dir=pairs_dir, smoothing=smoothing, eval_cache=eval_cache, data_dir=data_dir,
    )
    metrics_binary = metrics_six

    # a single observed class (smoothed base accuracy never reaches 1.0, so check the labels)
    for l in metrics_six["language"]:
        if language_data[l][1][target_col].astype(str).nunique() <= 1:
            trivial_langs[l] = dict(language_data[l][1][target_col].value_counts())

    # trivial-distribution languages are omitted from the table and listed in the "not in the table" list
    omit_langs = dict(trivial_langs)

    # an all-"yes" language above the undersized bar is real data with nothing to fit: keep it,
    # scored as a trivial tree (root = tree = smoothed 100%)
    # unk rows are dropped before the fit, so they don't count against "all yes"
    single_yes = {
        lang: dist for lang, dist in trivial_langs.items()
        if _fitted_labels(dist) == {"yes"}
        and _yes_count(dist) >= min_yes
    }
    for lang in single_yes:
        omit_langs.pop(lang, None)
    have = set(metrics_six["language"])
    extra = []
    for lang, dist in single_yes.items():
        if lang in have:
            continue
        n_raw = _yes_count(dist)
        n_pairs = 0
        if pairs_dir is not None:
            pairs_fn = os.path.join(pairs_dir, lang, "correct_swaps.parquet")
            if os.path.exists(pairs_fn):
                n_pairs = pq.ParquetFile(pairs_fn).metadata.num_rows
        acc = smoothed_accuracy(n_raw, n_raw, smoothing)
        extra.append({
            "language": lang, "base_acc": acc, "tree_acc": acc, "gain_acc": 0.0,
            "accuracy": 1.0, "test_acc": float("nan"), "test_n": 0, "n_raw": n_raw,
            "n_keep": n_raw, "n_pairs": n_pairs, "retried": False,
        })
    if extra:
        metrics_six = pd.concat([metrics_six, pd.DataFrame(extra)], ignore_index=True)
        metrics_six = metrics_six.sort_values("language")
        metrics_binary = metrics_six

    metrics_six = metrics_six[~metrics_six["language"].isin(omit_langs.keys())]
    metrics_binary = metrics_binary[~metrics_binary["language"].isin(omit_langs.keys())]

    # Pull undersized languages (real variation, even a real fitted tree
    # in some cases -- just too few "Yes" rows to mean much) out of the
    # table too, same as the trivial ones above: folded into omit_langs
    # so they share that one note/list instead of a separate per-row
    # badge (read_label_distribution works for any language that
    # reached write_label_distribution, fitted tree or not). A language
    # with real entropy (never "trivial" at all, e.g. a thin-but-real
    # split like Akuntsu/Assyrian) only ever shows up as an ordinary
    # metrics_six row, never in trivial_langs -- this check has to run
    # here, against metrics_six directly, to catch those too.
    undersized_mask = metrics_six["n_raw"].between(1, min_yes - 1)
    undersized_langs = set(metrics_six.loc[undersized_mask, "language"])
    for lang in undersized_langs:
        omit_langs.setdefault(lang, read_label_distribution(data_dir, lang) or {})
    metrics_six = metrics_six[~metrics_six["language"].isin(undersized_langs)]
    metrics_binary = metrics_binary[~metrics_binary["language"].isin(undersized_langs)]

    # an older fit of a language the current scan does not pass: out of the table, whatever else it looks like
    scan = load_scan_results("vp" if ":" in scan_key else "np") if scan_key else {}
    scan_failed = _scan_failed_langs(
        set(metrics_six["language"]) | set(omit_langs), scan, scan_key, scan_cfg)
    for lang in scan_failed & set(metrics_six["language"]):
        omit_langs[lang] = read_label_distribution(data_dir, lang) or {}
    metrics_six = metrics_six[~metrics_six["language"].isin(scan_failed)]
    metrics_binary = metrics_binary[~metrics_binary["language"].isin(scan_failed)]

    # Find corresponding HTML files — stem string -> Path
    html_files = {
        f.stem: f for f in html_path.glob("*.html") if f.name.lower() != "index.html"
    }

    # Build color lookup before generating plot data
    lang_colors = {}
    for _, row in metrics_six.iterrows():
        lang_name = row["language"].replace("_", " ")
        lang_colors[lang_name] = scatter_color(
            row, language_data, target_col, exclude_labels=exclude_labels
        )

    # the diagnostics panel needs per-language diagnostics
    diagnostics_by_lang = diagnostics_by_lang or {}
    diagnostics_enabled = bool(diagnostics_by_lang)

    def build_languages(metrics_df, lang_colors):
        """Per-language dicts for the diagnostics page's client-side
        renderer (agreement_trees.report.html.html_deprel's LANGUAGES blob) — the same
        data the page shows, kept as JSON so colours
        (which read CSS custom properties for light/dark theming) and
        sorting can both happen client-side instead of being baked in here.
        A language with no diagnostics_by_lang entry (e.g. create_pairs
        raised on it) still
        gets a row, with "diag": null — the page's JS renders that as a "no
        diagnostics available" state rather than failing. labelDistribution
        is independent of "diag" (see read_label_distribution) so the
        Distribution column still works in that case.
        """
        languages = []
        for _, row in metrics_df.iterrows():
            lang_name = row["language"].replace("_", " ")
            lang_file = html_files.get(row["language"])  # ← stem matches directly
            color = lang_colors.get(lang_name, "#2563eb")

            languages.append({
                "name": lang_name,
                "langUrl": f"{quote(lang_file.name)}" if lang_file else None,
                "color": color if color != "#2563eb" else None,
                "base": row["base_acc"],
                "tree": row["tree_acc"],
                "gain": row["gain_acc"],
                "test": row["test_acc"],
                "testN": int(row["test_n"]),
                # raw training accuracy: not shown in this page's table any more,
                # but scripts/overview/build_stats.py reads it for the dataset overview
                "acc": row["accuracy"],
                "nRaw": int(row["n_raw"]),
                "retried": bool(row["retried"]),
                "labelDistribution": read_label_distribution(data_dir, row["language"]),
                "nKeep": int(row["n_keep"]),
                "nPairs": int(row["n_pairs"]),
                "diag": diagnostics_by_lang.get(row["language"]),
            })
        return languages

    languages_json = json.dumps(build_languages(metrics_six, lang_colors))

    # scatter plot data
    def generate_plot_data(metrics_df):
        data = []
        for _, row in metrics_df.iterrows():
            lang_name = row["language"].replace("_", " ")
            lang_file = html_files.get(row["language"])  # ← stem matches directly
            url = f"{quote(lang_file.name)}" if lang_file else None

            data.append(
                {
                    "name": lang_name,
                    "base": row["base_acc"],
                    "tree": row["tree_acc"],
                    # marker size / hover count: n_raw ("Yes" items)
                    "n_items": int(row["n_raw"]),
                    "url": url,
                    "color": lang_colors.get(lang_name, "#2563eb"),
                }
            )
        return data

    plot_data_json = json.dumps(generate_plot_data(metrics_six))

    # Build legend / notes
    color_note = ""
    if "#e5c64d" in lang_colors.values():
        color_note = (
            '<p class="trivial-note">'
            '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
            'background:#e5c64d;margin-right:6px;vertical-align:middle;"></span>'
            "Languages shown in <strong>yellow</strong> only exhibit uninformative agreement "
            "(labels <code>--</code> and <code>+-</code>) — no Yes/No contrast was observed.</p>"
        )

    left_out = _left_out_rows(omit_langs, no_data_langs, data_dir, scan, scan_key,
                              shown=set(metrics_six["language"]), min_yes=min_yes, scan_failed=scan_failed)
    accounting = _accounting(set(metrics_six["language"]), set(omit_langs) | set(no_data_langs),
                             no_data_langs, scan, scan_key, config)
    notes = color_note + _left_out_disclosure(left_out, html_files, accounting)
    left_out_json = json.dumps(_left_out_export(left_out, html_files, pairs_dir, diagnostics_by_lang, scan_failed))

    html_content = create_html(
        plot_data_json,
        trivial_note=notes,
        languages_json=languages_json,
        left_out_json=left_out_json,
        show_diagnostics=diagnostics_enabled,
        leaf_threshold=leaf_threshold,
        agreement_label=agreement_label,
        head_role_label=head_role_label,
        subject_label=subject_label,
        nsubj_label=nsubj_label,
        overview_href=overview_href,
        stats_overview_href=stats_overview_href,
    )

    output_path = html_path / "index.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
