import os
from glob import glob
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

from multiblimp.agreement_pipeline_utils import read_label_distribution, wilson_lower_bound

from .entropy import (
    calculate_base_entropy, calculate_tree_entropy, leaf_weighted_entropy,
    calculate_base_accuracy, pooled_leaf_accuracy, DEFAULT_LEAF_MIN_ACCURACY,
)
from .html.html_deprel import create_html
from .utils import is_agreement_predictor

# A language/condition with SOME "Yes"/"Agree" candidates (n_raw > 0) but
# fewer than this many is pulled out of the table entirely and folded into
# the trivial-note list instead (see the undersized_langs block below) --
# same threshold html_deprel.py's client-side MIN_N_FOR_COLOR already fades
# N KEEP/N PAIRS's coverage colour at, for the same reason (too few rows for
# a per-language read to mean much).
UNDERSIZED_MAX_N_RAW = 10


def is_undersized(n_raw: int) -> bool:
    return 0 < n_raw < UNDERSIZED_MAX_N_RAW


# Same default scripts/sva_trees/agreement_candidates.py's own config scan
# uses (passes_agreement_bar's wilson_floor) -- a language with plenty of
# raw "Yes" rows (>= UNDERSIZED_MAX_N_RAW, so not already undersized) can
# still have a Wilson lower bound this low on both/total if those rows are
# a vanishingly thin slice of its whole treebank (e.g. Naija: 16 Yes out of
# 9,634 rows -- a 0.1% lower bound). Distinct from undersized: this is
# "enough Yes in absolute terms, but too sparse relative to everything
# else to be confident it's real signal rather than incidental tagging".
SPARSE_WILSON_FLOOR = 0.01

# Same chevron glyph html_deprel.py's own JS chevronSvg() renders, inlined
# here since this note is built server-side in plain HTML, never through
# that JS.
_CHEVRON_SVG = (
    '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M6 4l4 4-4 4" stroke="currentColor" stroke-width="1.75" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)


def _notes_disclosure(summary: str, body: str) -> str:
    """<details>-based collapse for a note that can get long (an omitted-
    languages list, potentially 50+ entries) -- collapsed by default via
    plain HTML semantics (no JS needed), with a chevron matching html_
    deprel.py's own per-language one for visual consistency."""
    return (
        '<details class="notes-disclosure"><summary>'
        f'<span class="chevron">{_CHEVRON_SVG}</span>{html_lib.escape(summary)}'
        f'</summary><div class="notes-disclosure-body">{body}</div></details>'
    )


def _dtype_coerced_metrics(dt, df, target_col, binary_entropy, smoothing,
                           eval_cache=None, cache_key=None, accuracy_measure=False):
    """Coerce df's feature columns to match the fitted pipeline's expected dtypes
    (mutates df in place), then compute (base, reduced, delta, accuracy).

    accuracy_measure=False (word-order pages): base/reduced/delta are entropies
    (lower reduced = better, delta = base - reduced).
    accuracy_measure=True (agreement pages): base/reduced/delta are smoothed
    accuracies, both smoothed once over the whole dataset (+0.5 per class) --
    base = the root node's own smoothed accuracy, "reduced" is the tree's
    (sum of leaf majority counts + 0.5) / (N + 1), delta = tree - base
    (never negative). Per-leaf smoothing is deliberately not stacked here. `accuracy` is always the raw training
    accuracy of the fitted tree.

    Shared by calculate_metrics and calculate_agreement_metrics, which only
    differ in the extra per-language stats appended after this.

    eval_cache: optional dict shared across calls; the tree's leaf ids and
    accuracy don't depend on the measure, so they're computed once per
    (cache_key, df).
    """
    def base_of(d):
        if accuracy_measure:
            return calculate_base_accuracy(d, target_col, smoothing=smoothing)
        return calculate_base_entropy(d, target_col, binary=binary_entropy, smoothing=smoothing)

    def reduced_of(leaf_ids, d):
        if accuracy_measure:
            return pooled_leaf_accuracy(leaf_ids, d, target_col, smoothing=smoothing)
        return leaf_weighted_entropy(leaf_ids, d, target_col, binary=binary_entropy, smoothing=smoothing)

    def delta_of(base, reduced):
        return reduced - base if accuracy_measure else base - reduced

    cached = eval_cache.get(cache_key) if eval_cache is not None else None
    if cached is not None and cached[0] is df:
        _, leaf_ids, accuracy = cached
        base_v = base_of(df)
        reduced_v = reduced_of(leaf_ids, df)
        return base_v, reduced_v, delta_of(base_v, reduced_v), accuracy

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

    base_v = base_of(df)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    if [name for name, _ in dt.steps] == ["preprocessor", "clf"]:
        # one transform serves both the leaf assignment and the accuracy
        X_t = dt.named_steps["preprocessor"].transform(X)
        leaf_ids = dt.named_steps["clf"].apply(X_t)
        reduced_v = reduced_of(leaf_ids, df)
        # Calculate accuracy on full training data
        accuracy = dt.named_steps["clf"].score(X_t, y)
        if eval_cache is not None:
            eval_cache[cache_key] = (df, leaf_ids, accuracy)
    else:
        if accuracy_measure:
            leaf_ids = dt.named_steps["clf"].apply(dt.named_steps["preprocessor"].transform(X))
            reduced_v = reduced_of(leaf_ids, df)
        else:
            reduced_v = calculate_tree_entropy(
                dt, df, target_col, binary=binary_entropy, smoothing=smoothing
            )
        accuracy = dt.score(X, y)

    return base_v, reduced_v, delta_of(base_v, reduced_v), accuracy


def _metrics_df(metrics: list, columns: list) -> pd.DataFrame:
    """pd.DataFrame(metrics).sort_values("language"), but safe when `metrics`
    is empty -- e.g. every language for this deprel turned out trivial (no
    fitted tree at all), which happens for real: French participles don't
    inflect for Person, so head_nsubj_Person_agreement over VerbForm=Part
    heads is degenerate for every French row. pd.DataFrame([]) has no
    columns at all, so .sort_values("language") on it raises KeyError
    instead of just yielding an empty (but correctly shaped) table.
    """
    if not metrics:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(metrics).sort_values("language")


def calculate_metrics(
    language_data: Dict[str, Tuple[Pipeline, pd.DataFrame]],
    target_col: str = "deprel_order",
    binary_entropy: bool = False,
    smoothing: float = 0.5,
    eval_cache: dict | None = None,
) -> pd.DataFrame:
    """Calculate metrics for all languages.

    Args:
        language_data: Dict mapping language_name -> (fitted_pipeline, dataframe)
        target_col: Column name containing word order labels
        binary_entropy: If True, use binary entropy. If False, use six-class.
        smoothing: Smoothing factor for entropy calculation

    Returns:
        DataFrame with columns: language, base_entropy, reduced_entropy,
                                delta_entropy, accuracy, n_items, n_flexible, n_fully_flexible, total_pairs
    """
    metrics = []

    for lang_name, (dt, df) in tqdm(language_data.items()):
        base_ent, reduced_ent, delta_ent, accuracy = _dtype_coerced_metrics(
            dt, df, target_col, binary_entropy, smoothing,
            eval_cache=eval_cache, cache_key=lang_name,
        )

        # Number of items
        n_items = len(df)

        # Swap statistics (if num_swaps column exists -- only set_dt_features_
        # in_df's word-order branch, i.e. target is not None, ever adds it;
        # a target=None caller, e.g. NPA's pairwise agreement columns or
        # viz_tree's own "binary attachment classifier" case, never has it)
        if "num_swaps" in df.columns:
            n_flexible = (df["num_swaps"] >= 1).sum()
            n_fully_flexible = (df["num_swaps"] == 4).sum()
            total_pairs = sum(df["num_swaps"])
        else:
            n_flexible = 0
            n_fully_flexible = 0
            total_pairs = 0

        metrics.append(
            {
                "language": lang_name,
                "base_entropy": base_ent,
                "reduced_entropy": reduced_ent,
                "delta_entropy": delta_ent,
                "accuracy": accuracy,
                "n_items": n_items,
                "n_flexible": n_flexible,
                "n_fully_flexible": n_fully_flexible,
                "total_pairs": total_pairs,
            }
        )

    return _metrics_df(metrics, [
        "language", "base_entropy", "reduced_entropy", "delta_entropy", "accuracy",
        "n_items", "n_flexible", "n_fully_flexible", "total_pairs",
    ])


def _agreement_row_stats(df, target_col, leaf_threshold, pairs_dir, lang_name):
    """(n_raw, n_keep, n_pairs, retried) for one language's agreement dataframe.

    n_raw: rows with target_col's positive label (candidates for a feature
        swap), labelled "yes" in both SVA and NPA.
    n_keep: of those, how many pass the same leaf_top1_acc > leaf_threshold
        and leaf_decision filter sva_trees.create_pairs.create_pairs uses to pick
        which rows to actually attempt to re-inflect. Trivial languages (no fitted
        tree, hence no leaf_top1_acc/leaf_decision columns) are single-class by
        construction, so every raw row counts as kept.
    n_pairs: rows actually turned into a re-inflected minimal pair by create_pairs,
        read from "<pairs_dir>/<language>/correct_swaps.parquet". 0 if pairs_dir is
        None or that file doesn't exist yet (create_pairs not run, or n_keep was 0).
    retried: whether sva_trees.second_chance attempted a laxer retry for this
        language, regardless of whether it produced any pairs (meta.json's
        "second_pass", set only on that attempt -- see sva_trees.pipeline).
    """
    is_yes = df[target_col] == "yes"
    n_raw = int(is_yes.sum())

    # A language's own meta.json threshold wins over the pipeline-wide one, so
    # a language that got a second, laxer pass is counted at that threshold.
    retried = False
    if pairs_dir is not None:
        meta_fn = os.path.join(pairs_dir, lang_name, "meta.json")
        if os.path.exists(meta_fn):
            with open(meta_fn) as f:
                meta = json.load(f)
            leaf_threshold = meta.get("leaf_min_acc", leaf_threshold)
            retried = bool(meta.get("second_pass"))

    if "leaf_top1_acc" in df.columns and "leaf_decision" in df.columns:
        # A second_chance depth-aware retry's meta.json stores {leaf_id: that
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
) -> pd.DataFrame:
    """Calculate metrics for the SVA/agreement pipeline.

    Unlike calculate_metrics's word-order-flexibility columns (n_flexible /
    n_fully_flexible, derived from num_swaps), which rely on get_all_orders's
    word-order permutation codes and don't apply to agreement Yes/No labels,
    this reports n_raw/n_keep/n_pairs — see _agreement_row_stats.

    Returns:
        DataFrame with columns: language, base_acc, tree_acc, gain_acc, accuracy,
                                test_acc, test_n, n_raw, n_keep, n_pairs, retried
        (base_acc/tree_acc are smoothed once -- see _dtype_coerced_metrics;
        accuracy is the raw training accuracy, test_acc the raw held-out one.)
    """
    metrics = []

    for lang_name, (dt, df) in tqdm(language_data.items()):
        base_acc, tree_acc, gain_acc, accuracy = _dtype_coerced_metrics(
            dt, df, target_col, False, smoothing,
            eval_cache=eval_cache, cache_key=lang_name, accuracy_measure=True,
        )
        # Held-out accuracy from the split the tree was fit with (fit_dt
        # attaches test_eval_); NaN for a model without one.
        test_eval = getattr(dt, "test_eval_", None)
        if test_eval is not None and len(test_eval):
            test_acc, test_n = float((test_eval["y"] == test_eval["pred"]).mean()), int(len(test_eval))
        else:
            test_acc, test_n = float("nan"), 0

        n_raw, n_keep, n_pairs, retried = _agreement_row_stats(
            df, target_col, leaf_threshold, pairs_dir, lang_name
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
    debug_view: bool = False,
) -> None:
    """Generate interactive overview page with metrics and language links.

    Args:
        language_data: Dict mapping language_name -> (fitted_pipeline, dataframe)
        html_directory: Directory to save the index.html file
        target_col: Column name containing word order labels
        smoothing: Smoothing factor for entropy calculation
        agreement_label: Prose label for the page subtitle on the diagnostics-enabled
            page (e.g. "Subject-Verb" / "Subject-Auxiliary") -- only meaningful
            alongside "agreement" in target_col. Defaults to "Subject-Verb" so every
            pre-existing SVA call site is unaffected without passing it explicitly.
        head_role_label: Prose label standing in for the generic "head" role in the
            diagnostics-enabled page's "Dropped before fitting" stats/tooltips and
            legend text (e.g. "Verb" for SVA, "Aux" for subj_aux) -- purely cosmetic,
            same idea as agreement_label. "head" (default) is a no-op.
        subject_label, nsubj_label: Prose labels standing in for the generic
            "subject"/"nsubj" role in the "ambiguous subject" bucket and "# nsubj
            unk" stat -- both name the same fixed/comparison role (the one
            create_pairs keeps constant while reinflecting the other), just two
            historically-different wordings for it. Defaults ("subject"/"nsubj")
            are no-ops; a non-SVA caller (e.g. NPA, where the fixed role can be
            any of its named roles, not just "the subject") should pass both.
        exclude_labels: Labels that, when they are the only labels present, mark a
            language as uninformative and cause it to be coloured yellow and omitted
            from the table (e.g. {"--", "+-"}).
        leaf_threshold, pairs_dir: only used when "agreement" is in target_col (the
            SVA pipeline). In that case the word-order "N 1 swap"/"N 4 swap"/"N Pairs"
            columns (based on get_all_orders word-order permutation codes, meaningless
            for Yes/No agreement labels — see calculate_agreement_metrics) are replaced
            with N RAW / N KEEP / N PAIRS. leaf_threshold must match the leaf_threshold
            create_pairs was/will be run with, and pairs_dir must match its save_to
            parent directory (i.e. "<save_to>/../"), for the counts to be accurate.
        diagnostics_by_lang: dict mapping raw language name -> the JSON-shaped dict from
            sva_trees.diagnostics.diagnostics_row_to_json, one entry per language that
            create_pairs successfully produced diagnostics for. Only meaningful alongside
            "agreement" in target_col; when given (and non-empty) for an agreement target,
            the page renders with an expandable per-language diagnostics panel instead of
            the plain table. A language present in the metrics but missing from this dict
            (e.g. create_pairs raised on it) still gets a row, just without a diagnostics
            panel. None/empty falls back to the plain table, same as before this existed.
        debug_view: the data-debugging mode (no create_pairs ever run, so
            diagnostics_by_lang is always empty) still wants the richer
            dark-mode/scatter-colouring page instead of falling back to the
            plain table -- just without the (necessarily empty) diagnostics
            panel. True renders that page (word_order.html.html_deprel's
            _create_diagnostics_html with show_diagnostics=False) for an
            agreement target regardless of diagnostics_by_lang. The
            Distribution column still works either way: it reads each
            language's plain labelDistribution (multiblimp.
            agreement_pipeline_utils.read_label_distribution, written
            unconditionally by the caller -- see sva_trees.pipeline.Pipeline/
            npa.agreement.run_agreement_pipeline), independent of "diag".
    """
    html_path = Path(html_directory)
    is_agreement = is_agreement_predictor(target_col)

    # "../" only reaches decision_trees/index.html for a page one directory
    # below decision_trees/ (e.g. "svNa/") -- NPA nests an extra "npa/"
    # level (e.g. "npa/HEAD-DET_N/"), so its back-button needs "../../".
    if "decision_trees" in html_path.parts:
        depth = len(html_path.parts) - html_path.parts.index("decision_trees") - 1
    else:
        depth = 1
    overview_href = "../" * depth
    # One tier further up than overview_href: decision_trees/ itself sits
    # directly under the site root (html/), where the sva-dt stats overview
    # (build_stats.py/render_html.py's index.html) lives.
    stats_overview_href = "../" * (depth + 1) + "index.html"

    # Detect trivial/too-small langs from placeholder HTML files;
    # placeholder page also covers "too few samples to fit a tree" for a
    # mixed label set, not just a genuine single label).
    trivial_langs = {}
    for html_file in html_path.glob("*.html"):
        if html_file.name.lower() == "index.html":
            continue
        dist = extract_trivial_distribution(html_file)
        if dist is not None:
            trivial_langs[html_file.stem] = dist  # ← stem string, not Path

    # Languages that produced literally zero raw instances (never got a
    # tree, pairs, or HTML page at all -- e.g. sva_trees.pipeline.Pipeline's
    # "Skipping {lang}, raw_df has no entries" console-only skip) -- see
    # Pipeline._write_no_data_markers. Scanned from data_dir (where the
    # marker is written, alongside the .joblib/.parquet files), not
    # html_path, since a no-data language has no HTML page to glob for.
    no_data_langs = {}
    for marker_fn in glob(os.path.join(data_dir, "*.no_data.json")):
        lang_name = Path(marker_fn).stem.removesuffix(".no_data")
        try:
            with open(marker_fn) as f:
                no_data_langs[lang_name] = json.load(f).get("reason", "no reason recorded")
        except (OSError, json.JSONDecodeError):
            continue

    if language_data is None:
        language_data = {}

        for model_fn in tqdm(glob(os.path.join(data_dir, "*.joblib"))):
            lang_name = Path(model_fn).stem

            if lang_name in trivial_langs:
                continue

            parquet_fn = os.path.join(data_dir, lang_name + ".parquet")

            if not os.path.exists(parquet_fn):
                continue

            language_data[lang_name] = (
                joblib.load(model_fn),
                pd.read_parquet(parquet_fn),
            )

    # Agreement pages: one set of smoothed-accuracy metrics (base/tree/gain).
    # Word-order pages: entropy, for BOTH entropy types (sharing each
    # language's tree evaluation between the two passes).
    eval_cache = {}
    if is_agreement:
        metrics_six = calculate_agreement_metrics(
            language_data, target_col, leaf_threshold=leaf_threshold,
            pairs_dir=pairs_dir, smoothing=smoothing, eval_cache=eval_cache,
        )
        metrics_binary = metrics_six
        BASE_COL, TREE_COL, GAIN_COL = "base_acc", "tree_acc", "gain_acc"
    else:
        metrics_six = calculate_metrics(
            language_data, target_col, binary_entropy=False, smoothing=smoothing,
            eval_cache=eval_cache,
        )
        metrics_binary = calculate_metrics(
            language_data, target_col, binary_entropy=True, smoothing=smoothing,
            eval_cache=eval_cache,
        )
        BASE_COL, TREE_COL, GAIN_COL = "base_entropy", "reduced_entropy", "delta_entropy"

    # Pick up any trivial langs not yet covered by placeholder files
    # single observed class (smoothed base accuracy never reaches exactly 1.0,
    # so this is checked on the labels directly rather than on the metric)
    for l in metrics_six["language"]:
        if language_data[l][1][target_col].astype(str).nunique() <= 1:
            trivial_langs[l] = dict(language_data[l][1][target_col].value_counts())

    # Every trivial-distribution language (including an all-one-label case
    # like 100% "Yes") is omitted from the table -- there used to be an
    # exception that coloured an all-"Yes" language green and kept it in
    # the table/scatter plot instead, but that never actually fires in
    # practice (verified: zero real occurrences across every condition
    # built so far), so it was pure dead weight. Omitted languages are
    # listed instead, with their label distribution, in the trivial-note
    # built below.
    omit_langs = dict(trivial_langs)
    metrics_six = metrics_six[~metrics_six["language"].isin(omit_langs.keys())]
    metrics_binary = metrics_binary[~metrics_binary["language"].isin(omit_langs.keys())]

    sparse_langs = {}
    if is_agreement:
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
        undersized_mask = metrics_six["n_raw"].apply(is_undersized)
        undersized_langs = set(metrics_six.loc[undersized_mask, "language"])
        for lang in undersized_langs:
            omit_langs.setdefault(lang, read_label_distribution(data_dir, lang) or {})
        metrics_six = metrics_six[~metrics_six["language"].isin(undersized_langs)]
        metrics_binary = metrics_binary[~metrics_binary["language"].isin(undersized_langs)]

        # Separately: languages with enough raw "Yes" rows to clear
        # undersized, but whose joint-tagging rate is still too sparse
        # relative to the full dataset -- see SPARSE_WILSON_FLOOR. Its own
        # category/note, not folded into omit_langs -- "too little data" and
        # "enough data but too thin a slice of a much bigger whole" are
        # different failure modes worth telling apart.
        sparse_langs = {}
        for lang in metrics_six["language"]:
            dist = read_label_distribution(data_dir, lang) or {}
            total = sum(dist.values())
            both = dist.get("yes", 0) + dist.get("no", 0)
            if total and wilson_lower_bound(both, total) < SPARSE_WILSON_FLOOR:
                sparse_langs[lang] = dist
        metrics_six = metrics_six[~metrics_six["language"].isin(sparse_langs.keys())]
        metrics_binary = metrics_binary[~metrics_binary["language"].isin(sparse_langs.keys())]

    # Find corresponding HTML files — stem string -> Path
    html_files = {
        f.stem: f for f in html_path.glob("*.html") if f.name.lower() != "index.html"
    }
    # Generate table rows for both entropy types
    def generate_rows(metrics_df, lang_colors):
        rows = []
        for _, row in metrics_df.iterrows():
            lang_name = row["language"].replace("_", " ")
            lang_file = html_files.get(row["language"])  # ← stem matches directly
            color = lang_colors.get(lang_name, "#2563eb")
            name_style = (
                f' style="color:{color};font-weight:600;"' if color != "#2563eb" else ""
            )

            if lang_file:
                lang_link = f'<a href="{quote(lang_file.name)}"{name_style}>{lang_name}</a>'
            else:
                lang_link = f"<span{name_style}>{lang_name}</span>"

            if is_agreement and row.get("retried"):
                lang_link += (
                    ' <span style="display:inline-block;font-size:0.68rem;'
                    'font-weight:600;color:#d97706;background:#fef3c7;'
                    'border-radius:999px;padding:0.05rem 0.5rem;vertical-align:middle;"'
                    ' title="sva_trees.second_chance retried this language at a laxer,'
                    ' depth-aware accuracy floor after the strict pass alone wasn\'t'
                    ' enough.">retried</span>'
                )

            if is_agreement:
                # N KEEP/N PAIRS (rows attempted for re-inflection / actually
                # turned into a pair) are dropped here -- generate_rows only
                # ever runs when diagnostics_enabled is False (see its "else"
                # call site below), which for an agreement target means
                # create_pairs never ran or raised for every language, so
                # both numbers would just be a meaningless, permanent 0.
                # N RAW (rows with the positive label) is a property of the
                # tree's own label distribution, not the pairs stage, so it
                # stays.
                count_cells = f"""
                <td data-sort="{row['n_raw']}">{row['n_raw']:,}</td>"""
            else:
                count_cells = f"""
                <td data-sort="{row['n_items']}">{row['n_items']:,}</td>
                <td data-sort="{row['n_flexible']}">{row['n_flexible']:,}</td>
                <td data-sort="{row['n_fully_flexible']}">{row['n_fully_flexible']:,}</td>
                <td data-sort="{row['total_pairs']}">{row['total_pairs']:,}</td>"""

            rows.append(
                f"""
            <tr>
                <td data-sort="{lang_name.lower()}">{lang_link}</td>
                <td data-sort="{row['base_entropy']:.4f}">{row['base_entropy']:.3f}</td>
                <td data-sort="{row['reduced_entropy']:.4f}">{row['reduced_entropy']:.3f}</td>
                <td data-sort="{row['delta_entropy']:.4f}">{row['delta_entropy']:.3f}</td>
                <td data-sort="{row['accuracy']:.4f}">{row['accuracy']:.1%}</td>{count_cells}
            </tr>
            """
            )
        return "".join(rows)

 # Build color lookup before generating rows/plot data
    lang_colors = {}
    for _, row in metrics_six.iterrows():
        lang_name = row["language"].replace("_", " ")
        lang_colors[lang_name] = scatter_color(
            row, language_data, target_col, exclude_labels=exclude_labels
        )

    # Only meaningful for the agreement/SVA target — word-order pages (and
    # agreement pages nobody bothered to pass diagnostics for) keep the
    # plain, table-based page unchanged.
    diagnostics_by_lang = diagnostics_by_lang or {}
    diagnostics_enabled = is_agreement and bool(diagnostics_by_lang)
    # The richer dark-mode/scatter-colouring page is also worth it for an
    # agreement target with no diagnostics at all, as long as the caller
    # says so explicitly (debug_view) -- just without the diagnostics panel
    # itself (show_diagnostics=diagnostics_enabled below, i.e. only when
    # real per-language diagnostics exist).
    use_rich_table = is_agreement

    def build_languages(metrics_df, lang_colors):
        """Per-language dicts for the diagnostics page's client-side
        renderer (word_order.html.html_deprel's LANGUAGES blob) — the same
        data generate_rows() renders as HTML, just kept as JSON so colours
        (which read CSS custom properties for light/dark theming) and
        sorting can both happen client-side instead of being baked in here.
        A language with no diagnostics_by_lang entry (e.g. create_pairs
        raised on it, or debug_view never ran create_pairs at all) still
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
                "base": row[BASE_COL],
                "tree": row[TREE_COL],
                "gain": row[GAIN_COL],
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

    languages_json = "[]"
    if use_rich_table:
        languages_json = json.dumps(build_languages(metrics_six, lang_colors))
        languages_six_json = languages_binary_json = "[]"
        rows_six = rows_binary = ""
    else:
        languages_six_json = languages_binary_json = "[]"
        rows_six = generate_rows(metrics_six, lang_colors)
        rows_binary = generate_rows(metrics_binary, lang_colors)

    # Generate scatter plot data for both entropy types
    def generate_plot_data(metrics_df):
        data = []
        for _, row in metrics_df.iterrows():
            lang_name = row["language"].replace("_", " ")
            lang_file = html_files.get(row["language"])  # ← stem matches directly
            url = f"{quote(lang_file.name)}" if lang_file else None

            data.append(
                {
                    "name": lang_name,
                    "base": row[BASE_COL],
                    "tree" if is_agreement else "reduced": row[TREE_COL],
                    # marker size / hover count — n_raw ("Yes" items) in agreement mode,
                    # since there's no "n_items" column there
                    "n_items": int(row["n_raw"] if is_agreement else row["n_items"]),
                    "url": url,
                    "color": lang_colors.get(lang_name, "#2563eb"),
                }
            )
        return data

    plot_data_six = generate_plot_data(metrics_six)
    plot_data_binary = generate_plot_data(metrics_binary)

    plot_data_six_json = json.dumps(plot_data_six)
    plot_data_binary_json = json.dumps(plot_data_binary)
    plot_data_json = plot_data_six_json

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

    def _fmt_dist(dist):
        return "; ".join(
            f"{value}: {n}" for value, n in sorted(dist.items(), key=lambda kv: -kv[1])
        )

    if omit_langs:
        skipped_links = []
        for lang, dist in sorted(omit_langs.items()):
            lang_display = lang.replace("_", " ")
            dist_str = _fmt_dist(dist)
            lang_file = html_files.get(lang)  # ← stem matches directly
            if lang_file:
                url = f"{quote(lang_file.name)}"
                skipped_links.append(
                    f'<a href="{url}" style="color:#b893de;font-weight:600;">{lang_display}</a> ({dist_str})'
                )
            else:
                skipped_links.append(
                    f'<span style="color:#b893de;font-weight:600;">{lang_display}</span> ({dist_str})'
                )

        skipped_items = "".join(f"<li>{x}</li>" for x in skipped_links)
        trivial_note = (
            '<p class="no-data-note">The following languages were omitted for having too few or '
            'uninformative agreement labels (label distribution shown):</p>'
            f'<ul class="no-data-list">{skipped_items}</ul>'
        )
    else:
        trivial_note = ""

    if sparse_langs:
        sparse_links = []
        for lang, dist in sorted(sparse_langs.items()):
            lang_display = lang.replace("_", " ")
            dist_str = _fmt_dist(dist)
            lang_file = html_files.get(lang)  # ← stem matches directly
            if lang_file:
                url = f"{quote(lang_file.name)}"
                sparse_links.append(
                    f'<a href="{url}" style="color:#60a5fa;font-weight:600;">{lang_display}</a> ({dist_str})'
                )
            else:
                sparse_links.append(
                    f'<span style="color:#60a5fa;font-weight:600;">{lang_display}</span> ({dist_str})'
                )

        sparse_items = "".join(f"<li>{x}</li>" for x in sparse_links)
        sparse_note = (
            '<p class="no-data-note">The following languages have enough raw "Yes" samples '
            f'(&ge; {UNDERSIZED_MAX_N_RAW}) but too sparse a joint-tagging rate relative to '
            "the full dataset to be statistically confident this isn't incidental tagging "
            f'(Wilson lower bound below {SPARSE_WILSON_FLOOR:.0%}; label distribution shown):</p>'
            f'<ul class="no-data-list">{sparse_items}</ul>'
        )
    else:
        sparse_note = ""

    if no_data_langs:
        by_reason = {}
        for lang, reason in no_data_langs.items():
            by_reason.setdefault(reason, []).append(lang.replace("_", " "))
        intro = (
            "The following languages have NO data at all for this condition "
            "(zero raw instances extracted -- no tree, pairs, or page was ever "
            "produced for them)"
        )
        if len(by_reason) == 1:
            ((reason, langs),) = by_reason.items()
            groups = [(f"{intro}. {html_lib.escape(reason)}", langs)]
        else:
            groups = [(f"{intro}. {html_lib.escape(r)}", ls) for r, ls in by_reason.items()]
        no_data_note = "".join(
            f'<p class="no-data-note">{head}</p><ul class="no-data-list four-cols">'
            + "".join(f"<li>{html_lib.escape(n)}</li>" for n in sorted(ls))
            + "</ul>"
            for head, ls in groups
        )
    else:
        no_data_note = ""

    trivial_disclosure = ""
    if trivial_note:
        n_trivial = len(omit_langs)
        trivial_disclosure = _notes_disclosure(
            f"{n_trivial} language{'s' if n_trivial != 1 else ''} omitted "
            "(too few or uninformative labels) — click to expand",
            trivial_note,
        )

    sparse_disclosure = ""
    if sparse_note:
        n_sparse = len(sparse_langs)
        sparse_disclosure = _notes_disclosure(
            f"{n_sparse} language{'s' if n_sparse != 1 else ''} with too sparse a "
            "joint-tagging rate to be confident — click to expand",
            sparse_note,
        )

    no_data_disclosure = ""
    if no_data_note:
        n_no_data = len(no_data_langs)
        no_data_disclosure = _notes_disclosure(
            f"{n_no_data} language{'s' if n_no_data != 1 else ''} with no data at all "
            "for this condition — click to expand",
            no_data_note,
        )

    notes = color_note + trivial_disclosure + sparse_disclosure + no_data_disclosure

    header_cells = [
        '<th class="sortable" data-column="0">Language</th>',
        '<th class="sortable" data-column="1">Base Entropy</th>',
        '<th class="sortable" data-column="2">Reduced Entropy</th>',
        '<th class="sortable" data-column="3">Δ Entropy</th>',
        '<th class="sortable" data-column="4">DT Acc%</th>',
    ]
    if is_agreement:
        # Only meaningful in the classic (non-diagnostics-panel) page --
        # _create_diagnostics_html doesn't take header_cells at all. See
        # generate_rows' matching count_cells comment above for why N KEEP/
        # N PAIRS are dropped whenever there's no real diagnostics data.
        header_cells += (
            [
                '<th class="sortable" data-column="5">N RAW</th>',
                '<th class="sortable" data-column="6">N KEEP</th>',
                '<th class="sortable" data-column="7">N PAIRS</th>',
            ] if diagnostics_enabled else [
                '<th class="sortable" data-column="5">N RAW</th>',
            ]
        )
    else:
        header_cells += [
            '<th class="sortable" data-column="5">N Items</th>',
            '<th class="sortable" data-column="6">N 1 swap</th>',
            '<th class="sortable" data-column="7">N 4 swap</th>',
            '<th class="sortable" data-column="8">N Pairs</th>',
        ]

    html_content = create_html(
        rows_six,
        rows_binary,
        plot_data_six_json,
        plot_data_binary_json,
        trivial_note=notes,
        header_cells="".join(header_cells),
        diagnostics_enabled=use_rich_table,
        show_diagnostics_panel=diagnostics_enabled,
        languages_six_json=languages_six_json,
        languages_binary_json=languages_binary_json,
        languages_json=languages_json,
        plot_data_json=plot_data_json,
        leaf_threshold=leaf_threshold if is_agreement else None,
        agreement_label=agreement_label,
        head_role_label=head_role_label,
        subject_label=subject_label,
        nsubj_label=nsubj_label,
        overview_href=overview_href,
        stats_overview_href=stats_overview_href,
    )

    output_path = html_path / "index.html"
    output_path.write_text(html_content, encoding="utf-8")
