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

from agreement_trees.pipeline_utils import read_label_distribution, wilson_lower_bound

from agreement_trees.fit.entropy import (
    chunked_leaf_ids,
    calculate_base_accuracy, pooled_leaf_accuracy, smoothed_accuracy, DEFAULT_LEAF_MIN_ACCURACY,
)
from agreement_trees.report.html.html_deprel import create_html
from agreement_trees.utils import is_agreement_predictor

# A condition with some "Yes" candidates but fewer than this is moved out of the table into the trivial note
# (the same threshold html_deprel's MIN_N_FOR_COLOR fades the coverage colour at).
UNDERSIZED_MAX_N_RAW = 10


def is_undersized(n_raw: int) -> bool:
    return 0 < n_raw < UNDERSIZED_MAX_N_RAW


# The scan's wilson_floor default: a language with enough "Yes" rows in absolute terms can still have a Wilson
# lower bound this low if they are a sliver of its treebank (Naija: 16 Yes in 9,634 rows).
SPARSE_WILSON_FLOOR = 0.01

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


def _agreement_row_stats(df, target_col, leaf_threshold, pairs_dir, lang_name):
    """(n_raw, n_keep, n_pairs, retried) for one language's agreement dataframe.

    n_raw: rows with target_col's positive label (candidates for a feature
        swap), labelled "yes" in both SVA and NPA.
    n_keep: of those, how many pass the same leaf_top1_acc > leaf_threshold
        and leaf_decision filter agreement_trees.pairs.create_pairs.create_pairs uses to pick
        which rows to actually attempt to re-inflect. Trivial languages (no fitted
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
) -> None:
    """Write the deprel index.html: per-language metrics, scatter and, with diagnostics, bucket breakdowns.

    data_dir: where the fitted models and frames were saved (language_data overrides). html_directory: output
    directory. target_col: the agreement predictor. leaf_threshold and pairs_dir (create_pairs' save_to parent) give N KEEP / N PAIRS.
    exclude_labels: label sets that mark a language uninformative (omitted from the table).
    diagnostics_by_lang: {language: diagnostics_row_to_json dict}.
    agreement_label, head_role_label, subject_label, nsubj_label: prose for the subtitle, the head role
    ("Verb", "Aux") and the fixed role in the "ambiguous subject" bucket and "# nsubj unk" stat (cosmetic;
    NPA passes its own roles).
    """
    if not is_agreement_predictor(target_col):
        raise ValueError(f"{target_col!r} is not an agreement predictor")
    html_path = Path(html_directory)

    # "../" reaches decision_trees/index.html from a page one directory below it; NPA nests an extra "npa/" level
    if "decision_trees" in html_path.parts:
        depth = len(html_path.parts) - html_path.parts.index("decision_trees") - 1
    else:
        depth = 1
    overview_href = "../" * depth
    # decision_trees/ sits directly under the site root, where the stats overview lives
    stats_overview_href = "../" * (depth + 1) + "index.html"

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
        pairs_dir=pairs_dir, smoothing=smoothing, eval_cache=eval_cache,
    )
    metrics_binary = metrics_six

    # a single observed class (smoothed base accuracy never reaches 1.0, so check the labels)
    for l in metrics_six["language"]:
        if language_data[l][1][target_col].astype(str).nunique() <= 1:
            trivial_langs[l] = dict(language_data[l][1][target_col].value_counts())

    # trivial-distribution languages are omitted from the table and listed with their label distribution in
    # the trivial note
    omit_langs = dict(trivial_langs)

    # an all-"yes" language above the undersized bar is real data with nothing to fit: keep it,
    # scored as a trivial tree (root = tree = smoothed 100%)
    single_yes = {
        lang: dist for lang, dist in trivial_langs.items()
        if {str(k).lower() for k in dist} == {"yes"}
        and not is_undersized(sum(dist.values()))
        and sum(dist.values()) >= UNDERSIZED_MAX_N_RAW
    }
    for lang in single_yes:
        omit_langs.pop(lang, None)
    have = set(metrics_six["language"])
    extra = []
    for lang, dist in single_yes.items():
        if lang in have:
            continue
        n_raw = int(sum(dist.values()))
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

    sparse_langs = {}
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

    html_content = create_html(
        plot_data_json,
        trivial_note=notes,
        languages_json=languages_json,
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
