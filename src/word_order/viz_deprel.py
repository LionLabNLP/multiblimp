import os
from glob import glob
from pathlib import Path
from urllib.parse import quote
from typing import Dict, Tuple
import re
import json

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from .entropy import calculate_base_entropy, calculate_tree_entropy
from .html.html_deprel import create_html
from .utils import is_agreement_predictor


def _dtype_coerced_metrics(dt, df, target_col, binary_entropy, smoothing):
    """Coerce df's feature columns to match the fitted pipeline's expected dtypes
    (mutates df in place), then compute (base_entropy, reduced_entropy,
    delta_entropy, accuracy).

    Shared by calculate_metrics and calculate_agreement_metrics, which only
    differ in the extra per-language stats appended after this.
    """
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

    # Calculate entropies
    base_ent = calculate_base_entropy(
        df, target_col, binary=binary_entropy, smoothing=smoothing
    )
    reduced_ent = calculate_tree_entropy(
        dt, df, target_col, binary=binary_entropy, smoothing=smoothing
    )
    delta_ent = base_ent - reduced_ent

    # Calculate accuracy on full training data
    X = df.drop(columns=[target_col])
    y = df[target_col]
    accuracy = dt.score(X, y)

    return base_ent, reduced_ent, delta_ent, accuracy


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
            dt, df, target_col, binary_entropy, smoothing
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
    """(n_raw, n_keep, n_pairs) for one language's agreement dataframe.

    n_raw: rows with target_col's positive label (candidates for a feature
        swap) -- SVA's convention is "Yes", NPA's pairwise columns (e.g.
        "HEAD-DET_Number") use lowercase "yes" instead, so this matches
        case-insensitively rather than "Yes" literally.
    n_keep: of those, how many pass the same leaf_top1_entropy < leaf_threshold
        and leaf_decision filter sva_trees.create_pairs.create_pairs uses to pick
        which rows to actually attempt to re-inflect. Trivial languages (no fitted
        tree, hence no leaf_top1_entropy/leaf_decision columns) are single-class by
        construction, so every raw row counts as kept.
    n_pairs: rows actually turned into a re-inflected minimal pair by create_pairs,
        read from "<pairs_dir>/<language>/correct_swaps.parquet". 0 if pairs_dir is
        None or that file doesn't exist yet (create_pairs not run, or n_keep was 0).
    """
    is_yes = df[target_col].astype(str).str.lower() == "yes"
    n_raw = int(is_yes.sum())

    if "leaf_top1_entropy" in df.columns and "leaf_decision" in df.columns:
        keep = (df["leaf_top1_entropy"] < leaf_threshold) & df["leaf_decision"]
        n_keep = int((keep & is_yes).sum())
    else:
        n_keep = n_raw

    n_pairs = 0
    if pairs_dir is not None:
        pairs_fn = os.path.join(pairs_dir, lang_name, "correct_swaps.parquet")
        if os.path.exists(pairs_fn):
            n_pairs = len(pd.read_parquet(pairs_fn))

    return n_raw, n_keep, n_pairs


def calculate_agreement_metrics(
    language_data: Dict[str, Tuple[Pipeline, pd.DataFrame]],
    target_col: str,
    leaf_threshold: float = 0.1,
    pairs_dir: str | None = None,
    binary_entropy: bool = False,
    smoothing: float = 0.5,
) -> pd.DataFrame:
    """Calculate metrics for the SVA/agreement pipeline.

    Unlike calculate_metrics's word-order-flexibility columns (n_flexible /
    n_fully_flexible, derived from num_swaps), which rely on get_all_orders's
    word-order permutation codes and don't apply to agreement Yes/No labels,
    this reports n_raw/n_keep/n_pairs — see _agreement_row_stats.

    Returns:
        DataFrame with columns: language, base_entropy, reduced_entropy,
                                delta_entropy, accuracy, n_raw, n_keep, n_pairs
    """
    metrics = []

    for lang_name, (dt, df) in tqdm(language_data.items()):
        base_ent, reduced_ent, delta_ent, accuracy = _dtype_coerced_metrics(
            dt, df, target_col, binary_entropy, smoothing
        )

        n_raw, n_keep, n_pairs = _agreement_row_stats(
            df, target_col, leaf_threshold, pairs_dir, lang_name
        )

        metrics.append(
            {
                "language": lang_name,
                "base_entropy": base_ent,
                "reduced_entropy": reduced_ent,
                "delta_entropy": delta_ent,
                "accuracy": accuracy,
                "n_raw": n_raw,
                "n_keep": n_keep,
                "n_pairs": n_pairs,
            }
        )

    return _metrics_df(metrics, [
        "language", "base_entropy", "reduced_entropy", "delta_entropy", "accuracy",
        "n_raw", "n_keep", "n_pairs",
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
    include_trivial_labels: set | None = None,
    leaf_threshold: float = 0.1,
    pairs_dir: str | None = None,
    diagnostics_by_lang: dict | None = None,
    agreement_label: str = "Subject-Verb",
    head_role_label: str = "head",
    subject_label: str = "subject",
    nsubj_label: str = "nsubj",
    url_path: str | None = None,
) -> None:
    """Generate interactive overview page with metrics and language links.

    Args:
        language_data: Dict mapping language_name -> (fitted_pipeline, dataframe)
        html_directory: Directory to save the index.html file
        target_col: Column name containing word order labels
        smoothing: Smoothing factor for entropy calculation
        url_path: The "/multiblimp/{url_path}/..." prefix this page's own
            per-language links (and word_order.viz_overview.
            generate_html_overview_index's link back to this page) should
            use. None (default) falls back to html_directory's own leaf
            folder name, exactly what every pre-existing caller already
            gets implicitly -- only needed when html_directory is nested
            more than one level below wherever "/multiblimp/" is served
            from (e.g. NPA's "decision_trees/npa/{ROLE1}-{ROLE2}_{Feat}",
            where the correct link prefix is "npa/{ROLE1}-{ROLE2}_{Feat}",
            not just the leaf folder name).
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
        include_trivial_labels: Trivial labels that are still linguistically interesting
            and should be kept in the table/scatter plot with a green marker instead of
            being omitted (e.g. {"Yes"}).
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
    """
    html_path = Path(html_directory)
    include_trivial_labels = include_trivial_labels or set()
    is_agreement = is_agreement_predictor(target_col)

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

    # Calculate metrics for BOTH entropy types
    if is_agreement:
        metrics_six = calculate_agreement_metrics(
            language_data, target_col, leaf_threshold=leaf_threshold,
            pairs_dir=pairs_dir, binary_entropy=False, smoothing=smoothing
        )
        metrics_binary = calculate_agreement_metrics(
            language_data, target_col, leaf_threshold=leaf_threshold,
            pairs_dir=pairs_dir, binary_entropy=True, smoothing=smoothing
        )
    else:
        metrics_six = calculate_metrics(
            language_data, target_col, binary_entropy=False, smoothing=smoothing
        )
        metrics_binary = calculate_metrics(
            language_data, target_col, binary_entropy=True, smoothing=smoothing
        )

    # Pick up any trivial langs not yet covered by placeholder files
    for l in metrics_six[metrics_six["base_entropy"] == 0.0]["language"]:
        trivial_langs[l] = dict(language_data[l][1][target_col].value_counts())

    # Split trivial langs: any language with at least one row in an
    # include_trivial_labels value (e.g. "Yes") goes back into the main
    # table/plot, even if the rest of its rows carry other values.
    include_trivial_langs = {
        k: v for k, v in trivial_langs.items() if set(v) & include_trivial_labels
    }
    omit_langs = {
        k: v for k, v in trivial_langs.items() if not (set(v) & include_trivial_labels)
    }

    # Remove omitted langs from metrics; include_trivial_langs are re-added below
    metrics_six = metrics_six[~metrics_six["language"].isin(omit_langs.keys())]
    metrics_binary = metrics_binary[~metrics_binary["language"].isin(omit_langs.keys())]
    # Also drop include_trivial_langs that were in language_data (they have no tree)
    metrics_six = metrics_six[
        ~metrics_six["language"].isin(include_trivial_langs.keys())
    ]
    metrics_binary = metrics_binary[
        ~metrics_binary["language"].isin(include_trivial_langs.keys())
    ]

    # Add include_trivial_langs back as synthetic rows
    if include_trivial_langs:
        trivial_rows = []
        for lang, dist in include_trivial_langs.items():
            if lang in language_data:
                lang_df = language_data[lang][1]
            else:
                # Reconstruct a df with the real per-value counts (not just the
                # dominant value repeated), so _agreement_row_stats' n_raw
                # correctly counts only the "Yes" rows for a mixed-label,
                # too-few-samples language.
                values = [v for v, n in dist.items() for _ in range(n)]
                lang_df = pd.DataFrame({target_col: values})

            if is_agreement:
                n_raw, n_keep, n_pairs = _agreement_row_stats(
                    lang_df, target_col, leaf_threshold, pairs_dir, lang
                )
                trivial_rows.append(
                    {
                        "language": lang,
                        "base_entropy": 0.0,
                        "reduced_entropy": 0.0,
                        "delta_entropy": 0.0,
                        "accuracy": 1.0,
                        "n_raw": n_raw,
                        "n_keep": n_keep,
                        "n_pairs": n_pairs,
                    }
                )
                continue

            n_items = len(lang_df)
            trivial_rows.append(
                {
                    "language": lang,
                    "base_entropy": 0.0,
                    "reduced_entropy": 0.0,
                    "delta_entropy": 0.0,
                    "accuracy": 1.0,
                    "n_items": n_items,
                    "n_flexible": 0,
                    "n_fully_flexible": 0,
                    "total_pairs": 0,
                }
            )
        trivial_df = pd.DataFrame(trivial_rows)
        metrics_six = (
            pd.concat([metrics_six, trivial_df])
            .sort_values("language")
            .reset_index(drop=True)
        )
        metrics_binary = (
            pd.concat([metrics_binary, trivial_df])
            .sort_values("language")
            .reset_index(drop=True)
        )

    # Find corresponding HTML files — stem string -> Path
    html_files = {
        f.stem: f for f in html_path.glob("*.html") if f.name.lower() != "index.html"
    }
    deprel = html_path.name
    url_path = url_path if url_path is not None else deprel


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
                lang_link = f'<a href="/multiblimp/{url_path}/{quote(lang_file.stem)}"{name_style}>{lang_name}</a>'
            else:
                lang_link = f"<span{name_style}>{lang_name}</span>"

            if is_agreement:
                count_cells = f"""
                <td data-sort="{row['n_raw']}">{row['n_raw']:,}</td>
                <td data-sort="{row['n_keep']}">{row['n_keep']:,}</td>
                <td data-sort="{row['n_pairs']}">{row['n_pairs']:,}</td>"""
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
        trivial_dist = include_trivial_langs.get(row["language"])
        # Green means "categorical agreement throughout"
        if trivial_dist is not None and set(trivial_dist) <= include_trivial_labels:
            lang_colors[lang_name] = "#31cb9f"  # green — matches palette_map "Yes"
        else:
            lang_colors[lang_name] = scatter_color(
                row, language_data, target_col, exclude_labels=exclude_labels
            )

    # Only meaningful for the agreement/SVA target — word-order pages (and
    # agreement pages nobody bothered to pass diagnostics for) keep the
    # plain, table-based page unchanged.
    diagnostics_by_lang = diagnostics_by_lang or {}
    diagnostics_enabled = is_agreement and bool(diagnostics_by_lang)

    def build_languages(metrics_df, lang_colors):
        """Per-language dicts for the diagnostics page's client-side
        renderer (word_order.html.html_deprel's LANGUAGES blob) — the same
        data generate_rows() renders as HTML, just kept as JSON so colours
        (which read CSS custom properties for light/dark theming) and
        sorting can both happen client-side instead of being baked in here.
        A language with no diagnostics_by_lang entry (e.g. create_pairs
        raised on it) still gets a row, with "diag": null — the page's JS
        renders that as a "no diagnostics available" state rather than
        failing.
        """
        languages = []
        for _, row in metrics_df.iterrows():
            lang_name = row["language"].replace("_", " ")
            lang_file = html_files.get(row["language"])  # ← stem matches directly
            color = lang_colors.get(lang_name, "#2563eb")

            languages.append({
                "name": lang_name,
                "langUrl": f"/multiblimp/{url_path}/{quote(lang_file.stem)}" if lang_file else None,
                "color": color if color != "#2563eb" else None,
                "base": row["base_entropy"],
                "reduced": row["reduced_entropy"],
                "delta": row["delta_entropy"],
                "acc": row["accuracy"],
                "nRaw": int(row["n_raw"]),
                "nKeep": int(row["n_keep"]),
                "nPairs": int(row["n_pairs"]),
                "diag": diagnostics_by_lang.get(row["language"]),
            })
        return languages

    if diagnostics_enabled:
        languages_six_json = json.dumps(build_languages(metrics_six, lang_colors))
        languages_binary_json = json.dumps(build_languages(metrics_binary, lang_colors))
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
            url = f"/multiblimp/{url_path}/{quote(lang_file.stem)}" if lang_file else None

            data.append(
                {
                    "name": lang_name,
                    "base": row["base_entropy"],
                    "reduced": row["reduced_entropy"],
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

    # Build legend / notes
    any_yellow_drawn = "#e5c64d" in lang_colors.values()
    color_note_parts = []
    if any_yellow_drawn:
        color_note_parts.append(
            '<p class="trivial-note">'
            '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
            'background:#e5c64d;margin-right:6px;vertical-align:middle;"></span>'
            "Languages shown in <strong>yellow</strong> only exhibit uninformative agreement "
            "(labels <code>--</code> and <code>+-</code>) — no Yes/No contrast was observed."
        )
    if include_trivial_labels:
        labels_str = ", ".join(
            f"<code>{l}</code>" for l in sorted(include_trivial_labels)
        )
        color_note_parts.append(
            ('<br>' if any_yellow_drawn else '<p class="trivial-note">')
            + '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
            'background:#31cb9f;margin-right:6px;margin-top:6px;vertical-align:middle;"></span>'
            f"Languages shown in <strong>green</strong> have categorical agreement throughout "
            f"— all samples share the label {labels_str}."
        )
    if color_note_parts:
        color_note_parts.append("</p>")
    color_note = "".join(color_note_parts)

    if omit_langs:
        def _fmt_dist(dist):
            # A single-value distribution IS the "uninformative label" the
            # intro sentence already names as the reason for omission -- just
            # show its count. Only mixed distributions need the value labeled.
            if len(dist) == 1:
                (n,) = dist.values()
                return str(n)
            return "; ".join(
                f"{value}: {n}" for value, n in sorted(dist.items(), key=lambda kv: -kv[1])
            )

        skipped_links = []
        for lang, dist in sorted(omit_langs.items()):
            lang_display = lang.replace("_", " ")
            dist_str = _fmt_dist(dist)
            lang_file = html_files.get(lang)  # ← stem matches directly
            if lang_file:
                url = f"/multiblimp/{url_path}/{quote(lang_file.stem)}"
                skipped_links.append(
                    f'<a href="{url}" style="color:#b893de;font-weight:600;">{lang_display}</a> ({dist_str})'
                )
            else:
                skipped_links.append(
                    f'<span style="color:#b893de;font-weight:600;">{lang_display}</span> ({dist_str})'
                )

        trivial_note = (
            f'<p class="trivial-note">The following languages were omitted for having too few or '
            f'uninformative agreement labels (label distribution shown): {", ".join(skipped_links)}.</p>'
        )
    else:
        trivial_note = ""

    notes = color_note + trivial_note

    header_cells = [
        '<th class="sortable" data-column="0">Language</th>',
        '<th class="sortable" data-column="1">Base Entropy</th>',
        '<th class="sortable" data-column="2">Reduced Entropy</th>',
        '<th class="sortable" data-column="3">Δ Entropy</th>',
        '<th class="sortable" data-column="4">DT Acc%</th>',
    ]
    if is_agreement:
        header_cells += [
            '<th class="sortable" data-column="5">N RAW</th>',
            '<th class="sortable" data-column="6">N KEEP</th>',
            '<th class="sortable" data-column="7">N PAIRS</th>',
        ]
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
        diagnostics_enabled=diagnostics_enabled,
        languages_six_json=languages_six_json,
        languages_binary_json=languages_binary_json,
        leaf_threshold=leaf_threshold if is_agreement else None,
        agreement_label=agreement_label,
        head_role_label=head_role_label,
        subject_label=subject_label,
        nsubj_label=nsubj_label,
    )

    output_path = html_path / "index.html"
    output_path.write_text(html_content, encoding="utf-8")
