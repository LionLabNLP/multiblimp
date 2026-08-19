import os
import re
import warnings

import numpy as np
import plotly.graph_objects as go
import scipy
import seaborn as sns
import pandas as pd

from matplotlib.colors import to_hex

from .entropy import order_entropy, calculate_base_entropy, calculate_tree_entropy
from .utils import get_all_orders, build_grew_link, split_pairwise_predictor
from .html.html_tree import create_html, write_placeholder_html


def clean_rule(rule, threshold=None, is_binary=False):
    numerical = rule.startswith("num__")
    rule = (
        rule.removeprefix("num__")
        .removeprefix("cat__")
        .replace("sibling-deprel", "sibling")
        .replace("_missing", "nan")
    )

    if rule == "in_question":
        return "in a question?"

    rule_elements = rule.split("_")
    rule_elements[0] = (
        rule_elements[0].replace("nsubj", "subject").replace("obj", "object")
    )

    if rule_elements[-1] == "nan":
        rule = ".".join(rule_elements[:-1]) + " is not set"
    elif rule_elements[-2] in ["sibling", "sibling-pos"]:
        rule = f"{rule_elements[0]} has {rule_elements[-1]} sibling?"
    elif rule_elements[-2] in ["sibling-L"]:
        rule = f"{rule_elements[0]} has left {rule_elements[-1]} sibling?"
    elif rule_elements[-2] in ["sibling-R"]:
        rule = f"{rule_elements[0]} has right {rule_elements[-1]} sibling?"
    elif len(rule_elements) == 3:
        rule = f"{rule_elements[0]}.{rule_elements[1]} = {rule_elements[2]}"
    elif "child" in rule_elements[1]:
        if rule_elements[1] == "child-feat" and len(rule_elements) >= 5:
            rule = f"{rule_elements[0]}'s {rule_elements[2]} child: {rule_elements[3]} = {rule_elements[4]}"
        else:
            rule = f"{rule_elements[0]} has {rule_elements[2]} child?"
    else:
        rule = "_".join(rule_elements[:-1]) + f" = {rule_elements[-1]}"

    if numerical and threshold is not None and not is_binary:
        rule += f" ≤ {threshold:.2f}"

    return rule


def get_correlated_features(prep, clf, dt_df):
    X = prep.transform(dt_df)

    if isinstance(X, scipy.sparse._csr.csr_matrix):
        return {}

    if X.dtype == object:
        try:
            X = X.astype(np.float64)
        except (ValueError, TypeError) as e:
            warnings.warn(
                f"Cannot convert object array to float: {e}. Skipping correlation computation.",
                UserWarning,
                stacklevel=2,
            )
            return {}

    X = X.astype(np.float64)
    feature_names = prep.get_feature_names_out()

    rule_names = [clean_rule(f) for f in feature_names]

    tree = clf.tree_
    node_indicator = clf.decision_path(X)

    result = {}

    for node_id in range(tree.node_count):
        feat_idx = tree.feature[node_id]
        if feat_idx == -2:  # leaf
            continue

        row_indices = node_indicator[:, node_id].nonzero()[0]
        if len(row_indices) < 2:
            result[node_id] = []
            continue

        X_node = X[row_indices]
        feat_col = X_node[:, feat_idx]
        feat_centered = feat_col - feat_col.mean()
        feat_std = feat_centered.std()

        if feat_std == 0:
            result[node_id] = []
            continue

        others_centered = X_node - X_node.mean(axis=0)
        others_std = others_centered.std(axis=0)

        with np.errstate(invalid="ignore", divide="ignore"):
            correlations = np.dot(feat_centered, others_centered) / (
                len(row_indices) * feat_std * others_std
            )

        result[node_id] = [
            rule_names[jdx]
            for jdx, c in enumerate(correlations)
            if jdx != feat_idx and np.abs(c) > 0.99
        ]

    return result


def get_sample_ids(prep, clf, dt_df, predictor_var, max_rows=100, seed=42):
    predictor_samples = {}

    for predictor_value, df in dt_df.groupby(predictor_var):
        X_model = prep.transform(df)

        rng = np.random.default_rng(seed)
        node_indicator = clf.decision_path(X_model)
        n_nodes = clf.tree_.node_count

        out = {}
        for node_id in range(n_nodes):
            rows = node_indicator[:, node_id].nonzero()[0]
            if rows.size > max_rows:
                rows = rng.choice(rows, size=max_rows, replace=False)
            out[node_id] = df.index[rows]

        predictor_samples[predictor_value] = out

    return predictor_samples


# ── Agreement swap highlighting (head <-> child feature, e.g. SVA) ────────────
# Applies only when predictor_var is an agreement predictor with a known
# highlight pair — mirrors the highlighting built for sva_trees.create_pairs's
# example tables. No-op for plain word-order targets.
#
# Two disjoint predictor_var conventions are recognized (SVA's own
# convention never contains "-", so checking for one first is unambiguous):
#   - Pairwise role-pair naming, e.g. npa_trees' "HEAD-DET_Number":
#     "{Role1}-{Role2}_{Feature}" -- highlights both roles directly by name,
#     in whatever case they appear in predictor_var (NOT lowercased: unlike
#     SVA's own always-lowercase deprel-based prefixes, e.g. "nsubj",
#     npa_trees' role prefixes are uppercase by convention -- "HEAD_idx",
#     "HEAD_Gender", etc. -- and _build_feat_columns/_add_sen_str_column
#     both key off the prefix exactly as it appears in the column name, so
#     the highlight-prefix casing has to match that exactly too), no target
#     needed.
#   - SVA's "head_{child_deprel}_{Feature}_agreement" (e.g.
#     "head_nsubj_Number_agreement"): requires "agreement" in the string
#     and a PredictionTarget with child_deprels set -- highlights the
#     literal "head" role plus target.child_deprels[0].

def _agreement_context(predictor_var, target):
    """Returns (swap_feature, highlight_prefixes) for an agreement
    predictor, else (None, ())."""
    if not predictor_var:
        return None, ()

    pair_match = split_pairwise_predictor(predictor_var)
    if pair_match:
        role1, role2, feat = pair_match
        return feat, (role1, role2)

    if "agreement" not in predictor_var or target is None or not target.child_deprels:
        return None, ()
    match = re.match(r".*_([A-Z][a-z]+)_.*", predictor_var)
    return (match.group(1) if match else None), ("head", target.child_deprels[0])


def _display_predictor_var(predictor_var, head_label):
    """predictor_var, cosmetically cleaned up for the info-panel "Predictor"
    row: its trailing "_agreement" dropped (redundant on this page -- every
    predictor here is one) and its "head" role token swapped for a
    stream-specific, human-readable label (e.g. "Verb" for SVA, "Aux" for
    subj_aux, "NP head" for npa). The underlying dataframe/column-name
    convention (always literally "head"/"HEAD", and always "_agreement"-
    suffixed for SVA/subj_aux) is untouched; only this display string
    changes.

    Handles both predictor_var conventions _agreement_context does: the
    pairwise "{Role1}-{Role2}_{Feature}" naming (relabels whichever role is
    "head", case-insensitively -- npa's is always uppercase "HEAD"; never
    "_agreement"-suffixed, so that part is a no-op) and SVA/subj_aux's
    "head_{child_deprel}_{Feature}_agreement" (relabels the leading "head"
    segment). head_label="head" (the default) skips the role relabeling but
    still drops "_agreement".
    """
    if not predictor_var:
        return predictor_var
    pair_match = split_pairwise_predictor(predictor_var)
    if pair_match:
        role1, role2, feat = pair_match
        if head_label != "head":
            role1 = head_label if role1.lower() == "head" else role1
            role2 = head_label if role2.lower() == "head" else role2
        return f"{role1}-{role2}_{feat}"
    display = predictor_var
    if display.endswith("_agreement"):
        display = display[: -len("_agreement")]
    if head_label != "head" and display.lower().startswith("head_"):
        display = head_label + display[len("head"):]
    return display


def _relabel_head_columns(records, head_label):
    """Renames "head_*"/"HEAD_*" keys in each dict of `records` (e.g.
    "head_form" -> "Aux_form") to head_label -- same cosmetic-only relabeling
    as _display_predictor_var, applied to the sample-rows table's column
    keys instead of the Predictor string. head_label="head" is a no-op.
    """
    if head_label == "head" or not records:
        return records

    def relabel_key(k):
        for prefix in ("head_", "HEAD_"):
            if k.startswith(prefix):
                return head_label + k[len(prefix) - 1:]
        return k

    return [{relabel_key(k): v for k, v in r.items()} for r in records]


def _highlighted_sen_str(sen, highlight_idx) -> str:
    return " ".join(
        f"<strong>{tok}</strong>" if i in highlight_idx else str(tok)
        for i, tok in enumerate(sen)
    )


def _add_sen_str_column(full_df, highlight_prefixes=()):
    """Create sentence string and embolden the agreement-relevant
    tokens (head/child) when highlight_prefixes is given.
    """
    if not highlight_prefixes:
        full_df["sen_str"] = [" ".join(sen) for sen in full_df["sen"]]
        return
    idx_cols = [f"{p}_idx" for p in highlight_prefixes if f"{p}_idx" in full_df.columns]
    idx_series = [full_df[c] for c in idx_cols]
    sen_strs = []
    for sen, *idx_vals in zip(full_df["sen"], *idx_series): # more efficient than iterrows on wide dataset
        highlight_idx = {int(v) - 1 for v in idx_vals if pd.notna(v)}
        sen_strs.append(_highlighted_sen_str(sen, highlight_idx))
    full_df["sen_str"] = sen_strs


def _build_feat_columns(full_df, swap_feature=None, highlight_prefixes=()):
    """Build {prefix}_features list columns (one "Feat=Val" string per node feature
    that's actually set, skipping None/NaN/"_" -- extract_node_features creates a
    "{prefix}_{feat}" column for every feat in the treebank's WHOLE feature
    vocabulary regardless of POS, e.g. a verb Tense column exists, and is
    null, on every noun row -- matches sva_trees/create_pairs.py's own
    _feats_summary, which skips null values the same way), bolding the
    swap-relevant feature's entry for prefixes in highlight_prefixes.

    Includes each node's upos (extract_node_features's "{prefix}_pos" column
    -- displayed as "upos=..." rather than the column's own "pos" label,
    first in the list, ahead of the morphological Feats) alongside the
    morphological feats proper.

    Every row still contributes a (possibly empty) list to every prefix's
    "{prefix}_features" entry, even a row with zero set features for that
    prefix -- feat_collect's lists are later assigned directly as DataFrame
    columns (full_df[k] = v in this function's callers), which requires
    exactly len(full_df) entries in row order; silently omitting a row
    whenever its features all happened to be null would misalign every
    row after it.
    """
    # Trailing (?:\[[a-z]+\])? admits UD's layered-feature suffix (e.g.
    # "Number[psor]", "Gender[subj]"). Prefix is [A-Za-z]+, not just [a-z]+:
    # SVA's own prefixes ("head", "nsubj", ...) are always lowercase, but
    # e.g. npa_trees' role-based prefixes ("HEAD", "DET", ...) are uppercase
    # by design (pools NOUN/PROPN/PRON-headed NPs under one "HEAD" role) --
    # this widens the match without narrowing it for any lowercase-prefixed
    # caller.
    morph_df = full_df.filter(regex=r"^[A-Za-z]+_[A-Z][a-zA-Z]+(?:\[[a-z]+\])?$", axis=1)
    pos_cols = [c for c in full_df.columns if re.match(r"^[A-Za-z]+_pos$", c)]
    feat_df = pd.concat([full_df[pos_cols], morph_df], axis=1) if pos_cols else morph_df
    feat_cols = list(feat_df.columns)
    prefixes = {label.split("_")[0] for label in feat_cols}
    feat_collect = {f"{p}_features": [] for p in prefixes}
    # itertuples(), not iterrows(): avoids rebuilding a full-width Series per row.
    for row_tuple in feat_df.itertuples(index=False, name=None):
        mf = {f"{p}_features": [] for p in prefixes}
        for label, val in zip(feat_cols, row_tuple):
            if pd.isna(val) or val == "_":
                continue
            prefix, feature = label.split("_")
            key = f"{prefix}_features"
            val_str = str(val)[:-2] if str(val).endswith(".0") else str(val)
            display_feature = "upos" if feature == "pos" else feature
            entry = f"{display_feature}={val_str}"
            if swap_feature and feature == swap_feature and prefix in highlight_prefixes:
                entry = f"<strong>{entry}</strong>"
            mf[key].append(entry)
        for k, v in mf.items():
            feat_collect[k].append(v)
    return feat_collect


def get_samples(
    prep,
    clf,
    full_df,
    label_distribution,
    class2idx,
    dt_df,
    predictor_var,
    max_rows=100,
    seed=42,
    show_features=False,
    extra_columns=None,
    target=None,
    head_label="head",
):
    sample_ids = get_sample_ids(prep, clf, dt_df, predictor_var, max_rows, seed)

    swap_feature, highlight_prefixes = _agreement_context(predictor_var, target)

    keep_columns = ["sen_str"]
    _add_sen_str_column(full_df, highlight_prefixes=highlight_prefixes)
    full_df["treebank_link"] = build_treebank_links(full_df)

    if show_features:
        feat_collect = _build_feat_columns(full_df, swap_feature=swap_feature,
                                            highlight_prefixes=highlight_prefixes)

        for k, v in feat_collect.items():
            full_df[k] = v
            keep_columns.extend([f"{k.split("_")[0]+"_form"}", k])
    else:
        keep_columns.extend([col for col in full_df if col.endswith("_form")])
    keep_columns.extend(["treebank_link"])

    if extra_columns:
        keep_columns.extend(
            [
                col
                for col in extra_columns
                if col in full_df.columns and col not in keep_columns
            ]
        )

    predictor_samples = {}
    for predictor, node_sample_ids in sample_ids.items():
        predictor_samples[predictor] = {
            int(node_idx): {
                "rows": _relabel_head_columns(
                    full_df.loc[sample_ids][keep_columns].to_dict("records"), head_label
                ),
                "count": len(sample_ids),
                "total_count": label_distribution[node_idx][class2idx[predictor]],
            }
            for node_idx, sample_ids in node_sample_ids.items()
        }

    return predictor_samples


def compute_tree_layout(tree):
    """
    Returns dicts: node_id -> x, node_id -> y, plus edge data with sample counts
    Root at top, leaves evenly spaced.
    """
    children_left = tree.children_left
    children_right = tree.children_right
    n_samples = tree.n_node_samples

    node_x = {}
    node_y = {}
    current_x = 0

    def dfs(node_id, depth):
        nonlocal current_x

        left = children_left[node_id]
        right = children_right[node_id]

        if left == -1:  # leaf
            node_x[node_id] = current_x
            node_y[node_id] = -depth
            current_x += 1
        else:
            dfs(left, depth + 1)
            dfs(right, depth + 1)
            node_x[node_id] = (node_x[left] + node_x[right]) / 2
            node_y[node_id] = -depth

    dfs(0, 0)

    node_ids = list(node_x.keys())

    edge_x_true, edge_y_true = [], []
    edge_x_false, edge_y_false = [], []
    edge_samples_true = []
    edge_samples_false = []

    for node_id in node_ids:
        left = children_left[node_id]
        right = children_right[node_id]

        if left != -1:
            # FALSE branch (left) - absolute sample count
            edge_x_false += [node_x[node_id], node_x[left], None]
            edge_y_false += [node_y[node_id], node_y[left], None]
            edge_samples_false += [n_samples[left], n_samples[left], None]

            # TRUE branch (right) - absolute sample count
            edge_x_true += [node_x[node_id], node_x[right], None]
            edge_y_true += [node_y[node_id], node_y[right], None]
            edge_samples_true += [n_samples[right], n_samples[right], None]

    return (
        node_x,
        node_y,
        edge_x_true,
        edge_y_true,
        edge_x_false,
        edge_y_false,
        edge_samples_true,
        edge_samples_false,
        node_ids,
    )


def interpolate_color(hex1, hex2, t):
    """Linearly interpolate between two hex colors."""
    hex1 = hex1.lstrip("#")
    hex2 = hex2.lstrip("#")

    r1, g1, b1 = int(hex1[0:2], 16), int(hex1[2:4], 16), int(hex1[4:6], 16)
    r2, g2, b2 = int(hex2[0:2], 16), int(hex2[2:4], 16), int(hex2[4:6], 16)

    r = round(r1 + (r2 - r1) * t)
    g = round(g1 + (g2 - g1) * t)
    b = round(b1 + (b2 - b1) * t)

    return f"#{r:02x}{g:02x}{b:02x}"


def build_treebank_links(full_df: pd.DataFrame) -> list[str]:
    """Build grew.fr query links for each row in full_df."""

    form_cols = [x for x in full_df.columns if x.endswith("_form")]
    treebanks = full_df["treebank"] if "treebank" in full_df.columns else [None] * len(full_df)
    sent_ids = full_df["sent_id"] if "sent_id" in full_df.columns else [None] * len(full_df)
    form_rows = zip(*(full_df[c] for c in form_cols)) if form_cols else [()] * len(full_df)
    tb_links = []
    for treebank, sent_id, form_values in zip(treebanks, sent_ids, form_rows):
        link = build_grew_link(treebank, sent_id, list(form_values))
        tb_links.append(link if link is not None else "&mdash;")
    return tb_links


def remove_censored(full_df):
    """Keep rows where at least one *_form column has a real word (not underscore/empty)."""
    form_cols = [x for x in full_df.columns if x.endswith("_form")]
    if not form_cols:
        return full_df
    stripped = full_df[form_cols].astype(str).apply(lambda s: s.str.strip())
    has_word = ~stripped.isin(["_", "", "nan"])
    return full_df[has_word.any(axis=1)]


def build_placeholder_args(
    dt_df, full_df, predictor_var, meta=None, show_features=False, target=None,
    head_label="head",
):
    """Compute all arguments needed for write_placeholder_html."""
    # dt_df's predictor values may not actually be uniform: this placeholder
    # path is also taken when fit_dt refuses to fit a tree for having too few
    # rows (below its min_df_len), regardless of how many distinct labels are
    # present. Report the true full-count distribution (not a subsample)
    # rather than assuming dt_df[0] speaks for every row.
    label = dict(dt_df[predictor_var].value_counts())

    if meta is None:
        meta = {}
    meta["Training samples"] = f"{len(dt_df):,}"
    meta["Predictor"] = _display_predictor_var(predictor_var, head_label)

    swap_feature, highlight_prefixes = _agreement_context(predictor_var, target)

    _add_sen_str_column(full_df, highlight_prefixes=highlight_prefixes)
    full_df["treebank_link"] = build_treebank_links(full_df)

    keep_columns = ["sen_str"]
    if show_features:
        feat_collect = _build_feat_columns(full_df, swap_feature=swap_feature,
                                            highlight_prefixes=highlight_prefixes)

        for k, v in feat_collect.items():
            full_df[k] = v
            keep_columns.extend([f"{k.split("_")[0]+"_form"}", k])
    else:
        keep_columns.extend([col for col in full_df if col.endswith("_form")])

    keep_columns.append("treebank_link")

    sample_rows = full_df.sample(min(50, len(full_df)), random_state=42)[
        keep_columns
    ].to_dict("records")
    sample_rows = _relabel_head_columns(sample_rows, head_label)

    return label, meta, sample_rows


def _finalize_tree_html(
    pipeline_model,
    data_df,
    classes,
    label_distribution,
    predictor_samples,
    meta,
    out_file,
    correlate_features=True,
    n_palette_colors=None,
    palette_map=None,
):
    """
    Core visualization engine shared by tree2html and pipeline2html.

    pipeline_model: fitted sklearn Pipeline with "preprocessor" and "clf" steps.
    data_df: feature DataFrame used for correlated-feature computation.
    classes: ordered display class list (determines legend order and colors).
    label_distribution: list[list[int]], shape (n_nodes, len(classes)).
    predictor_samples: per-class example dicts from get_samples(), or {} to skip.
    meta: info-panel key/value dict; must contain "accuracy".
    n_palette_colors: palette size override for color-stable cross-dataset comparisons.
    """
    prep = pipeline_model.named_steps["preprocessor"]
    clf = pipeline_model.named_steps["clf"]

    # Layout
    tree = clf.tree_
    (
        node_x,
        node_y,
        edge_x_true,
        edge_y_true,
        edge_x_false,
        edge_y_false,
        edge_samples_true,
        edge_samples_false,
        node_ids,
    ) = compute_tree_layout(tree)

    # Node metadata
    feature = tree.feature
    n_samples = tree.n_node_samples
    model_classes = list(clf.classes_)
    n_model_classes = len(model_classes)
    impurity = tree.impurity
    max_impurity = np.log2(n_model_classes) or 1
    tree_values = tree.value
    predicted_class_ids = [value[0].argmax() for value in tree_values]
    predicted_class = [str(clf.classes_[idx]) for idx in predicted_class_ids]

    classes = [
        str(c) for c in classes
    ]  # normalize to str (clf.classes_ may be numpy ints)
    n_classes = len(classes)
    class2idx = {c: idx for idx, c in enumerate(classes)}
    # Classes fit_dt dropped before fitting (e.g. "unk") only ever have a
    # real (non-root-node) count via the legend/root distribution -- every
    # other node's count for them is definitionally 0, since none of those
    # rows ever reached a split. Used below to keep them out of each node's
    # own dist/tooltip breakdown, where they'd just be permanent zero-clutter.
    model_classes_str = {str(c) for c in model_classes}

    feature_names = prep.get_feature_names_out()

    n_features_out = len(feature_names)
    cat_slice = prep.output_indices_.get("cat")
    binary_feature_indices = (
        set(range(*cat_slice.indices(n_features_out))) if cat_slice is not None else set()
    )

    if correlate_features:
        correlated_features = get_correlated_features(prep, clf, data_df)
    else:
        correlated_features = {}

    if palette_map:
        # Fixed canonical order & colours from palette_map, regardless of the data.
        # classes get reordered to palette_map's keys, so realign the passed-in
        # label_distribution (currently in `classes` order) to the new order.
        old_class2idx = {c: idx for idx, c in enumerate(classes)}
        classes = [str(c) for c in palette_map.keys()]
        hex_colors = list(palette_map.values())
        label_distribution = [
            [dist[old_class2idx[cls]] if cls in old_class2idx else 0 for cls in classes]
            for dist in label_distribution
        ]
        n_classes = len(classes)
        class2idx = {c: idx for idx, c in enumerate(classes)}
    else:
        # Always generate palette across full SVO space so colours are stable
        palette = sns.color_palette("husl", n_colors=n_palette_colors or n_classes)
        hex_colors = [to_hex(c) for c in palette[:n_classes]]

        # Grey out classes with zero samples at the root (absent from this language)
        root_dist = label_distribution[0]
        hex_colors = [
            color if root_dist[idx] > 0 else "#d4d4d4"
            for idx, color in enumerate(hex_colors)
        ]

    annotations = []
    node_data = {}

    # Build parent relationships and rules for path tracing
    children_left = tree.children_left
    children_right = tree.children_right
    parent_map = {}
    rule_map = {}  # node_id -> (rule_text, is_true_branch)

    # First pass: build parent map and rule map
    for node_id in node_ids:
        left_child = children_left[node_id]
        right_child = children_right[node_id]

        if left_child != -1:
            parent_map[left_child] = node_id
            parent_map[right_child] = node_id

            # Store the rule text for this internal node
            if feature[node_id] != -2:
                raw_feat = feature_names[feature[node_id]]
                rule_text = clean_rule(
                    raw_feat,
                    threshold=tree.threshold[node_id],
                    is_binary=feature[node_id] in binary_feature_indices,
                )
                rule_map[left_child] = (rule_text, False)  # False branch (left)
                rule_map[right_child] = (rule_text, True)  # True branch (right)

    for i in node_ids:
        # Get the predicted class name from the model
        predicted_class_name = predicted_class[i]
        # Map it to the display class index
        display_class_idx = class2idx[predicted_class_name]
        node_color = hex_colors[display_class_idx]
        relative_entropy = impurity[i] / max_impurity
        dist_i = label_distribution[i]
        # Entropy/frac must only ever reflect classes the tree actually
        # fit on -- at the root, dist_i also carries the dropped-class
        # (e.g. "unk") count injected for the legend, which would otherwise
        # inflate n_wrong/total_node there and desync this node's own H=/
        # frac numbers from the tree's real n_samples[i] and accuracy.
        dist_i_fitted = [
            cnt for cnt, cls in zip(dist_i, classes) if cls in model_classes_str
        ]
        n_right = max(dist_i_fitted)
        n_wrong = sum(dist_i_fitted) - n_right
        binary_entropy = order_entropy(n_right, n_wrong)
        node_color = interpolate_color(node_color, "#ffffff", relative_entropy * 0.72)

        total_node = sum(dist_i_fitted) or 1

        # Small dim style shared by node-id and stats. direction:ltr +
        # unicode-bidi:isolate: the rule/leaf-label line above this span can be
        # an RTL script (e.g. a Hebrew lemma, "lemma = כל"), and
        # without isolating this span the browser's bidi algorithm can let
        # that RTL run bleed into "[i]  n=...  H=...", mirroring the brackets
        # and reordering the digits (e.g. "[1]" rendering as "1]").
        meta_style = "color:#44403c;font-size:9px;direction:ltr;unicode-bidi:isolate"

        if feature[i] != -2:
            # Internal node: rule on top, [n] n= H= on second line
            rule = clean_rule(
                feature_names[feature[i]],
                threshold=tree.threshold[i],
                is_binary=feature[i] in binary_feature_indices,
            )
            corr = correlated_features.get(i, [])
            # Isolated for the same reason as meta_style: appended directly
            # after `rule` (possibly RTL) inside the same <b>, "[+]" would
            # otherwise be vulnerable to the same bracket-mirroring.
            corr_note = "<span style='direction:ltr;unicode-bidi:isolate'> [+]</span>" if corr else ""
            label = (
                f"<b>{rule}{corr_note}</b><br>"
                f"<span style='{meta_style}'>"
                f"<b>[{i}]</b>  n={n_samples[i]}  H={binary_entropy:.2f}</span>"
            )
            hover_corr = list(corr)
        else:
            # Leaf: include all classes with count >= 50% of the majority.
            # dist_i_fitted (not dist_i): a depth-0/trivial tree's single
            # leaf IS the root, so without this, a dropped class like "unk"
            # -- large enough at the root to pass the >=50%-of-majority bar
            # -- could show up right in the leaf's own predicted-class label.
            sorted_dist = sorted(
                zip(dist_i_fitted, [c for c in classes if c in model_classes_str]),
                key=lambda x: x[0], reverse=True,
            )
            top_cnt, top_cls = sorted_dist[0]
            qualifying = [top_cls] + [
                cls for cnt, cls in sorted_dist[1:] if cnt > 0 and cnt >= 0.5 * top_cnt
            ]
            leaf_label = f"<b>{' / '.join(qualifying)}</b>"
            label = (
                f"{leaf_label}<br>"
                f"<span style='{meta_style}'>"
                f"<b>[{i}]</b>  n={n_samples[i]}  H={binary_entropy:.2f}</span>"
            )
            hover_corr = []

        annotations.append(
            dict(
                x=node_x[i],
                y=node_y[i],
                text=label,
                name=str(i),  # node_id stored here for JS lookup
                showarrow=False,
                align="center",
                font=dict(size=11, family="JetBrains Mono, monospace"),
                bgcolor=node_color,
                bordercolor="rgba(0,0,0,0.14)",
                borderwidth=1,
                borderpad=7,
            )
        )

        node_data[i] = {
            "x": float(node_x[i]),
            "y": float(node_y[i]),
            "color": node_color,
            "dist": [
                {"cls": cls, "cnt": int(cnt), "frac": cnt / total_node, "color": color}
                for cls, cnt, color in zip(classes, label_distribution[i], hex_colors)
                if cls in model_classes_str
            ],
            "corr": hover_corr,
            "is_leaf": bool(feature[i] == -2),
            "parent": int(parent_map[i]) if i in parent_map else None,
            "rule": rule_map.get(i, None),
        }

    # ── Plot ──
    fig = go.Figure()
    fig.update_layout(clickmode="event")

    # Calculate edge widths based on absolute sample counts
    # Find max samples for normalization
    all_edges = edge_samples_true + edge_samples_false
    max_samples = (
        0 if len(all_edges) == 0 else max([s for s in all_edges if s is not None])
    )
    min_width = 0.5
    max_width = 8.0

    # Create individual edge traces with variable width
    def add_edges_with_variable_width(edge_x, edge_y, edge_samples, color, name):
        i = 0
        while i < len(edge_x):
            if edge_x[i] is not None:
                # Get the edge segment (2 points)
                x_segment = [edge_x[i], edge_x[i + 1]]
                y_segment = [edge_y[i], edge_y[i + 1]]
                samples = edge_samples[i]

                # Calculate width based on sample proportion
                if max_samples > 0:
                    width = min_width + (max_width - min_width) * (
                        samples / max_samples
                    )
                else:
                    width = min_width

                fig.add_trace(
                    go.Scatter(
                        x=x_segment,
                        y=y_segment,
                        mode="lines",
                        line=dict(width=width, color=color),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                i += 3  # Skip to next edge (current, next, None)
            else:
                i += 1

    # Add True branch edges (green)
    add_edges_with_variable_width(
        edge_x_true, edge_y_true, edge_samples_true, "#16a34a", "True"
    )

    # Add False branch edges (red)
    add_edges_with_variable_width(
        edge_x_false, edge_y_false, edge_samples_false, "#dc2626", "False"
    )

    fig.update_layout(annotations=annotations)

    # Large invisible hit-target markers — big enough to cover the full annotation box.
    # The hover shows correlated features (if any); click opens the example table.
    scatter_customdata = []
    scatter_hovertemplates = []
    for i in node_ids:
        nd = node_data[i]
        corr = nd["corr"]
        if corr:
            ht = (
                "<br>".join(f"≈ {c}" for c in corr)
                + "<br><i style='color:#9c9490'>Click to explore</i>"
            )
        else:
            ht = "<i style='color:#9c9490'>Click to explore examples</i>"
        scatter_customdata.append({"node_id": i, "hovertext": ht})

    fig.add_trace(
        go.Scatter(
            x=[node_x[i] for i in node_ids],
            y=[node_y[i] for i in node_ids],
            mode="markers",
            marker=dict(
                size=56,
                color="rgba(0,0,0,0)",
                line=dict(width=0),
            ),
            customdata=scatter_customdata,
            hovertemplate="%{customdata.hovertext}<extra></extra>",
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="#e7e5e4",
                font=dict(family="DM Sans, sans-serif", size=12, color="#1c1917"),
            ),
            showlegend=False,
        )
    )

    fig.update_layout(
        autosize=True,
        clickmode="event",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        # Transparent, not "white": the page (.tree-panel) paints its own
        # var(--bg) behind the chart, light or dark. Node boxes stay
        # light-toned regardless of theme (interpolate_color always blends
        # toward white), by design -- only the empty canvas behind them
        # needs to follow the page.
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family="DM Sans, sans-serif"),
    )

    write_html(
        fig,
        predictor_samples,
        node_data,
        out_file,
        classes,
        hex_colors,
        label_distribution[0],
        meta,
    )


def tree2html(
    pipeline_model,
    dt_df,
    full_df,
    predictor_var,
    out_file,
    target=None,
    max_rows=100,
    meta=None,
    only_show_real_orders=False,
    correlate_features=True,
    extra_columns=None,
    show_features=False,
    full_tree_html=True,
    palette_map=None,
    leaf_threshold=None,
    full_label_distribution=None,
    head_label="head",
):
    """
    pipeline_model:
        sklearn Pipeline with steps:
            - "preprocessor"
            - "clf" (DecisionTree*)

    dt_df / full_df:
        pandas DataFrames

    meta (optional dict):
        Extra info shown in the info panel, e.g.:
            {
                "Language": "Kurmanji",
                "Nodes": 23,
                "Depth": 5,
                "Training samples": 4486,
            }
        If not provided, these values are computed automatically where possible.
    leaf_threshold: the entropy cutoff sva_trees.create_pairs.create_pairs uses to
        decide "keep" (leaf_top1_entropy < leaf_threshold) -- shown in the info
        panel next to base/reduced entropy so it's visible right where it's
        needed, without cross-referencing the create_pairs run that produced
        this language's N Keep/minimal pairs. None omits the row (e.g. callers
        that don't know it, or aren't showing agreement diagnostics at all).
    full_label_distribution: {label: count} across the language's full
        predictor_var column, BEFORE word_order.decision_tree.fit_dt's
        UNK_LABELS drop (same dict sva_trees.pipeline already computes for
        the deprel overview's distribution bar). Classes fit_dt dropped
        (typically "unk", or "+-"/"--") never appear in the fitted model's
        own class list, so without this they'd show a misleading 0 in the
        root legend/distribution bar -- as if zero such rows existed, rather
        than however many were excluded before the tree ever saw them. Only
        applied to the root node (index 0): those rows never reach any split,
        so every other node's count for them is correctly 0. None (default)
        leaves dropped classes at 0, same as before this existed.
    head_label: cosmetic display label for the "head" role in the info-panel
        Predictor row and the sample table's column headers (e.g. "Verb" for
        SVA, "Aux" for subj_aux, "NP head" for npa) -- see
        _display_predictor_var/_relabel_head_columns. "head" (default) is a
        no-op, unchanged from before this existed.
    """
    if leaf_threshold is not None:
        meta = dict(meta or {})
        meta["Keep threshold"] = f"entropy &lt; {leaf_threshold:g}"

    if not full_tree_html:
        write_placeholder_html(
            out_file,
            _display_predictor_var(predictor_var, head_label),
            *build_placeholder_args(
                dt_df, full_df, predictor_var, meta, show_features=show_features, target=target,
                head_label=head_label,
            ),
        )
        return

    prep = pipeline_model.named_steps["preprocessor"]
    clf = pipeline_model.named_steps["clf"]
    tree = clf.tree_

    # Node metadata needed for class expansion
    model_classes = list(pipeline_model.classes_)
    class2idx_model = {c: idx for idx, c in enumerate(model_classes)}
    n_nodes = tree.node_count
    n_samples = tree.n_node_samples
    tree_values = tree.value
    # np.round, not .astype(int) (which truncates toward zero): tree_values[i][0]
    # is a float probability array, so the reconstructed count for a true integer
    # class count can land a hair under it (e.g. 1.9999999997 for a true 2) due to
    # ordinary floating-point noise -- truncating such values silently drops
    # samples from the displayed leaf distribution (and anything computed from
    # it, e.g. the on-node entropy annotation), diverging from the exact
    # dataframe-derived counts calculate_base_entropy/leaf_top1_entropy use.
    label_distribution_model = [
        np.round(tree_values[i][0] * n_samples[i]).astype(int).tolist() for i in range(n_nodes)
    ]

    # Pad to full classes permutation set so legend & nodes always show all classes.
    # Build display_classes in canonical class order; append any extras.
    # When target is None (e.g. binary attachment classifier) skip the canonical
    # word-order enumeration and use the model's own class list directly.
    if target is not None:
        all_orders = get_all_orders(predictor_var, target)
        if only_show_real_orders:
            all_orders = [
                order
                for order in all_orders
                if order in full_df[predictor_var].unique()
            ]
    else:
        all_orders = list(model_classes)

    classes = list(all_orders)
    for c in model_classes:
        if c not in classes:
            classes.append(c)
    # get_all_orders enumerates word-order permutation codes -- meaningless
    # for an agreement predictor_var (e.g. "Vs"/"sV"), so it and model_classes
    # (which never includes "unk"/"+-"/"--" now that fit_dt drops them before
    # fitting) never actually add a dropped class here. Append any class
    # full_label_distribution has that isn't already shown, so it gets a
    # legend entry/bar at all -- the counts themselves come from the root-node
    # backfill below.
    if full_label_distribution:
        for c in full_label_distribution:
            if c not in classes:
                classes.append(c)

    n_classes = len(classes)

    # Expand label_distribution to full display_classes (missing classes get 0,
    # except the root -- see full_label_distribution's docstring above).
    label_distribution = []
    for node_idx, dist in enumerate(label_distribution_model):
        full_dist = []
        for cls in classes:
            if cls in class2idx_model:
                full_dist.append(dist[class2idx_model[cls]])
            elif node_idx == 0 and full_label_distribution:
                full_dist.append(full_label_distribution.get(cls, 0))
            else:
                full_dist.append(0)
        label_distribution.append(full_dist)

    class2idx = {c: idx for idx, c in enumerate(classes)}

    # Auto-build meta if not provided
    if meta is None:
        meta = {}

    accuracy = pipeline_model.score(dt_df, dt_df[predictor_var])
    meta["accuracy"] = f"{accuracy * 100:.1f}%"

    base_ent = calculate_base_entropy(dt_df, predictor_var, binary=True)
    reduced_ent = calculate_tree_entropy(
        pipeline_model, dt_df, predictor_var, binary=True
    )
    meta["base entropy"] = f"{base_ent:.3f}"
    meta["reduced entropy"] = f"{reduced_ent:.3f}"

    tree_depth = clf.get_depth()
    n_leaves = clf.get_n_leaves()
    root_samples = n_samples[0]

    if "Nodes" not in meta:
        meta["Nodes"] = f"{n_nodes} ({n_leaves} leaves)"
    if "Depth" not in meta:
        meta["Depth"] = str(tree_depth)
    if "Training samples" not in meta:
        meta["Training samples"] = f"{root_samples:,}"
    if "Predictor" not in meta:
        meta["Predictor"] = _display_predictor_var(predictor_var, head_label)

    predictor_samples = get_samples(
        prep,
        clf,
        full_df,
        label_distribution,
        class2idx,
        dt_df,
        predictor_var,
        max_rows=max_rows,
        extra_columns=extra_columns,
        show_features=show_features,
        target=target,
        head_label=head_label,
    )

    _finalize_tree_html(
        pipeline_model,
        dt_df,
        classes,
        label_distribution,
        predictor_samples,
        meta,
        out_file,
        correlate_features=correlate_features,
        # Palette sized to all_orders so colours are stable across languages
        n_palette_colors=max(n_classes, len(all_orders)),
        palette_map=palette_map,
    )


def write_html(
    fig,
    node_samples,
    node_data,
    out_file,
    classes,
    hex_colors,
    root_dist_counts,
    meta,
):
    div_id = "tree-figure"

    html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
        div_id=div_id,
    )

    # Build legend items: swatch + label + count + horizontal bar
    max_root = max(root_dist_counts) or 1
    legend_items = ""
    for cls, color, cnt in zip(classes, hex_colors, root_dist_counts):
        bar_w = round((cnt / max_root) * 80)  # max 80px wide bar
        dimmed = "opacity:0.35;" if cnt == 0 else ""
        legend_items += f"""
        <div class="legend-item" style="{dimmed}">
          <span class="legend-swatch" style="background:{color};"></span>
          <span class="legend-label">{cls}</span>
          <span class="legend-count">{cnt}</span>
          <div class="legend-bar-track">
            <div class="legend-bar-fill" style="width:{bar_w}px;background:{color};"></div>
          </div>
        </div>"""

    # Build meta rows
    meta_rows = ""
    if meta:
        for key, val in meta.items():
            meta_rows += f'<div class="meta-row"><span class="meta-key">{key}</span><span class="meta-val">{val}</span></div>'

    meta["legend_items"] = legend_items
    meta["rows"] = meta_rows

    html += create_html(meta, node_samples, node_data, hex_colors, classes, div_id)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)
