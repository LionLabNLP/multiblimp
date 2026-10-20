import re
import warnings

import numpy as np
import scipy
import seaborn as sns
import pandas as pd

from matplotlib.colors import to_hex

from agreement_trees.fit.entropy import (
    order_entropy, smoothed_accuracy, calculate_base_entropy, calculate_tree_entropy,
    calculate_base_accuracy, calculate_tree_accuracy,
)
from agreement_trees.utils import build_grew_link, split_pairwise_predictor, is_agreement_predictor
from agreement_trees.extract.process_treebank import resolve_layered_head_key
from agreement_trees.report.html.html_tree import write_placeholder_html, write_html_v2


# "num__" features that are 0/1 presence flags (passed through, not one-hot): a split on them is always at
# 0.5, so no threshold is shown.
_BOOLEAN_NUM_FAMILIES = {"sibling", "sibling-L", "sibling-R", "sibling-pos", "under"}


def _relational_rule(e):
    """Readable label for features describing another node (child, sibling, head, grandhead, ancestor) or one
    of its attributes; None for the rest. `e`: the "_"-split feature name, "_missing" already turned into "nan"."""
    if len(e) < 3:
        return None
    role, fam = e[0], e[1]
    if fam in ("child-feat", "sibling-feat") and len(e) >= 5:
        noun = fam.split("-")[0]
        val = "_".join(e[4:])
        tail = "is not set" if val == "nan" else f"= {val}"
        return f"{role}'s {e[2]} {noun}: {e[3]} {tail}"
    if fam in ("child-lemma", "sibling-lemma") and len(e) >= 4:
        noun = fam.split("-")[0]
        lemma = "_".join(e[3:])
        if lemma == "None":
            return f"{role} has no {e[2]} {noun}?"
        tail = "is not set" if lemma == "nan" else f"= {lemma}"
        return f"{role}'s {e[2]} {noun}: lemma {tail}"
    if fam in ("head", "grandhead") and len(e) >= 4:
        val = "_".join(e[3:])
        tail = "is not set" if val == "nan" else f"= {val}"
        return f"{role}'s {fam}: {e[2]} {tail}"
    if fam == "under" and len(e) == 3:
        return f"{role} is under {e[2]}?"
    return None


def clean_rule(rule, threshold=None, is_binary=False, head_label="head"):
    numerical = rule.startswith("num__")
    rule = (
        rule.removeprefix("num__")
        .removeprefix("cat__")
        .replace("sibling-deprel", "sibling")
        .replace("_missing", "nan")
    )

    if rule == "in_question":
        return "in a question?"

    if rule == "child_head_distance":
        return "child-head distance" + (f" ≤ {threshold:.1f}" if threshold is not None else "")

    rule_elements = rule.split("_")
    if head_label != "head" and rule_elements[0].lower() == "head":
        rule_elements[0] = head_label
    rule_elements[0] = (
        rule_elements[0].replace("nsubj", "subject").replace("obj", "object")
    )

    if len(rule_elements) > 1 and rule_elements[1] in _BOOLEAN_NUM_FAMILIES:
        is_binary = True

    if (relational := _relational_rule(rule_elements)) is not None:
        rule = relational
    elif rule_elements[-1] == "nan":
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
        rule = f"{rule_elements[0]} has {rule_elements[2]} child?"
    else:
        rule = "_".join(rule_elements[:-1]) + f" = {rule_elements[-1]}"

    if numerical and threshold is not None and not is_binary:
        rule += f" ≤ {threshold:.2f}"

    return rule


def get_correlated_features(prep, clf, dt_df, head_label="head"):
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

    rule_names = [clean_rule(f, head_label=head_label) for f in feature_names]

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


# ── Agreement swap highlighting (head <-> child feature) ──────────────────────
# Applies only to agreement predictors, in two conventions (VP names never contain "-"): pairwise
# "{Role1}-{Role2}_{Feature}" (NPA; both roles highlighted by name, uppercase as in the column names), and
# VP's "head_{child_deprel}_{Feature}_agreement" (needs a PredictionTarget; highlights "head" and the child).

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
    """predictor_var for the info panel's "Predictor" row: "_agreement" dropped and the "head" role (either
    convention, case-insensitive) shown as head_label ("Verb", "Aux", "NP head"). The column names are untouched."""
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
    """Rename "head_*" / "HEAD_*" keys of the sample records ("head_form" -> "Aux_form") to head_label."""
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


# Verbal argument-indexing dimensions written as layered features (Number[obj], Person[erg], Gender[abs]);
# Case is the argument's own morphology and is never bracketed on the verb.
_LAYERED_DISPLAY_FEATS = ("Number", "Person", "Gender")


def _build_feat_columns(full_df, swap_feature=None, highlight_prefixes=(), target_deprel=None):
    """{prefix}_features list columns: one "Feat=Val" string per set node feature (None/NaN/"_" skipped), with
    the swap feature bold for prefixes in highlight_prefixes. upos, deprel, lemma (xpos) come first. Every row
    contributes a (possibly empty) list to every prefix, or the columns assigned by the callers misalign.
    """
    # the prefix is [A-Za-z]+ (VP prefixes are lowercase, NPA roles uppercase); the suffix admits layered features
    morph_df = full_df.filter(regex=r"^[A-Za-z]+_[A-Z][a-zA-Z]+(?:\[[^\]]+\])?$", axis=1)
    extra_cols = []
    for suffix in ("pos", "deprel", "lemma", "xpos"):
        extra_cols += [c for c in full_df.columns if re.match(rf"^[A-Za-z]+_{suffix}$", c)]
    feat_df = pd.concat([full_df[extra_cols], morph_df], axis=1) if extra_cols else morph_df
    feat_cols = list(feat_df.columns)
    # in column order, not set order (which changes with every process)
    prefixes = list(dict.fromkeys(label.split("_")[0] for label in feat_cols))
    feat_collect = {f"{p}_features": [] for p in prefixes}

    # for obj/iobj targets the plain head_{feat} is the subject's folded value, not this argument's marker (as
    # in process_treebank.extract_instances): suppress it and use the resolved bracketed key instead
    resolve_bracket_feats = bool(target_deprel) and target_deprel != "nsubj" and "head" in highlight_prefixes

    for row_tuple in feat_df.itertuples(index=False, name=None):
        mf = {f"{p}_features": [] for p in prefixes}
        row_vals = {}
        for label, val in zip(feat_cols, row_tuple):
            if pd.isna(val) or val == "_":
                continue
            prefix, feature = label.split("_", 1)
            val_str = str(val)[:-2] if str(val).endswith(".0") else str(val)
            row_vals[label] = val_str
            if resolve_bracket_feats and prefix == "head" and feature in _LAYERED_DISPLAY_FEATS:
                continue  # resolved below instead, from the real bracketed key
            key = f"{prefix}_features"
            display_feature = "upos" if feature == "pos" else feature
            entry = f"{display_feature}={val_str}"
            if swap_feature and feature == swap_feature and prefix in highlight_prefixes:
                entry = f"<strong>{entry}</strong>"
            mf[key].append(entry)

        if resolve_bracket_feats:
            for feat in _LAYERED_DISPLAY_FEATS:
                layered_key = resolve_layered_head_key(row_vals, row_vals, feat, target_deprel)
                if layered_key is None:
                    continue
                value = row_vals.get(f"head_{layered_key}")
                if value is None:
                    continue
                entry = f"{feat}={value}"
                if swap_feature and feat == swap_feature:
                    entry = f"<strong>{entry}</strong>"
                mf["head_features"].append(entry)

        for k, v in mf.items():
            feat_collect[k].append(v)

    # the child's deprel is constant, so the data build dropped it: put it back for display
    if target_deprel and target_deprel in prefixes and f"{target_deprel}_deprel" not in full_df.columns:
        feat_collect[f"{target_deprel}_features"] = [
            v + [f"deprel={target_deprel}"] for v in feat_collect[f"{target_deprel}_features"]
        ]
    return feat_collect


def _feature_sources(prep):
    """{output feature index: (raw column, category or None)} for the
    preprocessor's one-hot ("cat") and passthrough ("num") outputs, in
    output order. {} if the layout isn't what's expected."""
    src, idx = {}, 0
    for name, trans, cols in prep.transformers_:
        if name == "cat" and hasattr(trans, "categories_"):
            drops = trans.drop_idx_ if trans.drop_idx_ is not None else [None] * len(cols)
            for col, cats, drop in zip(cols, trans.categories_, drops):
                for j, cat in enumerate(cats):
                    if drop is not None and j == drop:
                        continue
                    src[idx] = (col, cat)
                    idx += 1
        elif name == "num":
            for col in cols:
                src[idx] = (col, None)
                idx += 1
    return src if idx == len(prep.get_feature_names_out()) else {}


def _fmt_route_value(v):
    if v is None or (not isinstance(v, str) and pd.isna(v)) or v in ("None", "_missing", "nan", ""):
        return "(not set)"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def _sample_routes(prep, clf, full_df, dt_df, ids):
    """{row id: {split node id (str): the row's raw value of the column that
    node splits on}} along each row's path through the tree."""
    src = _feature_sources(prep)
    ids = list(dict.fromkeys(ids))
    if not src or not ids:
        return {}
    X = prep.transform(dt_df.loc[ids])
    indicator = clf.decision_path(X)
    feature = clf.tree_.feature
    routes = {}
    for pos, sid in enumerate(ids):
        steps = {}
        for node in indicator[pos].indices:
            f = feature[node]
            if f < 0 or f not in src:
                continue
            col = src[f][0]
            source = full_df if col in full_df.columns else dt_df
            steps[str(int(node))] = _fmt_route_value(source.at[sid, col])
        routes[sid] = steps
    return routes


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

    all_ids = list(dict.fromkeys(
        sid for node_map in sample_ids.values() for ids in node_map.values() for sid in ids))
    # display columns are built for the sampled rows only
    shown_df = full_df.loc[all_ids].copy()

    keep_columns = ["sen_str"]
    _add_sen_str_column(shown_df, highlight_prefixes=highlight_prefixes)
    shown_df["treebank_link"] = build_treebank_links(shown_df)

    if show_features:
        target_deprel = target.child_deprels[0] if target and target.child_deprels else None
        feat_collect = _build_feat_columns(shown_df, swap_feature=swap_feature,
                                            highlight_prefixes=highlight_prefixes,
                                            target_deprel=target_deprel)

        for k, v in feat_collect.items():
            shown_df[k] = v
            keep_columns.extend([f"{k.split("_")[0]+"_form"}", k])
    else:
        keep_columns.extend([col for col in shown_df if col.endswith("_form")])
    keep_columns.extend(["treebank_link"])

    if extra_columns:
        keep_columns.extend(
            [
                col
                for col in extra_columns
                if col in shown_df.columns and col not in keep_columns
            ]
        )

    routes = _sample_routes(prep, clf, full_df, dt_df, all_ids)

    def _rows_with_routes(ids):
        records = _relabel_head_columns(shown_df.loc[ids][keep_columns].to_dict("records"), head_label)
        for rec, sid in zip(records, ids):
            rec["route"] = routes.get(sid, {})
        return records

    predictor_samples = {}
    for predictor, node_sample_ids in sample_ids.items():
        predictor_samples[predictor] = {
            int(node_idx): {
                "rows": _rows_with_routes(sample_ids),
                "count": len(sample_ids),
                "total_count": label_distribution[node_idx][class2idx[predictor]],
            }
            for node_idx, sample_ids in node_sample_ids.items()
        }

    return predictor_samples


def compute_tree_layout(tree):
    """(node_id -> x, node_id -> y, node ids): root at the top, leaves evenly spaced."""
    children_left = tree.children_left
    children_right = tree.children_right

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

    return node_x, node_y, list(node_x.keys())


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
    """Build grew.fr query links for each row in full_df.

    Prefers the untouched-casing "*_form_orig" columns over the lowercased
    "*_form" ones -- see build_grew_link.
    """

    orig_cols = [x for x in full_df.columns if x.endswith("_form_orig")]
    form_cols = orig_cols or [x for x in full_df.columns if x.endswith("_form")]
    treebanks = full_df["treebank"] if "treebank" in full_df.columns else [None] * len(full_df)
    sent_ids = full_df["sent_id"] if "sent_id" in full_df.columns else [None] * len(full_df)
    form_rows = zip(*(full_df[c] for c in form_cols)) if form_cols else [()] * len(full_df)
    tb_links = []
    for treebank, sent_id, form_values in zip(treebanks, sent_ids, form_rows):
        link = build_grew_link(treebank, sent_id, list(form_values))
        tb_links.append(link if link is not None else "&mdash;")
    return tb_links


PLACEHOLDER_ROWS_PER_CLASS = 10
_LABEL_DISPLAY = {"yes": "Agree", "no": "Disagree", "unk": "unknown"}


def _display_label(label) -> str:
    return _LABEL_DISPLAY.get(str(label).lower(), str(label))


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

    # up to PLACEHOLDER_ROWS_PER_CLASS examples of every class, so rare ones show up too
    sampled_idx, class_of = [], []
    for cls, group in dt_df.groupby(predictor_var, observed=True):
        picked = group.sample(min(len(group), PLACEHOLDER_ROWS_PER_CLASS), random_state=42)
        sampled_idx.extend(picked.index)
        class_of.extend([_display_label(cls)] * len(picked))

    # display columns are built for the sampled rows only
    shown_df = full_df.loc[sampled_idx].copy()
    _add_sen_str_column(shown_df, highlight_prefixes=highlight_prefixes)
    shown_df["treebank_link"] = build_treebank_links(shown_df)

    keep_columns = ["sen_str"]
    if show_features:
        target_deprel = target.child_deprels[0] if target and target.child_deprels else None
        feat_collect = _build_feat_columns(shown_df, swap_feature=swap_feature,
                                            highlight_prefixes=highlight_prefixes,
                                            target_deprel=target_deprel)

        for k, v in feat_collect.items():
            shown_df[k] = v
            keep_columns.extend([f"{k.split("_")[0]+"_form"}", k])
    else:
        keep_columns.extend([col for col in shown_df if col.endswith("_form")])

    keep_columns.append("treebank_link")

    sample_df = shown_df[keep_columns].copy()
    sample_df.insert(0, "category", class_of)
    sample_rows = sample_df.sort_values("category", kind="stable").to_dict("records")
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
    head_label="head",
    leaf_threshold=None,
    strict_leaf_threshold=None,
    correct_swaps_df=None,
):
    """Build the tree page (create_html_v2) for a fitted pipeline.

    data_df: frame for the correlated-feature computation. classes: ordered display classes (legend order,
    colours). label_distribution: per node, one count per class. predictor_samples: per-class example dicts
    from get_samples(). meta: info-panel dict, must contain "accuracy". n_palette_colors: palette size
    override for colour-stable comparisons across datasets.
    """
    prep = pipeline_model.named_steps["preprocessor"]
    clf = pipeline_model.named_steps["clf"]

    tree = clf.tree_
    node_x, node_y, node_ids = compute_tree_layout(tree)

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
    # classes fit_dt dropped ("unk") only have a count at the root (legend); other nodes never saw those rows
    model_classes_str = {str(c) for c in model_classes}

    feature_names = prep.get_feature_names_out()

    n_features_out = len(feature_names)
    cat_slice = prep.output_indices_.get("cat")
    binary_feature_indices = (
        set(range(*cat_slice.indices(n_features_out))) if cat_slice is not None else set()
    )

    if correlate_features:
        correlated_features = get_correlated_features(prep, clf, data_df, head_label=head_label)
    else:
        correlated_features = {}

    if palette_map:
        # fixed order and colours from palette_map; realign the label distribution to it
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
        palette = sns.color_palette("husl", n_colors=n_palette_colors or n_classes)
        hex_colors = [to_hex(c) for c in palette[:n_classes]]

        # grey out classes absent from this language
        root_dist = label_distribution[0]
        hex_colors = [
            color if root_dist[idx] > 0 else "#d4d4d4"
            for idx, color in enumerate(hex_colors)
        ]

    node_data = {}

    children_left = tree.children_left
    children_right = tree.children_right
    parent_map = {}
    rule_map = {}  # node_id -> (rule_text, is_true_branch)

    for node_id in node_ids:
        left_child = children_left[node_id]
        right_child = children_right[node_id]

        if left_child != -1:
            parent_map[left_child] = node_id
            parent_map[right_child] = node_id

            if feature[node_id] != -2:
                raw_feat = feature_names[feature[node_id]]
                rule_text = clean_rule(
                    raw_feat,
                    threshold=tree.threshold[node_id],
                    is_binary=feature[node_id] in binary_feature_indices,
                    head_label=head_label,
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
        # fractions and accuracy count only the classes the tree fit on (the root's dist also carries the dropped ones)
        dist_i_fitted = [
            cnt for cnt, cls in zip(dist_i, classes) if cls in model_classes_str
        ]
        n_right = max(dist_i_fitted)
        n_wrong = sum(dist_i_fitted) - n_right
        binary_entropy = order_entropy(n_right, n_wrong)
        node_acc = smoothed_accuracy(n_right, n_right + n_wrong)
        node_color = interpolate_color(node_color, "#ffffff", relative_entropy * 0.72)

        total_node = sum(dist_i_fitted) or 1

        if feature[i] != -2:
            rule = clean_rule(
                feature_names[feature[i]],
                threshold=tree.threshold[i],
                is_binary=feature[i] in binary_feature_indices,
                head_label=head_label,
            )
            hover_corr = list(correlated_features.get(i, []))
        else:
            hover_corr = []

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
            "n": int(n_samples[i]),
            "H": float(binary_entropy),
            "acc": float(node_acc),
            # "rule" above is the incoming branch condition; "own_rule" is this node's own split
            "own_rule": rule if feature[i] != -2 else None,
        }

    write_html_v2(
        predictor_samples,
        node_data,
        out_file,
        classes,
        hex_colors,
        label_distribution[0],
        dict(meta),
        leaf_threshold=leaf_threshold,
        strict_leaf_threshold=strict_leaf_threshold,
        head_label=head_label,
        correct_swaps_df=correct_swaps_df,
    )


def _root_card_page(dt_df, full_df, predictor_var, out_file, meta=None, **kwargs):
    """The tree page with no tree: a one-node tree whose root card carries the "no tree"
    notice, so the page keeps the tree layout, legend and per-class example tables."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier

    dt_df = dt_df.assign(_root=0)
    labels = dt_df[predictor_var].astype(str)
    clf = DecisionTreeClassifier(min_samples_split=len(dt_df) + 1)
    model = Pipeline([("preprocessor", ColumnTransformer([("root", "passthrough", ["_root"])])), ("clf", clf)])
    model.fit(dt_df, labels)
    discounted = set(kwargs.get("discounted_classes") or [])
    counted = labels[~labels.isin(discounted)]
    if discounted and labels.isin(discounted).any():
        names = [_display_label(c) for c in counted.value_counts().index]
        n = int(len(counted))
        rows = f"{n} sample{'s' if n != 1 else ''}"
        if not names:
            text = "Every class here is greyed out: none of these rows is counted in this view."
        elif len(names) == 1:
            text = f"Only {names[0]} is counted in this view: a single class, with {rows}, too few to fit a tree."
        else:
            text = f"Only {' and '.join(names)} are counted in this view, but their {rows} are too few to fit a tree."
        notice = {"heading": "Too few samples to fit a tree", "text": text}
    elif labels.nunique() == 1:
        notice = {"heading": "Single label",
                  "text": "All training samples share one agreement label, so no decision tree was needed."}
    else:
        notice = {"heading": "Too few samples to fit a tree",
                  "text": "The training samples don&rsquo;t share one agreement label, but there weren&rsquo;t "
                          "enough of them to fit a decision tree. Examples of every class are listed below."}
    meta = dict(meta or {})
    meta["rootNotice"] = notice
    tree2html(model, dt_df, full_df, predictor_var, out_file, meta=meta, correlate_features=False,
              full_tree_html=True, **kwargs)


def tree2html(
    pipeline_model,
    dt_df,
    full_df,
    predictor_var,
    out_file,
    target=None,
    max_rows=100,
    meta=None,
    correlate_features=True,
    extra_columns=None,
    show_features=False,
    full_tree_html=True,
    palette_map=None,
    leaf_threshold=None,
    strict_leaf_threshold=None,
    full_label_distribution=None,
    head_label="head",
    correct_swaps_df=None,
    treebank_nav=None,
    fit_excluded=None,
    discounted_classes=None,
):
    """Write the tree page for a fitted pipeline (a placeholder or one-node root card when full_tree_html is
    False or there is no tree).

    pipeline_model: sklearn Pipeline ("preprocessor", "clf"). dt_df: the tree's frame; full_df: all rows
    of the language/treebank, for sample tables and distributions. meta: extra info-panel entries
    (Language, Nodes, Depth, ...; computed where absent).
    discounted_classes: classes this view leaves out of its fit (unk in the as-is view); greyed out.
    leaf_threshold: the accuracy floor create_pairs keeps leaves by, shown in the info panel (None omits it).
    strict_leaf_threshold: the first-pass floor; with a second-chance retry leaf_threshold is the lax one and
        leaves at or below this value are tagged as kept only by the retry.
    full_label_distribution: {label: count} over the full predictor column before unk rows were dropped, so
        the root legend shows them (applied to the root only; no other node sees those rows).
    head_label: display name of the "head" role ("Verb", "Aux", "NP head"; header text only).
    correct_swaps_df: create_pairs' "correct_swaps" bucket, joined by leaf_id into the pairs section.
    """
    if leaf_threshold is not None and not isinstance(leaf_threshold, dict):
        meta = dict(meta or {})
        meta["Keep threshold"] = f"leaf accuracy &gt; {leaf_threshold:g}"

    if not full_tree_html:
        try:
            _root_card_page(
                dt_df, full_df, predictor_var, out_file, target=target, max_rows=max_rows, meta=meta,
                show_features=show_features, palette_map=palette_map,
                full_label_distribution=full_label_distribution, head_label=head_label,
                treebank_nav=treebank_nav, fit_excluded=fit_excluded,
                discounted_classes=discounted_classes)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"root-card page failed ({type(e).__name__}: {e}); writing the plain placeholder")
            write_placeholder_html(
                out_file,
                _display_predictor_var(predictor_var, head_label),
                *build_placeholder_args(
                    dt_df, full_df, predictor_var, meta, show_features=show_features, target=target,
                    head_label=head_label,
                ),
                nav=treebank_nav,
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
    # np.round, not astype(int): the reconstructed counts can land a hair under an integer
    label_distribution_model = [
        np.round(tree_values[i][0] * n_samples[i]).astype(int).tolist() for i in range(n_nodes)
    ]

    # display classes: the model's own classes, then any the fit dropped
    classes = list(model_classes)
    # add classes fit_dt dropped (unk, +-, --) so they get a legend entry; their counts come from the root backfill
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

    # Classes with rows in the data that the fit dropped (unk, when the run
    # drops it): shown greyed in the info-panel legend, for context only.
    excluded = [c for c in classes if c not in class2idx_model and label_distribution[0][class2idx[c]] > 0]
    excluded += [c for c in (discounted_classes or []) if c in classes and c not in excluded
                 and label_distribution[0][class2idx[c]] > 0]
    if excluded:
        meta["excludedClasses"] = excluded

    meta["measure"] = "accuracy" if is_agreement_predictor(predictor_var) else "entropy"
    accuracy = pipeline_model.score(dt_df, dt_df[predictor_var])
    meta["accuracy"] = f"{accuracy * 100:.1f}%"
    # Held-out accuracy of this same tree, from the test split it was fit with
    # (fit_dt attaches test_eval_ to the model); absent for older cached models.
    test_eval = getattr(pipeline_model, "test_eval_", None)
    if test_eval is not None and len(test_eval):
        test_acc = float((test_eval["y"] == test_eval["pred"]).mean())
        meta["testAccuracy"] = f"{test_acc * 100:.1f}%"
        meta["testN"] = int(len(test_eval))

    if meta["measure"] == "accuracy":
        meta["root accuracy"] = f"{calculate_base_accuracy(dt_df, predictor_var) * 100:.1f}%"
        meta["train accuracy (smoothed)"] = f"{calculate_tree_accuracy(pipeline_model, dt_df, predictor_var) * 100:.1f}%"
    else:
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
    if treebank_nav:
        meta["treebankNav"] = treebank_nav
    if fit_excluded:
        # columns kept out of the fit but shown in the feature tables with an "x" marker
        meta["fitExcluded"] = list(fit_excluded)

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
        n_palette_colors=n_classes,
        palette_map=palette_map,
        head_label=head_label,
        leaf_threshold=leaf_threshold,
        strict_leaf_threshold=strict_leaf_threshold,
        correct_swaps_df=correct_swaps_df,
    )


