import joblib
import json
import os
import re
import warnings

from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import _tree

import numpy as np
import pandas as pd

from agreement_trees.fit.entropy import smoothed_accuracy
from agreement_trees.extract.prediction_target import PredictionTarget
from agreement_trees.extract.process_treebank import (
    META_FEATURES, is_feature_col, is_attractor_col, is_attractor_slots_col,
)


from multiblimp.config import SEED as RND

# unseen categories at transform time are intentionally encoded as all zeros
warnings.filterwarnings(
    "ignore", message="Found unknown categories", category=UserWarning
)

# smallest df a tree is attempted on
MIN_DF_LEN = 10

# Tokens whose xpos is kept out of the tree fit: the head carries the agreement marking and some tagsets'
# xpos spells it out (label leak). The dependent's xpos stays available; excluded columns are marked in tables.
FIT_EXCLUDED_FEATS = ("head_xpos",)


def fit_excluded_for_display(omit_cols):
    """The token-level columns among a fit's omitted ones (e.g. head_Number,
    head_xpos) -- the ones a feature table cell can be marked for. The many
    relational (child/sibling) columns are left out."""
    return sorted(c for c in omit_cols
                  if re.fullmatch(r"[A-Za-z]+_(?:[A-Z][A-Za-z]*|xpos)(?:\[[^\]]+\])?", c))

UNK_LABELS = {"unk", "+-", "--"}


def _infer_column_types(cols: list[str]) -> tuple[list[str], list[str]]:
    """Split column names into (categorical_cols, numeric_cols) based on naming conventions."""
    numeric_cols = [
        col
        for col in cols
        if (
            "freq" in col
            or "#" in col
            or "under" in col
            or "question" in col
            or ("sibling" in col and not ("lemma" in col or "feat" in col))
            or "has_" in col
            or "distance" in col
        )
    ]
    categorical_cols = [col for col in cols if col not in set(numeric_cols)]
    return categorical_cols, numeric_cols


def _prune_rare_categoricals(
    X: pd.DataFrame, categorical_cols: list[str], min_support: int
) -> list[str]:
    """Drop categorical columns with fewer than min_support minority (non-mode) rows. The deprel/feat/pos
    cross-product makes many columns that are >99% one value: they survive nunique() > 1 but one-hot into a
    column per rare category (memory), and cannot produce a valid leaf anyway.
    """
    # dropna=False: NaN is its own bucket (a mode comparison breaks when the mode is NaN)
    return [
        col
        for col in categorical_cols
        if len(X) - X[col].value_counts(dropna=False).iloc[0] >= min_support
    ]


MISSING = "_missing"   # placeholder for an unstated feature in the fit frame (and so in dt_df)


def _fill_missing(
    X: pd.DataFrame, categorical_cols: list[str], numeric_cols: list[str]
) -> pd.DataFrame:
    """Fill NaN: categorical → MISSING (added as a category if needed), numeric → 0."""
    X = X.copy()
    for col in categorical_cols:
        if isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = X[col].cat.add_categories(MISSING)
            X[col] = X[col].fillna(MISSING)
        else:
            X[col] = X[col].fillna(MISSING)
        X[col] = X[col].astype(str)
    for col in numeric_cols:
        if isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X[numeric_cols] = X[numeric_cols].fillna(0)
    return X


def _compact_dtypes(X: pd.DataFrame, categorical_cols: list[str], numeric_cols: list[str]) -> pd.DataFrame:
    """The fit's input in cheaper dtypes, same values. Strings go through a category and back to object, so a
    column's cells share its few distinct string objects (sklearn's encoder otherwise builds one Python string per
    cell: lower peak RAM, faster fit than plain strings or a category input, which measured no better than strings).
    Integral numerics become the narrowest int (the 0/1 flags were int64/float64)."""
    X = X.copy()
    for col in categorical_cols:
        X[col] = X[col].astype("category").astype(object)
    for col in numeric_cols:
        values = X[col]
        if pd.api.types.is_float_dtype(values) and not (values == values.round()).all():
            continue
        X[col] = pd.to_numeric(values, downcast="integer")
    return X


def _build_and_fit_pipeline(
    model_type,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    categorical_cols: list[str],
    numeric_cols: list[str],
    max_depth: int,
    min_samples_leaf: int,
    min_impurity_decrease: float,
) -> Pipeline:
    """Build and fit an OHE + DecisionTree sklearn Pipeline on pre-filled X."""

    if model_type != "decision_tree":
        raise ValueError(f"unknown model_type {model_type!r}")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                categorical_cols,
            ),
            ("num", "passthrough", numeric_cols),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "clf",
                DecisionTreeClassifier(
                    criterion="entropy",
                    random_state=RND,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    min_impurity_decrease=min_impurity_decrease,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    return model


def fit_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    model_type,
    max_depth: int = 12,
    min_samples_leaf: int = 25,
    min_impurity_decrease: float = 0.005,
    test_size: float | None = None,
    random_state: int = RND,
    test_index=None,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame | None, pd.Series, pd.Series | None]:
    """
    Infer column types, fill missing values, optionally split, build and fit a DT Pipeline.
    Returns (model, X_train, X_test, y_train, y_test).
    X_test and y_test are None when test_size is None (no train/test split).
    test_index (row labels of X) overrides test_size/random_state with an
    explicit held-out set.
    """
    categorical_cols, numeric_cols = _infer_column_types(X.columns.tolist())
    min_support = max(min_samples_leaf, int(0.01 * len(X)))
    categorical_cols = _prune_rare_categoricals(X, categorical_cols, min_support)
    X = _fill_missing(X, categorical_cols, numeric_cols)

    for col in numeric_cols:
        if X[col].dtype == "object" or X[col].dtype.name == "category":
            X[col] = X[col].astype(int)
        elif X[col].dtype == "bool":
            X[col] = X[col].astype(int)
    X = _compact_dtypes(X, categorical_cols, numeric_cols)

    if test_index is not None:
        test_mask = X.index.isin(test_index)
        X_train, X_test = X[~test_mask], X[test_mask]
        y_train, y_test = y[~test_mask], y[test_mask]
    elif test_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    else:
        X_train, X_test, y_train, y_test = X, None, y, None

    model = _build_and_fit_pipeline(
        model_type,
        X_train,
        y_train,
        X_test,
        y_test,
        categorical_cols,
        numeric_cols,
        max_depth,
        min_samples_leaf,
        min_impurity_decrease,
    )
    return model, X_train, X_test, y_train, y_test


def fit_dt(
    full_df,
    model_type: str,
    target: PredictionTarget | None = None,
    deprel_kwargs=None,
    omit_feats=None,
    save_to=None,
    max_depth=12,
    min_samples_leaf=25,
    min_impurity_decrease=0.005,
    test_size=0.1,
    verbose=1,
    min_df_len=MIN_DF_LEN,
    predictor_var="deprel_order",
    drop_unk=True,
    test_index=None,
):
    """Fit the tree on full_df's rows (unk rows dropped with drop_unk) to predict `predictor_var`, from every
    column except the omitted ones (omit_feats plus META_FEATURES, *_idx, *_dir, *_form_orig and the other
    agreement labels). The omitted columns are re-attached to the returned frame. test_index: row labels
    held out instead of a random test_size split.

    Returns (model, X_train enriched by set_dt_features_in_df, y_train, unk_counts), or
    (None, None, None, unk_counts) when too few rows remain. unk_counts splits the dropped unk rows by
    which side lacks the feature (head_unk / nsubj_unk / both_unk).
    """
    sub_df = full_df.copy()
    if deprel_kwargs is not None:
        for k, v in deprel_kwargs.items():
            sub_df = sub_df[sub_df[k] == v]

    unk_counts = {"head_unk": 0, "nsubj_unk": 0, "both_unk": 0}
    unk_mask = sub_df[predictor_var].isin(UNK_LABELS)
    if unk_mask.any():
        feat_match = re.match(r".*_([A-Z][a-z]+)_.*", predictor_var)
        child_deprel = target.child_deprels[0] if target is not None else None
        head_col = f"head_{feat_match.group(1)}" if feat_match else None
        child_col = (
            f"{child_deprel}_{feat_match.group(1)}"
            if feat_match and child_deprel else None
        )
        if head_col in sub_df.columns and child_col in sub_df.columns:
            head_missing = sub_df[head_col].isna()
            child_missing = sub_df[child_col].isna()
            unk_counts["both_unk"] = int((unk_mask & head_missing & child_missing).sum())
            unk_counts["head_unk"] = int((unk_mask & head_missing & ~child_missing).sum())
            unk_counts["nsubj_unk"] = int((unk_mask & ~head_missing & child_missing).sum())
        # drop_unk=False keeps unk rows in the fit (the counts still report what would be dropped)
        if drop_unk:
            sub_df = sub_df[~unk_mask]

    if len(sub_df) < min_df_len:
        return None, None, None, unk_counts

    # "treebank" is metadata, but the tree may split on it
    omit_feats = (omit_feats or set()).union(set(META_FEATURES) - {"treebank"})
    omit_feats.update({col for col in full_df.columns if "idx" in col})
    omit_feats.update({col for col in full_df.columns if "_dir" in col})
    omit_feats.update({col for col in full_df.columns if col.endswith("_form_orig")})
    omit_feats.update({col for col in full_df.columns if is_attractor_col(col) or is_attractor_slots_col(col)})
    omit_feats.update(
        {
            col
            for col in full_df.columns
            if "agreement" in col and not predictor_var in col
        }
    )
    # head_feats columns are re-attached to dt_df although constant (dropped from X)
    if target is not None and target.head_feats:
        omit_feats.update({f"head_{feat}" for feat in target.head_feats})

    # frame column order, not set order (set order varies per process and would change split tie-breaking)
    sub_condition_on = [
        col for col in full_df.columns if col not in omit_feats and col != predictor_var
    ]

    X = sub_df[sub_condition_on].copy()
    X = X.loc[:, X.nunique() > 1].copy()

    dropped_form_cols = {
        col for col in sub_condition_on if col.endswith("_form") and col not in X.columns
    }
    omit_feats.update(dropped_form_cols)
    # constant feature columns are useless to the tree but create_pairs needs them (slot resolution, og_feats)
    omit_feats.update(
        col for col in sub_condition_on
        if is_feature_col(col) and col not in X.columns and full_df[col].notna().any()
    )
    # upos columns (head_pos, ...) are constant by construction but shown in the tables: restore them too
    omit_feats.update(
        col for col in sub_condition_on
        if col.endswith("_pos") and col not in X.columns and full_df[col].notna().any()
    )

    y = sub_df[predictor_var].copy()
    y[pd.isna(y)] = "None"

    model, X_train, X_test, y_train, y_test = fit_pipeline(
        X,
        y,
        model_type,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=min_impurity_decrease,
        test_size=test_size,
        test_index=test_index,
    )

    if X_test is not None and len(X_test):
        # per-row held-out results, so callers can compare models on shared rows
        model.test_eval_ = pd.DataFrame(
            {"y": y_test, "pred": model.predict(X_test)}, index=X_test.index
        )

    if model_type == "decision_tree":
        X_train = set_dt_features_in_df(
            model,
            X_train,
            full_df,
            predictor_var,
            additional_vars=omit_feats,
            omit_feats=omit_feats,
        )

    if save_to is not None:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        joblib.dump(model, save_to + ".joblib")
        X_train.to_parquet(save_to + ".parquet")
        with open(save_to + "_unk_counts.json", "w") as f:
            json.dump(unk_counts, f)

    if verbose > 0:
        print("Train acc", model.score(X_train, y_train))
        if X_test is not None:
            print("Test acc ", model.score(X_test, y_test))

    return model, X_train, y_train, unk_counts


def _node_depths(children_left, children_right):
    """Root-is-0 depth for every node id in a fitted tree's arrays (same
    convention as sklearn's own tree_.max_depth/get_depth())."""
    depths = np.zeros(len(children_left), dtype=int)
    stack = [(0, 0)]
    while stack:
        node_id, depth = stack.pop()
        depths[node_id] = depth
        left, right = children_left[node_id], children_right[node_id]
        if left != -1:
            stack.append((left, depth + 1))
        if right != -1:
            stack.append((right, depth + 1))
    return depths


def set_dt_features_in_df(
    model,
    df,
    full_df,
    predictor_var: str,
    additional_vars: set | None = None,
    omit_feats: set = set(),
):
    """df plus the tree-derived columns: leaf_id, leaf_depth, leaf_top1_acc (smoothed accuracy of the row's
    own class in its leaf, what the keep rule reads), leaf_decision (the leaf's majority class is the row's),
    leaf_rule, and the omitted columns of full_df (additional_vars, META_FEATURES, omit_feats, predictor_var).
    """
    X_trans = model.named_steps["preprocessor"].transform(df)
    leaf_ids = model.named_steps["clf"].apply(X_trans)

    tree = model.named_steps["clf"].tree_
    children_left = tree.children_left
    children_right = tree.children_right

    additional_vars = (additional_vars or set()).union(META_FEATURES).union(omit_feats)
    additional_vars.add(predictor_var)

    # skip columns already in df (e.g. "treebank" when fit on as a feature): concat would duplicate them
    new_cols = {
        col: full_df[col]
        for col in full_df.columns
        if col in additional_vars and col not in df.columns
    }
    new_cols["leaf_id"] = pd.Series(leaf_ids, index=df.index)
    # root-is-0 split count
    node_depths = _node_depths(children_left, children_right)
    new_cols["leaf_depth"] = pd.Series(node_depths[leaf_ids], index=df.index)

    class_to_idx = {c: i for i, c in enumerate(model.classes_)}
    predictor_series = full_df.loc[df.index, predictor_var]
    # the distribution depends only on the leaf: once per unique leaf
    leaf_distributions = {
        leaf_id: tree.value[leaf_id][0] * tree.n_node_samples[leaf_id]
        for leaf_id in sorted(set(leaf_ids))
    }

    leaf_top1_accs = []
    for leaf_id, label in zip(leaf_ids, predictor_series):
        leaf_distribution = leaf_distributions[leaf_id]
        n_right = leaf_distribution[class_to_idx[label]]
        leaf_top1_accs.append(smoothed_accuracy(n_right, sum(leaf_distribution)))
    new_cols["leaf_top1_acc"] = leaf_top1_accs

    tree_rules = get_tree_rules(model, tight=True)
    new_cols["leaf_rule"] = [tree_rules[node] for node in leaf_ids]

    leaf2var = {
        idx: model.classes_[var.item()] for idx, var in enumerate(tree.value.argmax(-1))
    }
    new_cols["leaf_decision"] = [
        label == leaf2var[leaf_id] for leaf_id, label in zip(leaf_ids, predictor_series)
    ]

    # the returned frame is stored and reused (pairs, pages): object columns back to category
    df = df.copy(deep=False)
    for col in df.columns[df.dtypes == object]:
        df[col] = df[col].astype("category")
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def get_rule_for_leaf(model, leaf_id, tight=False):
    """Returns a human-readable rule for a leaf node."""
    tree = model.named_steps["clf"].tree_
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    children_left = tree.children_left
    children_right = tree.children_right
    threshold = tree.threshold
    feature = tree.feature

    # Build parent pointers
    parent_position = {}
    stack = [0]  # root node
    while stack:
        node = stack.pop()
        for child in [children_left[node], children_right[node]]:
            if child != -1:
                parent_position[child] = node
                stack.append(child)

    path = []
    node = leaf_id

    while node in parent_position:
        parent = parent_position[node]

        # Determine direction
        direction = "<=" if children_left[parent] == node else ">"

        feat = feature[parent]
        if feat != _tree.TREE_UNDEFINED:
            rule = f"{feature_names[feat]} {direction} {threshold[parent]:.4f}"
            rule = rule.replace("num__", "").replace("cat__", "")
            items = rule.split("_")
            if "<=" in rule:
                rule = (
                    "_".join(items[:-1]) + " != " + items[-1].replace(" <= 0.5000", "")
                )
            elif ">" in rule:
                rule = "_".join(items[:-1]) + " = " + items[-1].replace(" > 0.5000", "")

            if tight:
                path.append(f"[{parent}]")
            else:
                path.append(f"[{parent}] {rule}")

        node = parent

    return path[::-1]


def get_tree_rules(model, tight=False):
    tree = model.named_steps["clf"].tree_
    leaf_ids = np.where((tree.children_left == -1) & (tree.children_right == -1))[0]

    leaf_rules = {
        int(leaf): get_rule_for_leaf(model, leaf, tight=tight) for leaf in leaf_ids
    }

    if tight:
        leaf_rules = {
            leaf: " -> ".join(rule + [f"[{leaf}]"]) for leaf, rule in leaf_rules.items()
        }

    return leaf_rules
