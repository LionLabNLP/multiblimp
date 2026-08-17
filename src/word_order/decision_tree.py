import joblib
import json
import os
import re

from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import _tree
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd

from .entropy import order_entropy
from .prediction_target import PredictionTarget
from .process_treebank import META_FEATURES
from .utils import get_all_orders


RND = 42

#OMIT_FEATURES = []#["subject_idx", "object_idx", "verb_idx"]
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
    """Drop categorical columns whose minority (non-mode) rows number fewer
    than min_support.

    fit_dt's X.nunique() > 1 filter only removes truly constant columns, but
    the deprel/feat/pos cross-product in extract_node_features generates many
    categorical columns that are >99% one value with just a handful of rows
    in the tail. Those survive nunique() > 1 yet one-hot encode into a column
    per rare category, which is what blows up ColumnTransformer/OneHotEncoder
    memory on languages with large, heterogeneous treebanks. A column whose
    minority class has fewer rows than min_support also can't produce a valid
    decision-tree leaf anyway, so pruning it costs no real signal.
    """
    # value_counts(dropna=False) treats NaN as its own bucket; a plain
    # `col != col.mode()` comparison breaks when the mode itself is NaN.
    return [
        col
        for col in categorical_cols
        if len(X) - X[col].value_counts(dropna=False).iloc[0] >= min_support
    ]


def _fill_missing(
    X: pd.DataFrame, categorical_cols: list[str], numeric_cols: list[str]
) -> pd.DataFrame:
    """Fill NaN: categorical → '_missing' (added as a category if needed), numeric → 0."""
    X = X.copy()
    for col in categorical_cols:
        if isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = X[col].cat.add_categories("_missing")
            X[col] = X[col].fillna("_missing")
        else:
            X[col] = X[col].fillna("_missing")
        X[col] = X[col].astype(str)
    for col in numeric_cols:
        if isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X[numeric_cols] = X[numeric_cols].fillna(0)
    return X


class ReconstructDF(BaseEstimator, TransformerMixin):
    """ColumnTransformer outputs a bare numpy array. This turns it back into a
    DataFrame with cleaned names and int-typed categoricals, in order cat+num."""

    def __init__(self, categorical_cols, numeric_cols):
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols

    def fit(self, X, y=None):
        ordered = list(self.categorical_cols) + list(self.numeric_cols)
        self.clean_cols_ = [self._clean(c) for c in ordered]
        self.clean_categorical_ = [self._clean(c) for c in self.categorical_cols]
        self.clean_numeric_ = [self._clean(c) for c in self.numeric_cols]
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=self.clean_cols_)
        df[self.clean_categorical_] = df[self.clean_categorical_].astype(int)
        df[self.clean_numeric_] = (
            df[self.clean_numeric_]
            .apply(pd.to_numeric, errors="coerce")
            .astype("float64")
        )
        return df

    @staticmethod
    def _clean(name):
        return re.sub(r"[^A-Za-z0-9_]", "_", str(name))


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

    if model_type == "gradient_boosting":
        import lightgbm as lgb
        from sklearn.preprocessing import OrdinalEncoder

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                    categorical_cols,
                ),
                ("num", "passthrough", numeric_cols),
            ]
        )

        prep = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("reconstruct", ReconstructDF(categorical_cols, numeric_cols)),
            ]
        )

        clf = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.025,
            num_leaves=31,
            min_child_samples=25,
            reg_alpha=0.5,
            reg_lambda=0.5,
            colsample_bytree=0.8,
            # subsample=0.8,   # bagging
            # subsample_freq=1,
            device="gpu",
            random_state=RND,
            verbose=-1,
        )

        # Transform once so eval_set / early stopping has access to preprocessed X_test
        X_train_t = prep.fit_transform(X_train, y_train)
        X_test_t = prep.transform(X_test)
        cat_feature = prep.named_steps["reconstruct"].clean_categorical_

        clf.fit(
            X_train_t,
            y_train,
            categorical_feature=cat_feature,
            eval_set=[(X_test_t, y_test)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )

        # Final pipeline that accepts RAW X (steps already fitted, no refit happens)
        model = Pipeline([("preprocessor", prep), ("clf", clf)])

    elif model_type == "decision_tree":
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
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame | None, pd.Series, pd.Series | None]:
    """
    Infer column types, fill missing values, optionally split, build and fit a DT Pipeline.
    Returns (model, X_train, X_test, y_train, y_test).
    X_test and y_test are None when test_size is None (no train/test split).
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

    if test_size is not None:
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
    min_df_len=10,
    leaf_threshold=0.1,
    predictor_var="deprel_order",
    drop_unk=True,
):
    sub_df = full_df.copy()
    if deprel_kwargs is not None:
        for k, v in deprel_kwargs.items():
            sub_df = sub_df[sub_df[k] == v]

    unk_counts = {"head_unk": 0, "nsubj_unk": 0, "both_unk": 0}
    if "agreement" in predictor_var:
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
            # drop_unk=False keeps unk rows in the fit (counts above still
            # report what WOULD be dropped)
            if drop_unk:
                sub_df = sub_df[~unk_mask]

    if len(sub_df) < min_df_len:
        return None, None, None, unk_counts

    omit_feats = (omit_feats or set()).union(set(META_FEATURES))
    omit_feats.update({col for col in full_df.columns if "idx" in col})
    omit_feats.update({col for col in full_df.columns if "_dir" in col})
    omit_feats.update(
        {
            col
            for col in full_df.columns
            if "agreement" in col and not predictor_var in col
        }
    )
    # Keep target.head_feats columns in omit_feats so set_dt_features_in_df
    # restores them into dt_df even though X.nunique() drops them as constant.
    if target is not None and target.head_feats:
        omit_feats.update({f"head_{feat}" for feat in target.head_feats})

    sub_condition_on = list(set(full_df.columns) - omit_feats - {predictor_var})

    X = sub_df[sub_condition_on].copy()
    X = X.loc[:, X.nunique() > 1].copy()

    dropped_form_cols = {
        col for col in sub_condition_on if col.endswith("_form") and col not in X.columns
    }
    omit_feats.update(dropped_form_cols)

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
    )

    if model_type == "decision_tree":
        X_train = set_dt_features_in_df(
            model,
            X_train,
            full_df,
            target,
            predictor_var,
            additional_vars=omit_feats,
            threshold=leaf_threshold,
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


def set_dt_features_in_df(
    model,
    df,
    full_df,
    target: PredictionTarget | None,
    predictor_var: str,
    additional_vars: set | None = None,
    threshold=0.1,
    omit_feats: set = set(),
):
    """
    Augments df with DT-derived columns and returns the enriched DataFrame.

    Args:
        model: fitted sklearn Pipeline with 'preprocessor' and 'clf' steps.
        df: feature DataFrame (e.g. X_train) to assign leaf ids and entropies for.
        full_df: original full DataFrame used to re-attach omitted columns (e.g. META_FEATURES).
        target: PredictionTarget used to enumerate all possible word orders.
        predictor_var: column name of the word-order variable being predicted.
        additional_vars: extra columns from full_df to carry over into the result.
        threshold: entropy threshold below which a leaf prediction is considered confident.

    Returns:
        A new DataFrame with added columns: leaf_id, leaf_full_entropy, leaf_top1_entropy,
        per-class entropies, leaf_rule, leaf_decision, keep, num_swaps, swap_order_candidates.
    """
    X_trans = model.named_steps["preprocessor"].transform(df)
    leaf_ids = model.named_steps["clf"].apply(X_trans)

    tree = model.named_steps["clf"].tree_
    impurity = tree.impurity
    children_left = tree.children_left
    children_right = tree.children_right

    is_leaf = (children_left == -1) & (children_right == -1)
    leaf_node_ids = np.where(is_leaf)[0]
    leaf_entropy_map = dict(zip(leaf_node_ids, impurity[leaf_node_ids]))
    leaf_full_entropy = np.array([leaf_entropy_map[node] for node in leaf_ids])

    additional_vars = (additional_vars or set()).union(META_FEATURES).union(omit_feats)
    additional_vars.add(predictor_var)

    new_cols = {col: full_df[col] for col in additional_vars if col in full_df.columns}
    new_cols["leaf_id"] = pd.Series(leaf_ids, index=df.index)
    new_cols["leaf_full_entropy"] = pd.Series(leaf_full_entropy, index=df.index)

    classes_list = list(model.classes_)
    class_to_idx = {c: i for i, c in enumerate(classes_list)}
    predictor_series = full_df.loc[df.index, predictor_var]
    # leaf_id repeats across rows; the distribution only depends on the leaf,
    # so compute it once per unique leaf instead of once per row.
    leaf_distributions = {
        leaf_id: tree.value[leaf_id][0] * tree.n_node_samples[leaf_id]
        for leaf_id in set(leaf_ids)
    }

    leaf_top1_entropies = []
    for leaf_id, deprel_order in zip(leaf_ids, predictor_series):
        leaf_distribution = leaf_distributions[leaf_id]
        n_right = leaf_distribution[class_to_idx[deprel_order]]
        n_wrong = sum(leaf_distribution) - n_right
        leaf_top1_entropies.append(order_entropy(n_right, n_wrong))
    new_cols["leaf_top1_entropy"] = leaf_top1_entropies

    tree_rules = get_tree_rules(model, tight=True)
    new_cols["leaf_rule"] = [tree_rules[node] for node in leaf_ids]

    leaf2var = {
        idx: model.classes_[var.item()] for idx, var in enumerate(tree.value.argmax(-1))
    }
    leaf_decision = [
        order == leaf2var[leaf_id] for leaf_id, order in zip(leaf_ids, predictor_series)
    ]
    new_cols["leaf_decision"] = leaf_decision
    new_cols["keep"] = [
        e < threshold and d
        for e, d in zip(new_cols["leaf_top1_entropy"], leaf_decision)
    ]

    if target is not None:
        # Word-order-specific columns: per-class entropies and swap candidates.
        all_orders = set(get_all_orders(predictor_var, target))
        unseen_classes = list(all_orders - set(model.classes_))

        for swapped_idx, swapped_class in enumerate(classes_list + unseen_classes):
            class_entropies = []
            for leaf_id, deprel_order in zip(leaf_ids, predictor_series):
                deprel_order_idx = class_to_idx[deprel_order]
                leaf_distribution = leaf_distributions[leaf_id]
                n_swapped = (
                    leaf_distribution[swapped_idx]
                    if swapped_idx < len(leaf_distribution)
                    else 0
                )
                class_entropies.append(
                    order_entropy(leaf_distribution[deprel_order_idx], n_swapped)
                )
            new_cols[f"{swapped_class}_entropy"] = class_entropies

        correct_num_swaps = []
        all_swap_order_candidates = []
        for i, (deprel_order, decision) in enumerate(
            zip(predictor_series, leaf_decision)
        ):
            swap_orders = all_orders - {deprel_order}
            swap_order_candidates = [
                arg_order
                for arg_order in swap_orders
                if new_cols[f"{arg_order}_entropy"][i] < threshold and decision
            ]
            correct_num_swaps.append(len(swap_order_candidates))
            all_swap_order_candidates.append(swap_order_candidates)
        new_cols["num_swaps"] = correct_num_swaps
        new_cols["swap_order_candidates"] = all_swap_order_candidates

    new_df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return new_df


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
