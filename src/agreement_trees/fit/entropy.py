import math

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

TRANSFORM_CHUNK_ROWS = 50_000


def chunked_leaf_ids(dt: Pipeline, X: pd.DataFrame, y=None):
    """(leaf ids of every row of X, number of rows whose prediction equals y or None without y).

    Transforms in row chunks: the one-hot matrix of a whole large treebank can raise MemoryError."""
    pre, clf = dt.named_steps["preprocessor"], dt.named_steps["clf"]
    parts, hits = [], 0
    for start in range(0, len(X), TRANSFORM_CHUNK_ROWS):
        whole = len(X) <= TRANSFORM_CHUNK_ROWS    # no slice, which would copy a wide frame
        X_t = pre.transform(X if whole else X.iloc[start:start + TRANSFORM_CHUNK_ROWS])
        parts.append(clf.apply(X_t))
        if y is not None:
            y_part = y if whole else y.iloc[start:start + TRANSFORM_CHUNK_ROWS]
            hits += int((clf.predict(X_t) == y_part.to_numpy()).sum())
    leaf_ids = np.concatenate(parts) if parts else np.empty(0, dtype=np.intp)
    return leaf_ids, (hits if y is not None else None)


def order_entropy(
    n_ab: float,
    n_ba: float,
    smoothing_a: float = 0.5,  # Jeffreys prior
) -> float:
    """Shannon entropy (bits) of two counts, each smoothed by `smoothing_a` (Jeffreys prior)."""
    n_ab += smoothing_a
    n_ba += smoothing_a

    total = n_ab + n_ba
    if total == 0:
        return 0.0

    p_ab = n_ab / total
    p_ba = n_ba / total

    def safe_term(p):
        return -p * math.log2(p) if p > 0 else 0.0

    return safe_term(p_ab) + safe_term(p_ba)


def calculate_base_entropy(
    df: pd.DataFrame, target_col: str, binary: bool = False, smoothing: float = 0.5
) -> float:
    """Entropy (bits) of the labels in `target_col`. binary: majority class vs the rest; otherwise all
    classes, each smoothed by `smoothing`."""
    # astype(str) drops unused categories, which would otherwise keep a one-class frame above 0 entropy
    value_counts = df[target_col].astype(str).value_counts()

    if binary:
        n_majority = value_counts.iloc[0]
        n_rest = len(df) - n_majority
        return order_entropy(n_majority, n_rest, smoothing_a=smoothing)
    else:
        counts = value_counts.values
        smoothed_counts = counts + smoothing
        total = smoothed_counts.sum()
        probabilities = smoothed_counts / total

        return sum(-p * math.log2(p) if p > 0 else 0.0 for p in probabilities)


def calculate_tree_entropy(
    dt: Pipeline,
    df: pd.DataFrame,
    target_col: str,
    binary: bool = False,
    smoothing: float = 0.5,
) -> float:
    """Leaf-weighted entropy of `target_col` over the fitted pipeline's leaves for `df`."""
    X = df.drop(columns=[target_col])

    leaf_ids, _ = chunked_leaf_ids(dt, X)

    return leaf_weighted_entropy(
        leaf_ids, df, target_col, binary=binary, smoothing=smoothing
    )


def leaf_weighted_entropy(
    leaf_ids: np.ndarray,
    df: pd.DataFrame,
    target_col: str,
    binary: bool = False,
    smoothing: float = 0.5,
) -> float:
    """Leaf-weighted average entropy of target_col (`leaf_ids`: one leaf id per row of `df`, from the
    tree's .apply()). Only target_col is read, as df can be thousands of columns wide."""
    labels = df[[target_col]]

    weighted_entropy = 0.0
    total_samples = len(df)

    for leaf_id in np.unique(leaf_ids):
        leaf_mask = leaf_ids == leaf_id
        leaf_df = labels[leaf_mask]
        leaf_weight = len(leaf_df) / total_samples
        leaf_entropy = calculate_base_entropy(
            leaf_df, target_col, binary=binary, smoothing=smoothing
        )
        weighted_entropy += leaf_weight * leaf_entropy

    return weighted_entropy


# --- accuracy-based leaf/tree measures (add-`smoothing` per class) ----------

# Leaf-keep floor: a leaf is kept when its smoothed accuracy is above this (a pure leaf of
# min_samples_leaf=10 has 10.5/11 = 0.955).
DEFAULT_LEAF_MIN_ACCURACY = 0.95


def smoothed_accuracy(n_right: float, n_total: float, smoothing: float = 0.5) -> float:
    """Add-`smoothing` accuracy of "right vs. rest": (n_right + s) / (n_total + 2s).
    Same smoothed p that order_entropy(n_right, n_total - n_right) uses."""
    return (n_right + smoothing) / (n_total + 2 * smoothing)


def calculate_base_accuracy(df: pd.DataFrame, target_col: str, smoothing: float = 0.5) -> float:
    """Smoothed accuracy of always predicting the majority class (the root node's own)."""
    counts = df[target_col].astype(str).value_counts()
    return smoothed_accuracy(counts.iloc[0], len(df), smoothing)


def pooled_leaf_accuracy(leaf_ids: np.ndarray, df: pd.DataFrame, target_col: str,
                         smoothing: float = 0.5) -> float:
    """Train accuracy of the tree smoothed once over the whole dataset, (sum of leaf majority counts + s) /
    (N + 2s): comparable with calculate_base_accuracy, so the gain cannot go negative (smoothing per leaf
    would stack a penalty per leaf). The keep rule and leaf labels use the per-leaf smoothed_accuracy."""
    labels = df[target_col].astype(str)
    majority = sum(
        labels[leaf_ids == leaf].value_counts().iloc[0] for leaf in np.unique(leaf_ids)
    )
    return smoothed_accuracy(majority, len(df), smoothing)


def calculate_tree_accuracy(dt: Pipeline, df: pd.DataFrame, target_col: str,
                            smoothing: float = 0.5) -> float:
    X = df.drop(columns=[target_col])
    leaf_ids, _ = chunked_leaf_ids(dt, X)
    return pooled_leaf_accuracy(leaf_ids, df, target_col, smoothing=smoothing)
