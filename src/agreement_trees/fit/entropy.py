import math

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline


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


def default_leaf_threshold(min_samples_leaf: int, smoothing_a: float = 0.5) -> float:
    """LEGACY (unused, superseded by DEFAULT_LEAF_MIN_ACCURACY): the entropy of a unanimous leaf of
    exactly `min_samples_leaf` samples, the entropy-era floor for keeping a leaf."""
    return order_entropy(min_samples_leaf, 0, smoothing_a)


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
    tree_model = dt.named_steps["clf"]

    X = df.drop(columns=[target_col])

    leaf_ids = tree_model.apply(dt.named_steps["preprocessor"].transform(X))

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
    leaf_ids = dt.named_steps["clf"].apply(dt.named_steps["preprocessor"].transform(X))
    return pooled_leaf_accuracy(leaf_ids, df, target_col, smoothing=smoothing)
