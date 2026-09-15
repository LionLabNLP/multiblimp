import math
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline


def order_entropy(
    n_ab: float,
    n_ba: float,
    smoothing_a: float = 0.5,  # Jeffreys prior
) -> float:
    """
    Compute Shannon entropy (in bits) for word order frequencies.

    Args:
        n_ab (int): Count of A>B order
        n_ba (int): Count of B>A order
        smoothing_a (float): Smoothing factor

    Returns:
        float: Entropy in bits
    """
    n_ab += smoothing_a
    n_ba += smoothing_a

    total = n_ab + n_ba
    if total == 0:
        return 0.0

    p_ab = n_ab / total
    p_ba = n_ba / total

    # avoid log(0) issues
    def safe_term(p):
        return -p * math.log2(p) if p > 0 else 0.0

    return safe_term(p_ab) + safe_term(p_ba)


def default_leaf_threshold(min_samples_leaf: int, smoothing_a: float = 0.5) -> float:
    """The entropy a leaf of exactly `min_samples_leaf` samples has when
    perfectly unanimous, under order_entropy's own smoothing -- i.e. the
    natural floor for "the tree said this is definitely one answer".

    A fixed constant threshold (this pipeline used a bare 0.12 for a while)
    implicitly demands a much larger leaf than min_samples_leaf actually
    allows before ANY leaf, however pure, can pass it: at
    min_samples_leaf=10, smoothing_a=0.5, a literal 0.12 requires ~30
    samples even at 100% purity (order_entropy(30, 0) ~= 0.119), silently
    orphaning every leaf in the 10-29 range min_samples_leaf says should be
    allowed to exist at all -- e.g. this was Basque svNa's entire blocker:
    97% tree accuracy, but its best leaf (666/677 rows, 97.7% pure) sat at
    entropy 0.159, just above 0.12, so num_ud_candidates_keep was 0.

    Deriving the threshold from the same smoothing formula instead keeps
    the "smallest allowed leaf, if unanimous, passes" invariant
    automatically for whatever min_samples_leaf/smoothing_a are actually in
    use, and doesn't loosen the bar for large, well-populated leaves at
    all: Jeffreys smoothing's influence decays as 1/n, so a large leaf's
    entropy is barely changed by it either way -- this raises the large-
    leaf raw-disagreement tolerance from ~1.6% at the old constant 0.12 to
    ~4.6% here (min_samples_leaf=10, smoothing_a=0.5), still a demanding
    bar, and a leaf with genuine higher disagreement (a real split the tree
    just can't resolve further, regardless of how much data backs it) is
    correctly still rejected.
    """
    return order_entropy(min_samples_leaf, 0, smoothing_a)


def calculate_base_entropy(
    df: pd.DataFrame, target_col: str, binary: bool = False, smoothing: float = 0.5
) -> float:
    """Calculate entropy of word order distribution.

    Args:
        df: DataFrame containing the data
        target_col: Column name containing word order labels
        binary: If True, calculate binary entropy (majority-class vs rest).
                If False, calculate six-class entropy.
        smoothing: Smoothing factor for entropy calculation (Jeffreys prior)

    Returns:
        Entropy value in bits
    """
    # astype(str) drops any unused categorical categories -- otherwise a
    # single-observed-class df[target_col] (e.g. a trivial language) can carry
    # phantom zero-count categories and never actually reach exactly 0 entropy.
    value_counts = df[target_col].astype(str).value_counts()

    if binary:
        # Binary entropy: majority class vs. rest
        n_majority = value_counts.iloc[0]  # Most frequent class
        n_rest = len(df) - n_majority
        return order_entropy(n_majority, n_rest, smoothing_a=smoothing)
    else:
        # Six-class entropy with smoothing
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
    """Calculate weighted entropy after decision tree split.

    Args:
        dt: Fitted sklearn Pipeline containing the decision tree
        df: DataFrame containing the features
        target_col: Column name containing word order labels
        binary: If True, calculate binary entropy. If False, six-class entropy.
        smoothing: Smoothing factor for entropy calculation

    Returns:
        Weighted average entropy of leaf nodes
    """
    # Get the decision tree classifier from the pipeline
    tree_model = dt.named_steps["clf"]

    # Prepare features (drop target column)
    X = df.drop(columns=[target_col])

    # Transform features through preprocessor and get leaf assignments
    leaf_ids = tree_model.apply(dt.named_steps["preprocessor"].transform(X))

    # Calculate entropy for each leaf
    weighted_entropy = 0.0
    total_samples = len(df)

    for leaf_id in np.unique(leaf_ids):
        leaf_mask = leaf_ids == leaf_id
        leaf_df = df[leaf_mask]
        leaf_weight = len(leaf_df) / total_samples
        leaf_entropy = calculate_base_entropy(
            leaf_df, target_col, binary=binary, smoothing=smoothing
        )
        weighted_entropy += leaf_weight * leaf_entropy

    return weighted_entropy
