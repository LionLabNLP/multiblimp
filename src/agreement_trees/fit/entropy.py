
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


# --- accuracy-based leaf/tree measures (add-`smoothing` per class) ----------

# Leaf-keep floor: a leaf is kept when its smoothed accuracy is at least this (a pure leaf of
# min_samples_leaf=10 has 10.5/11 = 0.955).
DEFAULT_LEAF_MIN_ACCURACY = 0.95


def smoothed_accuracy(n_right: float, n_total: float, smoothing: float = 0.5) -> float:
    """Add-`smoothing` accuracy of "right vs. rest": (n_right + s) / (n_total + 2s).
    """
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
