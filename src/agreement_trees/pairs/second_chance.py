"""Rule-based check for whether a language gets a second, laxer pass.

The first pass keeps only Yes rows in leaves with smoothed accuracy > leaf_threshold.
Low-resource treebanks with messy annotation can fail that bar even though the tree found
real structure. The retry uses one flat floor (lax_threshold) for every leaf, never stricter
than the strict floor. Settings: fit.second_chance in resources/agreement_config.json
(agreement_trees.cli.second_chance_config).
"""

import math
from dataclasses import dataclass

from agreement_trees.fit.decision_tree import UNK_LABELS


@dataclass
class SecondChanceConfig:
    min_keep_frac: float = 0.5    # first pass is fine if n_keep / n_yes reaches this
    max_unk_frac: float = 0.5     # refuse above this unk share of all treebank rows
    lax_threshold: float = 0.9    # retry keeps leaves with accuracy >= this


def second_chance_decision(dt_df, predictor_var, label_distribution,
                           strict_threshold, min_samples_leaf=10, cfg=None):
    """Decide whether to retry a language with a laxer, flat accuracy floor.

    Args:
        dt_df: create_pairs' input df (leaf_top1_acc/leaf_decision
            columns when a tree was fit).
        label_distribution: {label: count} over the whole treebank's predictor
            column (pipeline's label_distribution, unk rows included).
        strict_threshold: the first-pass leaf accuracy floor.
        min_samples_leaf: the tree's leaf size; fewer Yes rows than this can't
            fill even one leaf, so there is nothing to rescue.

    Returns:
        (retry: bool, threshold: float, info: dict) -- threshold is
        strict_threshold unchanged when retry is False, or the flat lax
        floor when True. info holds the inputs and the rule that decided,
        for logging/meta.json.
    """
    cfg = cfg or SecondChanceConfig()
    n_total = sum(label_distribution.values())
    n_unk = sum(v for k, v in label_distribution.items() if k in UNK_LABELS)
    is_yes = dt_df[predictor_var] == "yes"
    n_yes = int(is_yes.sum())
    info = {"n_total": n_total, "n_yes": n_yes,
            "unk_frac": n_unk / n_total if n_total else 0.0}

    def no(reason):
        return False, strict_threshold, {**info, "reason": reason}

    required_cols = {"leaf_top1_acc", "leaf_decision"}
    if not required_cols.issubset(dt_df.columns):
        info["n_keep"] = n_yes
        return no("no tree fit")
    acc, dec = dt_df["leaf_top1_acc"], dt_df["leaf_decision"]
    kept = (acc > strict_threshold) & dec & is_yes
    n_keep = int(kept.sum())
    info["n_keep"] = n_keep

    if n_yes < min_samples_leaf:
        return no("too little evidence")
    if n_keep / n_yes >= cfg.min_keep_frac:
        return no("first pass sufficient")
    if info["unk_frac"] > cfg.max_unk_frac:
        return no("too much unk")

    # Callers keep rows with accuracy > threshold, so step just below the
    # floor to make it inclusive (>=).
    lax = min(math.nextafter(cfg.lax_threshold, 0.0), strict_threshold)
    lax_keep = (acc > lax) & dec & is_yes
    n_rescued = int((lax_keep & ~kept).sum())
    info["n_rescued"] = n_rescued
    if n_rescued == 0:
        return no("nothing rescuable")
    return True, lax, {**info, "reason": "retry"}

