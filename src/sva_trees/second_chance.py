"""Rule-based check for whether a language gets a second, laxer pass.

The first pass keeps only Yes rows in leaves with smoothed accuracy > leaf_threshold.
Low-resource treebanks with messy annotation can fail that bar even though
the tree found real structure. The retry re-runs with one flat floor
(lax_threshold, accuracy >= 0.9 by default) for every leaf, never stricter
than the strict floor itself.
"""

import math
from dataclasses import dataclass

UNK_LABELS = {"unk", "+-", "--"}


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


def add_cli_args(parser):
    """The second-chance flag family, shared by sva_trees.cli and
    scripts/npa/fit_candidate.py. The retry is ON by default."""
    parser.add_argument("--no_second_chance", action="store_true",
                        help="Skip the second-chance retry. By default a language "
                             "whose strict pass under-filled "
                             "(sva_trees.second_chance) is retried at a laxer, "
                             "flat accuracy floor before giving up on it. With "
                             "this flag a language keeps only what clears the "
                             "strict floor.")
    parser.add_argument("--second_chance_lax_threshold", type=float,
                        default=SecondChanceConfig.lax_threshold,
                        help="Ignored with --no_second_chance: the retry keeps "
                             "leaves with smoothed accuracy >= this "
                             f"(default {SecondChanceConfig.lax_threshold:g}).")
    parser.add_argument("--second_chance_min_keep_frac", type=float,
                        default=SecondChanceConfig.min_keep_frac,
                        help="Ignored with --no_second_chance: skip the retry if the "
                             "strict pass already kept at least this fraction of "
                             f"Yes rows (default {SecondChanceConfig.min_keep_frac:g}).")
    parser.add_argument("--second_chance_max_unk_frac", type=float,
                        default=SecondChanceConfig.max_unk_frac,
                        help="Ignored with --no_second_chance: refuse to retry above "
                             "this unk share of the language's whole treebank "
                             f"(default {SecondChanceConfig.max_unk_frac:g}).")


def config_from_args(args):
    """SecondChanceConfig from add_cli_args' flags, None with --no_second_chance."""
    if args.no_second_chance:
        return None
    return SecondChanceConfig(
        min_keep_frac=args.second_chance_min_keep_frac,
        max_unk_frac=args.second_chance_max_unk_frac,
        lax_threshold=args.second_chance_lax_threshold,
    )
