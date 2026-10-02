"""Rule-based check for whether a language gets a second, laxer pass.

The first pass keeps only Yes rows in leaves with smoothed accuracy > leaf_threshold.
Low-resource treebanks with messy annotation can fail that bar even though
the tree found real structure. The retry re-runs with a per-leaf accuracy floor:
looser for a leaf the tree never got to split (or split only a little),
converging to the strict cap itself for a leaf reached only after several
splits, on the theory that a leaf the tree already worked hard to purify and
still couldn't is more likely genuinely noisy than one that just never got
the chance to split.

The cap is depth-scaled but size-anchored at min_samples_leaf, not at each
leaf's own real size -- like the strict floor's own convention. A leaf's leaf_top1_acc is already computed with its own
real size (smoothed lightly if large, heavily if small); holding every leaf
to a bar calibrated at the *smallest* allowed size is what actually gives a
big leaf more benefit of the doubt than a small one. Anchoring the bar to
each leaf's own size instead would cancel out exactly against that leaf's
own already-smoothed accuracy and add no size-based caution at all.
"""

from dataclasses import dataclass

from word_order.entropy import smoothed_accuracy

UNK_LABELS = {"unk", "+-", "--"}


@dataclass
class SecondChanceConfig:
    min_keep_frac: float = 0.5    # first pass is fine if n_keep / n_yes reaches this
    max_unk_frac: float = 0.5     # refuse above this unk share of all treebank rows
    max_lax_disagreement: float = 0.15  # laxest leaf error rate we ever accept, at depth 0
    depth_decay_ratio: float = 0.8      # disagreement budget *= this per split: 15% -> 12% -> 9.6% -> ...


def leaf_lax_threshold(cfg, strict_threshold, depth, min_samples_leaf):
    """The retry accuracy floor for a leaf at this depth: express the
    depth-decayed disagreement budget as real counts at min_samples_leaf
    (a fixed reference size, not this leaf's own actual size), then run it
    through smoothed_accuracy. Never stricter than the strict floor itself,
    so a retry can never be stricter than not retrying.
    """
    disagreement = cfg.max_lax_disagreement * (cfg.depth_decay_ratio ** depth)
    n_wrong = min_samples_leaf * disagreement
    n_right = min_samples_leaf - n_wrong
    return min(smoothed_accuracy(n_right, min_samples_leaf), strict_threshold)


def second_chance_decision(dt_df, predictor_var, label_distribution,
                           strict_threshold, min_samples_leaf=10, cfg=None):
    """Decide whether to retry a language with a laxer, per-leaf accuracy
    floor.

    Args:
        dt_df: create_pairs' input df (leaf_top1_acc/leaf_decision/leaf_depth
            columns when a tree was fit).
        label_distribution: {label: count} over the whole treebank's predictor
            column (pipeline's label_distribution, unk rows included).
        strict_threshold: the first-pass leaf accuracy floor.
        min_samples_leaf: the tree's leaf size; fewer Yes rows than this can't
            fill even one leaf, so there is nothing to rescue. Also the fixed
            reference size the retry cap is smoothed against (see module
            docstring) -- not any specific leaf's own real size.

    Returns:
        (retry: bool, threshold: float | dict[int, float], info: dict) --
        threshold is strict_threshold unchanged when retry is False, or a
        {leaf_id: this leaf's own lax threshold} map when True (every leaf
        with a Yes row gets an entry, not just rescued ones -- an
        already-strict-passing leaf's own value is simply >= strict_threshold
        too, so it stays kept either way). info holds the inputs and the
        rule that decided, for logging/meta.json.
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

    required_cols = {"leaf_top1_acc", "leaf_decision", "leaf_depth"}
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

    # One threshold per unique depth, not per leaf or per row -- the cap only
    # depends on depth now (min_samples_leaf is a fixed constant for the
    # whole language), so this is cheap regardless of row or leaf count.
    per_depth_threshold = {
        int(depth): leaf_lax_threshold(cfg, strict_threshold, int(depth), min_samples_leaf)
        for depth in dt_df["leaf_depth"].unique()
    }
    leaf_depths = dt_df[["leaf_id", "leaf_depth"]].drop_duplicates("leaf_id")
    per_leaf_threshold = {
        int(row.leaf_id): per_depth_threshold[int(row.leaf_depth)]
        for row in leaf_depths.itertuples()
    }
    row_threshold = dt_df["leaf_depth"].map(per_depth_threshold)
    lax_keep = (acc > row_threshold) & dec & is_yes
    n_rescued = int((lax_keep & ~kept).sum())
    info["n_rescued"] = n_rescued
    if n_rescued == 0:
        return no("nothing rescuable")
    return True, per_leaf_threshold, {**info, "reason": "retry"}


def add_cli_args(parser):
    """The second-chance flag family, shared by sva_trees.cli and
    scripts/npa/fit_candidate.py. The retry is ON by default."""
    parser.add_argument("--no_second_chance", action="store_true",
                        help="Skip the second-chance retry. By default a language "
                             "whose strict pass under-filled "
                             "(sva_trees.second_chance) is retried at a laxer, "
                             "per-leaf accuracy floor before giving up on it -- "
                             "looser for a leaf the tree never got to split, "
                             "converging to the strict floor for one reached only "
                             "after several splits. With this flag a language "
                             "keeps only what clears the strict floor.")
    parser.add_argument("--second_chance_depth_decay_ratio", type=float,
                        default=SecondChanceConfig.depth_decay_ratio,
                        help="Ignored with --no_second_chance: the retry's disagreement "
                             f"budget is multiplied by this per tree split (default "
                             f"{SecondChanceConfig.depth_decay_ratio:g}: 15%% at "
                             "depth 0, decaying every split after).")
    parser.add_argument("--second_chance_max_lax_disagreement", type=float,
                        default=SecondChanceConfig.max_lax_disagreement,
                        help="Ignored with --no_second_chance: the laxest leaf error "
                             f"rate ever accepted, at depth 0 (default "
                             f"{SecondChanceConfig.max_lax_disagreement:g}).")
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
        max_lax_disagreement=args.second_chance_max_lax_disagreement,
        depth_decay_ratio=args.second_chance_depth_decay_ratio,
    )
