import argparse
import dataclasses
import os
import random
import sys

from multiblimp.config import DEBUG_MODE
from multiblimp.languages import get_ud_langs
from npa.npa_config import load_npa_config, config_langs_for
from sva_trees.conditions import CONDITIONS
from sva_trees.pipeline import Pipeline
from sva_trees.second_chance import (
    add_cli_args as add_second_chance_args, config_from_args as second_chance_config_from_args,
)

random.seed(42)

# The 9 condition families (sv/sp/sa/ov/op/oa/iov/iop/ioa), derived from
# CONDITIONS itself (every family has an "Na" entry) rather than a separate
# hardcoded list that could drift from sva_trees.conditions._build_conditions.
FAMILY_PREFIXES = sorted(cid[:-2] for cid in CONDITIONS if cid.endswith("Na"))


def build_parser(description: str, is_aux: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--langs", "-l", nargs="*", help="Languages to process", default=[])
    parser.add_argument("--recache", action="store_true",
                        help="Always create new df's, decesion trees, and HTML files")
    parser.add_argument("--refit", action="store_true",
                        help="Redo everything downstream of the cached extraction "
                             "df -- split included/excluded, refit the decision "
                             "tree, regenerate minimal pairs and HTML -- without "
                             "forcing a fresh extraction pass the way --recache "
                             "does. For a fix that only changes post-extraction "
                             "behavior (e.g. an excluded_treebanks lookup bug, a "
                             "fit_dt/create_pairs change), the cached df is "
                             "already correct and re-extracting it is wasted "
                             "work; --refit reuses it instead. Combine with "
                             "--langs to scope to just the affected languages.")
    parser.add_argument("--distinguish_unk", "-d", action="store_true",
                        help="Keep +-/-- labels distinct instead of collapsing them "
                             "into 'unk' before fitting the decision tree (default: collapsed)")
    parser.add_argument("--n_jobs", "-j", type=int, default=1,
                        help="Languages processed in parallel (1=serial)")
    parser.add_argument("--max_worker_mem_gb", "-m", type=float, default=None,
                        help="Hard RAM cap per worker process, in GB. Default: "
                             "auto-derived from currently-available system RAM / n_jobs")
    parser.add_argument("--force", action="store_true",
                        help="Skip the startup check for other already-running "
                             "instances of this script. Only use for deliberate "
                             "concurrent runs.")
    parser.add_argument("--max_treebank_len", type=int, default=None,
                        help="Cap the number of treebank sentences read per "
                             "language. Default: no cap.")
    parser.add_argument("--keep_unk", action="store_true",
                        help="Keep unk-labeled rows (missing head/child feature "
                             "annotation) in the decision tree fit instead of "
                             "dropping them. Default: dropped.")
    parser.add_argument("--no_per_treebank", action="store_true",
                        help="Skip the analysis-only decision tree per treebank "
                             "(default: fit/render one for languages with several "
                             "treebanks, linked from the pooled tree page; "
                             "word_order.per_treebank).")
    parser.add_argument("--include_excluded", action="store_true",
                        help="Also fit/render trees for the language's excluded "
                             "treebanks (tagged with their exclusion reason; never "
                             "used for the pooled tree or minimal pairs). Split out "
                             "of the same unified <word_order_dir>/<lang>.parquet "
                             "cache at read time (sva_trees.pipeline.Pipeline."
                             "_split_excluded) -- no separate extraction needed. "
                             "Needs per-treebank trees.")
    parser.add_argument("--incl_unk_trees", action="store_true",
                        help="Also fit an incl.-unk variant of each per-treebank "
                             "decision tree (unk-labeled rows kept in, instead of "
                             "dropped) alongside the normal one -- word_order."
                             "per_treebank.fit_treebank_trees. Off by default: only "
                             "the normal drop-unk per-treebank trees are fit "
                             "(per_treebank itself is controlled separately by "
                             "--no_per_treebank).")
    parser.add_argument("--target_id", default=None,
                        help="Override the output dir name (decision_trees/<id>, "
                             "minimal_pairs/<id>, ...). Default: the condition id "
                             "(run_condition) or {family}{feature} (run_candidate_"
                             "condition). Useful for A/B runs that shouldn't "
                             "overwrite each other, e.g. --keep_unk comparisons.")
    parser.add_argument("--max_tasks_per_child", type=int, default=1,
                        help="Recycle each worker process after this many "
                             "languages. Default: 1 (fresh process per "
                             "language, so memory can't accumulate across a "
                             "full-corpus run). Raise to trade some of that "
                             "safety back for less per-language interpreter-"
                             "startup overhead.")
    parser.add_argument("--debug", action="store_true",
                        help="Data-debugging mode: never create minimal pairs, "
                             "and never let process_treebank.expand_anno fill in "
                             "UM/UD-derived annotation, so extracted features "
                             "reflect only the treebank's own original "
                             "annotation. When --incl_unk_trees is also given, "
                             "its per-treebank incl.-unk tree breaks unk down "
                             "into *why* (Head/Subject/Both unknown) instead of "
                             "one plain merged 'unk' class. Requires "
                             "MULTIBLIMP_DEBUG_MODE=1 "
                             "to already be set in the environment BEFORE this "
                             "script is invoked -- that's what actually sends "
                             "output/html to output_debug/html_debug (decided "
                             "at import time, in multiblimp.config), this flag "
                             "only double-checks the two agree so a run can't "
                             "silently end up in the wrong mode.")
    add_second_chance_args(parser)
    if is_aux:
        parser.add_argument("--single_aux_only", action="store_true",
                            help="Use single-aux instances only, excluding stacked-aux "
                                 "(2+) ones (default: single+stacked-aux combined)")
    return parser


def run_condition(condition_id: str, argv=None):
    condition = CONDITIONS[condition_id]
    args = build_parser(f"Run the {condition_id} condition.", condition.is_aux).parse_args(argv)
    if args.debug != DEBUG_MODE:
        raise SystemExit(
            "--debug was "
            + ("passed" if args.debug else "not passed")
            + f" but MULTIBLIMP_DEBUG_MODE={'1' if DEBUG_MODE else '0'} in the "
            "environment -- export MULTIBLIMP_DEBUG_MODE=1 before invoking this "
            "script for a debug run (it must be set before Python starts, so "
            "output/html resolve to output_debug/html_debug), or drop --debug "
            "for a normal run."
        )
    resource_dir = "../../resources"

    kwargs = {}
    target_id = args.target_id or condition_id
    if condition.is_aux:
        from subj_aux.pipeline import AuxPipeline
        pipeline_cls = AuxPipeline
        kwargs["aux_target"] = condition.aux_target
        kwargs["include_multi_aux"] = not args.single_aux_only
        if args.single_aux_only and not args.target_id:
            target_id += "_single"
    else:
        pipeline_cls = Pipeline

    second_chance = second_chance_config_from_args(args)

    pipeline = pipeline_cls(
        target=condition.make_target(),
        predictor_var=condition.predictor_var,
        langs=(args.langs if args.langs else get_ud_langs(resource_dir)),
        inflection_map=condition.inflection_map,
        unimorph_args=condition.unimorph_args,
        deprel_dir=condition.deprel,
        resource_dir=resource_dir,
        word_order_dir=condition.word_order_dir(),
        max_treebank_len=args.max_treebank_len,
        never_skip=args.recache,
        never_skip_fit=args.refit,
        rm_columns=list(condition.rm_columns),
        target_id=target_id,
        simplify=not args.distinguish_unk,
        n_jobs=args.n_jobs,
        max_worker_mem_gb=args.max_worker_mem_gb,
        force=args.force,
        drop_unk=not args.keep_unk,
        max_tasks_per_child=args.max_tasks_per_child,
        per_treebank=not args.no_per_treebank,
        include_excluded=args.include_excluded,
        generate_pairs=not args.debug,
        fetch_all=not args.debug,
        incl_unk=args.incl_unk_trees,
        detailed_unk=args.debug,
        second_chance=second_chance,
        **kwargs,
    )
    pipeline()


def run_candidate_condition(argv=None):
    """Run an ad-hoc condition for one (family, feature) pair that isn't one
    of the fixed Na/Ga/Pa (Number/Gender/Person) scripts -- a candidate
    scripts/sva_trees/agreement_candidates.py's scan surfaced (e.g. "Case",
    "Mood", "Animacy"), still unwired to any real condition. Reuses the
    family's existing word_order cache (extract_node_features already wrote
    every attested feature into it, not just Number/Gender/Person -- see
    vp_types.py's module docstring), so no new extraction happens UNLESS
    this feature's own head_{deprel}_{feature}_agreement column isn't in
    that cache yet (Pipeline.agreement_feats only ever computes columns for
    the features it's explicitly given -- see process_treebank.
    extract_instances), in which case exactly one real pass computes it and
    the shared cache gains that column for every future run, same as a
    brand-new svNa.py-style script's first run would.

    inflection_map=(feature, None): the generic "swap to any other observed
    value" shape multiblimp.swap_features' own "_any" constants already use
    (e.g. swap_case_any = ("Case", None), swap_number_subj_any = ([...],
    None)) and npa.agreement.build_role_inflector's generic per-target_col
    reinflection already relies on for the exact same reason -- there's no
    single fixed value-remapping (like swap_number's SG<->PL) to hardcode
    for an arbitrary candidate feature, so the underlying UniMorph
    reinflection engine is asked to find whatever alternate value the
    lemma's own paradigm attests instead.
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--family", "-f", choices=FAMILY_PREFIXES)
    pre_args, _ = pre_parser.parse_known_args(argv)
    is_aux = pre_args.family is not None and CONDITIONS[f"{pre_args.family}Na"].is_aux

    parser = build_parser(
        "Run an ad-hoc candidate condition for one (family, feature) pair "
        "found by scripts/sva_trees/agreement_candidates.py but not wired "
        "to a real Na/Ga/Pa script yet.",
        is_aux,
    )
    parser.add_argument("--family", "-f", required=True, choices=FAMILY_PREFIXES,
                         help="Condition family whose existing word_order cache "
                              "to reuse (sv/sp/sa/ov/op/oa/iov/iop/ioa).")
    parser.add_argument("--feature", required=True,
                         help='Morphological feature to swap, e.g. "Case" -- any '
                              "UD feature name attested in the family's cache "
                              "(scripts/sva_trees/agreement_candidates.py's "
                              "report/sva_agreement_config.json lists candidates).")
    parser.add_argument("--config", default="../../resources/sva_agreement_config.json",
                         help="sva_agreement_config.json (scripts/sva_trees."
                              "agreement_candidates.py) mapping family -> feature -> "
                              "the languages actually worth attempting it for. Used "
                              "to narrow the default language list (in place of "
                              "every UD language) when --langs isn't given; ignored "
                              "otherwise. A (family, feature) missing from this file "
                              "falls back to the full get_ud_langs() list.")
    args = parser.parse_args(argv)
    if args.debug != DEBUG_MODE:
        raise SystemExit(
            "--debug was "
            + ("passed" if args.debug else "not passed")
            + f" but MULTIBLIMP_DEBUG_MODE={'1' if DEBUG_MODE else '0'} in the "
            "environment -- export MULTIBLIMP_DEBUG_MODE=1 before invoking this "
            "script for a debug run (it must be set before Python starts, so "
            "output/html resolve to output_debug/html_debug), or drop --debug "
            "for a normal run."
        )
    resource_dir = "../../resources"
    condition = CONDITIONS[f"{args.family}Na"]
    target = dataclasses.replace(condition.make_target(), swap_feat=args.feature)
    predictor_var = f"head_{condition.deprel}_{args.feature}_agreement"

    kwargs = {}
    target_id = args.target_id or f"{args.family}{args.feature}"
    if condition.is_aux:
        from subj_aux.pipeline import AuxPipeline
        pipeline_cls = AuxPipeline
        kwargs["aux_target"] = condition.aux_target
        kwargs["include_multi_aux"] = not args.single_aux_only
        if args.single_aux_only and not args.target_id:
            target_id += "_single"
    else:
        pipeline_cls = Pipeline

    if args.langs:
        langs = args.langs
    else:
        family_config = load_npa_config(args.config).get(args.family, {})
        langs = config_langs_for(family_config, args.feature,
                                  fallback=get_ud_langs(resource_dir))

    second_chance = second_chance_config_from_args(args)

    pipeline = pipeline_cls(
        target=target,
        predictor_var=predictor_var,
        langs=langs,
        inflection_map=(args.feature, None),
        unimorph_args=condition.unimorph_args,
        deprel_dir=condition.deprel,
        resource_dir=resource_dir,
        word_order_dir=condition.word_order_dir(),
        max_treebank_len=args.max_treebank_len,
        never_skip=args.recache,
        never_skip_fit=args.refit,
        rm_columns=list(condition.rm_columns),
        target_id=target_id,
        simplify=not args.distinguish_unk,
        # Number/Gender/Person stay in the fit alongside the candidate
        # feature -- not just args.feature alone -- so a first run for this
        # feature doesn't drop the three agreement columns every other
        # condition already relies on out of the shared cache it rewrites.
        agreement_feats=["Number", "Gender", "Person", args.feature],
        n_jobs=args.n_jobs,
        max_worker_mem_gb=args.max_worker_mem_gb,
        force=args.force,
        drop_unk=not args.keep_unk,
        max_tasks_per_child=args.max_tasks_per_child,
        per_treebank=not args.no_per_treebank,
        include_excluded=args.include_excluded,
        generate_pairs=not args.debug,
        fetch_all=not args.debug,
        incl_unk=args.incl_unk_trees,
        detailed_unk=args.debug,
        second_chance=second_chance,
        **kwargs,
    )
    pipeline()
