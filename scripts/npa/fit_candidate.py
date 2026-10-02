import argparse
import os
import sys

sys.path.append("../../src/")

from npa.agreement import run_agreement_pipeline, canonical_target_col
from npa.npa_config import load_npa_config, config_langs_for
from multiblimp.languages import get_ud_langs, gblang2udlang
from multiblimp.config import TREEBANK_FEATURES_DIR, DEBUG_MODE
from word_order.entropy import DEFAULT_LEAF_MIN_ACCURACY
from sva_trees.second_chance import add_cli_args as add_second_chance_args, config_from_args as second_chance_config_from_args
from agreement_candidates import scan_qualifying_langs
from np_types import process_one_language, MAX_TREEBANK_LEN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit an NPA pairwise-agreement decision tree per language "
                     "(e.g. target_col='HEAD-DET_Number') from "
                     "treebank_features/npa/np_instances/*.parquet (built by "
                     "np_types.py's sweep -- not built here), render "
                     "word_order.viz_tree.tree2html for each, then (unless "
                     "--no_pairs) build minimal pairs per language "
                     "(create_npa_pairs_for_target_col -- reinflects the "
                     "non-head role when target_col involves HEAD, both roles "
                     "in turn otherwise) and the same diagnostics-enabled "
                     "deprel index + cross-target_col overview index the "
                     "SVA/subj_aux pipelines produce."
    )
    parser.add_argument("--target_col", "-t", required=True,
                        help='NPA pairwise agreement column, e.g. "HEAD-DET_Number"')
    parser.add_argument("--langs", "-l", nargs="*", help="Languages to process", default=[])
    parser.add_argument("--config", default="../../resources/npa_config.json",
                        help="npa_config.json (scripts/npa/agreement_candidates.py) "
                             "mapping target_col -> the languages actually worth "
                             "attempting for it. Used to narrow the default language "
                             "list (in place of every UD language) when --langs isn't "
                             "given; ignored otherwise. A target_col missing from "
                             "this file (e.g. a plain pooled 'HEAD-DET_Case' when the "
                             "config was built with --split_head_upos, so only "
                             "'HEAD:NOUN-DET_Case' etc are keys) triggers a live "
                             "scan_qualifying_langs scan instead of falling straight "
                             "back to every UD language -- see that scan's own "
                             "--wilson_floor/--min_both. A missing config FILE, or a "
                             "target_col the live scan also finds nothing for (the "
                             "column doesn't exist in any language's parquet at all), "
                             "still falls back to the full get_ud_langs() list.")
    parser.add_argument("--wilson_floor", type=float, default=0.01,
                        help="Only used for the live scan_qualifying_langs fallback "
                             "(see --config) -- same meaning as agreement_candidates."
                             "py's own flag of the same name.")
    parser.add_argument("--min_both", type=int, default=10,
                        help="Only used for the live scan_qualifying_langs fallback "
                             "(see --config) -- same meaning as agreement_candidates."
                             "py's own flag of the same name.")
    parser.add_argument("--min_yes", type=int, default=10,
                        help="Only used for the live scan fallback -- see agreement_candidates.py.")
    parser.add_argument("--yes_rate_floor", type=float, default=0.15,
                        help="Only used for the live scan fallback -- see agreement_candidates.py.")
    parser.add_argument("--recache", action="store_true",
                        help="Re-extract each language's np_instances parquet (and its "
                             "np_types CSV) from the treebank, then redo everything "
                             "downstream: refit trees / rebuild pairs / re-render HTML. "
                             "A language with no np_instances parquet yet is extracted "
                             "regardless (see --no_extract).")
    parser.add_argument("--refit", action="store_true",
                        help="Redo everything downstream of the cached np_instances "
                             "parquet -- refit the decision tree, regenerate minimal "
                             "pairs (and any second-chance retry) and HTML -- without "
                             "re-extracting from the treebank. Same as sva_trees' --refit.")
    parser.add_argument("--no_extract", action="store_true",
                        help="Never extract np_instances here: a language with no "
                             "parquet is skipped, and --recache behaves like --refit. "
                             "For sweeps (fit_all.py) over data that should already exist.")
    parser.add_argument("--max_treebank_len", type=int, default=MAX_TREEBANK_LEN,
                        help="Cap on treebank sentences read per language when "
                             "extracting np_instances (default 10000; uncapped data is "
                             "too large to merge/fit). Only used when extracting.")
    parser.add_argument("--keep_unk", action="store_true",
                        help="Keep unk-labeled rows in the tree fit instead of dropping them")
    parser.add_argument("--no_per_treebank", action="store_true",
                        help="Skip the analysis-only tree per treebank (default: fit/"
                             "render one for languages with several treebanks, linked "
                             "from the pooled page; word_order.per_treebank).")
    parser.add_argument("--incl_unk_trees", action="store_true",
                        help="Also fit an incl.-unk variant ({lang}__unk, per-treebank "
                             "and pooled) of each tree -- unk-labeled rows kept in, "
                             "instead of dropped -- alongside the normal one. Off by "
                             "default: only the normal drop-unk trees are fit (the "
                             "per-treebank trees themselves are controlled separately "
                             "by --no_per_treebank). Same as sva_trees' flag.")
    parser.add_argument("--include_excluded", action="store_true",
                        help="Also per-treebank trees for the language's excluded "
                             "treebanks, split off the unified np_instances parquet at "
                             "read time (word_order.per_treebank.split_excluded). "
                             "Never used for the pooled tree or pairs.")
    parser.add_argument("--max_depth", type=int, default=12)
    parser.add_argument("--min_samples_leaf", type=int, default=10)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--leaf_threshold", type=float, default=None,
                        help="Leaf-keep floor on smoothed leaf accuracy (leaves with "
                             "accuracy above it are kept). Default: 0.95, same as "
                             "sva_trees/subj_aux.")
    parser.add_argument("--no_index", action="store_true",
                        help="Skip this run's diagnostics table + deprel index.html "
                             "(fit_all.py passes this and builds every condition's "
                             "index once, in parallel, via scripts/overview/"
                             "generate_html_indexes.py afterwards).")
    parser.add_argument("--no_pairs", action="store_true",
                        help="Stop at tree-fitting -- skip create_npa_pairs_for_target_col "
                             "and the diagnostics/index-generation stage")
    parser.add_argument("--debug", action="store_true",
                        help="Data-debugging mode, kept parallel to sva_trees.cli's "
                             "--debug for cross-pipeline comparability: implies "
                             "--no_pairs, and switches the incl.-unk tree (always "
                             "fit here) from one plain merged 'unk' class to the "
                             "role1_unknown/role2_unknown/both_unknown breakdown. "
                             "NPA's own extraction (np_types.py) never fills in "
                             "UM/UD-derived annotation the way SVA's expand_anno "
                             "does, so there's no separate flag for that here -- "
                             "np_instances parquets already reflect only the "
                             "treebank's own annotation, in both modes. Requires "
                             "MULTIBLIMP_DEBUG_MODE=1 to already be set in the "
                             "environment BEFORE this script is invoked (that's "
                             "what sends output/html to output_debug/html_debug, "
                             "decided at import time in multiblimp.config); this "
                             "flag only double-checks the two agree.")
    parser.add_argument("--resource_dir", default="../../resources")
    add_second_chance_args(parser)
    args = parser.parse_args()

    canonical = canonical_target_col(args.target_col)
    if canonical != args.target_col:
        print(f"{args.target_col} doesn't exist as written (pairs are stored in role "
              f"priority order) -- using {canonical}")
        args.target_col = canonical

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

    instances_dir = os.path.join(TREEBANK_FEATURES_DIR, "npa", "np_instances")
    if args.langs:
        langs = args.langs
    else:
        all_langs = get_ud_langs(args.resource_dir)
        config = load_npa_config(args.config)
        if args.target_col in config:
            langs = config_langs_for(config, args.target_col, fallback=all_langs)
            print(f"npa_config: {len(langs)}/{len(all_langs)} languages "
                  f"worth attempting for {args.target_col} (cached)")
        else:
            print(f"{args.target_col} not in {args.config} -- scanning "
                  f"np_instances parquets directly for qualifying languages...")
            langs = scan_qualifying_langs(
                args.target_col, instances_dir=instances_dir,
                wilson_floor=args.wilson_floor, min_both=args.min_both,
                min_yes=args.min_yes, yes_rate_floor=args.yes_rate_floor,
            )
            if langs:
                print(f"live scan: {len(langs)}/{len(all_langs)} languages "
                      f"worth attempting for {args.target_col}")
            else:
                print(f"live scan found no qualifying languages for "
                      f"{args.target_col} -- falling back to the full "
                      f"{len(all_langs)}-language list")
                langs = all_langs

    if not args.no_extract:
        # Same on-demand extraction SVA does (sva_trees.pipeline.Pipeline.
        # _extract_df): build a language's np_instances when missing or on
        # --recache, always streaming (bounded memory) and capped.
        os.makedirs(instances_dir, exist_ok=True)
        for lang in langs:
            cached_lang = gblang2udlang.get(lang, lang).replace(" ", "_")
            if args.recache or not os.path.exists(os.path.join(instances_dir, f"{cached_lang}.parquet")):
                process_one_language(
                    lang, args.resource_dir, None, instances_dir,
                    args.max_treebank_len, streaming=True, chunk_size=500,
                    never_skip=args.recache,
                )

    run_agreement_pipeline(
        target_col=args.target_col,
        langs=langs,
        instances_dir=instances_dir,
        # save_dir/pairs_dir/diagnostics_csv left at their defaults --
        # run_agreement_pipeline derives them from target_col itself (see
        # npa.agreement.npa_id), nested under a shared "npa/" folder
        # (decision_trees/npa/{npa_id}, etc.) rather than one top-level
        # folder per target_col.
        resource_dir=args.resource_dir,
        never_skip=args.recache or args.refit,
        drop_unk=not args.keep_unk,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        test_size=args.test_size,
        leaf_threshold=(args.leaf_threshold if args.leaf_threshold is not None
                        else DEFAULT_LEAF_MIN_ACCURACY),
        build_pairs=not (args.no_pairs or args.debug),
        per_treebank=not args.no_per_treebank,
        include_excluded=args.include_excluded,
        second_chance=second_chance_config_from_args(args),
        build_index=not args.no_index,
        detailed_unk=args.debug,
        incl_unk=args.incl_unk_trees,
    )
