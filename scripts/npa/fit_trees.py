import argparse
import os
import sys

sys.path.append("../../src/")

from npa.agreement import run_agreement_pipeline
from npa.npa_config import load_npa_config, config_langs_for
from multiblimp.languages import get_ud_langs
from multiblimp.config import TREEBANK_FEATURES_DIR
from agreement_candidates import scan_qualifying_langs

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
    parser.add_argument("--never_skip", "-ns", action="store_true",
                        help="Always refit trees / rebuild pairs / re-render HTML, even if cached")
    parser.add_argument("--keep_unk", action="store_true",
                        help="Keep unk-labeled rows in the tree fit instead of dropping them")
    parser.add_argument("--max_depth", type=int, default=12)
    parser.add_argument("--min_samples_leaf", type=int, default=10)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--leaf_threshold", type=float, default=0.1)
    parser.add_argument("--no_pairs", action="store_true",
                        help="Stop at tree-fitting -- skip create_npa_pairs_for_target_col "
                             "and the diagnostics/index-generation stage")
    parser.add_argument("--resource_dir", default="../../resources")
    args = parser.parse_args()

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
            )
            if langs:
                print(f"live scan: {len(langs)}/{len(all_langs)} languages "
                      f"worth attempting for {args.target_col}")
            else:
                print(f"live scan found no qualifying languages for "
                      f"{args.target_col} -- falling back to the full "
                      f"{len(all_langs)}-language list")
                langs = all_langs

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
        never_skip=args.never_skip,
        drop_unk=not args.keep_unk,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        test_size=args.test_size,
        leaf_threshold=args.leaf_threshold,
        build_pairs=not args.no_pairs,
    )
