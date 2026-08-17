import argparse
import sys
import random
from functools import partial

sys.path.append("../../src/")

from sva_trees.pipeline import Pipeline
from multiblimp.languages import get_ud_langs
from multiblimp.swap_features import *
from word_order.prediction_target import *

random.seed(42)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", "-l", nargs="*", help="Languages to process", default=[])
    parser.add_argument("--never_skip", "-ns", action="store_true", 
                        help="Always create new df's, decesion trees, and HTML files")
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
    parser.add_argument("--target_id", default=None,
                        help="Override the output dir name (decision_trees/<id>, "
                             "minimal_pairs/<id>, ...). Default: this script's "
                             "filename (spGa). Useful for A/B runs that shouldn't "
                             "overwrite each other, e.g. --keep_unk comparisons.")
    args = parser.parse_args()

    target = nsubj_target
    target.head_feats = {"VerbForm": partial(filter_head_feats, require="Part")}
    target.swap_feat = "Gender"
    # target.child_feats = {}
    deprel_dir = "_".join(target.child_deprels)
    resource_dir = "../../resources"

    pipeline = Pipeline(target=target,
                        predictor_var=f"head_nsubj_{target.swap_feat}_agreement",
                        langs=(args.langs if args.langs else get_ud_langs(resource_dir)), 
                        inflection_map=swap_gender_any,
                        unimorph_args = {
                            "filter_entries": {
                                "upos": ["V"],
                                },
                                "combine_um_ud": True,
                                "remove_multiword_forms": True,
                                },
                        deprel_dir="_".join(target.child_deprels), 
                        resource_dir="../../resources", 
                        word_order_dir=f"../../treebank_features/{deprel_dir}_part",
                        max_treebank_len=args.max_treebank_len,
                        never_skip=args.never_skip,
                        rm_columns=["nsubj_child-deprel_conj",
                                    #"head_child-deprel_cop",
                                    #"head_child-deprel_aux"
                                    ],
                        target_id=args.target_id or sys.argv[0][:-3],
                        threshold=0.12,
                        simplify=not args.distinguish_unk,
                        n_jobs=args.n_jobs,
                        max_worker_mem_gb=args.max_worker_mem_gb,
                        force=args.force,
                        drop_unk=not args.keep_unk,
                        )
    pipeline()
