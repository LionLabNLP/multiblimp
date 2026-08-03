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
    args = parser.parse_args()

    target = nsubj_target
    target.head_feats = {"VerbForm": partial(filter_head_feats, exclude="Part")}
    # target.child_feats = {}
    deprel_dir = "_".join(target.child_deprels)
    resource_dir = "../../resources"

    pipeline = Pipeline(target=target,
                        predictor_var="head_nsubj_Gender_agreement",
                        langs=(args.langs if args.langs else get_ud_langs(resource_dir)), 
                        inflection_map=swap_number_subj_any,
                        unimorph_args = {
                            "filter_entries": {
                                "upos": ["V"],
                                },
                                "combine_um_ud": True,
                                "remove_multiword_forms": True,
                                },
                        deprel_dir="_".join(target.child_deprels), 
                        resource_dir="../../resources", 
                        word_order_dir=f"../../treebank_features/{deprel_dir}",
                        max_treebank_len=30_000,
                        never_skip=args.never_skip,
                        rm_columns=["nsubj_child-deprel_conj",
                                    #"head_child-deprel_cop",
                                    #"head_child-deprel_aux"
                                    ],
                        target_id=sys.argv[0][:-3])
    pipeline()
