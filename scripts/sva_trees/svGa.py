import argparse
import sys
import random

sys.path.append("../../src/")

from sva_trees.pipeline import Pipeline
from multiblimp.languages import get_ud_langs
from word_order.prediction_target import *

random.seed(42)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="*", help="Languages to process", default=[])
    args = parser.parse_args()

    target = nsubj_target
    deprel_dir = "_".join(target.child_deprels)
    resource_dir = "../../resources"

    pipeline = Pipeline(target=target,
                        predictor_var="head_nsubj_Gender_agreement",
                        langs=args.langs if args.langs else get_ud_langs(resource_dir), 
                        deprel_dir="_".join(target.child_deprels), 
                        resource_dir="../../resources", 
                        word_order_dir=f"../../treebank_features/{deprel_dir}",
                        max_treebank_len=30_000)
    pipeline()
