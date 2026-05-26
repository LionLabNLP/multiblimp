# %load_ext autoreload
# %autoreload 2
import argparse

import os
import sys
import random
import math

from tqdm import tqdm
# from glob import glob


sys.path.append("../../src/")

from multiblimp.languages import get_ud_langs
from word_order.process_treebank import load_treebank, create_word_order_df, read_df
from word_order.prediction_target import *
from word_order.decision_tree import fit_dt
from word_order.viz_tree import tree2html


random.seed(42)


def get_impurity(n, min_n=300, max_n=4000, max_val=0.1, min_val=0.01):
    if n <= min_n:
        return max_val
    if n >= max_n:
        return min_val

    t = (math.log(n) - math.log(min_n)) / (math.log(max_n) - math.log(min_n))
    
    return max_val - t * (max_val - min_val)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="*", help="Languages to process", default=[])

    args = parser.parse_args()

    target = nsubj_target #kwarg

    deprel_dir = "_".join(target.child_deprels)
    resource_dir = "../../resources"
    word_order_dir = f"../../treebank_features/{deprel_dir}"

    langs = args.langs if args.langs else get_ud_langs(resource_dir)

    max_treebank_len =  30_000

    for lang in tqdm(sorted(langs)):
        if not os.path.exists(f"{word_order_dir}/{lang.replace(' ', '_')}.csv"):
            treebank = load_treebank(lang, resource_dir, max_treebank_len=max_treebank_len)

            # mode must still be head_child (as opposed to multi_child) ##?
            df = create_word_order_df(
                lang=lang, 
                treebank=treebank,
                target=target, 
                resource_dir=resource_dir,
                save_to=word_order_dir,
                max_treebank_len=max_treebank_len,
                drop_singleton_columns=True,
            )

            print(lang, len(df))
        else:
            print(f"Skipping {lang}, df already found")

        # langs = [path.split('/')[-1].split('.')[0] for path in glob(word_order_dir+"/*.csv")]

        deprels = ['nsubj']
        dt_df_dir = "../../dt_df/"

        # all_base_entropy = []
        # all_model_entropy = []

        predictor_var = f"head_nsubj_Gender_agreement"
        tofeat = {"Number": "#", "Gender": "G", "Case": "C"}

        targetfeat = f"sv{tofeat[predictor_var.split("_")[2]]}a"

        html_file = f"../../decision_trees/{targetfeat}/html/{lang}.html"
        if not os.path.exists(html_file):

            print(lang)
            treebank = load_treebank(lang, resource_dir, max_treebank_len=max_treebank_len)
            raw_df = read_df(lang, word_order_dir=word_order_dir)

            full_df = raw_df[raw_df[predictor_var].notnull()]

            # full_df = full_df[full_df[predictor_var].str.len() == 3] # specific vor svo?
            # full_df = full_df[full_df['head_deprel'] == 'root'] # svo specific?

            ## idk
            # if 'child_sibling-deprel_aux' in full_df.columns:
            #     full_df = full_df[~full_df['child_sibling-deprel_aux']]
            # if 'child_sibling-deprel_cop' in full_df.columns:
            #     full_df = full_df[~full_df['child_sibling-deprel_cop']]

            omit_feats = None #{col for col in full_df.columns if ('nsubj' in col) or ('obj' in col) or ('form' in col) or ('lemma' in col)}

            min_impurity_decrease = get_impurity(len(full_df))

            for deprel in deprels:
                model, dt_df, predictor_df = fit_dt(
                    full_df, 
                    target, 
                    verbose=1, 
                    predictor_var=predictor_var,
                    min_impurity_decrease=min_impurity_decrease,
                    min_samples_leaf=10,
                    save_to=f"../../decision_trees/{targetfeat}/{targetfeat}_{deprel}/{lang}",
                    omit_feats=omit_feats,
                )

                if model is None:
                    print(f"Skipping {lang}, decision tree already found")
#??
            
                # ????
                # swap_df = create_pairs(
                #     model, 
                #     dt_df,
                #     full_df, 
                #     treebank, 
                #     predictor_var,
                #     swap_type="core_arg",
                #     # save_to_tight_keep=f"word_order/pairs/tight/{deprel}/{lang}.csv",
                #     # save_to_full_keep=f"word_order/pairs/full/{deprel}/{lang}.csv",
                #     save_to=os.path.join(dt_df_dir, f"{lang}.csv"),
                # )
                #tree2html(model, dt_df, full_df, predictor_var, html_file, max_rows=15)

                tree2html(
                model, 
                dt_df, 
                full_df, 
                predictor_var,
                target,
                html_file, 
                max_rows=15,
                meta={"Language": lang},
                only_show_real_orders=True,
                correlate_features=True,
            )

        else:
            print("skipping", lang)

    # generate_html_index('word_order/decision_trees/html/')