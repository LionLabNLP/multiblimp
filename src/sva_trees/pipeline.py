import os
import sys
import random
import math

from tqdm import tqdm

sys.path.append("../")

from word_order.process_treebank import load_treebank, create_word_order_df, read_df
from word_order.decision_tree import fit_dt
from word_order.viz_tree import tree2html
from multiblimp.languages import remove_diacritics_langs, remove_multiples_langs, lang2langcode
from multiblimp.unimorph import UnimorphInflector

random.seed(42)

def get_impurity(n, min_n=300, max_n=4000, max_val=0.1, min_val=0.01):
    if n <= min_n:
        return max_val
    if n >= max_n:
        return min_val

    t = (math.log(n) - math.log(min_n)) / (math.log(max_n) - math.log(min_n))
    
    return max_val - t * (max_val - min_val)

def load_inflector(lang: str, langcode: str, unimorph_args, inflection_map: dict, 
                   resource_dir: str): # TODO put in utils?
    num_form = 0
    num_lemma = 0
    skip_lang = False

    remove_diacritics = lang in remove_diacritics_langs
    remove_multiples = lang in remove_multiples_langs

    inflector = UnimorphInflector(
        langcode=langcode,
        inflection_map=inflection_map,
        resource_dir=resource_dir,
        load_from_pickle=True,
        remove_diacritics=remove_diacritics,
        remove_multiples=remove_multiples,
        **unimorph_args,
    )

    if len(inflector) == 0:
        skip_lang = True
        num_lemma = 0
        num_form = 0
    else:
        num_lemma = inflector.num_lemmas
        num_form = inflector.num_forms

    if not inflector.can_feature_swap:
        skip_lang = True

    return inflector, skip_lang, num_lemma, num_form

class Pipeline:
    def __init__(self, target, predictor_var, langs, inflection_map, unimorph_args,
                 deprel_dir, resource_dir, word_order_dir,
                 max_treebank_len, never_skip=False, rm_columns=[], target_id=False):
        self.target = target
        self.predictor_var = predictor_var
        self.langs = langs
        self.inflection_map = inflection_map
        self.unimorph_args = unimorph_args
        self.deprel_dir = deprel_dir
        self.resource_dir = resource_dir
        self.word_order_dir = word_order_dir
        self.max_treebank_len = max_treebank_len
        self.never_skip = never_skip
        self.rm_columns = rm_columns
        self.target_id = target_id if target_id else predictor_var.split("_")[2][0]

    def __call__(self):
        if not len(self.langs):
            raise ValueError("No langs specified/found")

        for lang in tqdm(sorted(self.langs)):
            raw_df = read_df(lang, word_order_dir=self.word_order_dir) if os.path.exists(f"{self.word_order_dir}/{lang.replace(' ', '_')}.csv") else False
            if self.never_skip or type(raw_df)==bool or not self.predictor_var in raw_df.columns:
                treebank = load_treebank(lang, self.resource_dir, max_treebank_len=self.max_treebank_len)
                # TODO
                inflector, skip_lang, num_lemma, num_form = load_inflector(lang=lang, langcode=lang2langcode(lang),
                               unimorph_args=self.unimorph_args,
                                inflection_map=self.inflection_map,
                                resource_dir=self.resource_dir)
                
                df = create_word_order_df(
                    lang=lang, 
                    treebank=treebank,
                    target=self.target, 
                    resource_dir=self.resource_dir,
                    save_to=self.word_order_dir,
                    max_treebank_len=self.max_treebank_len,
                    drop_singleton_columns=True,
                    inflector=inflector
                )

                print(lang, len(df))
                if not len(df):
                    print(f"Skipping {lang}, raw_df has no entries")
                    continue
            else:
                print(f"Skipping {lang} load_treebank(), df already found")

            # langs = [path.split('/')[-1].split('.')[0] for path in glob(word_order_dir+"/*.csv")]

            deprels = ['nsubj']
            html_file = f"../../decision_trees/{self.target_id}/html/{lang}.html"
            if self.never_skip or not os.path.exists(html_file):
                print(lang)

                #treebank = load_treebank(lang, self.resource_dir, max_treebank_len=self.max_treebank_len)
                try:
                    raw_df = read_df(lang, word_order_dir=self.word_order_dir)
                    full_df = raw_df[raw_df[self.predictor_var].notnull()]
                except FileNotFoundError:
                    print(f"Skipping {lang}, '../../treebank_features/nsubj/{lang}.csv' could'nt be found or df was 0")
                    continue

                # remove samples ; TODO must be head_ prefix for nsubj
                # if 'child_sibling-deprel_aux' in full_df.columns:
                #     full_df = full_df[~full_df['child_sibling-deprel_aux']]
                # if 'child_sibling-deprel_cop' in full_df.columns:
                #     full_df = full_df[~full_df['child_sibling-deprel_cop']]

                # drop all instances with a specific column=True value
                for col in self.rm_columns:
                    if col in full_df.columns:
                        full_df = full_df[~full_df[col]]

                omit_feats = None #{col for col in full_df.columns if ('nsubj' in col) or ('obj' in col) or ('form' in col) or ('lemma' in col)}
                min_impurity_decrease = get_impurity(len(full_df))

                for deprel in deprels:
                    if self.never_skip or not os.path.exists(f"../../decision_trees/{self.target_id}/{self.target_id}_{deprel}/{lang}"):
                        try:
                            model, dt_df, predictor_df = fit_dt(
                                full_df, 
                                self.target, 
                                verbose=1, 
                                predictor_var=self.predictor_var,
                                min_impurity_decrease=min_impurity_decrease,
                                min_samples_leaf=10,
                                save_to=f"../../decision_trees/{self.target_id}/{self.target_id}_{deprel}/{lang}",
                                omit_feats=omit_feats,
                            )

                            if model is None:
                                raise TypeError(f"no model returned for {lang, deprel}")
                        except TypeError:
                            continue
                        
                    else: 
                        print(f"Skipping {lang}, decision tree already found")
                    
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
                        self.predictor_var,
                        self.target,
                        html_file, 
                        max_rows=15,
                        meta={"Language": lang},
                        only_show_real_orders=True,
                        correlate_features=True,
                        show_features=True
                    )

            else:
                print(f"Skipping {lang}, HTML file found")

            # generate_html_index('word_order/decision_trees/html/')