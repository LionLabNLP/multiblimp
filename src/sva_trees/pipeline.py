import os
import sys
import random
import math
import io

from tqdm import tqdm
from pandas import DataFrame

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append("../")

from word_order.process_treebank import load_treebank, create_word_order_df, read_df
from word_order.decision_tree import fit_dt
from word_order.viz_tree import tree2html
from word_order.viz_deprel import generate_html_deprel_index
from word_order.viz_overview import generate_html_overview_index
from multiblimp.languages import remove_diacritics_langs, remove_multiples_langs, lang2langcode
from multiblimp.unimorph import UnimorphInflector
from word_order.create_pairs import create_pairs

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
        fill_unk_values=False,
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
        # with open("rivals.txt", "w", encoding="utf-8") as f :
        #     print(self.target_id, file=f)

        trivial_langs = dict()
        for lang in (sorted(self.langs)):
            print(lang)
            html_file = f"../../decision_trees/{self.target_id}/{lang}.html"
            if self.never_skip==False and os.path.exists(html_file):
                print(f"Skipping {lang}, -ns==False and html found.")
                continue

            df = read_df(lang, word_order_dir=self.word_order_dir) if (
                os.path.exists(f"{self.word_order_dir}/{lang.replace(' ', '_')}.parquet")
                ) else False
            #  os.path.join(word_order_dir or "", f"{lang}{deprel_suffix}.parquet")
            #  os.path.exists(f"{self.word_order_dir}/{lang.replace(' ', '_')}.csv")
            
            if self.never_skip or type(df)==bool or not self.predictor_var in df.columns:
                treebank = load_treebank(lang, self.resource_dir, max_treebank_len=self.max_treebank_len)
                # TODO
               # with open("rivals.txt", "a", encoding="utf-8") as f :
                #   print(lang, file=f)
                # get POS'es from prediction target, load one lookup_df for each 

                inflector, skip_lang, num_lemma, num_form = load_inflector(
                    lang=lang, 
                    langcode=lang2langcode(lang),
                    unimorph_args=self.unimorph_args,
                    inflection_map=self.inflection_map,
                    resource_dir=self.resource_dir,)

                # TODO: split by upos for quicker filter in expand_anno, which POS does language have ?
                um_df = inflector.load_unimorph_pickle("unimorph/um_pickles", filter={}) # default load without pickle
                # um_df = inflector.load_unimorph(
                #         remove_diacritics=False,
                #         filter={},
                #         remove_multiples=False,
                #         load_from_pickle= False,) # default load without pickle

                if type(um_df)==DataFrame:
                    um_data = {pos: um_df[um_df["upos"]==pos].dropna(axis=1, how="all") 
                               for pos in um_df["upos"].unique()}
                    um_data["full"] = um_df
                else:
                    um_data = None

                # with open("debug_mapping.txt", "a", encoding="utf-8") as f:
                #     print(um_data.keys(), file=f)
                #     print(um_data["ADJ"], file=f)
                #     print(um_data["ADJ"].dropna(axis=1, how="all"), file=f)
                #     raise KeyError

                um_df = inflector.load_unimorph_pickle("unimorph/um_pickles", filter={}) # default load without pickle
                if type(um_df)==DataFrame:
                    um_data = {pos: um_df[um_df["upos"]==pos].dropna(axis=1, how="all") 
                               for pos in um_df["upos"].unique()}
                    um_data["full"] = um_df
                else:
                    um_data = None

                df = create_word_order_df(
                    lang=lang, 
                    treebank=treebank,
                    target=self.target, 
                    resource_dir=self.resource_dir,
                    save_to=self.word_order_dir,
                    max_treebank_len=self.max_treebank_len,
                    drop_singleton_columns=True,
                    predictor_var=self.predictor_var,
                    lexicalize=True,
                    um_data=um_data,
                    fetch_all=False,
                )
                #raise ValueError

                #with open("rivals.txt", "a", encoding="utf-8") as f:
                 #   print("="*50, file=f)
                #raise FileExistsError

                print(lang, len(df))
                if not len(df):
                    print(f"Skipping {lang}, raw_df has no entries") # TODO still create dummy html?
                    continue
            else:
                print(f"Skipping {lang} load_treebank(), df already found")

            # langs = [path.split('/')[-1].split('.')[0] for path in glob(word_order_dir+"/*.csv")]

            if self.never_skip or not os.path.exists(html_file):
                print("Generating HTML file for", lang)
                #treebank = load_treebank(lang, self.resource_dir, max_treebank_len=self.max_treebank_len)
                try:
                    #raw_df = read_df(lang, word_order_dir=self.word_order_dir)
                    full_df = df[df[self.predictor_var].notnull()]
                except FileNotFoundError:
                    print(f"Skipping {lang}, '../../treebank_features/nsubj/{lang}.csv' could'nt be found or df was 0")
                    continue

                # drop all instances with a specific column=True value; greedily
                # match col name to also rm :pass or :tense items
                for col in full_df:
                    if (lambda x: any([x.startswith(y) for y in self.rm_columns]))(col):
                        full_df = full_df[~full_df[col]]

                omit_feats = None #{col for col in full_df.columns if ('nsubj' in col) or ('obj' in col) or ('form' in col) or ('lemma' in col)}
                min_impurity_decrease = get_impurity(len(full_df))

                for deprel in self.target.child_deprels:
                    trivial_langs[deprel] = trivial_langs.get(deprel, dict())
                    if self.never_skip or not os.path.exists(f"../../decision_trees/{self.target_id}/{self.target_id}_{deprel}/{lang}"):
                        learn_dt=False
                        pred_values = set(full_df[self.predictor_var].values)

                        if len(pred_values)>1: # TODO or label is ==yes
                            model, dt_df, predictor_df = fit_dt(
                                full_df=full_df,
                                model_type="decision_tree",
                                target=self.target,
                                verbose=1,
                                predictor_var=self.predictor_var,
                                min_impurity_decrease=min_impurity_decrease,
                                min_samples_leaf=10,
                                save_to=f"../../decision_trees/{self.target_id}/{self.target_id}_{deprel}/{lang}",
                                omit_feats={f"{"head"}_{self.target.swap_feat}",
                                            f"{deprel}_{self.target.swap_feat}"}
                                            )
                            if model: 
                                learn_dt=True
                        if len(pred_values)<=1 or learn_dt==False: # no decision to learn
                            model, predictor_df = None, None
                            dt_df = full_df
                            learn_dt = False
                            trivial_langs[deprel][lang] = trivial_langs[deprel].get(lang, list(pred_values)[0])

                    else: 
                        print(f"Skipping {lang}, decision tree already found")

                    tree2html(
                        pipeline_model=model, 
                        dt_df=dt_df, 
                        full_df=full_df, 
                        predictor_var=self.predictor_var,
                        target=self.target,
                        out_file=html_file, 
                        max_rows=15,
                        meta={"Language": lang},
                        only_show_real_orders=True,
                        correlate_features=True,
                        show_features=True,
                        full_tree_html=learn_dt,
                        palette_map = {"Yes": "#31cb9f", "No": "#f16393",
                                        "+-": "#e5c64d","--": "#b893de",}
                    )
 
            else:
                print(f"Skipping {lang}, HTML file found")

        generate_html_deprel_index(data_dir=f"../../decision_trees/{self.target_id}/{self.target_id}_nsubj",
                                    html_directory=f"../../decision_trees/{self.target_id}",
                                    target_col=self.predictor_var,
                                    exclude_labels={"--", "+-"}
                                    #trivial_langs=trivial_langs[deprel]
                                    )
        generate_html_overview_index(html_directory=f"../../decision_trees/")
