import os
import sys
import random
import math
import io
from concurrent.futures import ProcessPoolExecutor, as_completed

from pandas import DataFrame

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append("../")

from word_order.process_treebank import (
    load_treebank, create_word_order_df, read_df, clear_form_groups_cache,
)
from word_order.decision_tree import fit_dt
from word_order.viz_tree import tree2html
from word_order.viz_deprel import generate_html_deprel_index
from word_order.viz_overview import generate_html_overview_index
from multiblimp.languages import lang2langcode
from multiblimp.unimorph import load_inflector
from sva_trees.create_pairs import create_pairs

random.seed(42)

def get_impurity(n, min_n=300, max_n=4000, max_val=0.1, min_val=0.01):
    if n <= min_n:
        return max_val
    if n >= max_n:
        return min_val

    t = (math.log(n) - math.log(min_n)) / (math.log(max_n) - math.log(min_n))
    
    return max_val - t * (max_val - min_val)


def get_um_lookup_table(inflector):
    um_df = inflector.load_unimorph_pickle("unimorph/um_pickles", filter={})
    if type(um_df)==DataFrame:
        um_data = {pos: um_df[um_df["upos"]==pos].dropna(axis=1, how="all") 
                    for pos in um_df["upos"].unique()}
        um_data["full"] = um_df
    else:
        um_data = None
    return um_data

def get_data_df(lang, resource_dir, save_to_dir, target, predictor_var, inflector, max_treebank_len):
    treebank = load_treebank(lang, resource_dir, max_treebank_len=max_treebank_len)

    # get POS'es from prediction target, load one lookup_df for each
    # TODO alpha split or precompute?
    um_data = get_um_lookup_table(inflector)

    df = create_word_order_df(
        lang=lang, 
        treebank=treebank,
        target=target, 
        resource_dir=resource_dir,
        save_to=save_to_dir,
        max_treebank_len=max_treebank_len,
        drop_singleton_columns=True,
        predictor_var=predictor_var,
        lexicalize=True,
        um_data=um_data,
        fetch_all=True,
    )
    return df

class Pipeline:
    def __init__(self, target, predictor_var, langs, inflection_map, unimorph_args,
                 deprel_dir, resource_dir, word_order_dir,
                 max_treebank_len, never_skip=False, rm_columns=[], target_id=False,
                 threshold=0.12, simplify=False, n_jobs=1):
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
        self.leaf_threshold = threshold
        self.simplify = simplify  # collapse +-/-- to "unk" for decision tree
        self.n_jobs = n_jobs # parallelise langauge computation across this many processes; 1=serial, >1=parallel

    def __call__(self):
        if not len(self.langs):
            raise ValueError("No langs specified/found")

        langs = sorted(self.langs)
        if self.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(self._process_language, lang) for lang in langs]
                for future in as_completed(futures):
                    future.result()  # re-raise any worker exception here
        else:
            for lang in langs:
                self._process_language(lang)

        for deprel in self.target.child_deprels:
            flowchart_dir = f"../../decision_trees/{self.target_id}/{self.target_id}_{deprel}/flowcharts/"
            generate_html_deprel_index(data_dir=f"../../decision_trees/{self.target_id}/{self.target_id}_{deprel}",
                            html_directory=f"../../decision_trees/{self.target_id}",
                            target_col=self.predictor_var,
                            exclude_labels={"unk"} if self.simplify else {"--", "+-"},
                            flowchart_dir=flowchart_dir,
                            leaf_threshold=self.leaf_threshold,
                            pairs_dir=f"../../minimal_pairs/{self.target_id}/{self.target_id}_{deprel}",
                            )
        generate_html_overview_index(html_directory=f"../../decision_trees/")

    def _process_language(self, lang):
        # Each language gets its own um_data POS-slices (sva_trees.pipeline.
        # get_um_lookup_table), and the langs loop never revisits a language, so
        # process_treebank's per-DataFrame groupby cache can be dropped as soon as
        # we're done with this one rather than growing for the rest of the run.
        try:
            self._process_language_impl(lang)
        finally:
            clear_form_groups_cache()

    def _process_language_impl(self, lang):
        print(lang)

        html_file = f"../../decision_trees/{self.target_id}/{lang}.html"
        if self.never_skip==False and os.path.exists(html_file):
            print(f"HTML found and -ns==False.")
            return

        df = read_df(lang, word_order_dir=self.word_order_dir) if (
            os.path.exists(f"{self.word_order_dir}/{lang.replace(' ', '_')}.parquet")
            ) else False

        inflector, skip_lang, num_lemma, num_form = load_inflector(
                lang=lang,
                langcode=lang2langcode(lang),
                unimorph_args=self.unimorph_args,
                inflection_map=self.inflection_map,
                resource_dir=self.resource_dir,)

        if self.never_skip or type(df)==bool or not self.predictor_var in df.columns:
            df = get_data_df(lang, self.resource_dir, self.word_order_dir, self.target, self.predictor_var,
                            inflector, self.max_treebank_len)
            print(lang, len(df))
            if not len(df):
                print(f"Skipping {lang}, raw_df has no entries")
                return
        else:
            print(f"Skipping {lang} load_treebank(), df already found")

        print("Generating HTML file for", lang)

        full_df = df[df[self.predictor_var].notnull()]

        for col in full_df:
            # drop all instances with a specific column=True value; greedily
            # match col name to also rm :pass or :tense items
            if (lambda x: any([x.startswith(y) for y in self.rm_columns]))(col):
                full_df = full_df[~full_df[col]]

        if self.simplify:
            full_df[self.predictor_var] = (
                full_df[self.predictor_var]
                .astype(str)
                .replace({"+-": "unk", "--": "unk"})
            )

        min_impurity_decrease = get_impurity(len(full_df))

        for deprel in self.target.child_deprels:
            # Reset per-iteration: each deprel gets its own dt_df/model, never a
            # stale one carried over from a previous deprel in this loop (or left
            # unbound entirely on the first iteration).
            dt_df, model, learn_dt = None, None, False

            if self.never_skip or not os.path.exists(f"../../decision_trees/{self.target_id}/{self.target_id}_{deprel}/{lang}"):
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

            else:
                print(f"Skipping {lang}, decision tree already found")

            if dt_df is None:
                # The "already found" branch above didn't reload the tree from
                # disk, so there's nothing to render/generate pairs from this
                # iteration — skip rather than reuse a previous deprel's dt_df.
                print(f"Skipping {lang}/{deprel} tree2html/create_pairs, no dt_df available")
                continue

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
                palette_map = (
                    {"Yes": "#31cb9f", "No": "#f16393", "unk": "#b893de"}
                    if self.simplify else
                    {"Yes": "#31cb9f", "No": "#f16393",
                     "+-": "#e5c64d", "--": "#b893de"}
                )
            )

            print("Generating minimal pairs for", lang)
            flowchart_dir = f"../../decision_trees/{self.target_id}/{self.target_id}_{deprel}/flowcharts/"
            create_pairs(dt_df, swap_feat=self.predictor_var , inflector=inflector,
                        leaf_threshold=self.leaf_threshold,
                        save_to=f"../../minimal_pairs/{self.target_id}/{self.target_id}_{deprel}/{lang}",
                        flowchart_path = os.path.join(flowchart_dir, f"{lang}.html"),
                        verbose=True)
