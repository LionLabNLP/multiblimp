import os
import sys
import random
import math
import gc
import joblib
import traceback

from concurrent.futures import ProcessPoolExecutor, as_completed
from pandas import DataFrame

PYTHONIOENCODING="utf-8"

sys.path.append("../")

from word_order.process_treebank import (
    create_word_order_df, read_df, clear_form_groups_cache,
)
from word_order.decision_tree import fit_dt
from word_order.viz_tree import tree2html
from word_order.viz_deprel import generate_html_deprel_index
from word_order.viz_overview import generate_html_overview_index
from multiblimp.languages import lang2langcode, gblang2udlang
from multiblimp.unimorph import load_inflector
from multiblimp.agreement_pipeline_utils import (
    available_system_memory_bytes, find_other_running_instances,
    limit_process_memory, read_unk_counts,
)
from sva_trees.create_pairs import create_pairs
from sva_trees.diagnostics import generate_diagnostics_table, write_diagnostics_csv, diagnostics_row_to_json

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

def get_ud_lookup_table(inflector):
    """UD-derived UniMorph-schema data (expand_anno fallback for features
    um_data had no confident match for). Same shape as get_um_lookup_table's
    return value. None if `inflector` wasn't built with combine_um_ud=True
    (i.e. has no inflector.ud_inflector to source from).
    """
    if inflector.ud_inflector is None:
        return None
    ud_df = inflector.ud_inflector.load_unimorph_pickle("ud_unimorph/ud_pickles", filter={})
    if type(ud_df)==DataFrame:
        ud_data = {pos: ud_df[ud_df["upos"]==pos].dropna(axis=1, how="all")
                    for pos in ud_df["upos"].unique()}
        ud_data["full"] = ud_df
    else:
        ud_data = None
    return ud_data


class Pipeline:
    def __init__(self, target, predictor_var, langs, inflection_map, unimorph_args,
                 deprel_dir, resource_dir, word_order_dir,
                 max_treebank_len, never_skip=False, rm_columns=[], target_id=False,
                 threshold=0.12, simplify=False, n_jobs=1,
                 max_worker_mem_gb=None, mem_headroom=0.8, force=False,
                 agreement_feats=None, drop_unk=True, max_tasks_per_child=1):
        self.target = target
        self.predictor_var = predictor_var
        # Extract agreement var for all of these, df's can be shared between similar scripts.
        self.agreement_feats = (
            agreement_feats if agreement_feats is not None
            else ["Number", "Gender", "Person"]
        )
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
        # If False, unk-labeled rows stay in the DT fit instead of being
        # dropped (word_order.decision_tree.fit_dt's UNK_LABELS drop).
        self.drop_unk = drop_unk
        self.n_jobs = n_jobs # parallelise language computation across this many processes; 1=serial, >1=parallel
        # Per-worker RLIMIT_AS cap, in GB. None (default) auto-derives.
        self.max_worker_mem_gb = max_worker_mem_gb
        self.mem_headroom = mem_headroom
        # Recycle each worker after this many languages (ProcessPoolExecutor's
        # max_tasks_per_child). Default 1 == a fresh process per language, not
        # a long-lived one reused across the whole corpus. Matters even at
        # n_jobs=1: pandas/NumPy buffers and general heap fragmentation don't
        # reliably get handed back to the OS between languages, so a language
        # deep into a sorted 100+-language run can hit even a large fixed
        # memory cap purely from what earlier languages left behind, never
        # its own actual requirement -- a fresh process per language avoids
        # this entirely, since the OS fully reclaims a worker's memory the
        # moment it exits. Raise this (e.g. 5-10) to trade some of that
        # safety back for less per-language interpreter-startup overhead.
        self.max_tasks_per_child = max_tasks_per_child
        # If True, skip the startup check for other already-running instances
        # of this same script. Only meant for deliberate concurrent runs.
        self.force = force

    def _worker_mem_bytes(self):
        if self.max_worker_mem_gb is not None:
            return int(self.max_worker_mem_gb * 1024**3)
        available_mem = available_system_memory_bytes()
        if available_mem is None:
            return None
        return int((available_mem * self.mem_headroom) / self.n_jobs)

    def __call__(self):
        if not len(self.langs):
            raise ValueError("No langs specified/found")

        if not self.force:
            script_name = os.path.basename(sys.argv[0])
            other_pids = find_other_running_instances(script_name)
            if other_pids:
                raise RuntimeError(
                    f"Another instance of {script_name} appears to already be "
                    f"running (PID(s): {other_pids}). "
                    f"If they're stale, kill them first: kill {' '.join(map(str, other_pids))}"
                    f"Else, pass force=True to Pipeline."
                )

        langs = sorted(self.langs)
        worker_mem_bytes = self._worker_mem_bytes()
        if worker_mem_bytes is None:
            print("Could not detect system RAM; running without a memory cap")

        if worker_mem_bytes is not None:
            print(f"Capping each of {self.n_jobs} worker(s) to "
                  f"{worker_mem_bytes / 1024**3:.1f} GB RAM")
            initializer, initargs = limit_process_memory, (worker_mem_bytes,)
        else:
            initializer, initargs = None, ()
        # Always through ProcessPoolExecutor, even at n_jobs=1 -- see
        # max_tasks_per_child's docstring above for why a plain serial
        # for-loop in this process isn't safe for a full-corpus run.
        #
        # Per-language error isolation (matches subj_aux.pipeline.
        # SubjAuxPipeline): a failing language is caught and logged (message
        # + full traceback) rather than re-raised, so one bad language in a
        # long sweep doesn't take the diagnostics table/deprel index/
        # overview index below down with it for every OTHER language that
        # already succeeded -- those still get written to disk regardless
        # of how many other languages failed alongside them.
        with ProcessPoolExecutor(max_workers=self.n_jobs,
                                  max_tasks_per_child=self.max_tasks_per_child,
                                  initializer=initializer, initargs=initargs) as executor:
            future_to_lang = {
                executor.submit(self._process_language, lang): lang for lang in langs
            }
            for future in as_completed(future_to_lang):
                lang = future_to_lang[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"  FAILED: {lang}: {e}")
                    traceback.print_exc()

        for deprel in self.target.child_deprels:
            pairs_dir = f"../../minimal_pairs/{self.target_id}/{self.target_id}_{deprel}"
            decision_trees_dir = f"../../decision_trees/{self.target_id}/{self.target_id}_{deprel}"

            print("Generating diagnostics table for", deprel)
            diagnostics_df = generate_diagnostics_table(pairs_dir)
            write_diagnostics_csv(
                diagnostics_df,
                f"../../diagnostics/{self.target_id}/{self.target_id}_{deprel}.csv",
            )
            # Reshaped here (not inside word_order.viz_deprel) so that module
            # doesn't need to depend on sva_trees.
            diagnostics_by_lang = {
                row["Language"]: diagnostics_row_to_json(
                    row, lang_dir=os.path.join(pairs_dir, row["Language"])
                )
                for _, row in diagnostics_df.iterrows()
            }

            print("Generating deprel index for", deprel)
            generate_html_deprel_index(data_dir=decision_trees_dir,
                            html_directory=f"../../decision_trees/{self.target_id}",
                            target_col=self.predictor_var,
                            exclude_labels={"unk"} if self.simplify else {"--", "+-"},
                            include_trivial_labels={"Yes"},
                            leaf_threshold=self.leaf_threshold,
                            pairs_dir=pairs_dir,
                            diagnostics_by_lang=diagnostics_by_lang,
                            head_role_label="Verb",
                            )
        print("Generating overview index")
        generate_html_overview_index(html_directory=f"../../decision_trees/")

    def _process_language(self, lang):
        # Each language gets its own um_data POS-slices (sva_trees.pipeline.
        # get_um_lookup_table), can be dropped as soon as lang is done.
        try:
            self._process_language_impl(lang)
        finally:
            clear_form_groups_cache()
            gc.collect()

    def _process_language_impl(self, lang):
        # If the minimal pairs for all deprels already exist, skip this language
        if not self.never_skip and all(
            os.path.isdir(f"../../minimal_pairs/{self.target_id}/{self.target_id}_{deprel}/{lang}")
            for deprel in self.target.child_deprels
            ):
            print(f"{lang}: Skip, minimal pairs already found")
            return

        print(lang)

        cached_lang = gblang2udlang.get(lang, lang).replace(" ", "_")
        df = read_df(lang, word_order_dir=self.word_order_dir) if (
            os.path.exists(f"{self.word_order_dir}/{cached_lang}.parquet")
            ) else False

        inflector, skip_lang, num_lemma, num_form = load_inflector(
                lang=lang,
                langcode=lang2langcode(lang),
                unimorph_args=self.unimorph_args,
                inflection_map=self.inflection_map,
                resource_dir=self.resource_dir,)

        if self.never_skip or type(df)==bool or not self.predictor_var in df.columns:
            um_data = get_um_lookup_table(inflector)
            ud_data = get_ud_lookup_table(inflector)

            df = create_word_order_df(
                lang=lang,
                target=self.target,
                resource_dir=self.resource_dir,
                save_to=self.word_order_dir,
                max_treebank_len=self.max_treebank_len,
                drop_singleton_columns=True,
                predictor_var=self.predictor_var,
                agreement_feats=self.agreement_feats,
                lexicalize=True,
                um_data=um_data,
                ud_data=ud_data,
                fetch_all=True,
            )
            clear_form_groups_cache()

        if not len(df):
            print(f"Skipping {lang}, raw_df has no entries")
            return

        full_df = df[df[self.predictor_var].notnull()]
        del df

        for col in full_df:
            # drop all instances with a specific column=True value; greedily
            # match col name to also rm :pass or :tense items
            if any(col.startswith(y) for y in self.rm_columns):
                full_df = full_df[~full_df[col]]

        if self.simplify:
            full_df[self.predictor_var] = (
                full_df[self.predictor_var]
                .astype(str)
                .replace({"+-": "unk", "--": "unk"})
            )

        min_impurity_decrease = get_impurity(len(full_df))
        # Raw label distribution across ALL of full_df (before fit_dt drops unk rows)
        # for the deprel overview's per-language distribution bar.
        label_distribution = { str(k): int(v) for k, v in full_df[self.predictor_var].value_counts().items()}

        for deprel in self.target.child_deprels:
            # Reset per-iteration: each deprel gets its own dt_df/model
            dt_df, model, learn_dt, unk_counts = None, None, False, None
            decision_trees_dir = f"../../decision_trees/{self.target_id}/{self.target_id}_{deprel}"

            pred_values = set(full_df[self.predictor_var].values)

            if self.never_skip or not os.path.exists(f"{decision_trees_dir}/{lang}.joblib"):
                if len(pred_values)>1:
                    model, dt_df, predictor_df, unk_counts = fit_dt(
                        full_df=full_df,
                        model_type="decision_tree",
                        target=self.target,
                        verbose=1,
                        predictor_var=self.predictor_var,
                        min_impurity_decrease=min_impurity_decrease,
                        min_samples_leaf=10,
                        save_to=f"{decision_trees_dir}/{lang}",
                        omit_feats=set([col for col in full_df if col.endswith(f"_{self.target.swap_feat}") and not col.startswith("swap_")]),
                        drop_unk=self.drop_unk,
                        )
                    if model:
                        learn_dt=True
                if len(pred_values)<=1 or learn_dt==False: # no decision to learn
                    model, predictor_df = None, None
                    dt_df = full_df
                    learn_dt = False
            else:
                dt_df = read_df(lang, word_order_dir=decision_trees_dir)
                model = joblib.load(f"{decision_trees_dir}/{lang}.joblib")
                unk_counts = read_unk_counts(decision_trees_dir, lang)

            if self.never_skip or not os.path.exists(f"../../decision_trees/{self.target_id}/{lang}.html"):
                tree2html(
                    pipeline_model=model,
                    dt_df=dt_df,
                    full_df=full_df,
                    predictor_var=self.predictor_var,
                    target=self.target,
                    out_file=f"../../decision_trees/{self.target_id}/{lang}.html",
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
                    ),
                    leaf_threshold=self.leaf_threshold,
                    full_label_distribution=label_distribution,
                    head_label="Verb",
                )
            try:
                create_pairs(dt_df, swap_feat=self.predictor_var , inflector=inflector,
                        leaf_threshold=self.leaf_threshold,
                        save_to=f"../../minimal_pairs/{self.target_id}/{self.target_id}_{deprel}/{lang}",
                        num_lemma=num_lemma,
                        num_form=num_form,
                        full_df=full_df,
                        unk_counts=unk_counts,
                        label_distribution=label_distribution,
                        head_label="Verb")
            except KeyError:
                print(f"Skipping {lang} for {deprel}, missing column")