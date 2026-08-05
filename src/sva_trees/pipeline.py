import os
import sys
import random
import math
import gc
import joblib

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
from multiblimp.languages import lang2langcode
from multiblimp.unimorph import load_inflector
from sva_trees.create_pairs import create_pairs

random.seed(42)

def _total_system_memory_bytes():
    """Best-effort total physical RAM, POSIX only. None if undetectable
    (e.g. Windows). Only used as a fallback when /proc/meminfo isn't
    available — prefer _available_system_memory_bytes for anything
    memory-cap-related.
    """
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def _available_system_memory_bytes():
    """Best-effort currently-available memory."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, OSError, ValueError, IndexError):
        pass
    return _total_system_memory_bytes()


def _limit_worker_memory(max_bytes):
    """ProcessPoolExecutor initializer: caps this worker's address space so a
    runaway language (huge treebank/lookup tables) hits a catchable
    MemoryError instead of letting the OS OOM-killer kill processes system-
    wide (which is what takes down unrelated services, not just this pool).
    """
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
    except (ImportError, ValueError, OSError):
        pass


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
    """UD-derived UniMorph-schema data, used as an expand_anno fallback for
    features um_data didn't have a confident match for. Same shape as
    get_um_lookup_table's return value, just sourced from ud_unimorph/ud_pickles
    via inflector's internal UD-mode inflector instead of unimorph/um_pickles.

    None whenever there's no UD-mode inflector to source from — i.e. the
    caller didn't build `inflector` with combine_um_ud=True, same as passing
    use_ud_inflections=True directly would (see UnimorphInflector's assertion
    against combining both at once): either way there's a single, correct
    source of UD-derived data, inflector.ud_inflector.
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

def get_data_df(lang, resource_dir, save_to_dir, target, predictor_var, inflector, max_treebank_len):
    um_data = get_um_lookup_table(inflector)
   # ud_data = get_ud_lookup_table(inflector)

    # treebank is loaded inside create_word_order_df (treebank=None default)
    # rather than here, so this frame never holds its own reference to it —
    # otherwise `del treebank` there couldn't actually drop the refcount to
    # zero while this call is still on the stack.
    df = create_word_order_df(
        lang=lang,
        target=target,
        resource_dir=resource_dir,
        save_to=save_to_dir,
        max_treebank_len=max_treebank_len,
        drop_singleton_columns=True,
        predictor_var=predictor_var,
        lexicalize=True,
        um_data=um_data,
       # ud_data=ud_data,
        fetch_all=True,
    )
    return df

def _find_other_running_instances(script_name):
    """PIDs of other processes whose command line mentions script_name,
    excluding this process itself. Uses pgrep -f; returns [] (skips the
    check) if pgrep isn't available rather than blocking the run."""
    import subprocess
    try:
        result = subprocess.run(
            ["pgrep", "-f", script_name], capture_output=True, text=True
        )
    except (FileNotFoundError, OSError):
        return []
    pids = [int(p) for p in result.stdout.split() if p.strip().isdigit()]
    return [p for p in pids if p != os.getpid()]


class Pipeline:
    def __init__(self, target, predictor_var, langs, inflection_map, unimorph_args,
                 deprel_dir, resource_dir, word_order_dir,
                 max_treebank_len, never_skip=False, rm_columns=[], target_id=False,
                 threshold=0.12, simplify=False, n_jobs=1,
                 max_worker_mem_gb=None, mem_headroom=0.8, force=False):
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
        # Per-worker RLIMIT_AS cap, in GB. None (default) auto-derives one from
        # currently-available system RAM (mem_headroom fraction, split across
        # n_jobs).
        self.max_worker_mem_gb = max_worker_mem_gb
        self.mem_headroom = mem_headroom
        # If True, skip the startup check for other already-running instances
        # of this same script. Only meant for deliberate concurrent runs.
        self.force = force

    def _worker_mem_bytes(self):
        if self.max_worker_mem_gb is not None:
            return int(self.max_worker_mem_gb * 1024**3)
        available_mem = _available_system_memory_bytes()
        if available_mem is None:
            return None
        return int((available_mem * self.mem_headroom) / self.n_jobs)

    def __call__(self):
        if not len(self.langs):
            raise ValueError("No langs specified/found")

        if not self.force:
            script_name = os.path.basename(sys.argv[0])
            other_pids = _find_other_running_instances(script_name)
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

        if self.n_jobs > 1:
            if worker_mem_bytes is not None:
                print(f"Capping each of {self.n_jobs} workers to "
                      f"{worker_mem_bytes / 1024**3:.1f} GB RAM")
                initializer, initargs = _limit_worker_memory, (worker_mem_bytes,)
            else:
                initializer, initargs = None, ()
            with ProcessPoolExecutor(max_workers=self.n_jobs,
                                      initializer=initializer, initargs=initargs) as executor:
                futures = [executor.submit(self._process_language, lang) for lang in langs]
                for future in as_completed(futures):
                    future.result()  # re-raise any worker exception here
        else:
            if worker_mem_bytes is not None:
                print(f"Capping this process to {worker_mem_bytes / 1024**3:.1f} GB RAM")
                _limit_worker_memory(worker_mem_bytes)
            for lang in langs:
                self._process_language(lang)

        for deprel in self.target.child_deprels:
            print("Generating deprel index for", deprel)
            flowchart_dir = f"../../decision_trees/{self.target_id}/{self.target_id}_{deprel}/flowcharts/"
            generate_html_deprel_index(data_dir=f"../../decision_trees/{self.target_id}/{self.target_id}_{deprel}",
                            html_directory=f"../../decision_trees/{self.target_id}",
                            target_col=self.predictor_var,
                            exclude_labels={"unk"} if self.simplify else {"--", "+-"},
                            include_trivial_labels={"Yes"},
                            flowchart_dir=flowchart_dir,
                            leaf_threshold=self.leaf_threshold,
                            pairs_dir=f"../../minimal_pairs/{self.target_id}/{self.target_id}_{deprel}",
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
            # um_data/ud_data no longer needed
            clear_form_groups_cache()

        if not len(df):
            print(f"Skipping {lang}, raw_df has no entries")
            return

        full_df = df[df[self.predictor_var].notnull()]
        del df

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
            # Reset per-iteration: each deprel gets its own dt_df/model
            dt_df, model, learn_dt = None, None, False

            pred_values = set(full_df[self.predictor_var].values)

            if self.never_skip or not os.path.exists(f"../../decision_trees/{self.target_id}/{self.target_id}_{deprel}/{lang}"):
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
                # read from disk if it already exists
                dt_df = read_df(lang, word_order_dir=f"../../decision_trees/{self.target_id}/{self.target_id}_{deprel}")
                model = joblib.load(f"../../decision_trees/{self.target_id}/{self.target_id}_{deprel}/{lang}.joblib")

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
                    )
                )
            try:
                create_pairs(dt_df, swap_feat=self.predictor_var , inflector=inflector,
                        leaf_threshold=self.leaf_threshold,
                        save_to=f"../../minimal_pairs/{self.target_id}/{self.target_id}_{deprel}/{lang}",
                        flowchart_path = os.path.join(f"../../decision_trees/{self.target_id}/{self.target_id}_{deprel}/flowcharts/", f"{lang}.html"),
                        verbose=True)
            except KeyError:
                print(f"Skipping {lang} for {deprel}, missing column")