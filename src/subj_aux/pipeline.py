import dataclasses
import gc
import os
import sys
import traceback

from concurrent.futures import ProcessPoolExecutor, as_completed

import joblib

sys.path.append("../")

from word_order.process_treebank import (
    load_treebank, create_word_order_df, read_df, clear_form_groups_cache,
    drop_singleton_cols,
)
from word_order.decision_tree import fit_dt
from word_order.viz_tree import tree2html
from word_order.viz_deprel import generate_html_deprel_index
from word_order.viz_overview import generate_html_overview_index
from multiblimp.languages import gblang2udlang, lang2langcode
from multiblimp.unimorph import load_inflector
from multiblimp.agreement_pipeline_utils import (
    available_system_memory_bytes, find_other_running_instances,
    limit_process_memory, read_unk_counts,
)
from sva_trees.create_pairs import create_pairs
from sva_trees.pipeline import get_impurity, get_um_lookup_table, get_ud_lookup_table
from sva_trees.diagnostics import (
    generate_diagnostics_table, write_diagnostics_csv, diagnostics_row_to_json,
)
from multiblimp.config import (
    HTML_DECISION_TREES_DIR, OUTPUT_DECISION_TREES_DIR, OUTPUT_MINIMAL_PAIRS_DIR,
    OUTPUT_DIAGNOSTICS_DIR,
)

from .redirect import redirect_nsubj_to_aux, nsubj_aux_target, AUX_COUNT_FEAT


def extract_subj_aux_df(lang: str, resource_dir: str, save_to: str | None = None,
                         max_treebank_len: int | None = None,
                         agreement_feats: list[str] | None = None,
                         lexicalize: bool = True, never_skip: bool = False,
                         um_data=None, ud_data=None, fetch_all: bool = False):
    """Load `lang`'s treebank, redirect every nsubj to its clause's chosen
    auxiliary (see redirect_nsubj_to_aux), and extract the resulting
    nsubj-vs-aux instances via the ordinary create_word_order_df /
    extract_instances machinery, completely unmodified -- after redirection
    this is just an ordinary single-head/single-child extraction, the same
    shape word_order.prediction_target.nsubj_target already uses for VERB
    heads, so agreement labels (head_nsubj_{feat}_agreement) come out of
    the existing extract_instances logic for free.

    Caches (and, on a later call, reads back) ONE combined "{save_to}/
    {lang}.parquet" covering every instance -- single-aux, stacked-aux
    (2+), and the small "not redirected" bucket (= 0 aux).
    Which of those streams actually feeds decision-tree-fitting/minimal-
    pairs generation is a downstream concern,
    decided per SubjAuxPipeline run -- keeping the cache unsplit means that
    choice can change without re-extracting.

    agreement_feats defaults to ["Number", "Gender", "Person"], matching
    sva_trees.pipeline.Pipeline's default.

    um_data/ud_data/fetch_all: pass a loaded UniMorph/UD lookup table (see
    sva_trees.pipeline.get_um_lookup_table/get_ud_lookup_table) and
    fetch_all=True for the same UM/UD-lookup feature enrichment svNa.py
    etc. use. Defaults to fetch_all=False (raw UD feats only, no lookup)
    for callers that haven't built an inflector -- SubjAuxPipeline always
    passes fetch_all=True.

    create_word_order_df's own drop_singleton_columns is left off; instead
    word_order.process_treebank.drop_singleton_cols is applied here
    afterwards, with "head_{AUX_COUNT_FEAT}" protected -- see that
    function's docstring. Applied (and the result cached) before saving, so
    the cached parquet already reflects it.
    """
    agreement_feats = agreement_feats or ["Number", "Gender", "Person"]
    deprel = nsubj_aux_target.child_deprels[0]  # "nsubj"
    needed_cols = {f"head_{deprel}_{feat}_agreement" for feat in agreement_feats}

    cached_lang = gblang2udlang.get(lang, lang).replace(" ", "_")
    cache_path = os.path.join(save_to, f"{cached_lang}.parquet") if save_to else None
    if cache_path is not None and not never_skip and os.path.exists(cache_path):
        cached_df = read_df(lang, word_order_dir=save_to)
        # Matches Pipeline._process_language_impl's "not self.predictor_var
        # in df.columns" check: a cache built with a narrower agreement_feats
        # (or predating some other extraction change) is missing a column
        # this call needs -- fall through and re-extract rather than
        # silently handing back an incomplete dataframe that would KeyError
        # downstream.
        if needed_cols.issubset(cached_df.columns):
            return cached_df

    treebank = load_treebank(lang, resource_dir, max_treebank_len=max_treebank_len)
    redirect_nsubj_to_aux(treebank)

    df = create_word_order_df(
        treebank=treebank,
        lang=lang,
        target=nsubj_aux_target,
        resource_dir=resource_dir,
        save_to=None,
        agreement_feats=agreement_feats,
        lexicalize=lexicalize,
        drop_singleton_columns=False,
        um_data=um_data,
        ud_data=ud_data,
        fetch_all=fetch_all,
    )
    df = drop_singleton_cols(df, target=nsubj_aux_target, extra_always_keep={f"head_{AUX_COUNT_FEAT}"})

    if cache_path is not None and len(df) > 0:
        os.makedirs(save_to, exist_ok=True)
        df.to_parquet(cache_path, index=False)

    return df


class SubjAuxPipeline:
    """The subj_aux analogue of sva_trees.pipeline.Pipeline: same shape
    (configure in __init__, run everything via __call__), same per-language
    stages (extract, fit_dt, tree2html, create_pairs) and same post-loop
    stage (cross-language diagnostics table + HTML deprel index + overview
    index), reusing every one of those SVA/word_order functions completely
    unmodified -- just pointed at nsubj_aux_target (see redirect.py) and a
    treebank pre-processed by redirect_nsubj_to_aux instead of a plain
    load_treebank call.

    Differences from Pipeline, deliberate for now:
      - No `target`/`inflection_map`-agnostic genericity: this pipeline is
        inherently specific to nsubj-vs-aux, so `feature` (Number/Gender/
        Person) is the only per-script knob, plus `inflection_map` for the
        matching swap (each script sets both explicitly, the same way
        svNa.py/svGa.py/svPa.py pick their own).
      - `include_multi_aux` (single-aux-only vs single+stacked-aux
        combined, default True) is subj_aux-specific. Baked into
        the default target_id (a "_single" suffix when False) so switching
        the flag can't silently reuse a decision-tree/minimal-pairs cache
        fit under the other setting -- extraction's own cache is unaffected
        (it always holds every stream; the flag only selects downstream).
      - Per-language error isolation: a failing language is caught and
        logged ("FAILED: {lang}: {e}", plus the full traceback via
        traceback.print_exc() -- str(e) alone is often useless for
        pinpointing where a real bug lives, e.g. a pandas ValueError gives
        no line number) without aborting the rest of the sweep, for both
        n_jobs=1 and n_jobs>1 -- so one bad language doesn't take the
        diagnostics table/deprel index/overview index down with it for
        every other language that already succeeded. Pipeline's __call__
        now does the exact same thing (originally didn't -- a difference
        this docstring used to call out -- but re-raising the first worker
        exception meant one failing language blocked index generation for
        the whole run, including languages that had already finished; not
        worth the asymmetry).
    """

    def __init__(self, feature: str, inflection_map, langs, resource_dir: str,
                 word_order_dir: str, max_treebank_len: int | None = None,
                 never_skip: bool = False, target_id: str | None = None,
                 threshold: float = 0.12, simplify: bool = True, drop_unk: bool = True,
                 include_multi_aux: bool = True,
                 rm_columns=("nsubj_child-deprel_conj",),
                 unimorph_args: dict | None = None,
                 decision_trees_dir: str = OUTPUT_DECISION_TREES_DIR,
                 html_decision_trees_dir: str = HTML_DECISION_TREES_DIR,
                 minimal_pairs_dir: str = OUTPUT_MINIMAL_PAIRS_DIR,
                 diagnostics_dir: str = OUTPUT_DIAGNOSTICS_DIR,
                 n_jobs: int = 1, max_worker_mem_gb: float | None = None,
                 mem_headroom: float = 0.8, max_tasks_per_child: int = 1,
                 force: bool = False):
        self.feature = feature
        self.inflection_map = inflection_map
        self.target = dataclasses.replace(nsubj_aux_target, swap_feat=feature)
        self.deprel = nsubj_aux_target.child_deprels[0]  # "nsubj"
        self.predictor_var = f"head_{self.deprel}_{feature}_agreement"
        self.langs = langs
        self.resource_dir = resource_dir
        self.word_order_dir = word_order_dir
        self.max_treebank_len = max_treebank_len
        self.never_skip = never_skip
        self.include_multi_aux = include_multi_aux
        self.target_id = target_id or (
            f"sa{feature[0]}a" + ("" if include_multi_aux else "_single")
        )
        self.threshold = threshold
        self.simplify = simplify
        self.drop_unk = drop_unk
        self.rm_columns = rm_columns
        self.unimorph_args = unimorph_args or {
            "filter_entries": {"upos": ["V", "AUX"]},
            "combine_um_ud": True,
            "remove_multiword_forms": True,
        }
        self.decision_trees_dir = decision_trees_dir
        self.html_decision_trees_dir = html_decision_trees_dir
        self.minimal_pairs_dir = minimal_pairs_dir
        self.diagnostics_dir = diagnostics_dir
        self.n_jobs = n_jobs  # languages processed in parallel (1=serial)
        # Per-worker RLIMIT_AS cap, in GB. None (default) auto-derives from
        # currently-available system RAM / n_jobs, so raising n_jobs shrinks
        # each worker's ceiling and total usage across all workers stays
        # bounded -- same guarantee sva_trees.pipeline.Pipeline gives.
        self.max_worker_mem_gb = max_worker_mem_gb
        self.mem_headroom = mem_headroom
        # Recycle each worker after this many languages (see Pipeline's own
        # max_tasks_per_child for the full rationale: a fresh process per
        # language means the OS fully reclaims memory between languages,
        # rather than relying on pandas/NumPy to hand it back within a
        # long-lived worker).
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
                    f"Else, pass force=True to SubjAuxPipeline."
                )

        langs = sorted(self.langs)
        worker_mem_bytes = self._worker_mem_bytes()
        if worker_mem_bytes is not None:
            print(f"Capping each of {self.n_jobs} worker(s) to "
                  f"{worker_mem_bytes / 1024**3:.1f} GB RAM")
            initializer, initargs = limit_process_memory, (worker_mem_bytes,)
        else:
            print("Could not detect system RAM; running without a memory cap")
            initializer, initargs = None, ()

        # Always through ProcessPoolExecutor, even at n_jobs=1, for the same
        # reason Pipeline does: max_tasks_per_child's per-language process
        # recycling only applies via the pool, not a plain for-loop.
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

        self._generate_indexes()

    def _process_language(self, lang):
        print(lang)
        # Each language gets its own um_data POS-slices (get_um_lookup_table),
        # can be dropped as soon as lang is done -- mirrors
        # sva_trees.pipeline.Pipeline._process_language's cleanup wrapper.
        try:
            self._process_language_impl(lang)
        finally:
            clear_form_groups_cache()
            gc.collect()

    def _decision_trees_dir_full(self):
        return f"{self.decision_trees_dir}/{self.target_id}/{self.target_id}_{self.deprel}"

    def _pairs_dir(self, lang):
        return f"{self.minimal_pairs_dir}/{self.target_id}/{self.target_id}_{self.deprel}/{lang}"

    def _process_language_impl(self, lang: str):
        # Skip the whole language if its minimal pairs already exist --
        # matches Pipeline's top-of-language skip check.
        if not self.never_skip and os.path.isdir(self._pairs_dir(lang)):
            print(f"  skip, minimal pairs already found")
            return

        # Built once, up front, and reused both for extraction-time UM/UD
        # feature enrichment (fetch_all=True below) and for create_pairs'
        # reinflection later -- matches Pipeline's _process_language_impl,
        # which does the same for the analogous nsubj-vs-VERB case.
        inflector, skip_lang, num_lemma, num_form = load_inflector(
            lang=lang,
            langcode=lang2langcode(lang),
            unimorph_args=self.unimorph_args,
            inflection_map=self.inflection_map,
            resource_dir=self.resource_dir,
        )
        um_data = get_um_lookup_table(inflector)
        ud_data = get_ud_lookup_table(inflector)

        df = extract_subj_aux_df(lang, self.resource_dir, save_to=self.word_order_dir,
                                  max_treebank_len=self.max_treebank_len,
                                  never_skip=self.never_skip,
                                  um_data=um_data, ud_data=ud_data, fetch_all=True)

        aux_count_col =  f"head_{AUX_COUNT_FEAT}"
        if len(df) == 0 or not aux_count_col in df.columns:
            print(f"  no nsubj+aux instances found")
            return

        stream_df = (df[df[aux_count_col].isin(["1", "2+"])]
                     if self.include_multi_aux else df[df[aux_count_col] == "1"])
        full_df = stream_df[stream_df[self.predictor_var].notnull()].copy()

        for col in full_df:
            if any(col.startswith(y) for y in self.rm_columns):
                full_df = full_df[~full_df[col]]

        if self.simplify:
            full_df[self.predictor_var] = (
                full_df[self.predictor_var].astype(str).replace({"+-": "unk", "--": "unk"})
            )

        label_distribution = {
            str(k): int(v) for k, v in full_df[self.predictor_var].value_counts().items()
        }

        decision_trees_dir_full = self._decision_trees_dir_full()
        pred_values = set(full_df[self.predictor_var].values)

        model, dt_df, unk_counts, learn_dt = None, full_df, None, False

        joblib_path = f"{decision_trees_dir_full}/{lang}.joblib"
        if not self.never_skip and os.path.exists(joblib_path):
            dt_df = read_df(lang, word_order_dir=decision_trees_dir_full)
            model = joblib.load(joblib_path)
            unk_counts = read_unk_counts(decision_trees_dir_full, lang)
            learn_dt = True
        elif len(pred_values) > 1:
            min_impurity_decrease = get_impurity(len(full_df))
            model, dt_df, y_train, unk_counts = fit_dt(
                full_df=full_df,
                model_type="decision_tree",
                target=self.target,
                verbose=1,
                predictor_var=self.predictor_var,
                min_impurity_decrease=min_impurity_decrease,
                min_samples_leaf=10,
                save_to=f"{decision_trees_dir_full}/{lang}",
                omit_feats=set(
                    col for col in full_df
                    if col.endswith(f"_{self.feature}") and not col.startswith("swap_")
                ),
                drop_unk=self.drop_unk,
            )
            if model is not None:
                learn_dt = True
            else:
                dt_df = full_df

        # inflector/num_lemma/num_form already built at the top of this
        # method, for extraction-time enrichment -- reused here as-is. Run
        # before tree2html (not after, as before) so the v2 page's
        # generated-pairs section can join this same run's own
        # correct_swaps rows in by leaf_id -- create_pairs doesn't depend on
        # anything tree2html produces, so this reorder is safe.
        diagnostic_dfs = create_pairs(
            dt_df,
            swap_feat=self.predictor_var,
            inflector=inflector,
            target=self.target,
            leaf_threshold=self.threshold,
            save_to=self._pairs_dir(lang),
            num_lemma=num_lemma,
            num_form=num_form,
            full_df=full_df,
            unk_counts=unk_counts,
            label_distribution=label_distribution,
            head_label="Aux",
        )

        html_out = f"{self.html_decision_trees_dir}/{self.target_id}/{lang}.html"
        if self.never_skip or not os.path.exists(html_out):
            tree2html(
                pipeline_model=model,
                dt_df=dt_df,
                full_df=full_df,
                predictor_var=self.predictor_var,
                target=self.target,
                out_file=html_out,
                max_rows=15,
                meta={"Language": lang},
                only_show_real_orders=True,
                correlate_features=True,
                show_features=True,
                full_tree_html=learn_dt,
                palette_map=(
                    {"Yes": "#31cb9f", "No": "#f16393", "unk": "#b893de"}
                    if self.simplify else
                    {"Yes": "#31cb9f", "No": "#f16393", "+-": "#e5c64d", "--": "#b893de"}
                ),
                leaf_threshold=self.threshold,
                full_label_distribution=label_distribution,
                head_label="Aux",
                correct_swaps_df=diagnostic_dfs.get("correct_swaps"),
            )

    def _generate_indexes(self):
        pairs_dir = f"{self.minimal_pairs_dir}/{self.target_id}/{self.target_id}_{self.deprel}"

        print("Generating diagnostics table")
        diagnostics_df = generate_diagnostics_table(pairs_dir)
        write_diagnostics_csv(
            diagnostics_df,
            f"{self.diagnostics_dir}/{self.target_id}/{self.target_id}_{self.deprel}.csv",
        )

        diagnostics_by_lang = {
            row["Language"]: diagnostics_row_to_json(
                row, lang_dir=os.path.join(pairs_dir, row["Language"])
            )
            for _, row in diagnostics_df.iterrows()
        }

        print("Generating deprel index")
        generate_html_deprel_index(
            data_dir=self._decision_trees_dir_full(),
            html_directory=f"{self.html_decision_trees_dir}/{self.target_id}",
            target_col=self.predictor_var,
            exclude_labels={"unk"} if self.simplify else {"--", "+-"},
            include_trivial_labels={"Yes"},
            leaf_threshold=self.threshold,
            pairs_dir=pairs_dir,
            diagnostics_by_lang=diagnostics_by_lang,
            agreement_label="Subject-Auxiliary",
            head_role_label="Aux",
        )

        print("Generating overview index")
        generate_html_overview_index(html_directory=self.html_decision_trees_dir)
