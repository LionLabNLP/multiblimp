import os
import re
import sys
import random
import math
import gc
import joblib
import json
import traceback

from concurrent.futures import ProcessPoolExecutor, as_completed
from pandas import DataFrame

PYTHONIOENCODING="utf-8"

sys.path.append("../")

from word_order.process_treebank import (
    create_word_order_df, read_df, clear_form_groups_cache, load_treebank,
)
from word_order.decision_tree import fit_dt, FIT_EXCLUDED_FEATS, fit_excluded_for_display
from word_order.entropy import DEFAULT_LEAF_MIN_ACCURACY
from word_order.viz_tree import tree2html
from word_order.per_treebank import (
    fit_treebank_trees, render_treebank_pages, build_nav, split_excluded,
    CACHE_VERSION as PER_TREEBANK_CACHE_VERSION,
)
from word_order.viz_deprel import generate_html_deprel_index
from multiblimp.languages import (
    lang2langcode, gblang2udlang, treebank_row_exclusion_reason,
)
from multiblimp.unimorph import load_inflector
from multiblimp.condition_taxonomy import FEATURE_SUFFIXES
from multiblimp.config import (
    HTML_DECISION_TREES_DIR, OUTPUT_DECISION_TREES_DIR, OUTPUT_MINIMAL_PAIRS_DIR,
    OUTPUT_DIAGNOSTICS_DIR,
)
from multiblimp.agreement_pipeline_utils import (
    available_system_memory_bytes, find_other_running_instances,
    limit_process_memory, read_unk_counts, write_label_distribution,
)
from sva_trees.create_pairs import create_pairs
from sva_trees.second_chance import second_chance_decision, SecondChanceConfig
from sva_trees.diagnostics import generate_diagnostics_table, write_diagnostics_csv, diagnostics_row_to_json

random.seed(42)

# Prose labels for generate_html_deprel_index's "subject"/"nsubj" wording,
# keyed by the target's own child deprel -- without this, a report for
# obj_agr_target/iobj_agr_target (e.g. ovNa, iovNa) would say "subject"/
# "nsubj" throughout despite actually being object or indirect-object
# agreement. Falls back to the "nsubj" entry (both html_deprel.py params'
# own no-op defaults) for any deprel not listed here.
_DEPREL_ROLE_LABELS = {
    "nsubj": ("subject", "nsubj"),
    "obj": ("object", "obj"),
    "iobj": ("indirect object", "iobj"),
}
# Group label per target_id *prefix*, not deprel -- nsubj is the same
# dependency relation regardless of whether the subject agrees with a
# finite verb (sv) or a participle (sp), so the previous deprel-keyed
# lookup (both under "nsubj") collapsed sp's own pages into "Subject-Verb"
# too. Suffix (the feature -- Number/Gender/Person) comes from
# multiblimp.condition_taxonomy's shared FEATURE_SUFFIXES, appended
# separately (see the agreement_label assembly below) so a report actually
# says e.g. "Subject-Participle Number" instead of leaving every condition
# under one category indistinguishable from its siblings.
_TARGET_PREFIX_GROUP_LABELS = {
    "sv": "Subject-Verb",
    "sp": "Subject-Participle",
    "ov": "Object-Verb",
    "iov": "Indirect-Object-Verb",
    "sa": "Subject-Auxiliary",
    "op": "Object-Participle",
    "oa": "Object-Auxiliary",
    "iop": "Indirect-Object-Participle",
    "ioa": "Indirect-Object-Auxiliary",
}


def get_impurity(n, min_n=300, max_n=4000, max_val=0.1, min_val=0.01):
    if n <= min_n:
        return max_val
    if n >= max_n:
        return min_val

    t = (math.log(n) - math.log(min_n)) / (math.log(max_n) - math.log(min_n))

    return max_val - t * (max_val - min_val)


def get_um_lookup_table(inflector):
    return inflector.load_pos_lookup("unimorph/um_pickles")

def get_ud_lookup_table(inflector):
    """UD-derived UniMorph-schema data (expand_anno fallback for features
    um_data had no confident match for). Same shape as get_um_lookup_table's
    return value. None if `inflector` wasn't built with combine_um_ud=True
    (i.e. has no inflector.ud_inflector to source from).
    """
    if inflector.ud_inflector is None:
        return None
    return inflector.ud_inflector.load_pos_lookup("ud_unimorph/ud_pickles")


def refresh_deprel_index(target_id, deprel, predictor_var, head_label="Verb",
                         simplify=True, leaf_threshold=DEFAULT_LEAF_MIN_ACCURACY,
                         has_pairs=True):
    """Rebuild one SVA condition x deprel's diagnostics CSV + deprel
    index.html purely from the on-disk output/decision_trees +
    output/minimal_pairs artifacts -- the tail of Pipeline.__call__, extracted
    so it can run without any fitting (see scripts/overview/
    generate_html_indexes.py, which runs it in parallel across conditions;
    NPA's counterpart is npa.agreement.refresh_deprel_index).

    has_pairs=False (the data-debugging mode) skips the pairs-derived
    diagnostics and renders the index without the pairs columns/panel.
    """
    pairs_dir = os.path.join(OUTPUT_MINIMAL_PAIRS_DIR, target_id, f"{target_id}_{deprel}")
    decision_trees_dir = os.path.join(OUTPUT_DECISION_TREES_DIR, target_id, f"{target_id}_{deprel}")

    # Data-debugging mode: no pairs were ever created, so there's
    # nothing for the pairs-derived diagnostics table to summarize --
    # generate_html_deprel_index still runs below (pairs_dir=None,
    # diagnostics_by_lang=None), just without the pairs/diagnostics
    # columns and panel, so the accuracy-scatter overview keeps working.
    diagnostics_by_lang = None
    if has_pairs:
        print("Generating diagnostics table for", deprel)
        diagnostics_df = generate_diagnostics_table(pairs_dir)
        write_diagnostics_csv(
            diagnostics_df,
            os.path.join(OUTPUT_DIAGNOSTICS_DIR, target_id, f"{target_id}_{deprel}.csv"),
        )
        # Reshaped here (not inside word_order.viz_deprel) so that
        # module doesn't need to depend on sva_trees.
        diagnostics_by_lang = {
            row["Language"]: diagnostics_row_to_json(
                row, lang_dir=os.path.join(pairs_dir, row["Language"])
            )
            for _, row in diagnostics_df.iterrows()
        }

    subject_label, nsubj_label = _DEPREL_ROLE_LABELS.get(deprel, ("subject", deprel))
    # target_id is e.g. "svNa"/"spGa"/"ovPa" -- prefix (all but
    # the last 2 chars) selects the group, suffix (the last 2) the
    # feature, so this naturally covers every prefix length ("sv"/
    # "sp"/"ov" at 2 chars, "iov" at 3) without hardcoding either.
    condition_id = target_id.removesuffix("_single")
    group_label = _TARGET_PREFIX_GROUP_LABELS.get(condition_id[:-2], "Subject-Verb")
    feature_label = FEATURE_SUFFIXES.get(condition_id[-2:], "")
    agreement_label = f"{group_label} {feature_label}".strip()
    print("Generating deprel index for", deprel)
    generate_html_deprel_index(data_dir=decision_trees_dir,
                    html_directory=os.path.join(HTML_DECISION_TREES_DIR, target_id),
                    target_col=predictor_var,
                    exclude_labels={"unk"} if simplify else {"--", "+-"},
                    leaf_threshold=leaf_threshold,
                    pairs_dir=pairs_dir if has_pairs else None,
                    diagnostics_by_lang=diagnostics_by_lang,
                    head_role_label=head_label,
                    agreement_label=agreement_label,
                    subject_label=subject_label,
                    nsubj_label=nsubj_label,
                    debug_view=not has_pairs,
                    )


class Pipeline:
    # Wording of the agreeing head in reports/pages; subclasses override.
    head_label = "Verb"

    def __init__(self, target, predictor_var, langs, inflection_map, unimorph_args,
                 deprel_dir, resource_dir, word_order_dir,
                 max_treebank_len, never_skip=False, never_skip_fit=False, rm_columns=[],
                 target_id=False,
                 threshold=None, min_samples_leaf=10, simplify=False, n_jobs=1,
                 max_worker_mem_gb=None, mem_headroom=0.8, force=False,
                 agreement_feats=None, drop_unk=True, max_tasks_per_child=1,
                 second_chance=None, per_treebank=True, include_excluded=False,
                 generate_pairs=True, fetch_all=True, incl_unk=False,
                 detailed_unk=False, build_index=True, head_label=None):
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
        # Redo everything downstream of the cached extraction df (split
        # included/excluded, re-prepare, refit the decision tree, regenerate
        # pairs/HTML) without forcing a fresh extraction pass the way
        # never_skip does -- for a fix that only changes _split_excluded/
        # _prepare_full_df/fit_dt/create_pairs behavior (e.g. the exclusion-
        # key bug), extraction output is already correct and re-running it
        # is wasted work; this flag reuses it instead. _extract_df's own
        # cache check is intentionally NOT gated by this -- see that method.
        self.never_skip_fit = never_skip_fit
        self.rm_columns = rm_columns
        self.target_id = target_id if target_id else predictor_var.split("_")[2][0]
        self.min_samples_leaf = min_samples_leaf
        # None (default): DEFAULT_LEAF_MIN_ACCURACY (0.95). A pure leaf of
        # the default min_samples_leaf=10 has smoothed accuracy 0.955, so it
        # passes; a smaller min_samples_leaf could never reach 0.95. Pass an
        # explicit value to override.
        self.leaf_threshold = (
            threshold if threshold is not None
            else DEFAULT_LEAF_MIN_ACCURACY
        )
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
        # SecondChanceConfig to enable a laxer retry for languages whose
        # first pass yields too few pairs; None (default) disables it.
        self.second_chance = second_chance
        # Also fit/render one analysis-only tree per treebank (see
        # word_order.per_treebank) for languages with several treebanks.
        self.per_treebank = per_treebank
        # With per_treebank: also fit/render trees for the language's
        # excluded treebanks (multiblimp.languages.excluded_treebanks). Split
        # out of the same unified _extract_df cache at read time (see
        # _split_excluded) -- never reach the pooled tree or the minimal
        # pairs.
        self.include_excluded = include_excluded
        # If False, skip create_pairs entirely (no minimal pairs anywhere,
        # for any deprel/language) and the pairs-derived diagnostics/deprel
        # index built from them -- the data-debugging mode, which only wants
        # trees, not pairs.
        self.generate_pairs = generate_pairs
        # Passed through to create_word_order_df: if False, skip
        # process_treebank.expand_anno's UM/UD-derived annotation filling,
        # so extracted features (and anything fit/rendered from them) reflect
        # only the treebank's own original annotation.
        self.fetch_all = fetch_all
        # With per_treebank: also fit/render the incl.-unk tree (pooled and
        # per-treebank) -- see word_order.per_treebank's module docstring.
        self.incl_unk = incl_unk
        # incl_unk's own granularity: False (default) predicts a single
        # merged "unk" class, same label set the as-is tree drops. True
        # switches to _unk_split_config()'s finer Head/Subject/Both-unknown
        # breakdown -- reserved for the data-debugging mode, since acting on
        # *why* a row is unk is a debugging question most default-mode
        # viewers don't need.
        self.detailed_unk = detailed_unk
        # False: skip this run's own diagnostics table + deprel index (a
        # multi-condition sweep builds them once, in parallel, afterwards via
        # scripts/overview/generate_html_indexes.py).
        self.build_index = build_index
        if head_label:
            self.head_label = head_label

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
            other_pids = find_other_running_instances(sys.argv[0])
            if other_pids:
                raise RuntimeError(
                    f"Another instance of {script_name} appears to already be "
                    f"running (PID(s): {other_pids}). "
                    f"If they're stale, kill them first: kill {' '.join(map(str, other_pids))}"
                    f"Else, pass force=True to Pipeline."
                )

        langs = sorted(self.langs)

        # Filtered here, in the main process, rather than inside
        # _process_language_impl: that check is just a couple of os.path
        # calls, but running it there means it only fires after
        # ProcessPoolExecutor has already spawned a fresh worker for the
        # language (max_tasks_per_child recycles a worker per task -- see
        # below), which on macOS/spawn re-imports pandas/sklearn/etc. from
        # scratch. That import cost dwarfs the check itself, so an
        # already-done language should never pay it.
        langs_to_run = []
        for lang in langs:
            if self._is_done(lang):
                print(f"{lang}: Skip, minimal pairs already found")
            else:
                langs_to_run.append(lang)

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
                executor.submit(self._process_language, lang): lang for lang in langs_to_run
            }
            for future in as_completed(future_to_lang):
                lang = future_to_lang[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"  FAILED: {lang}: {e}")
                    traceback.print_exc()

        if self.build_index:
            for deprel in self.target.child_deprels:
                refresh_deprel_index(
                    self.target_id, deprel, self.predictor_var,
                    head_label=self.head_label, simplify=self.simplify,
                    leaf_threshold=self.leaf_threshold, has_pairs=self.generate_pairs,
                )
        # Cross-pipeline overview index (word_order.viz_overview.
        # generate_html_overview_index) is no longer rebuilt here -- it
        # rescans every condition's index.html recursively, so doing it once
        # per condition run is wasted work across a multi-condition sweep.
        # scripts/overview/generate_html_indexes.py now does this exactly once, after
        # all conditions are (re)built.

    def _is_done(self, lang):
        # True if the minimal pairs for all deprels already exist. Also
        # requires the tree HTML to exist -- pairs and HTML come from two
        # separate steps below (create_pairs vs. tree2html) that can fall out
        # of sync (e.g. a language whose HTML got written as a "too few
        # samples" placeholder despite a valid cached model existing, from a
        # since-fixed bug); without this, such a language would stay stuck on
        # its placeholder forever, since this check alone would keep skipping
        # it on every future run.
        return not self.never_skip and not self.never_skip_fit and all(
            (not self.generate_pairs or os.path.isdir(os.path.join(
                OUTPUT_MINIMAL_PAIRS_DIR, self.target_id, f"{self.target_id}_{deprel}", lang)))
            and os.path.exists(os.path.join(HTML_DECISION_TREES_DIR, self.target_id, f"{lang}.html"))
            and (not self.per_treebank or self._treebanks_done(
                os.path.join(OUTPUT_DECISION_TREES_DIR, self.target_id, f"{self.target_id}_{deprel}",
                             "treebanks", f"{lang}.json")))
            for deprel in self.target.child_deprels
            )

    def _treebanks_done(self, summary_fn):
        """Per-treebank summary (or the marker for a language without one)
        exists, was built with the current include_excluded/incl_unk/
        detailed_unk settings, AND under the current per_treebank.
        CACHE_VERSION -- without that last check, a version bump (e.g. adding
        unk_split) would never actually take effect for an already-"done"
        language: fit_treebank_trees' OWN version check never even gets a
        chance to run, since this top-level pre-filter (checked before
        ProcessPoolExecutor spawns anything, to skip re-importing pandas/
        sklearn for nothing) would keep marking the language done and skip it
        outright."""
        if not os.path.exists(summary_fn):
            return False
        with open(summary_fn) as f:
            cached = json.load(f)
        if not isinstance(cached, dict):
            return False
        return (cached.get("include_excluded", False) == self.include_excluded
                and cached.get("incl_unk", False) == self.incl_unk
                and cached.get("unk_split", False) == self.detailed_unk
                and cached.get("version") == PER_TREEBANK_CACHE_VERSION)

    def _process_language(self, lang):
        # Each language gets its own um_data POS-slices (sva_trees.pipeline.
        # get_um_lookup_table), can be dropped as soon as lang is done.
        try:
            self._process_language_impl(lang)
        finally:
            clear_form_groups_cache()
            gc.collect()

    def _unk_split_config(self):
        """word_order.per_treebank.fit_treebank_trees' unk_split config: an
        incl.-unk tree splits "unk" into Head/{Subject,Object,Indirect
        object} unknown (Both unknown when neither side has the swap
        feature annotated), rather than fitting/rendering it as one merged
        class -- same head_col/child_col NaN check the diagnostics table's
        head_unk/nsubj_unk/both_unk counts already use (see
        word_order.decision_tree.read_unk_counts), promoted to an actual
        class here."""
        feat, deprel = self.target.swap_feat, self.target.child_deprels[0]
        head_label = f"{self.head_label} unknown"
        dep_prose = _DEPREL_ROLE_LABELS.get(deprel, ("subject", deprel))[0]
        dep_label = f"{dep_prose[0].upper()}{dep_prose[1:]} unknown"
        grey = "#94a1b2"
        return {
            "col_a": f"head_{feat}", "label_a": head_label,
            "col_b": f"{deprel}_{feat}", "label_b": dep_label,
            "both_label": "Both unknown", "other_label": "unknown (other)",
            # "unknown (other)" itself (an ambiguous/multi-valued-overlap unk,
            # per decision_tree.split_unk_reasons -- expected-empty) is
            # deliberately left out of the palette: _finalize_tree_html forces
            # the legend's class list to exactly palette_map's keys, so a
            # permanent 0-count "unknown (other)" swatch would otherwise show
            # on literally every incl.-unk page. Still labeled that string if
            # it ever genuinely occurs (other_label above), just without its
            # own legend entry/color -- falls back to the default unk color.
            "palette": {
                "yes": "#31cb9f", "no": "#f16393",
                head_label: "#f6c453", dep_label: "#5aa9e6",
                "Both unknown": grey,
            },
        }

    def _swap_feat_cols(self, df):
        """Columns kept out of the fit: every plain or bracketed column of the
        swap feature (head_Number, head_Number[abs], sibling/child variants...),
        which is what the label is computed from, except the dependent's own
        (nsubj_Number / obj_Number / iobj_Number), which stays a predictor."""
        feat, deprel = self.target.swap_feat, self.target.child_deprels[0]
        pattern = re.compile(rf"_{feat}(\[[^\]]+\])?$")
        swap_cols = {col for col in df if pattern.search(col)
                     and not col.startswith("swap_")
                     and col != f"{deprel}_{feat}"}
        # plus the head token's xpos (see word_order.decision_tree.FIT_EXCLUDED_FEATS)
        return swap_cols | {col for col in FIT_EXCLUDED_FEATS if col in df}

    def _select_stream(self, df):
        """Which of the extracted instances feed the trees and pairs."""
        return df

    def _prepare_full_df(self, df):
        full_df = self._select_stream(df)
        full_df = full_df[full_df[self.predictor_var].notnull()]

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
        return full_df

    def _build_df(self, lang, save_to, inflector, treebank=None):
        """Extract the feature df of EVERY one of the language's treebanks --
        selected and excluded alike, tagged by the existing "treebank" column
        -- cached under `save_to` as one unified parquet. One treebank parse
        + one extract_node_features pass per language, not two: a prior
        version built the selected and excluded subsets as two completely
        separate cached parquets ({word_order_dir}/{lang}.parquet and
        {word_order_dir}/excluded/{lang}.parquet, the latter only on demand),
        which (a) doubled extraction cost whenever --include_excluded was
        used and (b) meant an excluded_treebanks (multiblimp.languages)
        change only ever took effect on whichever of the two parquets
        actually got --recache'd -- miss one and the two caches silently
        disagree about which treebank is excluded. Splitting back into
        included/excluded now happens at READ time instead (_split_excluded,
        _prepare_full_df/_excluded_df below), straight off the "treebank"
        column against the LIVE excluded_treebanks config -- an exclusion
        change takes effect on the next read, no re-extraction required.

        treebank: reuse an already-loaded Treebank instead of loading one
        here (scripts/sva_trees/vp_types.py shares one load_treebank() call
        across every non-aux family for the same language, since this
        method's own extraction never mutates it -- see that script's
        cache_language for why AuxPipeline, whose _build_df DOES mutate the
        treebank via redirect_to_aux, never shares one this way). None
        (default): load it here, same as always.
        """
        if treebank is None:
            treebank = load_treebank(
                lang, self.resource_dir, self.max_treebank_len, use_selected_treebanks=False
            )
        return create_word_order_df(
            lang=lang,
            treebank=treebank,
            target=self.target,
            resource_dir=self.resource_dir,
            save_to=save_to,
            drop_singleton_columns=True,
            predictor_var=self.predictor_var,
            agreement_feats=self.agreement_feats,
            lexicalize=True,
            um_data=get_um_lookup_table(inflector),
            ud_data=get_ud_lookup_table(inflector),
            fetch_all=self.fetch_all,
        )

    def empty_marker_path(self, lang):
        """Sentinel path for "already extracted, genuinely 0 matching
        instances" -- create_word_order_df never saves an empty df, so
        without it such a language is re-parsed on every run (see
        scripts/sva_trees/vp_types.py, which writes it in bulk)."""
        cached_lang = gblang2udlang.get(lang, lang).replace(" ", "_")
        return os.path.join(self.word_order_dir, f"{cached_lang}.empty.json")

    def _extract_df(self, lang, inflector):
        """Cached feature df of EVERY one of the language's treebanks (see
        _build_df) -- None if there is nothing to extract. A language with an
        empty-sentinel is answered None without re-parsing; --recache
        (never_skip) bypasses both the cache and the sentinel."""
        cache_dir = self.word_order_dir
        cached_lang = gblang2udlang.get(lang, lang).replace(" ", "_")
        marker = self.empty_marker_path(lang)
        if not self.never_skip:
            if os.path.exists(os.path.join(cache_dir, f"{cached_lang}.parquet")):
                df = read_df(lang, word_order_dir=cache_dir)
                if self.predictor_var in df.columns:
                    return df
            elif os.path.exists(marker):
                return None
        df = self._build_df(lang, cache_dir, inflector)
        clear_form_groups_cache()
        if df is None or not len(df):
            os.makedirs(cache_dir, exist_ok=True)
            with open(marker, "w") as f:
                json.dump({"predictor_var": self.predictor_var}, f)
        elif os.path.exists(marker):
            os.remove(marker)
        return df

    def _split_excluded(self, df):
        """(included_df, excluded_df) halves of a unified _extract_df result
        -- a read-time split, see _build_df's docstring and
        word_order.per_treebank.split_excluded."""
        return split_excluded(df)

    def _no_data_marker_paths(self, lang):
        """One {lang}.no_data.json path per child_deprel -- data/html are
        organized per-deprel throughout this file, and the deprel index page
        (generate_html_deprel_index) is generated per-deprel too, so a
        language skipped for having no raw instances needs a marker in each
        deprel's own decision_trees_dir to show up on every one of its
        pages."""
        return [
            os.path.join(
                OUTPUT_DECISION_TREES_DIR, self.target_id, f"{self.target_id}_{deprel}",
                f"{lang}.no_data.json",
            )
            for deprel in self.target.child_deprels
        ]

    def _clear_no_data_markers(self, lang):
        """Drop any stale "no data" marker from a previous run -- this run
        is about to redetermine lang's status from scratch, and a language
        that now has real data shouldn't keep showing up on the index page
        as a no-data skip."""
        for path in self._no_data_marker_paths(lang):
            if os.path.exists(path):
                os.remove(path)

    def _write_no_data_markers(self, lang, reason):
        """Record why `lang` produced zero rows, so generate_html_deprel_
        index can list it instead of silently omitting it (previously: a
        "Skipping {lang}, raw_df has no entries" console print only, with
        no trace anywhere in the generated HTML -- e.g. Beja on svNa)."""
        for path in self._no_data_marker_paths(lang):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump({"reason": reason}, f)

    def _describe_empty_reason(self, stage: str) -> str:
        """Best-effort, cheap explanation for a zero-row extraction --- read
        off this pipeline's own target/predictor_var filters, not re-derived
        from the treebank itself (that would mean re-parsing it just to
        explain a zero). Names what COULD be the cause, not a verified one:
        e.g. a head_feats requirement like VerbForm=Fin reads as the likely
        cause when the treebank simply never annotates VerbForm at all,
        which is exactly what a genuine mismatch would also look like."""
        target = self.target
        parts = []
        if target.head_pos:
            parts.append(f"head upos in {sorted(target.head_pos)}")
        if target.head_deprel:
            parts.append(f"head deprel == {target.head_deprel!r}")
        if target.child_pos:
            parts.append(f"child upos filter ({target.child_pos})")
        for feat, fn in (target.head_feats or {}).items():
            kw = getattr(fn, "keywords", {})
            if kw.get("require") is not None:
                parts.append(f"head {feat} starting with {kw['require']!r}")
            elif kw.get("exclude") is not None:
                parts.append(f"head {feat} NOT starting with {kw['exclude']!r}")
        filters = "; ".join(parts) if parts else "no structural filters beyond child_deprels"
        if stage == "raw":
            return (
                f"0 raw instances of child_deprels={target.child_deprels} ({filters}) "
                "-- likely this treebank has no attested instance matching every one "
                "of these filters (e.g. a required feature like VerbForm might simply "
                "never be annotated here at all, rather than genuinely disagreeing)."
            )
        return (
            f"raw instances existed ({filters}), but every row's {self.predictor_var} "
            "came back null after _prepare_full_df (or rm_columns dropped every row) "
            "-- the agreement column itself never got computed for any of them."
        )

    def _process_language_impl(self, lang):
        print(lang)

        cached_lang = gblang2udlang.get(lang, lang).replace(" ", "_")
        self._clear_no_data_markers(lang)

        inflector, skip_lang, num_lemma, num_form = load_inflector(
                lang=lang,
                langcode=lang2langcode(lang),
                unimorph_args=self.unimorph_args,
                inflection_map=self.inflection_map,
                resource_dir=self.resource_dir,)

        df = self._extract_df(lang, inflector)

        if df is None or not len(df):
            print(f"Skipping {lang}, raw_df has no entries")
            self._write_no_data_markers(lang, self._describe_empty_reason("raw"))
            return

        included_raw, excluded_raw = self._split_excluded(df)
        full_df = self._prepare_full_df(included_raw)
        # Computed once per language, reused for every deprel below (a prior
        # version re-fetched/re-prepared this per-deprel via a separate
        # _excluded_df(lang, inflector) call each iteration -- same excluded
        # rows every time, just re-read from a second cache file).
        extra_df_base = self._prepare_full_df(excluded_raw) if excluded_raw is not None else None
        if extra_df_base is not None and not len(extra_df_base):
            extra_df_base = None
        del df, included_raw, excluded_raw

        if not len(full_df):
            print(f"Skipping {lang}, no instances of the target")
            self._write_no_data_markers(lang, self._describe_empty_reason("full"))
            return

        min_impurity_decrease = get_impurity(len(full_df))
        # Raw label distribution across ALL of full_df (before fit_dt drops unk rows)
        # for the deprel overview's per-language distribution bar.
        label_distribution = { str(k): int(v) for k, v in full_df[self.predictor_var].value_counts().items()}

        swap_feat_cols = self._swap_feat_cols(full_df)

        for deprel in self.target.child_deprels:
            # Reset per-iteration: each deprel gets its own dt_df/model
            dt_df, model, learn_dt, unk_counts = None, None, False, None
            decision_trees_dir = os.path.join(OUTPUT_DECISION_TREES_DIR, self.target_id, f"{self.target_id}_{deprel}")
            lang_html_file = os.path.join(HTML_DECISION_TREES_DIR, self.target_id, f"{lang}.html")
            write_label_distribution(decision_trees_dir, lang, label_distribution)

            pred_values = set(full_df[self.predictor_var].values)

            if (self.never_skip or self.never_skip_fit
                    or not os.path.exists(f"{decision_trees_dir}/{lang}.joblib")):
                if len(pred_values)>1:
                    model, dt_df, predictor_df, unk_counts = fit_dt(
                        full_df=full_df,
                        model_type="decision_tree",
                        target=self.target,
                        verbose=1,
                        predictor_var=self.predictor_var,
                        min_impurity_decrease=min_impurity_decrease,
                        min_samples_leaf=self.min_samples_leaf,
                        save_to=f"{decision_trees_dir}/{lang}",
                        omit_feats=swap_feat_cols,
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
                learn_dt = model is not None

            # Run before tree2html (not after, as before) so the v2 page's
            # generated-pairs section can join this same run's own
            # correct_swaps rows in by leaf_id -- create_pairs doesn't depend
            # on anything tree2html produces, so this reorder is safe.
            correct_swaps_df = None
            leaf_threshold = self.leaf_threshold
            if self.generate_pairs:
                pairs_kwargs = dict(
                    swap_feat=self.predictor_var, inflector=inflector,
                    target=self.target,
                    save_to=os.path.join(OUTPUT_MINIMAL_PAIRS_DIR, self.target_id, f"{self.target_id}_{deprel}", lang),
                    num_lemma=num_lemma,
                    num_form=num_form,
                    full_df=full_df,
                    unk_counts=unk_counts,
                    label_distribution=label_distribution,
                    head_label=self.head_label)
                try:
                    diagnostic_dfs = create_pairs(dt_df, leaf_threshold=leaf_threshold, **pairs_kwargs)
                    if self.second_chance is not None:
                        retry, lax, info = second_chance_decision(
                            dt_df, self.predictor_var, label_distribution,
                            strict_threshold=leaf_threshold,
                            min_samples_leaf=self.min_samples_leaf, cfg=self.second_chance)
                        print(f"{lang} {deprel}: second chance: {info['reason']}")
                        if retry:
                            # Adopted unconditionally, even with zero correct_swaps:
                            # the strict pass's own n_keep/diagnostics undercount
                            # what the tree actually supports once a retry was
                            # attempted, so the lax pass's numbers -- keep count,
                            # bucket breakdown (e.g. no_candidates), leaf_threshold
                            # -- are the ones worth keeping on record even when
                            # reinflection itself came up empty.
                            second_dfs = create_pairs(
                                dt_df, leaf_threshold=lax,
                                extra_meta={"second_pass": True, "second_chance_info": info},
                                second_chance_threshold=leaf_threshold,
                                **pairs_kwargs)
                            leaf_threshold, diagnostic_dfs = lax, second_dfs
                            if not len(second_dfs["correct_swaps"]):
                                print(f"{lang} {deprel}: retry attempted, yielded no pairs")
                    correct_swaps_df = diagnostic_dfs.get("correct_swaps")
                except KeyError:
                    print(f"Skipping {lang} for {deprel}, missing column")

            tree_html_kwargs = dict(
                    predictor_var=self.predictor_var,
                    target=self.target,
                    max_rows=15,
                    only_show_real_orders=True,
                    correlate_features=True,
                    show_features=True,
                    full_tree_html=learn_dt,
                    palette_map = (
                        {"yes": "#31cb9f", "no": "#f16393", "unk": "#b893de"}
                        if self.simplify else
                        {"yes": "#31cb9f", "no": "#f16393",
                        "+-": "#e5c64d", "--": "#b893de"}
                    ),
                    leaf_threshold=leaf_threshold,
                    # Always the pipeline's own strict bar, regardless of
                    # whether second_chance retried -- lets the v2 page tag
                    # leaves that only cleared the (possibly lax) leaf_threshold
                    # above because of the retry.
                    strict_leaf_threshold=self.leaf_threshold,
                    full_label_distribution=label_distribution,
                    head_label=self.head_label,
                    fit_excluded=fit_excluded_for_display(swap_feat_cols),
            )

            treebank_nav, treebanks_fresh = None, False
            extra_df = (extra_df_base
                        if self.per_treebank and self.include_excluded and learn_dt else None)
            if (self.per_treebank and learn_dt and "treebank" in full_df.columns
                    and (full_df["treebank"].nunique() > 1 or extra_df is not None)):
                tb_cache_dir = os.path.join(decision_trees_dir, "treebanks")
                unk_split = self._unk_split_config() if self.detailed_unk else None
                tb_summary, treebanks_fresh = fit_treebank_trees(
                    full_df, lang, model, self.predictor_var, self.target, tb_cache_dir,
                    fit_kwargs=dict(min_samples_leaf=self.min_samples_leaf),
                    omit_feats_fn=self._swap_feat_cols,
                    impurity_fn=get_impurity, pooled_impurity=min_impurity_decrease,
                    drop_unk=self.drop_unk, never_skip=self.never_skip or self.never_skip_fit,
                    extra_df=extra_df, include_excluded=self.include_excluded,
                    reason_fn=treebank_row_exclusion_reason,
                    unk_split=unk_split, incl_unk=self.incl_unk)
                tb_html_dir = os.path.join(HTML_DECISION_TREES_DIR, self.target_id)
                if (treebanks_fresh or self.never_skip or self.never_skip_fit
                        or not os.path.isdir(os.path.join(tb_html_dir, lang))):
                    render_treebank_pages(tb_summary, lang, full_df, tb_cache_dir, tb_html_dir,
                                          tree_html_kwargs, extra_df=extra_df,
                                          unk_split=unk_split)
                treebank_nav = build_nav(tb_summary, lang)
            elif self.per_treebank:
                # marks the language as handled (see _is_done)
                os.makedirs(os.path.join(decision_trees_dir, "treebanks"), exist_ok=True)
                with open(os.path.join(decision_trees_dir, "treebanks", f"{lang}.json"), "w") as f:
                    json.dump({"version": PER_TREEBANK_CACHE_VERSION,
                               "include_excluded": self.include_excluded,
                               "incl_unk": self.incl_unk, "unk_split": self.detailed_unk,
                               "treebanks": []}, f)

            if (self.never_skip or self.never_skip_fit or treebanks_fresh
                    or not os.path.exists(lang_html_file)):
                tree2html(
                    pipeline_model=model,
                    dt_df=dt_df,
                    full_df=full_df,
                    out_file=lang_html_file,
                    meta={"Language": lang},
                    correct_swaps_df=correct_swaps_df,
                    treebank_nav=treebank_nav,
                    **tree_html_kwargs,
                )