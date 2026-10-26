import os
import re
import sys
import math
import functools
import gc
import joblib
import pyarrow.parquet as pq
import json

from agreement_trees.extract.process_treebank import (
    create_word_order_df, read_df, load_treebank,
)
from agreement_trees.fit.decision_tree import fit_dt, FIT_EXCLUDED_FEATS, fit_excluded_for_display
from agreement_trees.fit.entropy import DEFAULT_LEAF_MIN_ACCURACY
from agreement_trees.report.viz_tree import tree2html
from agreement_trees.extract.per_treebank import (
    fit_treebank_trees, render_treebank_pages, build_nav, split_excluded, restrict_treebanks,
    treebanks_done, write_treebanks_marker, pooled_page_stem,
)
from agreement_trees.report.viz_deprel import generate_html_deprel_index
from multiblimp.excluded_treebanks import treebank_row_exclusion_reason
from multiblimp.languages import lang2langcode, gblang2udlang
from multiblimp.ud_inflector import load_ud_inflector, nominal_inflector
from multiblimp.ud_table import ud_lookup_table, um_lookup_table
from agreement_trees.pipeline_utils import parquet_current, sentinel_current, write_sentinel
from agreement_trees.config import lexical_features
from agreement_trees.taxonomy import FEATURE_SUFFIXES, OTHER_VERB_FORMS
from multiblimp.config import (
    HTML_DECISION_TREES_DIR, OUTPUT_DECISION_TREES_DIR, OUTPUT_MINIMAL_PAIRS_DIR,
    OUTPUT_DIAGNOSTICS_DIR,
)
from agreement_trees.pipeline_utils import (
    find_other_running_instances, read_unk_counts, run_in_pool,
    worker_mem_bytes, write_label_distribution,
)
from agreement_trees.pairs.create_pairs import create_pairs, MissingColumnError, pairs_current, read_cached_pairs
from agreement_trees.pairs.second_chance import second_chance_decision
from agreement_trees.pairs.diagnostics import generate_diagnostics_table, write_diagnostics_csv, diagnostics_row_to_json

# Prose labels for generate_html_deprel_index's subject/nsubj wording, keyed by child deprel.
_DEPREL_ROLE_LABELS = {
    "nsubj": ("subject", "nsubj"),
    "obj": ("object", "obj"),
    "iobj": ("indirect object", "iobj"),
}
# Group label per target_id prefix; the feature suffix is appended separately.
_TARGET_PREFIX_GROUP_LABELS = {
    "sv": "Subject-Verb",
    "sp": "Subject-Participle",
    **{prefix: f"Subject-{label}" for prefix, (_, label) in OTHER_VERB_FORMS.items()},
    "ov": "Object-Verb",
    "iov": "Indirect-Object-Verb",
    "sa": "Subject-Auxiliary",
    "op": "Object-Participle",
    "oa": "Object-Auxiliary",
    "iop": "Indirect-Object-Participle",
    "ioa": "Indirect-Object-Auxiliary",
}


_CONDITION_ID_RE = re.compile(r"^(iov|iop|ioa|sv|sp|sx[a-z]+|sa|ov|op|oa)([A-Z][A-Za-z]*)")


def _split_condition_id(condition_id):
    """("sv", "Na") for "svNa"; ("sv", "Form") for the ad-hoc "svForm"."""
    match = _CONDITION_ID_RE.match(condition_id)
    return match.groups() if match else (condition_id[:-2], condition_id[-2:])


def get_impurity(n, min_n=300, max_n=4000, max_val=0.1, min_val=0.01):
    if n <= min_n:
        return max_val
    if n >= max_n:
        return min_val

    t = (math.log(n) - math.log(min_n)) / (math.log(max_n) - math.log(min_n))

    return max_val - t * (max_val - min_val)


def refresh_deprel_index(target_id, deprel, predictor_var, head_label="Verb",
                         leaf_threshold=DEFAULT_LEAF_MIN_ACCURACY):
    """Rebuild one condition x deprel's diagnostics CSV and deprel index.html from the
    on-disk output (no fitting); NPA's counterpart is agreement_trees.npa.agreement.refresh_deprel_index.
    """
    pairs_dir = os.path.join(OUTPUT_MINIMAL_PAIRS_DIR, target_id, f"{target_id}_{deprel}")
    decision_trees_dir = os.path.join(OUTPUT_DECISION_TREES_DIR, target_id, f"{target_id}_{deprel}")

    print("Generating diagnostics table for", deprel)
    diagnostics_df = generate_diagnostics_table(pairs_dir)
    write_diagnostics_csv(
        diagnostics_df,
        os.path.join(OUTPUT_DIAGNOSTICS_DIR, target_id, f"{target_id}_{deprel}.csv"),
    )
    # reshaped here so agreement_trees.report.viz_deprel needn't depend on sva_trees
    diagnostics_by_lang = {
        row["Language"]: diagnostics_row_to_json(
            row, lang_dir=os.path.join(pairs_dir, row["Language"])
        )
        for _, row in diagnostics_df.iterrows()
    }

    subject_label, nsubj_label = _DEPREL_ROLE_LABELS.get(deprel, ("subject", deprel))
    condition_id = target_id.removesuffix("_single")
    prefix, suffix = _split_condition_id(condition_id)
    group_label = _TARGET_PREFIX_GROUP_LABELS.get(prefix, "Subject-Verb")
    feature_label = FEATURE_SUFFIXES.get(suffix) or re.sub(r"(?<=[a-z])(?=[A-Z])", " ", suffix)
    agreement_label = f"{group_label} {feature_label}".strip()
    print("Generating deprel index for", deprel)
    generate_html_deprel_index(data_dir=decision_trees_dir,
                    html_directory=os.path.join(HTML_DECISION_TREES_DIR, target_id),
                    target_col=predictor_var,
                    exclude_labels={"unk"},
                    leaf_threshold=leaf_threshold,
                    pairs_dir=pairs_dir,
                    diagnostics_by_lang=diagnostics_by_lang,
                    head_role_label=head_label,
                    agreement_label=agreement_label,
                    subject_label=subject_label,
                    nsubj_label=nsubj_label,
                    )


CONTEXT_UPOS = ["NOUN", "PROPN", "PRON"]   # the child_pos of every VP target


class Pipeline:
    # Wording of the agreeing head in reports/pages; subclasses override.
    head_label = "Verb"

    def __init__(self, target, predictor_var, langs, inflection_map, unimorph_args,
                 deprel_dir, resource_dir, word_order_dir,
                 max_treebank_len, never_skip=False, never_skip_fit=False, rm_columns=[],
                 target_id=False,
                 threshold=None, min_samples_leaf=10, n_jobs=1,
                 max_worker_mem_gb=None, mem_headroom=0.8, force=False,
                 agreement_feats=None, drop_unk=True, max_tasks_per_child=1,
                 second_chance=None, per_treebank=True, include_excluded=False,
                 fetch_all=True, incl_unk=False,
                 build_index=True, head_label=None,
                 max_depth=12, test_size=0.1, impurity=None, only_treebanks=(), redo_pairs=False):
        self.only_treebanks = list(only_treebanks)
        self.redo_pairs = redo_pairs
        self.target = target
        self.predictor_var = predictor_var
        # all agreement features are extracted so one cache serves every target
        self.agreement_feats = (
            agreement_feats if agreement_feats is not None
            else "all"
        )
        self.langs = langs
        self.inflection_map = inflection_map
        self.unimorph_args = unimorph_args
        self.deprel_dir = deprel_dir
        self.resource_dir = resource_dir
        self.word_order_dir = word_order_dir
        self.max_treebank_len = max_treebank_len
        self.never_skip = never_skip
        # redo everything downstream of the cached extraction df (fit, pairs, HTML)
        self.never_skip_fit = never_skip_fit
        self.rm_columns = rm_columns
        self.target_id = target_id if target_id else predictor_var.split("_")[2][0]
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.test_size = test_size
        self.impurity_fn = functools.partial(get_impurity, **(impurity or {}))
        # defaults: 95%, and min 10 samples per leaf
        self.leaf_threshold = (
            threshold if threshold is not None
            else DEFAULT_LEAF_MIN_ACCURACY
        )
        # False keeps unk-labeled rows in the fit
        self.drop_unk = drop_unk
        self.n_jobs = n_jobs # parallelise language computation across this many processes; 1=serial, >1=parallel
        # per-worker memory cap in GB; None derives it
        self.max_worker_mem_gb = max_worker_mem_gb
        self.mem_headroom = mem_headroom
        # languages per worker process; 1 = fresh process each, which returns memory to the OS
        self.max_tasks_per_child = max_tasks_per_child
        # skip the check for other running instances of this script
        self.force = force
        # SecondChanceConfig: laxer retry for languages with too few pairs; None disables
        self.second_chance = second_chance
        # also fit one analysis-only tree per treebank
        self.per_treebank = per_treebank
        # with per_treebank: also trees for excluded treebanks (never reach the pooled tree or pairs)
        self.include_excluded = include_excluded
        # False skips expand_anno's annotation filling
        self.fetch_all = fetch_all
        # with per_treebank: also the incl.-unk tree (see agreement_trees.extract.per_treebank)
        self.incl_unk = incl_unk
        # False: leave the diagnostics table and deprel index to scripts/overview/generate_html_indexes.py
        self.build_index = build_index
        if head_label:
            self.head_label = head_label

    def _worker_mem_bytes(self):
        return worker_mem_bytes(self.n_jobs, self.max_worker_mem_gb, self.mem_headroom)

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

        # filtered here so done languages don't pay for a fresh worker process
        langs_to_run = []
        for lang in langs:
            if self._is_done(lang):
                print(f"{lang}: Skip, minimal pairs already found")
            else:
                langs_to_run.append(lang)

        mem_bytes = self._worker_mem_bytes()
        if mem_bytes is None:
            print("Could not detect system RAM; running without a memory cap")
        else:
            print(f"Capping each of {self.n_jobs} worker(s) to {mem_bytes / 1024**3:.1f} GB RAM")

        # always through worker processes, even at n_jobs=1; a failing language is logged
        def report_failure(lang, _result, error):
            if error is not None:
                print(f"  FAILED: {lang}: {error}")

        run_in_pool(self._process_language, {lang: (lang,) for lang in langs_to_run},
                    self.n_jobs, mem_bytes, self.max_tasks_per_child, on_done=report_failure)

        if self.build_index:
            for deprel in self.target.child_deprels:
                refresh_deprel_index(
                    self.target_id, deprel, self.predictor_var,
                    head_label=self.head_label,
                    leaf_threshold=self.leaf_threshold,
                )

    def _is_done(self, lang):
        # done = current pairs, tree HTML and per-treebank summaries exist for every deprel
        return not self.never_skip and not self.never_skip_fit and not self.redo_pairs and all(
            pairs_current(os.path.join(
                OUTPUT_MINIMAL_PAIRS_DIR, self.target_id, f"{self.target_id}_{deprel}", lang))
            and os.path.exists(os.path.join(HTML_DECISION_TREES_DIR, self.target_id, f"{lang}.html"))
            and (not self.per_treebank or self._treebanks_done(
                os.path.join(OUTPUT_DECISION_TREES_DIR, self.target_id, f"{self.target_id}_{deprel}",
                             "treebanks", f"{lang}.json")))
            for deprel in self.target.child_deprels
            )

    def _treebanks_done(self, summary_fn):
        return treebanks_done(summary_fn, self.include_excluded, self.incl_unk)

    def _process_language(self, lang):
        # per-language lookup slices are dropped once the language is done
        try:
            self._process_language_impl(lang)
        finally:
            gc.collect()

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
        # plus the head token's xpos (see agreement_trees.fit.decision_tree.FIT_EXCLUDED_FEATS)
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

        # collapse +-/-- to "unk" for the decision tree
        full_df[self.predictor_var] = (
            full_df[self.predictor_var]
            .astype(str)
            .replace({"+-": "unk", "--": "unk"})
        )
        return full_df

    def _build_df(self, lang, save_to, inflector, treebank=None):
        """Extract the feature df of every treebank of the language, selected and excluded
        alike (tagged by the "treebank" column), cached under `save_to` as one parquet.
        Included/excluded are split at read time (_split_excluded).

        treebank: reuse an already-loaded Treebank (scripts/agreement_trees/
        cache_treebank_features.py shares one across families; AuxPipeline mutates the
        treebank, so it never does).
        """
        if treebank is None:
            treebank = load_treebank(
                lang, self.resource_dir, self.max_treebank_len, use_selected_treebanks=False
            )
        treebank, target, extra_always_keep = self._extraction_setup(treebank)
        return create_word_order_df(
            lang=lang,
            treebank=treebank,
            target=target,
            resource_dir=self.resource_dir,
            save_to=save_to,
            drop_singleton_columns=True,
            predictor_var=self.predictor_var,
            agreement_feats=self.agreement_feats,
            lexicalize=True,
            um_data=um_lookup_table(inflector) if inflector is not None else None,
            ud_data=ud_lookup_table(inflector) if inflector is not None else None,
            fetch_all=self.fetch_all,
            extra_always_keep=extra_always_keep,
        )

    def _extraction_setup(self, treebank):
        """(treebank, extraction target, columns to keep when dropping singleton columns); AuxPipeline
        re-points the treebank at the auxiliaries and extracts through its own target."""
        return treebank, self.target, ()

    def _cache_stem(self, lang):
        return gblang2udlang.get(lang, lang).replace(" ", "_")

    def cache_path(self, lang):
        return os.path.join(self.word_order_dir, f"{self._cache_stem(lang)}.parquet")

    def empty_marker_path(self, lang):
        """Sentinel for "extracted, zero matching instances": no empty parquet
        is ever written, so without it such a language is re-parsed every run."""
        return os.path.join(self.word_order_dir, f"{self._cache_stem(lang)}.empty.json")

    def _columns_ok(self, names):
        """A cache is valid if it has this pipeline's agreement column and, for every
        feature on both head and child, that feature's agreement column. Lexical features
        (scan.exclude_features) never get one, so they are not checked."""
        names = set(names)
        if self.predictor_var not in names:
            return False
        lexical = lexical_features()
        for deprel in self.target.child_deprels:
            for col in names:
                feat = col[len("head_"):]
                if (col.startswith("head_") and re.fullmatch(r"[A-Z][a-zA-Z]*", feat)
                        and feat not in lexical
                        and f"{deprel}_{feat}" in names
                        and f"head_{deprel}_{feat}_agreement" not in names):
                    return False
        return True

    def cached_row_count(self, lang):
        """Row count of a valid cache for `lang` (0 via the empty sentinel), else None,
        read off parquet metadata only. never_skip ignores both."""
        if self.never_skip:
            return None
        if os.path.exists(self.cache_path(lang)):
            pf = pq.ParquetFile(self.cache_path(lang))
            if self._columns_ok(pf.schema_arrow.names) and parquet_current(self.cache_path(lang)):
                return pf.metadata.num_rows
            return None
        if sentinel_current(self.empty_marker_path(lang)):
            return 0
        return None

    def record_extraction(self, lang, df):
        """Write the empty sentinel if `df` is empty, else clear a stale one."""
        marker = self.empty_marker_path(lang)
        if df is None or not len(df):
            os.makedirs(self.word_order_dir, exist_ok=True)
            write_sentinel(marker, {"predictor_var": self.predictor_var})
        elif os.path.exists(marker):
            os.remove(marker)

    def _extract_df(self, lang, get_inflector):
        """Cached feature df of EVERY one of the language's treebanks (see
        _build_df), or None if there is nothing to extract."""
        if not self.never_skip:
            if os.path.exists(self.cache_path(lang)):
                df = read_df(lang, word_order_dir=self.word_order_dir)
                if self._columns_ok(df.columns) and parquet_current(self.cache_path(lang)):
                    return df
            elif sentinel_current(self.empty_marker_path(lang)):
                return None
        df = self._build_df(lang, self.word_order_dir, get_inflector()[0])
        self.record_extraction(lang, df)
        return df

    def _split_excluded(self, df):
        """(included_df, excluded_df) halves of a unified _extract_df result."""
        included, excluded = split_excluded(df)
        if self.only_treebanks:
            included, excluded = restrict_treebanks(included, self.only_treebanks), None
        return included, excluded

    def _no_data_marker_paths(self, lang):
        """One {lang}.no_data.json path per child_deprel."""
        return [
            os.path.join(
                OUTPUT_DECISION_TREES_DIR, self.target_id, f"{self.target_id}_{deprel}",
                f"{lang}.no_data.json",
            )
            for deprel in self.target.child_deprels
        ]

    def _clear_no_data_markers(self, lang):
        """Drop any stale "no data" marker from a previous run."""
        for path in self._no_data_marker_paths(lang):
            if os.path.exists(path):
                os.remove(path)

    def _write_no_data_markers(self, lang, reason):
        """Record why `lang` produced zero rows, so the deprel index can list it."""
        for path in self._no_data_marker_paths(lang):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump({"reason": reason}, f)

    def _describe_empty_reason(self, stage: str) -> str:
        """Best-effort explanation for a zero-row extraction, read off this pipeline's
        filters (a likely cause, not a verified one)."""
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

        self._clear_no_data_markers(lang)

        loaded = []

        def get_inflector():
            """(inflector, lemmas, forms), loaded on first use: only extraction and building pairs need it."""
            if not loaded:
                loaded.append(load_ud_inflector(
                    lang=lang,
                    langcode=lang2langcode(lang),
                    unimorph_args=self.unimorph_args,
                    inflection_map=self.inflection_map,
                    resource_dir=self.resource_dir,
                    also_upos=CONTEXT_UPOS))
            return loaded[0]

        df = self._extract_df(lang, get_inflector)

        if df is None or not len(df):
            print(f"Skipping {lang}, raw_df has no entries")
            self._write_no_data_markers(lang, self._describe_empty_reason("raw"))
            return

        included_raw, excluded_raw = self._split_excluded(df)
        full_df = self._prepare_full_df(included_raw)
        # computed once per language, reused for every deprel
        extra_df_base = self._prepare_full_df(excluded_raw) if excluded_raw is not None else None
        if extra_df_base is not None and not len(extra_df_base):
            extra_df_base = None
        del df, included_raw, excluded_raw

        if not len(full_df):
            print(f"Skipping {lang}, no instances of the target")
            self._write_no_data_markers(lang, self._describe_empty_reason("full"))
            return

        min_impurity_decrease = self.impurity_fn(len(full_df))
        # label distribution over all of full_df (before fit_dt drops unk rows)
        label_distribution = {str(k): int(v) for k, v in full_df[self.predictor_var].value_counts().items()}

        swap_feat_cols = self._swap_feat_cols(full_df)

        for deprel in self.target.child_deprels:
            # Reset per-iteration: each deprel gets its own dt_df/model
            dt_df, model, learn_dt, unk_counts = None, None, False, None
            decision_trees_dir = os.path.join(OUTPUT_DECISION_TREES_DIR, self.target_id, f"{self.target_id}_{deprel}")
            lang_html_file = os.path.join(HTML_DECISION_TREES_DIR, self.target_id, f"{lang}.html")
            write_label_distribution(decision_trees_dir, lang, label_distribution)

            pred_values = set(full_df[self.predictor_var].values)

            fit_rebuilt = (self.never_skip or self.never_skip_fit
                           or not os.path.exists(f"{decision_trees_dir}/{lang}.joblib"))
            if fit_rebuilt:
                if len(pred_values)>1:
                    model, dt_df, _predictor_df, unk_counts = fit_dt(
                        full_df=full_df,
                        model_type="decision_tree",
                        target=self.target,
                        verbose=1,
                        predictor_var=self.predictor_var,
                        min_impurity_decrease=min_impurity_decrease,
                        min_samples_leaf=self.min_samples_leaf,
                        max_depth=self.max_depth, test_size=self.test_size,
                        save_to=f"{decision_trees_dir}/{lang}",
                        omit_feats=swap_feat_cols,
                        drop_unk=self.drop_unk,
                        )
                    if model:
                        learn_dt=True
                if len(pred_values)<=1 or learn_dt==False: # no decision to learn
                    for stale in (f"{decision_trees_dir}/{lang}.joblib", f"{decision_trees_dir}/{lang}.parquet"):
                        if os.path.exists(stale):
                            os.remove(stale)
                    model = None
                    dt_df = full_df
                    learn_dt = False
            else:
                dt_df = read_df(lang, word_order_dir=decision_trees_dir)
                model = joblib.load(f"{decision_trees_dir}/{lang}.joblib")
                unk_counts = read_unk_counts(decision_trees_dir, lang)
                learn_dt = model is not None

            # before tree2html so the page can join this run's correct_swaps rows by leaf_id
            correct_swaps_df = None
            leaf_threshold = self.leaf_threshold
            pairs_lang_dir = os.path.join(OUTPUT_MINIMAL_PAIRS_DIR, self.target_id, f"{self.target_id}_{deprel}", lang)
            if (fit_rebuilt or self.never_skip or self.never_skip_fit or self.redo_pairs
                    or not pairs_current(pairs_lang_dir)):
                inflector, num_lemma, num_form = get_inflector()
                pairs_kwargs = dict(
                    swap_feat=self.predictor_var, inflector=inflector,
                    target=self.target,
                    save_to=pairs_lang_dir,
                    num_lemma=num_lemma,
                    num_form=num_form,
                    full_df=full_df,
                    unk_counts=unk_counts,
                    label_distribution=label_distribution,
                    head_label=self.head_label,
                    # which values of the swapped feature the argument's own form can carry (ambiguous_subjects)
                    context_inflector=nominal_inflector(inflector, CONTEXT_UPOS, (self.target.swap_feat, None)))
                try:
                    diagnostic_dfs = create_pairs(dt_df, leaf_threshold=leaf_threshold, **pairs_kwargs)
                    if self.second_chance is not None:
                        retry, lax, info = second_chance_decision(
                            dt_df, self.predictor_var, label_distribution,
                            strict_threshold=leaf_threshold,
                            min_samples_leaf=self.min_samples_leaf, cfg=self.second_chance)
                        print(f"{lang} {deprel}: second chance: {info['reason']}")
                        if retry:
                            # adopted even with zero correct_swaps: the lax pass's numbers are the ones to record
                            second_dfs = create_pairs(
                                dt_df, leaf_threshold=lax,
                                extra_meta={"second_pass": True, "second_chance_info": info},
                                second_chance_threshold=leaf_threshold,
                                **pairs_kwargs)
                            leaf_threshold, diagnostic_dfs = lax, second_dfs
                            if not len(second_dfs["correct_swaps"]):
                                print(f"{lang} {deprel}: retry attempted, yielded no pairs")
                    correct_swaps_df = diagnostic_dfs.get("correct_swaps")
                except MissingColumnError as err:
                    print(f"Skipping {lang} for {deprel}, missing column {err}")
            else:
                # pairs of an earlier run: the page reads them back, with the cutoffs they were built with
                correct_swaps_df, cached_threshold = read_cached_pairs(pairs_lang_dir)
                if cached_threshold is not None:
                    leaf_threshold = cached_threshold

            tree_html_kwargs = dict(
                    discounted_classes=[c for c in label_distribution if c not in ("yes", "no")] if self.drop_unk else None,
                    predictor_var=self.predictor_var,
                    target=self.target,
                    max_rows=15,
                    correlate_features=True,
                    show_features=True,
                    full_tree_html=learn_dt,
                    palette_map={"yes": "#31cb9f", "no": "#f16393", "unk": "#b893de"},
                    leaf_threshold=leaf_threshold,
                    # the strict bar even after a second-chance retry, to tag leaves that only cleared the lax one
                    strict_leaf_threshold=self.leaf_threshold,
                    full_label_distribution=label_distribution,
                    head_label=self.head_label,
                    fit_excluded=fit_excluded_for_display(swap_feat_cols),
            )

            treebank_nav, treebanks_fresh = None, False
            extra_df = (extra_df_base
                        if self.per_treebank and self.include_excluded and learn_dt else None)
            if (self.per_treebank and (learn_dt or self.incl_unk) and "treebank" in full_df.columns
                    and (full_df["treebank"].nunique() > 1 or extra_df is not None or self.incl_unk)):
                tb_cache_dir = os.path.join(decision_trees_dir, "treebanks")
                tb_summary, treebanks_fresh = fit_treebank_trees(
                    full_df, lang, model, self.predictor_var, self.target, tb_cache_dir,
                    fit_kwargs=dict(min_samples_leaf=self.min_samples_leaf, max_depth=self.max_depth,
                                      test_size=self.test_size),
                    omit_feats_fn=self._swap_feat_cols,
                    impurity_fn=self.impurity_fn, pooled_impurity=min_impurity_decrease,
                    drop_unk=self.drop_unk, never_skip=self.never_skip or self.never_skip_fit,
                    extra_df=extra_df, include_excluded=self.include_excluded,
                    reason_fn=treebank_row_exclusion_reason,
                    incl_unk=self.incl_unk)
                tb_html_dir = os.path.join(HTML_DECISION_TREES_DIR, self.target_id)
                if (treebanks_fresh or self.never_skip or self.never_skip_fit
                        or not os.path.isdir(os.path.join(tb_html_dir, lang))):
                    render_treebank_pages(tb_summary, lang, full_df, tb_cache_dir, tb_html_dir,
                                          tree_html_kwargs, extra_df=extra_df)
                treebank_nav = build_nav(tb_summary, lang)
                lang_html_file = os.path.join(HTML_DECISION_TREES_DIR, self.target_id,
                                              f"{pooled_page_stem(lang)}.html")
            elif self.per_treebank:
                # marks the language as handled (see _is_done)
                write_treebanks_marker(os.path.join(decision_trees_dir, "treebanks", f"{lang}.json"),
                                       self.include_excluded, self.incl_unk)

            if (self.never_skip or self.never_skip_fit or self.redo_pairs or treebanks_fresh
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