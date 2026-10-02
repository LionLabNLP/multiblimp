"""Extraction-only caching sweep for the sv/sp/sa/ov/op/oa/iov/iop/ioa
word_order caches -- the SVA-side analogue of scripts/npa/np_types.py.

Running any condition via fit_candidate.py (e.g. `fit_candidate.py svNa`)
already populates its family's cache as a side effect (extract_node_features writes
every attested feature, not just the one condition's own target -- see
sva_trees/agreement_candidates.py's module docstring), but only by also
paying for that condition's decision-tree fit, minimal-pair generation and
second-chance retries on top. This script does *only* the caching --
Pipeline._build_df / AuxPipeline._build_df, the same tested extraction code
the condition scripts use (via _extract_df), called directly instead of
through Pipeline.__call__ -- so a language's caches for every family can be
(re)built once, cheaply, before scripts/sva_trees/agreement_candidates.py
scans them.

Groups mirror the condition families in sva_trees/conditions.py:
  --verb        sv, ov, iov   (finite verb agreement)
  --participle  sp, op, iop   (participle agreement)
  --aux         sa, oa, ioa   (auxiliary agreement)
  --all         all of the above (default if no group flag is given)

Usage:
  python3 vp_types.py                       # all 9 families, all languages
  python3 vp_types.py --verb --aux          # just those 6 families
  python3 vp_types.py --langs German French # scope to specific languages
  python3 vp_types.py --recache             # ignore existing cache parquets

Efficiency (see cached_row_count/_empty_marker_path/cache_language): a
language already fully cached for every requested family is answered from
parquet metadata alone, in the main process -- no worker dispatch, no
re-extraction, no re-read or re-write of the existing file. A language with
only SOME families cached only sends its still-uncached pipelines to the
worker (the cached ones are reported from the same metadata, never touched
again) -- and once there, every non-aux family shares one load_treebank()
pickle-read rather than repeating it per family (see cache_language). A
language with genuinely zero matching instances (create_word_order_df never
saves an empty df) gets a sentinel marker so it isn't re-extracted from
scratch on every future run either. Only languages that still need real
work touch a worker at all, and only the pipelines that still need it pay
for extraction.

Output identity: every pipeline is built with the exact target/unimorph_
args/max_treebank_len/rm_columns a single condition run (e.g. `python3
fit_candidate.py svNa --recache`) would use, and extraction goes through
that same Pipeline._extract_df/_build_df -- the cache this script writes
for a given (language, family) is identical to what running any one of
that family's own Na/Ga/Pa conditions with --recache would independently
produce for the same file (verified: see conversation).

Safety (see multiblimp.agreement_pipeline_utils, shared with sva_trees.
pipeline.Pipeline.__call__): refuses to start if another instance (or an
orphaned worker left behind by a killed one) is already running, unless
--force; each worker's RAM is hard-capped (RLIMIT_AS) so a runaway
language raises a catchable MemoryError in just that worker instead of the
system OOM-killer or swap taking the whole machine down; a crashing/OOM'd
language is caught and logged rather than aborting the whole sweep.
"""
import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow.parquet as pq

sys.path.append("../../src/")

from multiblimp.languages import get_ud_langs, lang2langcode, gblang2udlang
from multiblimp.unimorph import load_inflector
from multiblimp.agreement_pipeline_utils import (
    find_other_running_instances, limit_process_memory,
)
from sva_trees.conditions import CONDITIONS
from sva_trees.pipeline import Pipeline
from subj_aux.pipeline import AuxPipeline
from word_order.process_treebank import load_treebank

GROUPS = {
    "verb": ["sv", "ov", "iov"],
    "participle": ["sp", "op", "iop"],
    "aux": ["sa", "oa", "ioa"],
}


def build_pipeline(prefix: str, langs: list[str], resource_dir: str,
                    max_treebank_len: int | None, never_skip: bool,
                    n_jobs: int = 1, max_worker_mem_gb: float | None = None,
                    mem_headroom: float = 0.8) -> Pipeline:
    """One (Aux)Pipeline per family, built the same way sva_trees.cli.
    run_condition does -- Na is an arbitrary pick among that family's Na/Ga/
    Pa conditions (see conditions.py: they share word_order_dir/deprel/
    unimorph_args/inflection_map-irrelevance for extraction purposes, only
    feature_suffix differs, and that only matters for which predictor_var
    names the agreement column -- see module docstring).

    predictor_var is set to this Na condition's own (e.g.
    "head_nsubj_Number_agreement"), NOT None: Pipeline._extract_df's skip
    check reads an existing cache parquet but only actually returns it early
    when `self.predictor_var in df.columns` -- with predictor_var=None that
    membership test can never succeed (None is never a real column name), so
    the check silently falls through to a full re-extraction every time,
    defeating caching entirely regardless of what's already on disk. The
    column itself is unaffected either way: Pipeline.agreement_feats already
    defaults to ["Number", "Gender", "Person"] regardless of predictor_var,
    so extraction always computes all three agreement columns; this only
    changes which one the skip check looks for.
    """
    condition = CONDITIONS[f"{prefix}Na"]
    kwargs = dict(
        target=condition.make_target(),
        predictor_var=condition.predictor_var,
        langs=langs,
        inflection_map=condition.inflection_map,
        unimorph_args=condition.unimorph_args,
        deprel_dir=condition.deprel,
        resource_dir=resource_dir,
        word_order_dir=condition.word_order_dir(),
        max_treebank_len=max_treebank_len,
        never_skip=never_skip,
        rm_columns=list(condition.rm_columns),
        target_id=f"{prefix}_vp_types",
        fetch_all=True,
        n_jobs=n_jobs,
        max_worker_mem_gb=max_worker_mem_gb,
        mem_headroom=mem_headroom,
    )
    if condition.is_aux:
        return AuxPipeline(aux_target=condition.aux_target, **kwargs)
    return Pipeline(**kwargs)


def _empty_marker_path(pipeline: Pipeline, lang: str) -> str:
    """Sentinel path for "already extracted, genuinely 0 matching instances"
    -- see cached_row_count/cache_language. create_word_order_df only ever
    writes a parquet when len(df) > 0 (Pipeline._build_df has no code path
    that saves an empty df), so a language with zero matches for this
    pipeline's target leaves no file at all, indistinguishable on disk from
    "never checked" -- meaning every future run re-pays the full extraction
    cost (a real treebank parse + feature-extraction pass) just to
    rediscover the same zero, forever. This sentinel is the difference
    between those two states.
    """
    return pipeline.empty_marker_path(lang)


def cached_row_count(pipeline: Pipeline, lang: str) -> int | None:
    """None if `pipeline`'s cache for `lang` doesn't (yet) satisfy Pipeline.
    _extract_df's own skip condition (path exists AND predictor_var is a
    column) -- otherwise its row count (0 for a confirmed-empty language,
    via the sentinel file). Mirrors _extract_df's check exactly, but from
    the main process, off parquet metadata alone (schema + num_rows), never
    reading the actual data -- so a language that's already fully cached for
    every requested pipeline can be identified, and its result printed,
    without ever spawning a worker for it at all. Before this, a fully-
    cached language still paid a full ProcessPoolExecutor dispatch (and,
    every --max_tasks_per_child'th language, a fresh interpreter + pandas/
    sklearn/pyarrow import) just to have the worker re-derive the same
    "yes, already cached" answer _extract_df would give it anyway -- ~2-3s
    of pure overhead per language, which dominates entirely once most of the
    corpus is already cached (see conversation: 152 languages in 6:51 with
    zero new files written).

    --recache (pipeline.never_skip) bypasses both the real cache and the
    empty-sentinel, same as it always has for the real cache -- the general
    mechanism for invalidating a cache that's stale for any reason (a filter
    change, e.g.), not something sentinel-specific.
    """
    if pipeline.never_skip:
        return None
    cached_lang = gblang2udlang.get(lang, lang).replace(" ", "_")
    path = os.path.join(pipeline.word_order_dir, f"{cached_lang}.parquet")
    if os.path.exists(path):
        pf = pq.ParquetFile(path)
        if pipeline.predictor_var in pf.schema_arrow.names:
            return pf.metadata.num_rows
        return None
    if os.path.exists(_empty_marker_path(pipeline, lang)):
        return 0
    return None


def cache_language(lang: str, pipelines: list[Pipeline], resource_dir: str) -> list[tuple[str, int]]:
    """Build the df for `lang` under every pipeline in `pipelines`, straight
    through Pipeline._build_df/AuxPipeline._build_df rather than
    pipeline._extract_df -- callers (see the __main__ dispatch loop) only
    ever pass pipelines cached_row_count already proved (from parquet
    metadata alone) are NOT validly cached, so _extract_df's own "does a
    cache already exist" recheck can only ever land on its rebuild branch
    here anyway; going there would, in the one case where a stale parquet
    exists but lacks the current predictor_var column, still pay a full
    read_df() just to rediscover what cached_row_count's cheap schema check
    already knew.

    One treebank load shared across every non-AuxPipeline instance here
    (same lang, same max_treebank_len -- both identical across all pipelines
    built by vp_types.build_pipeline for a given run): Pipeline._build_df's
    own extraction (extract_features -> merge_subj_layered_feats) never
    mutates the treebank in a target-specific way, so one load_treebank()
    pickle-read serves sv/sp/ov/op/iov/iop alike instead of each repeating
    it. AuxPipeline._build_df is the one exception -- it calls
    redirect_to_aux(treebank, deprels), which destructively rewrites
    token["head"] for its own target's deprels -- so every AuxPipeline
    (sa/oa/ioa) still loads and mutates its own private treebank, never this
    shared one.

    One inflector per distinct unimorph_args (id()-keyed: sva_trees.
    conditions.VERB_UNIMORPH_ARGS/AUX_UNIMORPH_ARGS are shared module-level
    dicts, so every "verb"/"participle" pipeline resolves to the same
    inflector object here, and every "aux" one to the other) -- UniMorph
    pickle loading is the expensive part of building an inflector, and
    inflection_map (which DOES differ per prefix) only affects reinflection/
    can_feature_swap, never the um_data/ud_data lookup tables this
    extraction-only run actually uses (multiblimp.unimorph.load_unimorph_
    pickle reads purely off langcode/resource_dir/filter_entries, never
    self.inflection_map) -- see module docstring. skip_lang (also inflection_
    map-dependent) is ignored here for the same reason: it only gates
    whether a language can be SWAPPED for, irrelevant to caching raw
    features.
    """
    results = []
    inflectors = {}
    shared_treebank = None
    if any(not isinstance(p, AuxPipeline) for p in pipelines):
        shared_treebank = load_treebank(
            lang, resource_dir, pipelines[0].max_treebank_len, use_selected_treebanks=False
        )
    for pipeline in pipelines:
        key = id(pipeline.unimorph_args)
        if key not in inflectors:
            inflector, _skip_lang, _num_lemma, _num_form = load_inflector(
                lang=lang,
                langcode=lang2langcode(lang),
                unimorph_args=pipeline.unimorph_args,
                inflection_map=pipeline.inflection_map,
                resource_dir=resource_dir,
            )
            inflectors[key] = inflector
        inflector = inflectors[key]

        if isinstance(pipeline, AuxPipeline):
            df = pipeline._build_df(lang, pipeline.word_order_dir, inflector)
        else:
            df = pipeline._build_df(lang, pipeline.word_order_dir, inflector, treebank=shared_treebank)
        n = 0 if df is None else len(df)
        results.append((pipeline.target_id, n))
        if n == 0:
            # create_word_order_df never saves an empty df, so without this
            # marker a genuinely-empty language gets fully re-extracted
            # (real treebank parse included) on every future run forever --
            # see _empty_marker_path's docstring.
            os.makedirs(pipeline.word_order_dir, exist_ok=True)
            with open(_empty_marker_path(pipeline, lang), "w") as f:
                json.dump({"predictor_var": pipeline.predictor_var}, f)
        else:
            # A stale sentinel from before some filter change (e.g. the
            # VerbForm=Fin fix) would otherwise sit next to the new real
            # parquet, contradicting it -- harmless functionally
            # (cached_row_count checks the real parquet first) but
            # confusing to find on disk.
            stale_marker = _empty_marker_path(pipeline, lang)
            if os.path.exists(stale_marker):
                os.remove(stale_marker)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verb", action="store_true", help="Cache sv/ov/iov (finite verb)")
    parser.add_argument("--participle", action="store_true", help="Cache sp/op/iop")
    parser.add_argument("--aux", action="store_true", help="Cache sa/oa/ioa")
    parser.add_argument("--all", action="store_true",
                        help="Cache all 9 families (default if no other group/--families flag is given)")
    parser.add_argument("--families", "-f", nargs="*", default=[],
                        choices=[p for fams in GROUPS.values() for p in fams],
                        help="Cache exactly these families (e.g. --families sv), "
                             "instead of a whole --verb/--participle/--aux group. "
                             "Combines with the group flags if both are given.")
    parser.add_argument("--langs", "-l", nargs="*", default=[])
    parser.add_argument("--recache", action="store_true",
                        help="Force fresh extraction even where a cached parquet "
                             "already exists (default: skip languages already "
                             "cached -- same flag name/meaning as sva_trees.cli's "
                             "own --recache).")
    parser.add_argument("--max_treebank_len", type=int, default=None,
                        help="Cap sentences read per language. Default: no cap.")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Languages processed in parallel (1=serial, but "
                             "still through a recycled worker process -- see "
                             "--max_tasks_per_child). No per-worker memory cap "
                             "here (unlike sva_trees.pipeline.Pipeline.__call__'s "
                             "own -j) -- keep this modest on a large "
                             "--max_treebank_len run.")
    parser.add_argument("--max_tasks_per_child", type=int, default=1,
                        help="Recycle each worker process after this many "
                             "languages (default 1: fresh process per language). "
                             "Matters even at --n_jobs 1 -- see sva_trees.pipeline."
                             "Pipeline's own max_tasks_per_child docstring: pandas/"
                             "NumPy buffers and general heap fragmentation don't "
                             "reliably get handed back to the OS between "
                             "languages in a long-lived process, which is why "
                             "this always runs through ProcessPoolExecutor rather "
                             "than a plain for-loop, even serially. Raise this "
                             "to trade some of that safety back for less "
                             "per-language interpreter-startup overhead.")
    parser.add_argument("--max_worker_mem_gb", type=float, default=None,
                        help="Hard RAM cap per worker process, in GB (RLIMIT_AS, "
                             "same mechanism as sva_trees.pipeline.Pipeline's own "
                             "-- multiblimp.agreement_pipeline_utils."
                             "limit_process_memory). A worker that exceeds it "
                             "gets a catchable MemoryError instead of triggering "
                             "the system OOM-killer or swapping the whole machine "
                             "into unresponsiveness. Default: auto-derived from "
                             "currently-available system RAM / --n_jobs x "
                             "--mem_headroom.")
    parser.add_argument("--mem_headroom", type=float, default=0.8,
                        help="Fraction of available RAM to divide across workers "
                             "when auto-deriving the per-worker cap (only with "
                             "--n_jobs>1... well, always, but only matters when "
                             "--max_worker_mem_gb is unset). Same default as "
                             "sva_trees.pipeline.Pipeline.")
    parser.add_argument("--force", action="store_true",
                        help="Skip the startup check for other already-running "
                             "instances of this script (multiblimp."
                             "agreement_pipeline_utils.find_other_running_"
                             "instances -- also catches orphaned workers left "
                             "behind by a run that was killed rather than left "
                             "to exit cleanly). Only use for deliberate "
                             "concurrent runs (e.g. one for --verb, another for "
                             "--participle, run by different people/terminals).")
    args = parser.parse_args()

    if not args.force:
        other_pids = find_other_running_instances(sys.argv[0])
        if other_pids:
            script_name = os.path.basename(sys.argv[0])
            raise SystemExit(
                f"Another instance of {script_name} appears to already be "
                f"running (PID(s): {other_pids}). If they're stale (a killed "
                f"run's orphaned workers), kill them first: "
                f"kill {' '.join(map(str, other_pids))}. If this is a "
                f"deliberate concurrent run, pass --force."
            )

    resource_dir = "../../resources"
    groups = [g for g, flag in (("verb", args.verb), ("participle", args.participle),
                                 ("aux", args.aux)) if flag]
    prefixes = list(dict.fromkeys([p for g in groups for p in GROUPS[g]] + args.families))
    if args.all or not prefixes:
        prefixes = [p for fams in GROUPS.values() for p in fams]

    langs = args.langs if args.langs else get_ud_langs(resource_dir)
    pipelines = [build_pipeline(p, langs, resource_dir, args.max_treebank_len, args.recache,
                                n_jobs=args.n_jobs, max_worker_mem_gb=args.max_worker_mem_gb,
                                mem_headroom=args.mem_headroom)
                 for p in prefixes]

    print(f"Caching {prefixes} for {len(langs)} language(s)")

    # Filter out languages already fully cached for every pipeline *before*
    # touching the worker pool at all -- see cached_row_count's docstring.
    # A language only goes to a worker if at least one pipeline still needs
    # real work; the rest are answered directly, at main-process/metadata-
    # read cost instead of a full dispatch + possible interpreter respawn.
    #
    # For a language that DOES go to a worker, only its still-uncached
    # pipelines are sent along (needed_pipelines/cached_counts below) --
    # cached_row_count already proved the rest are done, cheaply, off
    # parquet metadata alone, so re-handing those to cache_language would
    # just pay a full parquet read per already-cached family for nothing
    # (see cache_language's docstring).
    to_process = []
    needed_pipelines = {}
    cached_counts = {}
    for lang in langs:
        counts = [(p, p.target_id, cached_row_count(p, lang)) for p in pipelines]
        if all(n is not None for _, _, n in counts):
            print(f"{lang}: " + ", ".join(f"{tid}={n}" for _, tid, n in counts))
        else:
            to_process.append(lang)
            needed_pipelines[lang] = [p for p, _, n in counts if n is None]
            cached_counts[lang] = {tid: n for _, tid, n in counts if n is not None}

    if not to_process:
        print("Nothing left to extract -- every language already cached for every requested family.")
    else:
        print(f"{len(to_process)}/{len(langs)} language(s) need real extraction")
        # Always through ProcessPoolExecutor, even at n_jobs=1 -- see
        # --max_tasks_per_child's help text above for why a plain serial
        # for-loop in this process isn't safe for a full-corpus run (matches
        # sva_trees.pipeline.Pipeline.__call__'s own reasoning/pattern, see
        # its comment at "Always through ProcessPoolExecutor, even at
        # n_jobs=1").
        # Reuses Pipeline._worker_mem_bytes() (every pipeline carries the same
        # n_jobs/max_worker_mem_gb/mem_headroom, set in build_pipeline) rather
        # than re-deriving the same "available RAM x headroom / n_jobs"
        # formula by hand -- one formula, one place, same as the real
        # condition scripts use.
        worker_mem_bytes = pipelines[0]._worker_mem_bytes()
        if worker_mem_bytes is None:
            print("Could not detect system RAM; running without a memory cap")
            initializer, initargs = None, ()
        else:
            print(f"Capping each of {args.n_jobs} worker(s) to "
                  f"{worker_mem_bytes / 1024**3:.1f} GB RAM")
            initializer, initargs = limit_process_memory, (worker_mem_bytes,)

        with ProcessPoolExecutor(max_workers=args.n_jobs,
                                 max_tasks_per_child=args.max_tasks_per_child,
                                 initializer=initializer, initargs=initargs) as pool:
            futures = {
                pool.submit(cache_language, lang, needed_pipelines[lang], resource_dir): lang
                for lang in to_process
            }
            # Per-language error isolation (matches Pipeline.__call__'s own
            # pattern): a crashing/OOM'd language is caught and logged
            # rather than re-raised, so it doesn't take the whole sweep
            # (and every other language's already-submitted future) down
            # with it.
            for future in as_completed(futures):
                lang = futures[future]
                try:
                    counts = cached_counts[lang] | dict(future.result())
                    # Reported in the pipelines' own order, not whatever
                    # order cached vs. freshly-extracted happen to merge in.
                    print(f"{lang}: " + ", ".join(
                        f"{p.target_id}={counts[p.target_id]}" for p in pipelines))
                except Exception as e:
                    print(f"  FAILED: {lang}: {e}")
                    traceback.print_exc()
