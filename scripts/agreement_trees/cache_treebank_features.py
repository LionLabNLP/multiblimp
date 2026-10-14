"""Extraction-only caching sweep for the treebank feature caches.

  vp  verb-phrase caches: sv/sp/sa/ov/op/oa/iov/iop/ioa word_order parquets
      (Pipeline._build_df, the code the condition scripts use, without the
      tree fit, minimal pairs or second-chance retries)
  np  noun-phrase cache: treebank_features/npa/np_instances parquets
      (agreement_trees.npa.np_types)

Settings live in the shared agreement config (resources/agreement_config.json;
modes, langs, workers, vp, np; see agreement_trees.config), so a rerun
uses the same settings as the last one. Command line, for one-off or
machine-specific use: --config, --vp / --np (only that mode), --n_jobs and
--langs (override the config), --recache, --dry_run, --force.

Annotation filling (fetch_all: UniMorph and its UD-derived fallback) is on in both
modes. With MULTIBLIMP_DEBUG_MODE=1 it is off and the caches go to
treebank_features_debug/. A cache holds all features, so it serves every target.

Usage (any directory):
  python3 cache_treebank_features.py
  python3 cache_treebank_features.py --langs German --recache
  python3 cache_treebank_features.py --dry_run

Languages already cached for everything requested are answered in the main
process (parquet metadata / sentinel files). The rest go to one worker each,
which runs only the still-uncached parts; VP families share one treebank load.
Languages with no matching instances get an empty sentinel so they aren't
re-parsed. Workers are RAM-guarded; a language that fails or kills its worker
is logged to output/logs/cache_error_log.txt and the sweep continues. Refuses
to start if another instance is running (unless --force). Run log:
output/logs/cache_treebank_features.log.
"""
import argparse
import json
import os
import sys
import traceback
from collections import Counter

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_repo_root, "src"))

from agreement_trees.config import AGREEMENT_CONFIG_PATH, apply_mode_flags, load_agreement_config
from agreement_trees.pipeline_utils import (
    find_other_running_instances, run_in_pool, worker_mem_bytes,
)
from multiblimp.config import DEBUG_MODE, NP_REPORT_DIR, RESOURCE_DIR
from multiblimp.languages import get_ud_langs, lang2langcode
from multiblimp.script_log import log_cache_error, script_logger
from multiblimp.ud_inflector import load_ud_inflector
from agreement_trees.npa.np_cache import cache_np_language, np_cache_status, np_paths, read_counts, save_counts
from agreement_trees.vp.conditions import CONDITIONS
from agreement_trees.vp.pipeline import Pipeline
from agreement_trees.aux.pipeline import AuxPipeline
from agreement_trees.extract.process_treebank import load_treebank

# ---------------------------------------------------------------- VP

def build_vp_pipeline(prefix: str, langs: list[str], max_treebank_len: int | None,
                      never_skip: bool, n_jobs: int, max_worker_mem_gb: float | None,
                      mem_headroom: float) -> Pipeline:
    """One (Aux)Pipeline per family, built as agreement_trees.cli.run_vp does.
    The family's Na condition stands in for Na/Ga/Pa: they share everything
    extraction depends on. predictor_var must be that condition's own column,
    because the cache-validity check looks for it."""
    condition = CONDITIONS[f"{prefix}Na"]
    kwargs = dict(
        target=condition.make_target(),
        predictor_var=condition.predictor_var,
        langs=langs,
        inflection_map=condition.inflection_map,
        unimorph_args=condition.unimorph_args,
        deprel_dir=condition.deprel,
        resource_dir=RESOURCE_DIR,
        word_order_dir=condition.word_order_dir(),
        max_treebank_len=max_treebank_len,
        never_skip=never_skip,
        rm_columns=list(condition.rm_columns),
        target_id=f"{prefix}_vp_types",
        fetch_all=not DEBUG_MODE,
        n_jobs=n_jobs,
        max_worker_mem_gb=max_worker_mem_gb,
        mem_headroom=mem_headroom,
    )
    if condition.is_aux:
        return AuxPipeline(aux_target=condition.aux_target, **kwargs)
    return Pipeline(**kwargs)


def cache_vp_language(lang: str, pipelines: list[Pipeline]) -> list[tuple[str, int]]:
    """Build and cache `lang`'s df for each pipeline (all known to lack a valid
    cache). Non-aux pipelines share one treebank load; AuxPipeline rewrites
    heads in place (redirect_to_aux), so it loads its own. Inflectors are
    shared per distinct unimorph_args: extraction only uses the lookup tables,
    which don't depend on inflection_map."""
    results = []
    inflectors = {}
    shared_treebank = None
    if any(not isinstance(p, AuxPipeline) for p in pipelines):
        shared_treebank = load_treebank(
            lang, RESOURCE_DIR, pipelines[0].max_treebank_len, use_selected_treebanks=False
        )
    for pipeline in pipelines:
        key = id(pipeline.unimorph_args)
        if key not in inflectors and DEBUG_MODE:
            inflectors[key] = None
        elif key not in inflectors:
            inflectors[key], *_ = load_ud_inflector(
                lang=lang,
                langcode=lang2langcode(lang),
                unimorph_args=pipeline.unimorph_args,
                inflection_map=pipeline.inflection_map,
                resource_dir=RESOURCE_DIR,
            )
        inflector = inflectors[key]

        if isinstance(pipeline, AuxPipeline):
            df = pipeline._build_df(lang, pipeline.word_order_dir, inflector)
        else:
            df = pipeline._build_df(lang, pipeline.word_order_dir, inflector, treebank=shared_treebank)
        pipeline.record_extraction(lang, df)
        results.append((pipeline.target_id, 0 if df is None else len(df)))

    return results


# ---------------------------------------------------------------- driver

def cache_language(lang: str, vp_pipelines: list[Pipeline], np_job: dict | None) -> dict:
    """Worker task: run the VP and NP parts still needed for `lang`. A failure
    in one part doesn't lose the other's result."""
    out = {}
    if vp_pipelines:
        try:
            out["vp"] = ("processed", cache_vp_language(lang, vp_pipelines))
        except Exception as e:
            print(f"  FAILED (vp): {lang}", flush=True)
            traceback.print_exc()
            log_cache_error("vp", lang, f"{type(e).__name__}: {e}", traceback.format_exc())
            out["vp"] = ("failed", None)
    if np_job is not None:
        out["np"] = cache_np_language(lang, **np_job)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=AGREEMENT_CONFIG_PATH, help="Agreement config JSON (see module docstring).")
    parser.add_argument("--vp", action="store_true", help="Cache only the verb-phrase families.")
    parser.add_argument("--np", action="store_true", help="Cache only the noun-phrase instances.")
    parser.add_argument("--langs", "-l", nargs="*", default=None, help="Override the config's languages.")
    parser.add_argument("--n_jobs", type=int, default=None,
                        help="Languages processed in parallel; overrides the config's n_jobs "
                             "(machine-dependent, so usually set per run).")
    parser.add_argument("--recache", action="store_true", help="Re-extract even where a cache already exists.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only report which languages and caches would be extracted; write nothing.")
    parser.add_argument("--force", action="store_true", help="Skip the check for other running instances.")
    return parser.parse_args()


def main():
    args = parse_args()
    log = script_logger("cache_treebank_features")

    if not args.force:
        other_pids = find_other_running_instances(sys.argv[0])
        if other_pids:
            raise SystemExit(
                f"Another instance of {os.path.basename(sys.argv[0])} appears to be running "
                f"(PID(s): {other_pids}). If they're stale, kill them first: "
                f"kill {' '.join(map(str, other_pids))}. For a deliberate concurrent run, pass --force."
            )

    config = apply_mode_flags(load_agreement_config(args.config), args.vp, args.np)
    workers = config["workers"]
    if args.n_jobs is not None:
        workers["n_jobs"] = args.n_jobs
    do_vp, do_np = "vp" in config["modes"], "np" in config["modes"]
    vp_cfg, np_cfg = config["vp"], config["np"]
    requested_langs = args.langs if args.langs is not None else config["langs"]
    langs = sorted(requested_langs) if requested_langs else get_ud_langs(RESOURCE_DIR)
    log.info(f"Config {args.config}:\n{json.dumps(config, indent=2)}")

    vp_pipelines = []
    if do_vp:
        vp_pipelines = [
            build_vp_pipeline(p, langs, vp_cfg["max_treebank_len"], args.recache, workers["n_jobs"],
                              workers["max_worker_mem_gb"], workers["mem_headroom"])
            for p in vp_cfg["families"]
        ]
    report_dir = NP_REPORT_DIR if np_cfg["report"] else None
    log.info(f"Caching vp={[p.target_id for p in vp_pipelines] if do_vp else 'off'} "
             f"np={'on' if do_np else 'off'} for {len(langs)} language(s)")

    # Answer fully cached languages here; only what is still uncached goes to a worker.
    tasks, cached_vp, np_results = {}, {}, {}
    for lang in langs:
        counts = [(p, p.cached_row_count(lang)) for p in vp_pipelines]
        cached_vp[lang] = {p.target_id: n for p, n in counts if n is not None}
        needed = [p for p, n in counts if n is None]
        np_job = None
        if do_np:
            status = np_cache_status(lang, args.recache)
            if status is None:
                np_job = dict(report_dir=report_dir, max_treebank_len=np_cfg["max_treebank_len"],
                              streaming=np_cfg["streaming"], chunk_size=np_cfg["chunk_size"],
                              fetch_all=not DEBUG_MODE)
            else:
                np_results[lang] = (status, None)
        if needed or np_job is not None:
            tasks[lang] = (lang, needed, np_job)
        elif vp_pipelines:
            log.info(f"{lang}: " + ", ".join(f"{p.target_id}={cached_vp[lang][p.target_id]}" for p in vp_pipelines))

    vp_failed = []
    if args.dry_run:
        n_vp = sum(len(t[1]) for t in tasks.values())
        n_np = sum(t[2] is not None for t in tasks.values())
        log.info(f"Dry run: {len(tasks)} language(s) would run, {n_vp} VP cache(s) and {n_np} NP cache(s) to build")
        if do_np:
            counts = Counter(st for st, _ in np_results.values())
            log.info(f"NP already cached: {dict(counts)}")
        return
    if not tasks:
        log.info("Nothing left to extract -- every language already cached.")
    else:
        log.info(f"{len(tasks)}/{len(langs)} language(s) need real extraction")
        mem_bytes = worker_mem_bytes(workers["n_jobs"], workers["max_worker_mem_gb"], workers["mem_headroom"])
        if mem_bytes is None:
            log.info("Could not detect system RAM; running without a memory cap")
        else:
            log.info(f"Capping each of {workers['n_jobs']} worker(s) to {mem_bytes / 1024**3:.1f} GB RAM")

        def report(lang, result, error):
            if error is not None:
                log.info(f"  FAILED: {lang}: {error}")
                if tasks[lang][1]:
                    vp_failed.append(lang)
                    log_cache_error("vp", lang, f"worker lost: {error}")
                if tasks[lang][2] is not None:
                    np_results[lang] = ("failed", None)
                    log_cache_error("np", lang, f"worker lost: {error}")
                return
            if "vp" in result:
                status, rows = result["vp"]
                if status == "failed":
                    vp_failed.append(lang)
                else:
                    cached_vp[lang].update(dict(rows))
            if "np" in result:
                np_results[lang] = result["np"]
            if vp_pipelines and lang not in vp_failed:
                log.info(f"{lang}: " + ", ".join(
                    f"{p.target_id}={cached_vp[lang].get(p.target_id)}" for p in vp_pipelines))

        run_in_pool(cache_language, tasks, workers["n_jobs"], mem_bytes, workers["max_tasks_per_child"], on_done=report)

    if vp_failed:
        log.info(f"\nVP failed for {len(vp_failed)} language(s): {', '.join(sorted(vp_failed))}")
    if do_np:
        by_status = {s: [l for l, (st, _) in np_results.items() if st == s]
                     for s in ("processed", "skipped", "empty", "failed")}
        log.info(f"\nNP: processed {len(by_status['processed'])}, skipped (cached) {len(by_status['skipped'])}, "
                 f"empty {len(by_status['empty'])}, failed {len(by_status['failed'])} of {len(langs)} languages")
        if by_status["failed"]:
            log.info("NP failed: " + ", ".join(sorted(by_status["failed"])))
        if np_cfg["report"]:
            macro, n_csv = Counter(), 0
            for lang in langs:
                csv_path = np_paths(lang, NP_REPORT_DIR)[2]
                if os.path.exists(csv_path):
                    macro.update(read_counts(csv_path))
                    n_csv += 1
            if macro:
                save_counts(macro, os.path.join(NP_REPORT_DIR, "_macro.csv"))
                log.info(f"\nReport: {n_csv}/{len(langs)} language CSVs pooled into {NP_REPORT_DIR}/_macro.csv "
                         "(languages cached before 'report' was on have none; --recache to build them)")


if __name__ == "__main__":
    main()
