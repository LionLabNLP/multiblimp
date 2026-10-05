import argparse
import csv
import json
import os
import sys
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append("../../src/")

from npa.np_types import build_np_data, build_np_data_streaming
from multiblimp.languages import get_ud_langs, gblang2udlang
from multiblimp.agreement_pipeline_utils import (
    available_system_memory_bytes, find_other_running_instances, guard_process_memory,
)
from multiblimp.config import TREEBANK_FEATURES_DIR

REPORT_DIR = "../../report/np_types"
MAX_TREEBANK_LEN = 10_000


def save_counts(counts: Counter, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    total = sum(counts.values())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["np_type", "count", "proportion"])
        for np_type, count in counts.most_common():
            writer.writerow([np_type, count, f"{count / total:.6f}"])


def _empty_marker_path(instances_dir: str, cached_lang: str) -> str:
    """Sentinel for "already extracted, no np_instances parquet came out of
    it" (no NOUN/PROPN/PRON heads at all, or heads but none with a qualifying
    dependent -- build_np_data_streaming writes no parquet then). Without
    it that outcome is indistinguishable on disk from "never extracted", so
    every later run re-parses the treebank just to rediscover it. Same idea
    as scripts/sva_trees/vp_types.py's {lang}.empty.json."""
    return os.path.join(instances_dir, f"{cached_lang}.empty.json")


def _read_counts(path: str) -> Counter:
    counts = Counter()
    with open(path) as f:
        for row in csv.DictReader(f):
            counts[row["np_type"]] = int(row["count"])
    return counts


def process_one_language(lang: str, resource_dir: str, report_dir: str | None, instances_dir: str,
                          max_treebank_len: int | None, streaming: bool, chunk_size: int,
                          never_skip: bool) -> tuple[str, str, Counter | None]:
    """One language's worth of the original serial loop body, factored out
    so it can run either inline (n_jobs=1) or as a ProcessPoolExecutor task
    (n_jobs>1 -- see __main__). Returns (lang, status, counts) rather than
    mutating shared processed/skipped/failed lists or macro_counts directly,
    since those aren't safely shared across worker processes; __main__
    folds every language's (status, counts) back together once all are
    done, exactly as the serial loop did inline before.

    report_dir: None (default of the CLI) writes only the np_instances
    parquet. With a directory, the language's np_type counts CSV is also
    written there (the optional --report) and a language only counts as
    cached once that CSV exists too.

    status is one of "skipped" (already cached), "processed", "empty" (no
    NOUN/PROPN/PRON heads at all, fresh or per the empty-sentinel), or
    "failed" (exception during extraction -- traceback already printed
    here). counts is the language's np_type Counter for "processed", and for
    "skipped" when report_dir is set (read back from its CSV); None otherwise.

    streaming=True calls build_np_data_streaming (writes instances_path
    directly, bounded memory -- see that function's docstring) instead of
    build_np_data (returns the full df, written here). Both write the same
    on-disk shape (one {lang}.parquet under instances_dir), so a mixed run
    (some languages cached from one mode, freshly processed in the other)
    stays consistent for every downstream reader.
    """
    cached_lang = gblang2udlang.get(lang, lang).replace(" ", "_")
    out_path = os.path.join(report_dir, f"{cached_lang}.csv") if report_dir else None
    instances_path = os.path.join(instances_dir, f"{cached_lang}.parquet")
    empty_marker = _empty_marker_path(instances_dir, cached_lang)

    # With a report both artifacts come from one treebank pass -- only skip
    # a language once both are present, else redo the pass so they stay in
    # sync. Without one the parquet (or the empty-sentinel) alone decides.
    report_done = out_path is None or os.path.exists(out_path)
    if not never_skip and report_done and os.path.exists(instances_path):
        return lang, "skipped", _read_counts(out_path) if out_path else None
    if not never_skip and os.path.exists(empty_marker):
        if out_path and os.path.exists(out_path):
            return lang, "skipped", _read_counts(out_path)
        with open(empty_marker) as f:
            has_np_heads = json.load(f).get("has_np_heads", False)
        if out_path is None or not has_np_heads:
            return lang, "empty", None

    print(lang, flush=True)
    try:
        if streaming:
            os.makedirs(instances_dir, exist_ok=True)
            counts = build_np_data_streaming(
                lang, resource_dir, instances_path,
                max_treebank_len=max_treebank_len, chunk_size=chunk_size,
            )
        else:
            counts, df = build_np_data(
                lang, resource_dir, max_treebank_len=max_treebank_len)
    except Exception:
        print(f"  FAILED: {lang}", flush=True)
        traceback.print_exc()
        return lang, "failed", None

    os.makedirs(instances_dir, exist_ok=True)
    if counts:
        if out_path:
            save_counts(counts, out_path)
        if not streaming:
            df.to_parquet(instances_path, index=False)
    else:
        print(f"  no NOUN/PROPN/PRON heads found for {lang}", flush=True)

    if os.path.exists(instances_path):
        if os.path.exists(empty_marker):
            os.remove(empty_marker)  # stale: from before a filter change
    else:
        with open(empty_marker, "w") as f:
            json.dump({"has_np_heads": bool(counts)}, f)
    return (lang, "processed", counts) if counts else (lang, "empty", None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distribution of NOUN/PROPN/PRON-headed NP UPOS-sequence types "
                     "(e.g. 'DET NOUN', 'DET ADJ ADJ NOUN') per language and "
                     "macro (pooled across languages)."
    )
    parser.add_argument("--langs", "-l", nargs="*", default=[])
    parser.add_argument("--recache", action="store_true",
                        help="Re-count languages even if a cached CSV exists")
    parser.add_argument("--max_treebank_len", "-m", type=int, default=10_000)
    parser.add_argument("--top_n", "-n", type=int, default=20,
                        help="Number of top types to print for the macro summary")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True,
                        help="Use build_np_data_streaming (flush every --chunk_size "
                             "NP-instance records to disk, ~constant peak memory per "
                             "language) instead of build_np_data (holds the whole "
                             "language's records in memory at once -- can reach tens "
                             "of GB for a large, morphologically rich language). "
                             "On by default; pass --no-streaming for the in-memory path.")
    parser.add_argument("--chunk_size", type=int, default=500,
                        help="Records per flush under --streaming (ignored otherwise).")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Languages to process in parallel (ProcessPoolExecutor). "
                             "Each language is independent (its own output file), so "
                             "this is embarrassingly parallel; combine with --streaming "
                             "to keep total memory (n_jobs x per-language peak) in "
                             "check -- see --max_worker_mem_gb.")
    parser.add_argument("--max_worker_mem_gb", type=float, default=None,
                        help="Per-worker RLIMIT_AS cap, in GB (applies at any --n_jobs, incl. 1). "
                             "None (default) auto-derives from available system RAM / "
                             "n_jobs x --mem_headroom, same as sva_trees.pipeline.Pipeline.")
    parser.add_argument("--mem_headroom", type=float, default=0.8,
                        help="Fraction of available RAM to divide across workers when "
                             "auto-deriving the per-worker cap (at any --n_jobs "
                             "and --max_worker_mem_gb unset).")
    parser.add_argument("--force", action="store_true",
                        help="Skip the already-running-instance check (applies at any --n_jobs, incl. 1).")
    parser.add_argument("--report", action="store_true",
                        help=f"Also write the per-language np_type counts CSVs and the "
                             f"pooled _macro.csv to {REPORT_DIR} (and print the macro "
                             "summary). Off by default: nothing downstream reads them "
                             "except np_morph_pct.py, so the default run only writes the "
                             "np_instances parquets. With --report, a language only "
                             "counts as cached once its CSV exists too.")
    args = parser.parse_args()

    resource_dir = "../../resources"
    out_dir = REPORT_DIR if args.report else None
    instances_dir = os.path.join(TREEBANK_FEATURES_DIR, "npa", "np_instances")

    langs = sorted(args.langs) if args.langs else get_ud_langs(resource_dir)

    macro_counts = Counter()
    processed, skipped, empty, failed = [], [], [], []

    if not args.force:
        script_name = os.path.basename(sys.argv[0])
        other_pids = find_other_running_instances(sys.argv[0])
        if other_pids:
            raise RuntimeError(
                f"Another instance of {script_name} appears to already be "
                f"running (PID(s): {other_pids}). If they're stale, kill them "
                f"first: kill {' '.join(map(str, other_pids))} "
                f"Else, pass --force."
            )

    if args.max_worker_mem_gb is not None:
        worker_mem_bytes = int(args.max_worker_mem_gb * 1024**3)
    else:
        available_mem = available_system_memory_bytes()
        worker_mem_bytes = (
            int(available_mem * args.mem_headroom / args.n_jobs)
            if available_mem is not None else None
        )

    if worker_mem_bytes is not None:
        print(f"Capping each of {args.n_jobs} worker(s) to "
              f"{worker_mem_bytes / 1024**3:.1f} GB RAM")
        initializer, initargs = guard_process_memory, (worker_mem_bytes,)
    else:
        print("Could not detect system RAM; running without a memory cap")
        initializer, initargs = None, ()

    results = {}
    # max_tasks_per_child=1: a fresh process per language, not a
    # long-lived one reused across the whole corpus -- matches
    # sva_trees.pipeline.Pipeline's own reasoning (pandas/NumPy buffers
    # and heap fragmentation don't reliably get handed back to the OS
    # between languages, so a language deep into a long run can hit
    # even a large fixed memory cap purely from what earlier languages
    # left behind; a fresh process per language avoids that entirely).
    with ProcessPoolExecutor(max_workers=args.n_jobs, max_tasks_per_child=1,
                              initializer=initializer, initargs=initargs) as executor:
        future_to_lang = {
            executor.submit(
                process_one_language, lang, resource_dir, out_dir, instances_dir,
                args.max_treebank_len, args.streaming, args.chunk_size, args.recache,
            ): lang
            for lang in langs
        }
        for future in as_completed(future_to_lang):
            lang = future_to_lang[future]
            try:
                _, status, counts = future.result()
            except Exception as e:
                print(f"  FAILED: {lang}: {e}")
                traceback.print_exc()
                status, counts = "failed", None
            results[lang] = (status, counts)

    for lang in langs:
        status, counts = results[lang]
        if status in ("skipped", "processed"):
            (skipped if status == "skipped" else processed).append(lang)
            if counts:
                macro_counts.update(counts)
        elif status == "empty":
            empty.append(lang)
        else:
            failed.append(lang)

    if args.report:
        save_counts(macro_counts, os.path.join(out_dir, "_macro.csv"))

    print(f"\nProcessed {len(processed)}, skipped (cached) {len(skipped)}, "
          f"empty {len(empty)}, failed {len(failed)} of {len(langs)} languages")
    if failed:
        print("Failed:", ", ".join(failed))

    total = sum(macro_counts.values())
    if args.report and total:
        print(f"\nMacro distribution (top {args.top_n} of {len(macro_counts)} "
              f"types, {total} NPs total):")
        for np_type, count in macro_counts.most_common(args.top_n):
            print(f"  {count / total:6.2%}  {count:>10}  {np_type}")
