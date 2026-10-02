"""Rebuild every built condition's diagnostics CSV + deprel index.html (SVA
and NPA alike), in parallel, then the cross-pipeline decision_trees/index.html
overview once -- purely from on-disk output/decision_trees +
output/minimal_pairs artifacts, no refitting.

The fit scripts (sva_trees/fit_all.py, npa/fit_all.py) pass --no_index to
every condition they fit, so a sweep never builds 100+ indexes inline,
one after the other; this script is the single place they get built.
A lone fit_candidate.py run still builds its own condition's index unless
--no_index is given.

Why it is cheap relative to a full pipeline run: both pipelines' index
builders (sva_trees.pipeline.refresh_deprel_index, npa.agreement.
refresh_deprel_index -- each the extracted tail of its pipeline) read only
the decision-tree-stage .joblib/.parquet cache and the minimal-pairs cache,
never the extraction caches (np_instances/, treebank_features/).

Efficiency:
 - A condition is skipped unless something that feeds its index changed
   since it was built: any of its .joblib/.parquet/meta.json inputs, or the
   index-generating code itself (viz_deprel.py, html_deprel.py,
   diagnostics.py) -- see needs_refresh. --force bypasses the check.
 - Stale conditions are independent and read-only on the caches, so they run
   in a process pool (-j, default min(cpu_count, 8)); each worker is
   memory-capped like the fit pipelines' workers, gets a fresh process per
   condition, and its log is printed as one block when it finishes.
 - The cross-pipeline overview rescans every index.html, so it runs exactly
   once, after all conditions.

Condition discovery is from disk: SVA conditions are output/decision_trees/
<target_id>/<target_id>_<deprel>/ (a fixed id from sva_trees.conditions, or
an ad-hoc <family><Feature> candidate); NPA ones are output/decision_trees/
npa/<npa_id>/. A directory this script can't map back to a condition is
reported and skipped.

Run from anywhere: python3 scripts/overview/generate_html_indexes.py (also the
first step of scripts/overview/run_all.sh, which then builds the dataset
overview from the pages this refreshes).
"""
import argparse
import contextlib
import glob
import io
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Absolute, not "../src": a relative entry resolves against the caller's cwd.
# REPO_ROOT too: src/ imports the top-level `resources` package, which its own
# cwd-relative "../" / "../../" entries only find from scripts/ or scripts/<dir>/.
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "src"))

from npa.agreement import refresh_deprel_index as refresh_npa_index, npa_id, FEATURE_ABBREV  # noqa: E402
from sva_trees.pipeline import refresh_deprel_index as refresh_sva_index  # noqa: E402
from sva_trees.conditions import CONDITIONS  # noqa: E402
from multiblimp.condition_taxonomy import GROUP_PREFIXES  # noqa: E402
from multiblimp.agreement_pipeline_utils import (  # noqa: E402
    available_system_memory_bytes, limit_process_memory,
)
from word_order.viz_overview import generate_html_overview_index  # noqa: E402
from multiblimp.config import (  # noqa: E402
    OUTPUT_DECISION_TREES_DIR, HTML_DECISION_TREES_DIR,
    OUTPUT_MINIMAL_PAIRS_DIR, OUTPUT_DIAGNOSTICS_DIR, DEBUG_MODE,
)

INV_FEATURE_ABBREV = {v: k for k, v in FEATURE_ABBREV.items()}

# Source files whose edits change what an index looks like even when none of
# its data inputs did.
_CODE_FILES = [
    os.path.join(REPO_ROOT, "src", "word_order", "viz_deprel.py"),
    os.path.join(REPO_ROOT, "src", "word_order", "html", "html_deprel.py"),
    os.path.join(REPO_ROOT, "src", "sva_trees", "diagnostics.py"),
]


def _npa_identifier_to_target_col(npa_identifier: str) -> str:
    """Inverse of npa.agreement.npa_id -- "HEAD-DET_N" -> "HEAD-DET_Number".
    The on-disk directory name is the abbreviated identifier, not the real
    target_col (run_agreement_pipeline's own argument), so this has to be
    reversed rather than assumed. Relies on FEATURE_ABBREV being injective
    (true today: N/G/P/C/Def/Deg/PT/NT/Pos are all distinct)."""
    roles, abbr = npa_identifier.rsplit("_", 1)
    return f"{roles}_{INV_FEATURE_ABBREV.get(abbr, abbr)}"


def _infer_simplify(save_dir: str) -> bool:
    """Whether a condition was fit with unk collapsed ("unk" label, the
    default) or kept distinct ("+-"/"--", sva_trees' -d flag) -- read off any
    language's saved label distribution, since the flag itself isn't stored."""
    for fn in glob.glob(os.path.join(save_dir, "*_label_distribution.json")):
        try:
            with open(fn) as f:
                labels = set(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
        return not (labels & {"+-", "--"})
    return True


def _sva_spec(target_id: str, deprel: str):
    """(predictor_var, head_label) for an SVA target_id/deprel directory, or
    None if it maps to no known condition. A fixed id (svNa, saPa_single, ...)
    comes from sva_trees.conditions; anything else must be an ad-hoc
    <family><Feature> candidate (e.g. svCase), whose predictor_var follows the
    same head_{deprel}_{feature}_agreement convention."""
    cid = target_id.removesuffix("_single")
    if cid in CONDITIONS:
        cond, feature = CONDITIONS[cid], CONDITIONS[cid].swap_feat
    else:
        family = next((p for p in sorted(GROUP_PREFIXES, key=len, reverse=True)
                       if cid.startswith(p) and len(cid) > len(p)), None)
        if family is None or f"{family}Na" not in CONDITIONS:
            return None
        cond, feature = CONDITIONS[f"{family}Na"], cid[len(family):]
    return f"head_{deprel}_{feature}_agreement", cond.head_label


def built_sva_tasks(has_pairs: bool) -> tuple[list[dict], list[str]]:
    """(tasks, unrecognized dirs) for every SVA condition x deprel with at
    least one fitted language on disk."""
    tasks, unknown = [], []
    for tid_dir in sorted(glob.glob(os.path.join(OUTPUT_DECISION_TREES_DIR, "*"))):
        target_id = os.path.basename(tid_dir)
        if target_id == "npa" or not os.path.isdir(tid_dir):
            continue
        for save_dir in sorted(glob.glob(os.path.join(tid_dir, f"{target_id}_*"))):
            if not glob.glob(os.path.join(save_dir, "*.joblib")):
                continue
            deprel = os.path.basename(save_dir)[len(target_id) + 1:]
            spec = _sva_spec(target_id, deprel)
            if spec is None:
                unknown.append(os.path.relpath(save_dir, OUTPUT_DECISION_TREES_DIR))
                continue
            predictor_var, head_label = spec
            tasks.append(dict(
                kind="sva", name=f"{target_id}/{deprel}", aliases=[target_id],
                save_dir=save_dir,
                pairs_dir=os.path.join(OUTPUT_MINIMAL_PAIRS_DIR, target_id, f"{target_id}_{deprel}"),
                html_index=os.path.join(HTML_DECISION_TREES_DIR, target_id, "index.html"),
                diagnostics_csv=os.path.join(OUTPUT_DIAGNOSTICS_DIR, target_id, f"{target_id}_{deprel}.csv"),
                kwargs=dict(target_id=target_id, deprel=deprel, predictor_var=predictor_var,
                            head_label=head_label, simplify=_infer_simplify(save_dir),
                            has_pairs=has_pairs),
            ))
    return tasks, unknown


def built_npa_tasks(has_pairs: bool) -> list[dict]:
    """One task per NPA target_col with at least one fitted language on disk
    (read off output/decision_trees/npa/*/*.joblib, not npa_config.json,
    which lists CANDIDATE languages, most not yet run)."""
    tasks = []
    for cond_dir in sorted(glob.glob(os.path.join(OUTPUT_DECISION_TREES_DIR, "npa", "*"))):
        if not os.path.isdir(cond_dir) or not glob.glob(os.path.join(cond_dir, "*.joblib")):
            continue
        identifier = os.path.basename(cond_dir)
        target_col = _npa_identifier_to_target_col(identifier)
        tasks.append(dict(
            kind="npa", name=target_col, aliases=[identifier],
            save_dir=cond_dir,
            pairs_dir=os.path.join(OUTPUT_MINIMAL_PAIRS_DIR, "npa", identifier),
            html_index=os.path.join(HTML_DECISION_TREES_DIR, "npa", identifier, "index.html"),
            diagnostics_csv=os.path.join(OUTPUT_DIAGNOSTICS_DIR, "npa", f"{identifier}.csv"),
            kwargs=dict(target_col=target_col, has_pairs=has_pairs),
        ))
    return tasks


def _latest_mtime(*patterns: str) -> float:
    """Max mtime across every file matched by the given glob patterns, or
    -1 if none match."""
    mtimes = [
        os.path.getmtime(p)
        for pattern in patterns
        for p in glob.glob(pattern)
        if os.path.isfile(p)
    ]
    return max(mtimes) if mtimes else -1.0


def needs_refresh(task: dict) -> bool:
    """True unless the condition's index.html (and diagnostics CSV, when it
    has pairs) is newer than every data input AND the index-generating code --
    i.e. unless nothing that feeds it has changed since the last build."""
    outputs = [task["html_index"]]
    if task["kwargs"]["has_pairs"]:
        outputs.append(task["diagnostics_csv"])
    if not all(os.path.exists(p) for p in outputs):
        return True
    output_mtime = min(os.path.getmtime(p) for p in outputs)
    input_mtime = _latest_mtime(
        os.path.join(task["save_dir"], "*.joblib"),
        os.path.join(task["save_dir"], "*.parquet"),
        os.path.join(task["pairs_dir"], "*", "*.parquet"),
        os.path.join(task["pairs_dir"], "*", "meta.json"),
        *_CODE_FILES,
    )
    return input_mtime > output_mtime


def _refresh_one(task: dict) -> tuple[str, bool, str, float]:
    """Worker body: (name, ok, captured log, seconds). Output is captured so
    parallel conditions' logs print as contiguous blocks, not interleaved."""
    start, log = time.time(), io.StringIO()
    ok = True
    with contextlib.redirect_stdout(log):
        try:
            if task["kind"] == "sva":
                refresh_sva_index(**task["kwargs"])
            else:
                refresh_npa_index(**task["kwargs"])
        except Exception:
            ok = False
            traceback.print_exc(file=log)
    return task["name"], ok, log.getvalue(), time.time() - start


def _report(result: tuple[str, bool, str, float], done: int, total: int, verbose: bool) -> bool:
    name, ok, log, seconds = result
    print(f"[{done}/{total}] {'ok  ' if ok else 'FAIL'} {name} ({seconds:.0f}s)", flush=True)
    if verbose or not ok:
        print("".join(f"    {line}\n" for line in log.strip().splitlines()), end="", flush=True)
    return ok


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--condition", "--target-col", "-c", "-t", dest="conditions", action="append",
        help="Limit the sweep to this condition -- repeatable. Matches an SVA "
             "target_id (svNa) or target_id/deprel (svNa/nsubj), or an NPA "
             "target_col (HEAD-DET_Number) or id (HEAD-DET_N). Default: every "
             "condition with at least one fitted language on disk.",
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Refresh every selected condition regardless of needs_refresh "
             "-- use after touching a condition's cache without also "
             "touching its output (e.g. hand-editing a .joblib), which the "
             "mtime check can't otherwise see.",
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="Print which selected conditions would be refreshed and exit "
             "without building or writing anything.",
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=None,
        help="Conditions refreshed in parallel (default min(cpu_count, 8), "
             "never more than the number to refresh; 1 runs inline).",
    )
    parser.add_argument(
        "--max_worker_mem_gb", type=float, default=None,
        help="Per-worker memory cap in GB (default: 80%% of available RAM "
             "split across the workers). Each worker holds every fitted "
             "language of one condition in memory at once.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print each condition's full build log, not just its status line "
             "(failures always print theirs).",
    )
    parser.add_argument(
        "--skip_overview", action="store_true",
        help="Don't rebuild the cross-pipeline decision_trees/index.html.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    has_pairs = not DEBUG_MODE

    sva_tasks, unknown = built_sva_tasks(has_pairs)
    tasks = sva_tasks + built_npa_tasks(has_pairs)
    if unknown:
        print(f"Skipping {len(unknown)} unrecognized SVA dir(s) (no matching condition): {unknown}")

    if args.conditions:
        wanted = set(args.conditions)
        unmatched = sorted(w for w in wanted
                           if not any(w == t["name"] or w in t["aliases"] for t in tasks))
        if unmatched:
            sys.exit(f"Not built on disk, so nothing to refresh: {unmatched}")
        tasks = [t for t in tasks if t["name"] in wanted or wanted & set(t["aliases"])]

    n_sva = sum(t["kind"] == "sva" for t in tasks)
    print(f"{len(tasks)} condition(s) selected ({n_sva} sva, {len(tasks) - n_sva} npa)")

    to_refresh = [t for t in tasks if args.force or needs_refresh(t)]
    if len(to_refresh) < len(tasks):
        print(f"Already up to date, skipping {len(tasks) - len(to_refresh)}")

    if args.list:
        print(f"Would refresh {len(to_refresh)}: {[t['name'] for t in to_refresh]}")
        return 0

    failed = []
    if to_refresh:
        jobs = max(1, min(args.jobs or min(os.cpu_count() or 1, 8), len(to_refresh)))
        print(f"Refreshing {len(to_refresh)} condition(s) with {jobs} worker(s)", flush=True)
        total = len(to_refresh)
        if jobs == 1:
            for done, task in enumerate(to_refresh, 1):
                if not _report(_refresh_one(task), done, total, args.verbose):
                    failed.append(task["name"])
        else:
            if args.max_worker_mem_gb is not None:
                mem_bytes = int(args.max_worker_mem_gb * 1024**3)
            else:
                available = available_system_memory_bytes()
                mem_bytes = int(available * 0.8 / jobs) if available else None
            initializer, initargs = (limit_process_memory, (mem_bytes,)) if mem_bytes else (None, ())
            if mem_bytes:
                print(f"Capping each worker to {mem_bytes / 1024**3:.1f} GB RAM")
            # Largest conditions first, so the pool's tail isn't one slow straggler.
            order = sorted(to_refresh, key=lambda t: -len(glob.glob(os.path.join(t["save_dir"], "*.joblib"))))
            with ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=1,
                                     initializer=initializer, initargs=initargs) as pool:
                futures = {pool.submit(_refresh_one, t): t for t in order}
                for done, future in enumerate(as_completed(futures), 1):
                    task = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:  # worker died (e.g. hit the memory cap)
                        result = (task["name"], False, f"{type(e).__name__}: {e}", 0.0)
                    if not _report(result, done, total, args.verbose):
                        failed.append(task["name"])

    if not args.skip_overview:
        print("\nGenerating cross-pipeline overview index")
        generate_html_overview_index(html_directory=HTML_DECISION_TREES_DIR)

    if failed:
        print(f"\nFailed to refresh {len(failed)}: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
