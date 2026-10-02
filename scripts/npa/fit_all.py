"""Fit every NPA target_col in npa_config.json with fit_candidate.py, then
rebuild the overview indexes. Named to match sva_trees/fit_all.py (the
sva_trees analogue, which similarly drives sva_trees/fit_candidate.py).

Run from scripts/npa. Per-treebank trees are on by default; this driver also
passes --include_excluded to every target_col (excluded treebanks' rows live
in the same np_instances parquet as the selected ones and are split off at
read time). fit_candidate.py extracts a missing np_instances parquet on
demand, but this sweep driver passes --no_extract by default, so a full run
never silently starts a treebank-parsing sweep: languages without a parquet
are skipped (and --recache acts as --refit). Pass --extract here to allow it.
Anything else on the command line is passed through to
every fit_candidate.py call, e.g.
  python3 fit_all.py --langs English German
  python3 fit_all.py --target_cols HEAD-DET_Number HEAD-ADJ_Gender
"""
import argparse
import json
import subprocess
import sys

DEFAULT_FLAGS = ["--include_excluded"]
NO_INDEX_FLAG = ["--no_index"]  # indexes are built once, in parallel, after the loop
CONFIG = "../../resources/npa_config.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target_cols", nargs="*", default=None,
                        help="Default: every target_col in npa_config.json")
    parser.add_argument("--extract", action="store_true",
                        help="Let fit_candidate.py extract missing np_instances "
                             "parquets (and re-extract on --recache); off by default")
    parser.add_argument("--skip_indexes", action="store_true",
                        help="Build no indexes at all: every fit gets --no_index and "
                             "scripts/overview/generate_html_indexes.py is not run "
                             "afterwards (run it, or overview/run_all.sh, yourself)")
    parser.add_argument("--index_jobs", type=int, default=None,
                        help="Parallel workers for the final index build (default: "
                             "generate_html_indexes.py's own, min(cpu_count, 8)); "
                             "separate from -j, which parallelises languages "
                             "within each fit")
    args, passthrough = parser.parse_known_args()

    target_cols = args.target_cols
    if target_cols is None:
        with open(CONFIG) as f:
            target_cols = sorted(json.load(f))

    flags = DEFAULT_FLAGS if args.extract else [*DEFAULT_FLAGS, "--no_extract"]
    flags = [*flags, *NO_INDEX_FLAG]
    failed = []
    for target_col in target_cols:
        cmd = [sys.executable, "fit_candidate.py", "--target_col", target_col, *flags, *passthrough]
        print(f"\n=== {' '.join(cmd)}", flush=True)
        if subprocess.run(cmd).returncode:
            failed.append(target_col)

    if not args.skip_indexes:
        # Every fit above skipped its own index (--no_index); this builds all
        # of them once, in parallel, plus the cross-pipeline overview.
        cmd = [sys.executable, "../overview/generate_html_indexes.py"]
        if args.index_jobs:
            cmd += ["-j", str(args.index_jobs)]
        print(f"\n=== {' '.join(cmd)}", flush=True)
        subprocess.run(cmd)

    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
