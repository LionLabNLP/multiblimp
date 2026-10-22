"""Rebuild the surface form of multiword tokens in correct_swaps pairs (see agreement_trees.eval.surface).

Pair creation now writes the surface forms itself (sen_str, wrong_sen_str, surface_status, and the invalid_surface bucket); this
backfills pair directories built before that, and reports on any directory. Writes <target dir>/surface_strings.parquet
(pair_key, sen_str, wrong_sen_str, surface_status) for the pairs in sentences with multiword tokens, dropped pairs marked
"dropped:...", and a per-target report CSV; build_benchmark.py applies the file where it exists.

Usage (any directory):
  python3 validate_surface.py --langs French German
"""
import argparse
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_repo_root, "src"))

from agreement_trees.eval.surface import run_validation
from multiblimp.config import OUTPUT_DIAGNOSTICS_DIR, OUTPUT_MINIMAL_PAIRS_DIR, RESOURCE_DIR, UD_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_dir", default=OUTPUT_MINIMAL_PAIRS_DIR)
    parser.add_argument("--ud_dir", default=os.path.join(RESOURCE_DIR, UD_PATH))
    parser.add_argument("--langs", nargs="+", default=None)
    parser.add_argument("--conds", nargs="+", default=None)
    parser.add_argument("--report", default=os.path.join(OUTPUT_DIAGNOSTICS_DIR, "mwt", "surface_report.csv"))
    args = parser.parse_args()

    run_validation(args.pairs_dir, args.ud_dir, args.langs, args.conds, args.report)
