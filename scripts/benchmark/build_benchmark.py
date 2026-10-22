"""Downsample the correct_swaps of the minimal pairs into the benchmark directory.

Stage 2 of the benchmark: run after the pairs are built (and, for pair directories from before pair creation wrote the
written forms, after validate_surface.py). Keeps the <cond>/<cond>_<deprel>/<lang>/ layout, so
the benchmark reads like the pipeline output; manifest.json records the settings and the counts before and after.

Usage (any directory):
  python3 build_benchmark.py --max_per_cond 1000
"""
import argparse
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_repo_root, "src"))

from agreement_trees.eval.benchmark import build_benchmark
from multiblimp.config import OUTPUT_BENCHMARK_DIR, OUTPUT_MINIMAL_PAIRS_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_dir", default=OUTPUT_MINIMAL_PAIRS_DIR)
    parser.add_argument("--out_dir", default=OUTPUT_BENCHMARK_DIR)
    parser.add_argument("--langs", nargs="+", default=None)
    parser.add_argument("--conds", nargs="+", default=None)
    parser.add_argument("--max_per_cond", type=int, default=1000, help="per lang x cond")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    build_benchmark(args.pairs_dir, args.out_dir, args.langs, args.conds, args.max_per_cond, args.seed)
