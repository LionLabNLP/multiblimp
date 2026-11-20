"""Sample the benchmark from the correct_swaps of the minimal pairs: up to --n_items pairs per language, balanced
across its conditions and over the markers listed in multiblimp.downsample.

Stage 2 of the benchmark: run after the pairs are built (and, for pair directories from before pair creation wrote the
written forms, after validate_surface.py). Writes two files per language: <Language>.parquet (the sentences, the
phenomenon and the balance markers) and features/<Language>.parquet (every column), both keyed by pair_id, plus
meta/<Language>.json (each condition's pipeline meta.json without its HTML examples); manifest.json records the
settings and the counts before and after, stats.json feeds render_page.py.

Usage (any directory):
  python3 build_benchmark.py --n_items 2000
"""
import argparse
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_repo_root, "src"))

from agreement_trees.eval.benchmark import build_benchmark
from multiblimp.config import OUTPUT_BENCHMARK_DIR, OUTPUT_MINIMAL_PAIRS_DIR
from multiblimp.downsample import LEMMA_CAP, MAX_TOKENS, MIN_PAIRS, N_ITEMS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_dir", default=OUTPUT_MINIMAL_PAIRS_DIR)
    parser.add_argument("--out_dir", default=OUTPUT_BENCHMARK_DIR)
    parser.add_argument("--langs", nargs="+", default=None)
    parser.add_argument("--conds", nargs="+", default=None)
    parser.add_argument("--n_items", type=int, default=N_ITEMS, help="pairs per language, across its conditions")
    parser.add_argument("--lemma_cap", type=float, default=LEMMA_CAP, help="largest share of a language's sample one lemma may take")
    parser.add_argument("--min_pairs", type=int, default=MIN_PAIRS, help="leave out languages with fewer pairs in total")
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS, help="pairs whose sentence has more tokens are never drawn")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    build_benchmark(args.pairs_dir, args.out_dir, args.langs, args.conds, args.n_items, args.lemma_cap, args.seed, args.min_pairs,
                    args.max_tokens)
