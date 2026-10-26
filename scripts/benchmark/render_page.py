"""Render the benchmark page (html/index.html) from the stats.json and manifest.json build_benchmark.py wrote.

Stage 3 of the benchmark, after build_benchmark.py. Also writes pairs/<Language>.json next to the page, the pair browser's data.

Usage (any directory):
  python3 render_page.py
"""
import argparse
import json
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_repo_root, "src"))

from agreement_trees.report.html.html_benchmark import render_benchmark_page, write_pair_files
from multiblimp.config import HTML_DIR, OUTPUT_BENCHMARK_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_dir", default=OUTPUT_BENCHMARK_DIR, help="build_benchmark.py output")
    parser.add_argument("--out", default=os.path.join(HTML_DIR, "index.html"))
    parser.add_argument("--no_pairs", action="store_true", help="do not (re)write the per-language pair files next to the page")
    args = parser.parse_args()

    with open(os.path.join(args.bench_dir, "stats.json"), encoding="utf-8") as f:
        stats = json.load(f)
    with open(os.path.join(args.bench_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(render_benchmark_page(stats, manifest))
    print(f"Wrote {args.out}")
    if not args.no_pairs:
        counts = write_pair_files(args.bench_dir, os.path.join(os.path.dirname(args.out), "pairs"))
        print(f"Wrote {len(counts)} pair files ({sum(counts.values())} pairs) to {os.path.join(os.path.dirname(args.out), 'pairs')}")
