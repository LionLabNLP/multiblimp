"""Render output/overview/stats.json (see build_stats.py) into a self-
contained html/data.html — samples/minimal-pair counts per language, per
condition, and per language x condition, for the sva-dt project.

Run from the repo root: python scripts/overview/render_html.py
(after build_stats.py, or use scripts/overview/run_all.sh for both)
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(REPO_ROOT, "src"))
from agreement_trees.report.html.html_dataset import render_dataset_page  # noqa: E402
from multiblimp.config import HTML_DIR, OUTPUT_OVERVIEW_DIR  # noqa: E402

STATS_PATH = os.path.join(OUTPUT_OVERVIEW_DIR, "stats.json")
OUT_PATH = os.path.join(HTML_DIR, "data.html")


def main():
    with open(STATS_PATH, encoding="utf-8") as f:
        stats = json.load(f)
    html = render_dataset_page(stats)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
