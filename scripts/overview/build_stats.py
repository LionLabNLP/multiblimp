"""Aggregate samples / minimal-pair counts across every condition into one JSON blob (output/overview/stats.json)
for the dataset overview. The source is html/decision_trees/: each condition's index.html LANGUAGES blob, the
per-language diagnostics dict of agreement_trees.pairs.diagnostics.diagnostics_row_to_json. Conditions and NP subgroups are
discovered from what is on disk (_discover_conditions), so a rebuilt condition appears automatically;
scripts/overview/generate_html_indexes.py (re)builds those pages without refitting.

Run from the repo root: python scripts/overview/build_stats.py
"""
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_repo_root, "src"))

from agreement_trees.report.overview import build_stats

if __name__ == "__main__":
    build_stats()
