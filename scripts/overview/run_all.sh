#!/usr/bin/env bash
# Finalise the html: refresh every built NPA condition's deprel index and the
# cross-pipeline decision_trees overview (up-to-date conditions are skipped;
# pass --force / --target-col / --list to this script to forward them), then
# build the dataset overview from those pages.
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 scripts/overview/generate_html_indexes.py "$@"
python3 scripts/overview/build_stats.py
python3 scripts/overview/render_html.py
