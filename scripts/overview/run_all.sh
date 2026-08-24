#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 scripts/overview/build_stats.py
python3 scripts/overview/render_html.py
