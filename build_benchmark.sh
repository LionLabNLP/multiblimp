#!/usr/bin/env bash
# Pairs -> benchmark: run after setup.sh and cache_data.sh.
set -euo pipefail
cd "$(dirname "$0")"

# build_benchmark.py never clears its output directory, so stale files from an earlier build would stay
if [ -d output/benchmark ]; then
  echo "output/benchmark already exists; move or delete it first (e.g. mv output/benchmark output/benchmark_prev)" >&2
  exit 1
fi

# Decision trees, minimal pairs and pages for every scanned target (finished languages are skipped), then the indexes
(cd scripts/agreement_trees && python3 fit_all.py)

# Benchmark sample from the pairs
python3 scripts/benchmark/build_benchmark.py

# Overview indexes, dataset overview and the benchmark page
bash scripts/overview/run_all.sh
