#!/usr/bin/env bash
# Usage: scripts/benchmark/eval_goldfish.sh [gpu] [--dry-run] [-- extra eval_model.py args]
# Scores each Goldfish model on its own language only, one after the other.
# Languages without output/benchmark/<Language>.parquet are skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU=${1:-0}; shift $(( $# > 0 ? 1 : 0 ))
DRY=""; [[ ${1:-} == --dry-run ]] && { DRY=1; shift; }
[[ ${1:-} == -- ]] && shift
EXTRA=("$@")

# Model per language: resources/goldfish_map.py (GOLDFISH_MAP)
MODELS=$(python3 -I -c '
import runpy
for lang, v in runpy.run_path("resources/goldfish_map.py")["GOLDFISH_MAP"].items():
    print(v["model"], lang)
')

SKIPPED=()
while read -r model lang; do
  [[ -z $model ]] && continue
  if [[ ! -f output/benchmark/$lang.parquet ]]; then
    SKIPPED+=("$lang")
    continue
  fi
  if [[ -n $DRY ]]; then
    echo "./eval.sh $model $GPU --langs $lang ${EXTRA[*]-}"
  else
    WAIT=1 ./eval.sh "$model" "$GPU" --langs "$lang" ${EXTRA[@]+"${EXTRA[@]}"} || echo "FAILED $model ($lang)" >&2
  fi
done <<< "$MODELS"

(( ${#SKIPPED[@]} )) && echo "not in benchmark, skipped: ${SKIPPED[*]}" >&2
exit 0
