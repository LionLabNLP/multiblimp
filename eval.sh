#!/usr/bin/env bash
# Usage: ./eval.sh <hf model id> [gpu] [extra eval_model.py args]
# Scores the benchmark in the background; the log is output/model_results/logs/<model>__<dtype>_run.log. DTYPE=float32 overrides the default.
# WAIT=1 runs in the foreground (for sequential loops).
set -euo pipefail
cd "$(dirname "$0")"

MODEL=${1:?usage: ./eval.sh <hf model id> [gpu] [extra eval_model.py args]}
GPU=${2:-0}
shift $(( $# < 2 ? $# : 2 ))
EXTRA=("$@")
DTYPE=${DTYPE:-bfloat16}

LOG=output/model_results/logs/${MODEL//\//__}__${DTYPE}_run.log
mkdir -p "$(dirname "$LOG")"

export CUDA_VISIBLE_DEVICES=$GPU
run() {
  nohup bash -c '
    echo "start: $(date "+%F %T")"
    status=0
    python3 scripts/benchmark/eval_model.py "$@" || status=$?
    echo "end: $(date "+%F %T") (exit $status)"
    exit $status
  ' _ --model "$MODEL" --dtype "$DTYPE" --softmax_dtype float32 --batch_size auto --max_batch_size 8 ${EXTRA[@]+"${EXTRA[@]}"} \
    >> "$LOG" 2>&1
}

if [[ -n ${WAIT:-} ]]; then
  run
else
  run &
  echo "started $MODEL on GPU $GPU (pid $!), log: $LOG"
fi
