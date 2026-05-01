#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

# Model
PRESET="${PRESET:-othello_main}"
ARCHES="${ARCHES:-memory_update}" # transformer memory_tape memory_update memory_concat
MEMORY_TAPE_GATE="${MEMORY_TAPE_GATE:-scalar}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"

# Optimization
LR="${LR:-0.0003}"
STEPS="${STEPS:-500_000}"
BATCH_SIZE="${BATCH_SIZE:-128}"

EVAL_BATCHES="${EVAL_BATCHES:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"

# Misc.
DEVICE="${DEVICE:-mps}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

for ARCH in $ARCHES; do
  CMD=(
    python3 -m experiments.train_trace
    --preset "$PRESET"
    --architecture "$ARCH"
    --device "$DEVICE"
    --batch-size "$BATCH_SIZE"
    --train-steps "$STEPS"
    --eval-interval "$EVAL_INTERVAL"
    --eval-batches "$EVAL_BATCHES"
    --lr "$LR"
    --run-dir "$RESULTS_ROOT/trace/othello/$ARCH/$RUN_ID"
  )
  if [[ "$ARCH" == "memory_tape" ]]; then
    CMD+=(--memory-tape-gate "$MEMORY_TAPE_GATE")
  fi

  "${CMD[@]}"
done
