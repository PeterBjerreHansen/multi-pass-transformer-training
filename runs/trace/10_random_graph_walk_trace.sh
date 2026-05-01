#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# Model
PRESET="${PRESET:-random_graph_walk_main}"
ARCHES="${ARCHES:-transformer memory_tape}" # transformer memory_tape memory_update memory_concat
MEMORY_TAPE_GATE="${MEMORY_TAPE_GATE:-scalar}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"

# Optimization
LR="${LR:-0.0001}"
STEPS="${STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-64}"

# Data/logging
EVAL_BATCHES="${EVAL_BATCHES:-4}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"

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
    --run-dir "$RESULTS_ROOT/trace/random_graph_walk/$ARCH/$RUN_ID"
  )
  if [[ "$ARCH" == "memory_tape" ]]; then
    CMD+=(--memory-tape-gate "$MEMORY_TAPE_GATE")
  fi

  "${CMD[@]}"
done
