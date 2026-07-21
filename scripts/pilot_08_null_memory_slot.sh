#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/lib/local_pilot.sh

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export RESULT_ROOT="${RESULT_ROOT:-results/local_pilots/08_null_memory_slot/${RUN_ID}}"
export DEVICE="$(local_pilot_device)"
export SEED="${SEED:-1337}"
export TRAIN_STEPS="${TRAIN_STEPS:-250}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-${TRAIN_STEPS}}"
export EVAL_BATCHES="${EVAL_BATCHES:-1}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export PILOT_PRESET=random_graph_walk_main
export PILOT_ARCHITECTURE=memory_tape

run_trace_pilot_variant null_off \
  --null-memory-slot off
run_trace_pilot_variant null_on \
  --null-memory-slot on

python scripts/summarize_learning_runs.py --root "${RESULT_ROOT}"
