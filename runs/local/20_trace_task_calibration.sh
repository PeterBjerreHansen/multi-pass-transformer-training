#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
source scripts/lib/local_pilot.sh

DEVICE="$(local_pilot_device)"
SEEDS="${SEEDS:-1337}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
EVAL_BATCHES="${EVAL_BATCHES:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ARCHITECTURES="${ARCHITECTURES:-transformer memory_tape}"
TASKS="${TASKS:-random_graph_walk shortest_path}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-results/local_pilots/trace_calibration/${RUN_ID}}"
MIN_RELATIVE_LOSS_DROP="${MIN_RELATIVE_LOSS_DROP:-0.05}"

for task in ${TASKS}; do
  for architecture in ${ARCHITECTURES}; do
    for seed in ${SEEDS}; do
      PILOT_PRESET="${task}_main" \
      PILOT_ARCHITECTURE="${architecture}" \
      DEVICE="${DEVICE}" \
      SEED="${seed}" \
      TRAIN_STEPS="${TRAIN_STEPS}" \
      EVAL_INTERVAL="${EVAL_INTERVAL}" \
      EVAL_BATCHES="${EVAL_BATCHES}" \
      BATCH_SIZE="${BATCH_SIZE}" \
      RESULT_ROOT="${RESULT_ROOT}" \
        run_trace_pilot_variant "${task}/${architecture}"
    done
  done
done

python scripts/summarize_learning_runs.py \
  --root "${RESULT_ROOT}" \
  --strict \
  --min-relative-loss-drop "${MIN_RELATIVE_LOSS_DROP}"
