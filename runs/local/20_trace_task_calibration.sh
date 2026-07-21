#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
source scripts/lib/local_pilot.sh

DEVICE="$(local_pilot_device)"
SEEDS="${SEEDS:-1337}"
TRAIN_STEPS="${TRAIN_STEPS:-50000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-${EVAL_BATCHES:-2}}"
QUAL_EVAL_BATCHES="${QUAL_EVAL_BATCHES:-16}"
DIAGNOSTIC_EVAL_BATCHES="${DIAGNOSTIC_EVAL_BATCHES:-${TRAIN_EVAL_BATCHES}}"
MIN_QUAL_EXAMPLES="${MIN_QUAL_EXAMPLES:-256}"
ARCHITECTURES="${ARCHITECTURES:-transformer memory_tape}"
TASKS="${TASKS:-random_graph_walk shortest_path}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-results/local_pilots/trace_calibration/${RUN_ID}}"
MIN_RELATIVE_LOSS_DROP="${MIN_RELATIVE_LOSS_DROP:-0.05}"

qualification_examples=$((BATCH_SIZE * QUAL_EVAL_BATCHES))
if (( qualification_examples < MIN_QUAL_EXAMPLES )); then
  printf 'qualification uses %d examples; require at least %d (BATCH_SIZE * QUAL_EVAL_BATCHES)\n' \
    "${qualification_examples}" "${MIN_QUAL_EXAMPLES}" >&2
  exit 2
fi

for task in ${TASKS}; do
  for architecture in ${ARCHITECTURES}; do
    for seed in ${SEEDS}; do
      PILOT_PRESET="${task}_main" \
      PILOT_ARCHITECTURE="${architecture}" \
      DEVICE="${DEVICE}" \
      SEED="${seed}" \
      TRAIN_STEPS="${TRAIN_STEPS}" \
      EVAL_INTERVAL="${EVAL_INTERVAL}" \
      TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES}" \
      QUAL_EVAL_BATCHES="${QUAL_EVAL_BATCHES}" \
      DIAGNOSTIC_EVAL_BATCHES="${DIAGNOSTIC_EVAL_BATCHES}" \
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
