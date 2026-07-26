#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/lib/local_pilot.sh

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export RESULT_ROOT="${RESULT_ROOT:-results/local_pilots/generation_aligned_training/${RUN_ID}}"
export DEVICE="$(local_pilot_device)"
export SEED="${SEED:-1337}"
export TRAIN_STEPS="${TRAIN_STEPS:-250}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-${TRAIN_STEPS}}"
export EVAL_BATCHES="${EVAL_BATCHES:-1}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export PILOT_PRESET=random_graph_walk_main
export PILOT_ARCHITECTURE=memory_tape
TREATMENT_PROB="${TREATMENT_PROB:-0.25}"
APPEND_MICROBATCH_SIZE="${APPEND_MICROBATCH_SIZE:-4}"
APPEND_HORIZON="${APPEND_HORIZON:-4}"
APPEND_WARMUP_STEPS="${APPEND_WARMUP_STEPS:-50}"
APPEND_RAMP_STEPS="${APPEND_RAMP_STEPS:-50}"

run_trace_pilot_variant p0 \
  --append-train-prob 0
run_trace_pilot_variant treatment \
  --append-train-prob "${TREATMENT_PROB}" \
  --append-train-microbatch-size "${APPEND_MICROBATCH_SIZE}" \
  --append-train-horizon "${APPEND_HORIZON}" \
  --append-train-warmup-steps "${APPEND_WARMUP_STEPS}" \
  --append-train-ramp-steps "${APPEND_RAMP_STEPS}"

python scripts/summarize_learning_runs.py --root "${RESULT_ROOT}"
