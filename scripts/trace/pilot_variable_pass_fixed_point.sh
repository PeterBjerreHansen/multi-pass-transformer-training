#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
source scripts/lib/local_pilot.sh

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export RESULT_ROOT="${RESULT_ROOT:-results/local_pilots/01_variable_pass_fixed_point/${RUN_ID}}"
export DEVICE="$(local_pilot_device)"
export SEED="${SEED:-1337}"
export TRAIN_STEPS="${TRAIN_STEPS:-250}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-${TRAIN_STEPS}}"
export EVAL_BATCHES="${EVAL_BATCHES:-1}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export PILOT_PRESET=shortest_path_main
export PILOT_ARCHITECTURE=memory_tape

run_trace_pilot_variant fixed_k4 \
  --train-pass-mode fixed --eval-pass-mode fixed \
  --min-n-pass 2 --max-n-pass 4 --pass-loss-weights 0 0 0 1
run_trace_pilot_variant uniform_k2_k6 \
  --train-pass-mode uniform --eval-pass-mode fixed \
  --min-n-pass 2 --max-n-pass 6
run_trace_pilot_variant fixed_point_k2_k6 \
  --train-pass-mode fixed_point --eval-pass-mode fixed \
  --min-n-pass 2 --max-n-pass 6 \
  --fixed-point-residual-threshold "${RESIDUAL_THRESHOLD:-0.1}" \
  --fixed-point-kl-threshold "${LOGIT_KL_THRESHOLD:-0.001}"

python scripts/summarize_learning_runs.py --root "${RESULT_ROOT}"
