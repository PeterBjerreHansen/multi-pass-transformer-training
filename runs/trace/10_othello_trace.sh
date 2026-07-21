#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

SEED="${SEED:-1337}"
ARCHITECTURES="${ARCHITECTURES:-transformer memory_tape joint_memory_tape memory_concat memory_update}"
RESULT_ROOT="${RESULT_ROOT:-results/trace/othello}"

extra_args=()
[[ -n "${DEVICE:-}" ]] && extra_args+=(--device "${DEVICE}")
[[ -n "${TRAIN_STEPS:-}" ]] && extra_args+=(--train-steps "${TRAIN_STEPS}")
[[ -n "${EVAL_INTERVAL:-}" ]] && extra_args+=(--eval-interval "${EVAL_INTERVAL}")
[[ -n "${EVAL_BATCHES:-}" ]] && extra_args+=(--eval-batches "${EVAL_BATCHES}")
[[ -n "${BATCH_SIZE:-}" ]] && extra_args+=(--batch-size "${BATCH_SIZE}")
[[ -n "${TOKEN_SELECTION:-}" ]] && extra_args+=(--token-selection "${TOKEN_SELECTION}")
[[ -n "${OTHELLO_DATA_DIR:-}" ]] && extra_args+=(--othello-data-dir "${OTHELLO_DATA_DIR}")
[[ -n "${OTHELLO_TRAIN_GAMES:-}" ]] && extra_args+=(--othello-train-games "${OTHELLO_TRAIN_GAMES}")
[[ -n "${OTHELLO_VAL_GAMES:-}" ]] && extra_args+=(--othello-val-games "${OTHELLO_VAL_GAMES}")

for ARCH in ${ARCHITECTURES}; do
  python -m experiments.train_trace \
    --preset othello_main \
    --architecture "${ARCH}" \
    --seed "${SEED}" \
    --run-dir "${RESULT_ROOT}/${ARCH}/seed_${SEED}" \
    "${extra_args[@]}"
done
