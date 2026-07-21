#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

TASK="${TASK:-pointer_chasing}"
SEED="${SEED:-1337}"
ARCHITECTURES="${ARCHITECTURES:-transformer memory_tape joint_memory_tape memory_concat memory_update}"
RESULT_ROOT="${RESULT_ROOT:-results/bbh/${TASK}}"

extra_args=()
[[ -n "${DEVICE:-}" ]] && extra_args+=(--device "${DEVICE}")
[[ -n "${TRAIN_STEPS:-}" ]] && extra_args+=(--train-steps "${TRAIN_STEPS}")
[[ -n "${EVAL_INTERVAL:-}" ]] && extra_args+=(--eval-interval "${EVAL_INTERVAL}")
[[ -n "${EVAL_BATCHES:-}" ]] && extra_args+=(--eval-batches "${EVAL_BATCHES}")
[[ -n "${BATCH_SIZE:-}" ]] && extra_args+=(--batch-size "${BATCH_SIZE}")

for ARCH in ${ARCHITECTURES}; do
  python -m experiments.train_bbh \
    --preset "${TASK}_main" \
    --architecture "${ARCH}" \
    --seed "${SEED}" \
    --run-dir "${RESULT_ROOT}/${ARCH}/seed_${SEED}" \
    "${extra_args[@]}"
done
