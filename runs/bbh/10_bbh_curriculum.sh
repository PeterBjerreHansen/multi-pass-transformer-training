#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

# TASK remains a backwards-compatible way to select one task. TASKS is the
# multi-task interface used by the full launcher.
TASKS="${TASKS:-${TASK:-permutation tracking pointer_chasing state_machine}}"
SEED="${SEED:-1337}"
ARCHITECTURES="${ARCHITECTURES:-transformer memory_tape joint_memory_tape memory_concat memory_update}"
RESULT_ROOT="${RESULT_ROOT:-results/bbh}"

extra_args=()
[[ -n "${DEVICE:-}" ]] && extra_args+=(--device "${DEVICE}")
[[ -n "${TRAIN_STEPS:-}" ]] && extra_args+=(--train-steps "${TRAIN_STEPS}")
[[ -n "${EVAL_INTERVAL:-}" ]] && extra_args+=(--eval-interval "${EVAL_INTERVAL}")
[[ -n "${EVAL_BATCHES:-}" ]] && extra_args+=(--eval-batches "${EVAL_BATCHES}")
[[ -n "${BATCH_SIZE:-}" ]] && extra_args+=(--batch-size "${BATCH_SIZE}")
[[ -n "${MEMORY_GATE_INIT:-}" ]] && extra_args+=(--memory-gate-init "${MEMORY_GATE_INIT}")

for task in ${TASKS}; do
  for ARCH in ${ARCHITECTURES}; do
    python -m experiments.train_bbh \
      --preset "${task}_main" \
      --architecture "${ARCH}" \
      --seed "${SEED}" \
      --run-dir "${RESULT_ROOT}/${task}/${ARCH}/seed_${SEED}" \
      "${extra_args[@]}"
  done
done
