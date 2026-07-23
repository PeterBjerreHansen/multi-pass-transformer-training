#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

# TASK remains a backwards-compatible way to select one task. TASKS is the
# multi-task interface used by the full launcher.
TASKS="${TASKS:-${TASK:-permutation tracking pointer_chasing state_machine}}"
SEEDS="${SEEDS:-${SEED:-1337}}"
ARCHITECTURES="${ARCHITECTURES:-transformer memory_tape joint_memory_tape memory_concat memory_update}"
RESULT_ROOT="${RESULT_ROOT:-results/bbh}"

runtime_args=()
[[ -n "${DEVICE:-}" ]] && runtime_args+=(--device "${DEVICE}")

for task in ${TASKS}; do
  for ARCH in ${ARCHITECTURES}; do
    for seed in ${SEEDS}; do
      python -m experiments.train_bbh \
        --preset "${task}_main" \
        --architecture "${ARCH}" \
        --seed "${seed}" \
        --run-dir "${RESULT_ROOT}/${task}/${ARCH}/seed_${seed}" \
        "${runtime_args[@]}"
    done
  done
done
