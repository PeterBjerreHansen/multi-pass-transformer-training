#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

SEEDS="${SEEDS:-${SEED:-1337}}"
ARCHITECTURES="${ARCHITECTURES:-transformer memory_tape joint_memory_tape memory_concat memory_update}"
RESULT_ROOT="${RESULT_ROOT:-results/trace/othello}"

runtime_args=()
[[ -n "${DEVICE:-}" ]] && runtime_args+=(--device "${DEVICE}")

for ARCH in ${ARCHITECTURES}; do
  for seed in ${SEEDS}; do
    python -m experiments.train_trace \
      --preset othello_main \
      --architecture "${ARCH}" \
      --seed "${seed}" \
      --run-dir "${RESULT_ROOT}/${ARCH}/seed_${seed}" \
      "${runtime_args[@]}"
  done
done
