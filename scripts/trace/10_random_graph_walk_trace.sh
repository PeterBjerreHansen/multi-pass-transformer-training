#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
source "${ROOT}/scripts/lib/model_matrix.sh"

SEEDS="${SEEDS:-${SEED:-1337}}"
ARCHITECTURES="${ARCHITECTURES:-transformer memory_tape joint_memory_tape memory_concat memory_add memory_state memory_update}"
RESULT_ROOT="${RESULT_ROOT:-results/trace/random_graph_walk}"

runtime_args=()
[[ -n "${DEVICE:-}" ]] && runtime_args+=(--device "${DEVICE}")

read -r -a architecture_matrix <<< "${ARCHITECTURES}"
read -r -a seed_matrix <<< "${SEEDS}"
validate_architecture_matrix "${architecture_matrix[@]}"

for ARCH in "${architecture_matrix[@]}"; do
  for seed in "${seed_matrix[@]}"; do
    python -m experiments.train_trace \
      --preset random_graph_walk_main \
      --architecture "${ARCH}" \
      --seed "${seed}" \
      --run-dir "${RESULT_ROOT}/${ARCH}/seed_${seed}" \
      "${runtime_args[@]+"${runtime_args[@]}"}"
  done
done
