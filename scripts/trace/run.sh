#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
source "${ROOT}/scripts/lib/model_matrix.sh"

TASKS="${TASKS:-shortest_path othello}"
SEEDS="${SEEDS:-1337}"
ARCHITECTURES="${ARCHITECTURES:-transformer memory_tape joint_memory_tape memory_concat memory_add memory_state memory_update}"
RESULT_ROOT="${RESULT_ROOT:-results/trace}"

runtime_args=()
[[ -n "${DEVICE:-}" ]] && runtime_args+=(--device "${DEVICE}")

read -r -a task_matrix <<< "${TASKS}"
read -r -a architecture_matrix <<< "${ARCHITECTURES}"
read -r -a seed_matrix <<< "${SEEDS}"
validate_architecture_matrix "${architecture_matrix[@]}"

for task in "${task_matrix[@]}"; do
  if [[ "${task}" != "shortest_path" && "${task}" != "othello" ]]; then
    echo "invalid trace task: ${task}" >&2
    echo "valid trace tasks: shortest_path othello" >&2
    exit 2
  fi
done

for task in "${task_matrix[@]}"; do
  for architecture in "${architecture_matrix[@]}"; do
    for seed in "${seed_matrix[@]}"; do
      python -m experiments.train_trace \
        --preset "${task}_main" \
        --architecture "${architecture}" \
        --seed "${seed}" \
        --run-dir "${RESULT_ROOT}/${task}/main/${architecture}/seed_${seed}" \
        "${runtime_args[@]+"${runtime_args[@]}"}"
    done
  done
done
