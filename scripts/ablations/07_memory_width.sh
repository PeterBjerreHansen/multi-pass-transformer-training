#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

DEVICE="${DEVICE:-mps}"
SEEDS="${SEEDS:-1337 2027 4099}"
RESULT_ROOT="${RESULT_ROOT:-results/ablations/07_memory_width}"
TRAIN_STEPS=50000

run_eval() {
  local run_dir="$1" seed="$2"
  for mode in recompute append_recurrent; do
    python -m experiments.eval_trace_drift --input-run-dir "${run_dir}" \
      --run-dir "${run_dir}/drift/${mode}" --inference-mode "${mode}" \
      --token-selection argmax --device "${DEVICE}" --seed "${seed}"
  done
  python -m experiments.eval_diagnostics --input-run-dir "${run_dir}" \
    --output "${run_dir}/diagnostics.json" --device "${DEVICE}" --seed "${seed}"
}

for seed in ${SEEDS}; do
  for variant in legacy dm128 dm64 dm32; do
    run_dir="${RESULT_ROOT}/${variant}/seed_${seed}"
    width_args=()
    case "${variant}" in
      dm128) width_args+=(--n-memory-embd 128) ;;
      dm64) width_args+=(--n-memory-embd 64) ;;
      dm32) width_args+=(--n-memory-embd 32) ;;
    esac
    python -m experiments.train_trace --preset random_graph_walk_main --architecture memory_tape \
      --token-selection argmax --train-steps "${TRAIN_STEPS}" --device "${DEVICE}" \
      --seed "${seed}" --run-dir "${run_dir}" "${width_args[@]}"
    run_eval "${run_dir}" "${seed}"
  done
done

python -m experiments.summarize_ablation --root "${RESULT_ROOT}" --control dm128 \
  --variants legacy dm64 dm32 --recommendation-mode pareto
