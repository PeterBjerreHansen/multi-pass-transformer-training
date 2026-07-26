#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

DEVICE="${DEVICE:-mps}"
SEEDS="${SEEDS:-1337 2027 4099}"
RESULT_ROOT="${RESULT_ROOT:-results/ablations/01_variable_depth}"
TRAIN_STEPS=50000

run_eval() {
  local run_dir="$1" seed="$2"
  for mode in recompute append_recurrent; do
    python -m experiments.eval_trace_drift --input-run-dir "${run_dir}" \
      --run-dir "${run_dir}/drift/${mode}" --inference-mode "${mode}" \
      --token-selection argmax --device "${DEVICE}" --seed "${seed}"
  done
  python -m experiments.eval_diagnostics --input-run-dir "${run_dir}" \
    --output "${run_dir}/diagnostics.json" --extra-passes 6 --device "${DEVICE}" --seed "${seed}"
}

for seed in ${SEEDS}; do
  control_dir="${RESULT_ROOT}/fixed_k4/seed_${seed}"
  python -m experiments.train_trace --preset random_graph_walk_main --architecture memory_tape \
    --pass-loss-weights 0 0 0.3 0.7 --token-selection argmax --train-steps "${TRAIN_STEPS}" \
    --device "${DEVICE}" --seed "${seed}" --run-dir "${control_dir}"
  run_eval "${control_dir}" "${seed}"

  treatment_dir="${RESULT_ROOT}/uniform_k2_k6/seed_${seed}"
  python -m experiments.train_trace --preset random_graph_walk_main --architecture memory_tape \
    --pass-loss-weights 0 0 0.3 0.7 --train-pass-range 2 6 \
    --sampled-tail-loss-weights 0.3 0.7 --token-selection argmax --train-steps "${TRAIN_STEPS}" \
    --device "${DEVICE}" --seed "${seed}" --run-dir "${treatment_dir}"
  run_eval "${treatment_dir}" "${seed}"
done

python -m experiments.summarize_ablation --root "${RESULT_ROOT}" --control fixed_k4 \
  --variants uniform_k2_k6 --recommendation-mode quality-only
