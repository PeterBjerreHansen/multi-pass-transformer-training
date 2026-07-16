#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-mps}"
SEEDS="${SEEDS:-1337 2027 4099}"
TRAIN_STEPS="${TRAIN_STEPS:-50000}"
RESULT_ROOT="${RESULT_ROOT:-results/ablations/13_memory_read_layers}"

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
  for pattern in all early middle late; do
    run_dir="${RESULT_ROOT}/${pattern}/seed_${seed}"
    python -m experiments.train_trace --preset random_graph_walk_main --architecture memory_tape \
      --memory-read-pattern "${pattern}" --token-selection argmax --train-steps "${TRAIN_STEPS}" \
      --device "${DEVICE}" --seed "${seed}" --run-dir "${run_dir}"
    run_eval "${run_dir}" "${seed}"
  done
done

python -m experiments.summarize_ablation --root "${RESULT_ROOT}" --control all \
  --variants early middle late --recommendation-mode pareto
