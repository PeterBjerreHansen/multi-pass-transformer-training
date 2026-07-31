#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

DEVICE="${DEVICE:-mps}"
SEEDS="${SEEDS:-1337 2027 4099}"
RESULT_ROOT="${RESULT_ROOT:-results/ablations/01_variable_depth}"
TRAIN_STEPS="${TRAIN_STEPS:-200000}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-4000}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-4}"
FINAL_EVAL_BATCHES="${FINAL_EVAL_BATCHES:-64}"
DIAGNOSTIC_BATCHES="${DIAGNOSTIC_BATCHES:-4}"

run_eval() {
  local run_dir="$1" seed="$2"
  for mode in recompute append_recurrent; do
    python -m experiments.eval_trace --input-run-dir "${run_dir}" \
      --output-dir "${run_dir}/drift/${mode}" --inference-mode "${mode}" \
      --checkpoint best --eval-batches "${FINAL_EVAL_BATCHES}" \
      --token-selection argmax --device "${DEVICE}" --seed "${seed}"
  done
  python -m experiments.diagnose_memory --input-run-dir "${run_dir}" \
    --output "${run_dir}/diagnostics.json" --checkpoint best \
    --eval-batches "${DIAGNOSTIC_BATCHES}" --extra-passes 6 \
    --device "${DEVICE}" --seed "${seed}"
}

for seed in ${SEEDS}; do
  control_dir="${RESULT_ROOT}/fixed_k4/seed_${seed}"
  python -m experiments.train_trace --preset shortest_path_main --architecture memory_tape \
    --pass-loss-weights 0 0 0.3 0.7 --token-selection argmax --train-steps "${TRAIN_STEPS}" \
    --lr-warmup-steps "${LR_WARMUP_STEPS}" --lr-decay-steps "${TRAIN_STEPS}" \
    --eval-batches "${TRAIN_EVAL_BATCHES}" \
    --device "${DEVICE}" --seed "${seed}" --run-dir "${control_dir}"
  run_eval "${control_dir}" "${seed}"

  treatment_dir="${RESULT_ROOT}/uniform_k2_k6/seed_${seed}"
  python -m experiments.train_trace --preset shortest_path_main --architecture memory_tape \
    --pass-loss-weights 0 0 0.3 0.7 --train-pass-range 2 6 \
    --sampled-tail-loss-weights 0.3 0.7 --token-selection argmax --train-steps "${TRAIN_STEPS}" \
    --lr-warmup-steps "${LR_WARMUP_STEPS}" --lr-decay-steps "${TRAIN_STEPS}" \
    --eval-batches "${TRAIN_EVAL_BATCHES}" \
    --device "${DEVICE}" --seed "${seed}" --run-dir "${treatment_dir}"
  run_eval "${treatment_dir}" "${seed}"
done

python -m experiments.summarize_ablation --root "${RESULT_ROOT}" --control fixed_k4 \
  --variants uniform_k2_k6 --recommendation-mode quality-only
