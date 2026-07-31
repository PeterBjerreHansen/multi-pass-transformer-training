#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

DEVICE="${DEVICE:-mps}"
SEEDS="${SEEDS:-1337 2027 4099}"
RESULT_ROOT="${RESULT_ROOT:-results/ablations/memory_read_layers}"
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
    --eval-batches "${DIAGNOSTIC_BATCHES}" --device "${DEVICE}" --seed "${seed}"
}

for seed in ${SEEDS}; do
  for pattern in all early middle late; do
    run_dir="${RESULT_ROOT}/${pattern}/seed_${seed}"
    python -m experiments.train_trace --preset shortest_path_main --architecture memory_tape \
      --memory-read-pattern "${pattern}" --token-selection argmax --train-steps "${TRAIN_STEPS}" \
      --lr-warmup-steps "${LR_WARMUP_STEPS}" --lr-decay-steps "${TRAIN_STEPS}" \
      --eval-batches "${TRAIN_EVAL_BATCHES}" \
      --device "${DEVICE}" --seed "${seed}" --run-dir "${run_dir}"
    run_eval "${run_dir}" "${seed}"
  done
done

python -m experiments.summarize_ablation --root "${RESULT_ROOT}" --control all \
  --variants early middle late --recommendation-mode pareto
