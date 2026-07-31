#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

DEVICE="${DEVICE:-mps}"
SEEDS="${SEEDS:-1337 2027 4099}"
RESULT_ROOT="${RESULT_ROOT:-results/ablations/conditional_memory_gates}"
TRAIN_STEPS="${TRAIN_STEPS:-200000}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-4000}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-4}"
FINAL_EVAL_BATCHES="${FINAL_EVAL_BATCHES:-64}"
DIAGNOSTIC_BATCHES="${DIAGNOSTIC_BATCHES:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"

run_evaluations() {
  local run_dir="$1" seed="$2"
  for inference_mode in recompute append_recurrent; do
    python -m experiments.eval_trace \
      --input-run-dir "${run_dir}" \
      --checkpoint best \
      --inference-mode "${inference_mode}" \
      --token-selection argmax \
      --device "${DEVICE}" \
      --eval-batches "${FINAL_EVAL_BATCHES}" \
      --seed "${seed}" \
      --output-dir "${run_dir}/drift/${inference_mode}"
  done
  python -m experiments.diagnose_memory \
    --input-run-dir "${run_dir}" \
    --checkpoint best \
    --device "${DEVICE}" \
    --eval-batches "${DIAGNOSTIC_BATCHES}" \
    --seed "${seed}" \
    --output "${run_dir}/diagnostics.json"
}

for variant in gate_off gate_on; do
  gate=off
  [[ "${variant}" == "gate_on" ]] && gate=on
  for seed in ${SEEDS}; do
    run_dir="${RESULT_ROOT}/${variant}/seed_${seed}"
    python -m experiments.train_trace \
      --preset shortest_path_main \
      --architecture memory_tape \
      --conditional-memory-gate "${gate}" \
      --token-selection argmax \
      --train-steps "${TRAIN_STEPS}" \
      --lr-warmup-steps "${LR_WARMUP_STEPS}" \
      --lr-decay-steps "${TRAIN_STEPS}" \
      --eval-batches "${TRAIN_EVAL_BATCHES}" \
      --batch-size "${BATCH_SIZE}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      --run-dir "${run_dir}"
    run_evaluations "${run_dir}" "${seed}"
  done
done

python -m experiments.summarize_ablation \
  --root "${RESULT_ROOT}" \
  --control gate_off \
  --variants gate_on \
  --recommendation-mode conditional-gate
