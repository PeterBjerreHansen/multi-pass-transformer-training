#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

DEVICE="${DEVICE:-mps}"
SEEDS="${SEEDS:-1337 2027 4099}"
RESULT_ROOT="${RESULT_ROOT:-results/ablations/01_variable_pass_fixed_point}"
TRAIN_STEPS="${TRAIN_STEPS:-200000}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-4000}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-4}"
FINAL_EVAL_BATCHES="${FINAL_EVAL_BATCHES:-64}"
DIAGNOSTIC_BATCHES="${DIAGNOSTIC_BATCHES:-4}"
RESIDUAL_THRESHOLD="${RESIDUAL_THRESHOLD:-0.1}"
LOGIT_KL_THRESHOLD="${LOGIT_KL_THRESHOLD:-0.001}"

run_eval() {
  local run_dir="$1" seed="$2"
  for inference_mode in recompute append_recurrent; do
    local adaptive_eval_name
    if [[ "${inference_mode}" == "recompute" ]]; then
      adaptive_eval_name="adaptive_recompute"
    else
      adaptive_eval_name="adaptive_prefill"
    fi
    python -m experiments.eval_trace --input-run-dir "${run_dir}" \
      --output-dir "${run_dir}/drift/${inference_mode}" \
      --inference-mode "${inference_mode}" --eval-pass-mode fixed \
      --min-n-pass 2 --max-n-pass 4 --checkpoint best \
      --eval-batches "${FINAL_EVAL_BATCHES}" --token-selection argmax \
      --device "${DEVICE}" --seed "${seed}"
    python -m experiments.eval_trace --input-run-dir "${run_dir}" \
      --output-dir "${run_dir}/${adaptive_eval_name}" \
      --inference-mode "${inference_mode}" --eval-pass-mode fixed_point \
      --min-n-pass 2 --max-n-pass 6 \
      --fixed-point-residual-threshold "${RESIDUAL_THRESHOLD}" \
      --fixed-point-kl-threshold "${LOGIT_KL_THRESHOLD}" \
      --checkpoint best --eval-batches "${FINAL_EVAL_BATCHES}" \
      --token-selection argmax --device "${DEVICE}" --seed "${seed}"
  done
  python -m experiments.diagnose_memory --input-run-dir "${run_dir}" \
    --output "${run_dir}/diagnostics.json" --checkpoint best \
    --eval-batches "${DIAGNOSTIC_BATCHES}" --extra-passes 6 \
    --device "${DEVICE}" --seed "${seed}"
}

run_training() {
  local variant="$1" seed="$2"
  shift 2
  local run_dir="${RESULT_ROOT}/${variant}/seed_${seed}"
  python -m experiments.train_trace --preset shortest_path_main \
    --architecture memory_tape --token-selection argmax \
    --train-steps "${TRAIN_STEPS}" --lr-warmup-steps "${LR_WARMUP_STEPS}" \
    --lr-decay-steps "${TRAIN_STEPS}" --eval-batches "${TRAIN_EVAL_BATCHES}" \
    --device "${DEVICE}" --seed "${seed}" --run-dir "${run_dir}" "$@"
  run_eval "${run_dir}" "${seed}"
}

for seed in ${SEEDS}; do
  run_training fixed_k4 "${seed}" \
    --train-pass-mode fixed --eval-pass-mode fixed \
    --min-n-pass 2 --max-n-pass 4 --pass-loss-weights 0 0 0 1
  run_training uniform_k2_k6 "${seed}" \
    --train-pass-mode uniform --eval-pass-mode fixed \
    --min-n-pass 2 --max-n-pass 6
  run_training fixed_point_k2_k6 "${seed}" \
    --train-pass-mode fixed_point --eval-pass-mode fixed \
    --min-n-pass 2 --max-n-pass 6 \
    --fixed-point-residual-threshold "${RESIDUAL_THRESHOLD}" \
    --fixed-point-kl-threshold "${LOGIT_KL_THRESHOLD}"
done

python -m experiments.summarize_ablation --root "${RESULT_ROOT}" \
  --control fixed_k4 --variants uniform_k2_k6 fixed_point_k2_k6 \
  --recommendation-mode quality-only
