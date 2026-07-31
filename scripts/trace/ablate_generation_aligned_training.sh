#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

DEVICE="${DEVICE:-mps}"
SEEDS="${SEEDS:-1337 2027 4099}"
RESULT_ROOT="${RESULT_ROOT:-results/ablations/generation_aligned_training}"
TRAIN_STEPS="${TRAIN_STEPS:-200000}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-4000}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-4}"
FINAL_EVAL_BATCHES="${FINAL_EVAL_BATCHES:-64}"
DIAGNOSTIC_BATCHES="${DIAGNOSTIC_BATCHES:-4}"
APPEND_MICROBATCH_SIZE="${APPEND_MICROBATCH_SIZE:-8}"
APPEND_HORIZON="${APPEND_HORIZON:-4}"
APPEND_LOSS_WEIGHT="${APPEND_LOSS_WEIGHT:-1.0}"
APPEND_WARMUP_STEPS="${APPEND_WARMUP_STEPS:-5000}"
APPEND_RAMP_STEPS="${APPEND_RAMP_STEPS:-5000}"

VARIANTS=(
  "p0:0.0"
  "p10:0.1"
  "p25:0.25"
  "p50:0.5"
)

for specification in "${VARIANTS[@]}"; do
  variant="${specification%%:*}"
  probability="${specification##*:}"
  for seed in ${SEEDS}; do
    run_dir="${RESULT_ROOT}/${variant}/seed_${seed}"

    python -m experiments.train_trace \
      --preset shortest_path_main \
      --architecture memory_tape \
      --append-train-prob "${probability}" \
      --append-train-microbatch-size "${APPEND_MICROBATCH_SIZE}" \
      --append-train-horizon "${APPEND_HORIZON}" \
      --append-train-loss-weight "${APPEND_LOSS_WEIGHT}" \
      --append-train-warmup-steps "${APPEND_WARMUP_STEPS}" \
      --append-train-ramp-steps "${APPEND_RAMP_STEPS}" \
      --token-selection argmax \
      --train-steps "${TRAIN_STEPS}" \
      --lr-warmup-steps "${LR_WARMUP_STEPS}" \
      --lr-decay-steps "${TRAIN_STEPS}" \
      --eval-batches "${TRAIN_EVAL_BATCHES}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      --run-dir "${run_dir}"

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
  done
done

python -m experiments.summarize_ablation \
  --root "${RESULT_ROOT}" \
  --control p0 \
  --variants p10 p25 p50 \
  --recommendation-mode generation-aligned
