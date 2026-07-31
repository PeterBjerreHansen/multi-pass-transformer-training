#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-1337}"
RESULT_ROOT="${RESULT_ROOT:-results/local_pilots/conditional_memory_gates/${RUN_ID}}"
TRAIN_STEPS="${TRAIN_STEPS:-250}"
EVAL_INTERVAL="${EVAL_INTERVAL:-${TRAIN_STEPS}}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-1}"
FINAL_EVAL_BATCHES="${FINAL_EVAL_BATCHES:-1}"
DIAGNOSTIC_BATCHES="${DIAGNOSTIC_BATCHES:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
schedule_args=(--lr-schedule constant)
if (( TRAIN_STEPS > 1 )); then
  LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-$((TRAIN_STEPS / 50))}"
  if (( LR_WARMUP_STEPS < 1 )); then
    LR_WARMUP_STEPS=1
  fi
  if (( LR_WARMUP_STEPS >= TRAIN_STEPS )); then
    LR_WARMUP_STEPS=$((TRAIN_STEPS - 1))
  fi
  schedule_args=(
    --lr-schedule warmup_cosine
    --lr-warmup-steps "${LR_WARMUP_STEPS}"
    --lr-decay-steps "${TRAIN_STEPS}"
  )
fi

for variant in gate_off gate_on; do
    gate=off
    [[ "${variant}" == "gate_on" ]] && gate=on
    run_dir="${RESULT_ROOT}/${variant}/seed_${SEED}"
    python -m experiments.train_trace \
      --preset shortest_path_main \
      --architecture memory_tape \
      --conditional-memory-gate "${gate}" \
      --train-steps "${TRAIN_STEPS}" \
      "${schedule_args[@]}" \
      --eval-interval "${EVAL_INTERVAL}" \
      --eval-batches "${TRAIN_EVAL_BATCHES}" \
      --batch-size "${BATCH_SIZE}" \
      --seed "${SEED}" \
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
        --seed "${SEED}" \
        --output-dir "${run_dir}/drift/${inference_mode}"
    done
    python -m experiments.diagnose_memory \
      --input-run-dir "${run_dir}" \
      --checkpoint best \
      --device "${DEVICE}" \
      --eval-batches "${DIAGNOSTIC_BATCHES}" \
      --seed "${SEED}" \
      --output "${run_dir}/diagnostics.json"
done

python -m experiments.summarize_ablation \
  --root "${RESULT_ROOT}" \
  --control gate_off \
  --variants gate_on \
  --recommendation-mode conditional-gate
