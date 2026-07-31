#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

DEVICE="${DEVICE:-mps}"
SEEDS="${SEEDS:-1337 2027 4099}"
RESULT_ROOT="${RESULT_ROOT:-results/ablations/06_position_offsets}"
TRAIN_STEPS="${TRAIN_STEPS:-200000}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-4000}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-4}"
FINAL_EVAL_BATCHES="${FINAL_EVAL_BATCHES:-64}"
DIAGNOSTIC_BATCHES="${DIAGNOSTIC_BATCHES:-4}"

run_eval() {
  local run_dir="$1" seed="$2"
  for offset in 0 16 32 64; do
    for mode in recompute append_recurrent; do
      output_dir="${run_dir}/drift/offset_${offset}/${mode}"
      if [[ "${offset}" == 0 ]]; then
        output_dir="${run_dir}/drift/${mode}"
      fi
      python -m experiments.eval_trace --input-run-dir "${run_dir}" \
        --output-dir "${output_dir}" --inference-mode "${mode}" --eval-position-offset "${offset}" \
        --checkpoint best --eval-batches "${FINAL_EVAL_BATCHES}" \
        --token-selection argmax --device "${DEVICE}" --seed "${seed}"
    done
  done
  python -m experiments.diagnose_memory --input-run-dir "${run_dir}" \
    --output "${run_dir}/diagnostics.json" --checkpoint best \
    --eval-batches "${DIAGNOSTIC_BATCHES}" --device "${DEVICE}" --seed "${seed}"
}

for seed in ${SEEDS}; do
  for variant in offset_zero offset_uniform_0_64; do
    run_dir="${RESULT_ROOT}/${variant}/seed_${seed}"
    extra_args=()
    if [[ "${variant}" == offset_uniform_0_64 ]]; then
      extra_args+=(--train-position-offset-max 64)
    fi
    python -m experiments.train_trace --preset shortest_path_main --architecture memory_tape \
      --max-position-embeddings 209 --token-selection argmax --train-steps "${TRAIN_STEPS}" \
      --lr-warmup-steps "${LR_WARMUP_STEPS}" --lr-decay-steps "${TRAIN_STEPS}" \
      --eval-batches "${TRAIN_EVAL_BATCHES}" \
      --device "${DEVICE}" --seed "${seed}" --run-dir "${run_dir}" "${extra_args[@]}"
    run_eval "${run_dir}" "${seed}"
  done
done

python -m experiments.summarize_ablation --root "${RESULT_ROOT}" --control offset_zero \
  --variants offset_uniform_0_64 --recommendation-mode position-offset
