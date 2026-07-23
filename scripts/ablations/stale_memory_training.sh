#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

DEVICE="${DEVICE:-mps}"
SEEDS="${SEEDS:-1337 2027 4099}"
RESULT_ROOT="${RESULT_ROOT:-results/ablations/stale_memory_training}"
TRAIN_STEPS=50000
EVAL_BATCHES=4

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
      --preset random_graph_walk_main \
      --architecture memory_tape \
      --stale-memory-prob "${probability}" \
      --token-selection argmax \
      --train-steps "${TRAIN_STEPS}" \
      --eval-batches "${EVAL_BATCHES}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      --run-dir "${run_dir}"

    for inference_mode in recompute append_recurrent; do
      python -m experiments.eval_trace_drift \
        --input-run-dir "${run_dir}" \
        --inference-mode "${inference_mode}" \
        --token-selection argmax \
        --device "${DEVICE}" \
        --eval-batches "${EVAL_BATCHES}" \
        --seed "${seed}" \
        --run-dir "${run_dir}/drift/${inference_mode}"
    done

    python -m experiments.eval_diagnostics \
      --input-run-dir "${run_dir}" \
      --device "${DEVICE}" \
      --eval-batches "${EVAL_BATCHES}" \
      --seed "${seed}" \
      --output "${run_dir}/diagnostics.json"
  done
done

python -m experiments.summarize_ablation \
  --root "${RESULT_ROOT}" \
  --control p0 \
  --variants p10 p25 p50 \
  --recommendation-mode quality-only
