#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
source scripts/lib/local_pilot.sh

DEVICE="$(local_pilot_device)"
SEEDS="${SEEDS:-1337 2027 4099}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-results/ablations/memory_gate_init/${RUN_ID}}"

# These are fixed parts of the experiment contract, not environment overrides.
QUALIFICATION_EVAL_BATCHES=16
DIAGNOSTIC_EVAL_BATCHES=2
DIAGNOSTIC_BATCH_SIZE=16

for specification in \
  "control:shortest_path_gate_init_control" \
  "unit:shortest_path_gate_init_unit"; do
  IFS=: read -r variant preset <<<"${specification}"
  for seed in ${SEEDS}; do
    run_dir="${RESULT_ROOT}/${variant}/seed_${seed}"

    python -m experiments.train_trace \
      --preset "${preset}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      --run-dir "${run_dir}"

    for inference_mode in recompute append_recurrent; do
      python -m experiments.eval_trace_drift \
        --input-run-dir "${run_dir}" \
        --inference-mode "${inference_mode}" \
        --token-selection argmax \
        --device "${DEVICE}" \
        --eval-batches "${QUALIFICATION_EVAL_BATCHES}" \
        --seed "${seed}" \
        --run-dir "${run_dir}/drift/${inference_mode}"
    done

    python -m experiments.eval_diagnostics \
      --input-run-dir "${run_dir}" \
      --device "${DEVICE}" \
      --batch-size "${DIAGNOSTIC_BATCH_SIZE}" \
      --eval-batches "${DIAGNOSTIC_EVAL_BATCHES}" \
      --seed "${seed}" \
      --output "${run_dir}/diagnostics.json"
  done
done

python -m experiments.summarize_ablation \
  --root "${RESULT_ROOT}" \
  --control control \
  --variants unit \
  --recommendation-mode quality-only \
  --quality-metric drift.append_recurrent.optimal_path
