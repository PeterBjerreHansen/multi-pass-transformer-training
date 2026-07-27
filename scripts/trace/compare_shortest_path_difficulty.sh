#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

DEVICE="${DEVICE:-$(python -c 'from experiments.common import auto_device; print(auto_device())')}"
SEEDS="${SEEDS:-1337}"
TRAIN_STEPS="${TRAIN_STEPS:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-${EVAL_BATCHES:-2}}"
QUAL_EVAL_BATCHES="${QUAL_EVAL_BATCHES:-64}"
DIAGNOSTIC_EVAL_BATCHES="${DIAGNOSTIC_EVAL_BATCHES:-${TRAIN_EVAL_BATCHES}}"
MIN_QUAL_EXAMPLES="${MIN_QUAL_EXAMPLES:-4096}"
ARCHITECTURES="${ARCHITECTURES:-transformer}"
SHORTEST_PATH_DISTRIBUTIONS="${SHORTEST_PATH_DISTRIBUTIONS:-easy}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-results/trace/shortest_path_difficulty/${RUN_ID}}"

qualification_examples=$((BATCH_SIZE * QUAL_EVAL_BATCHES))
if (( qualification_examples < MIN_QUAL_EXAMPLES )); then
  printf 'qualification uses %d examples; require at least %d (BATCH_SIZE * QUAL_EVAL_BATCHES)\n' \
    "${qualification_examples}" "${MIN_QUAL_EXAMPLES}" >&2
  exit 2
fi

for distribution in ${SHORTEST_PATH_DISTRIBUTIONS}; do
  if [[ "${distribution}" != "easy" && "${distribution}" != "main" ]]; then
    printf 'invalid shortest-path distribution: %s\n' "${distribution}" >&2
    exit 2
  fi
  for architecture in ${ARCHITECTURES}; do
    for seed in ${SEEDS}; do
      run_dir="${RESULT_ROOT}/${distribution}/${architecture}/seed_${seed}"

      python -m experiments.train_trace \
        --preset shortest_path_main \
        --shortest-path-distribution "${distribution}" \
        --architecture "${architecture}" \
        --token-selection argmax \
        --train-steps "${TRAIN_STEPS}" \
        --eval-interval "${EVAL_INTERVAL}" \
        --eval-batches "${TRAIN_EVAL_BATCHES}" \
        --batch-size "${BATCH_SIZE}" \
        --seed "${seed}" \
        --device "${DEVICE}" \
        --run-dir "${run_dir}"

      modes=(recompute append_recurrent)
      if [[ "${architecture}" == "transformer" ]]; then
        modes=(recompute)
      fi
      for inference_mode in "${modes[@]}"; do
        python -m experiments.eval_trace_drift \
          --input-run-dir "${run_dir}" \
          --inference-mode "${inference_mode}" \
          --token-selection argmax \
          --device "${DEVICE}" \
          --eval-batches "${QUAL_EVAL_BATCHES}" \
          --seed "${seed}" \
          --run-dir "${run_dir}/drift/${inference_mode}"
      done

      if [[ "${RUN_DIAGNOSTICS}" == "1" && "${architecture}" != "transformer" ]]; then
        python -m experiments.eval_diagnostics \
          --input-run-dir "${run_dir}" \
          --device "${DEVICE}" \
          --batch-size "${BATCH_SIZE}" \
          --eval-batches "${DIAGNOSTIC_EVAL_BATCHES}" \
          --seed "${seed}" \
          --output "${run_dir}/diagnostics.json"
      fi
    done
  done
done

python scripts/summarize_learning_runs.py --root "${RESULT_ROOT}"
