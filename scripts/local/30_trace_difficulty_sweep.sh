#!/usr/bin/env bash
set -euo pipefail

# Exploratory local workflow: use fixed ablation scripts for reported results.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
source scripts/lib/local_pilot.sh

DEVICE="$(local_pilot_device)"
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
RESULT_ROOT="${RESULT_ROOT:-results/local_pilots/shortest_path_learning/${RUN_ID}}"

qualification_examples=$((BATCH_SIZE * QUAL_EVAL_BATCHES))
if (( qualification_examples < MIN_QUAL_EXAMPLES )); then
  printf 'qualification uses %d examples; require at least %d (BATCH_SIZE * QUAL_EVAL_BATCHES)\n' \
    "${qualification_examples}" "${MIN_QUAL_EXAMPLES}" >&2
  exit 2
fi

run_variant() {
  local preset="$1"
  local variant="$2"
  local architecture="$3"
  local seed="$4"
  shift 4

  PILOT_PRESET="${preset}" \
  PILOT_ARCHITECTURE="${architecture}" \
  DEVICE="${DEVICE}" \
  SEED="${seed}" \
  TRAIN_STEPS="${TRAIN_STEPS}" \
  EVAL_INTERVAL="${EVAL_INTERVAL}" \
  TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES}" \
  QUAL_EVAL_BATCHES="${QUAL_EVAL_BATCHES}" \
  DIAGNOSTIC_EVAL_BATCHES="${DIAGNOSTIC_EVAL_BATCHES}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS}" \
  RESULT_ROOT="${RESULT_ROOT}" \
    run_trace_pilot_variant "${variant}/${architecture}" "$@"
}

for distribution in ${SHORTEST_PATH_DISTRIBUTIONS}; do
  if [[ "${distribution}" != "easy" && "${distribution}" != "main" ]]; then
    printf 'invalid shortest-path distribution: %s\n' "${distribution}" >&2
    exit 2
  fi
  for architecture in ${ARCHITECTURES}; do
    for seed in ${SEEDS}; do
      run_variant shortest_path_main \
        "shortest_path/${distribution}" "${architecture}" "${seed}" \
        --shortest-path-distribution "${distribution}"
    done
  done
done

python scripts/summarize_learning_runs.py --root "${RESULT_ROOT}"
