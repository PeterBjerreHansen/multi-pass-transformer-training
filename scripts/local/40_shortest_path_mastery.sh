#!/usr/bin/env bash
set -euo pipefail

# Train and evaluate the two shortest-path distributions in order. Main is
# launched only after every architecture reaches exact mastery on smoke.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
source scripts/lib/local_pilot.sh
source scripts/lib/model_matrix.sh

DEVICE="$(local_pilot_device)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-results/local_pilots/shortest_path_mastery/${RUN_ID}}"
SEED=1337
ARCHITECTURES=(
  transformer
  memory_tape
  joint_memory_tape
  memory_concat
  memory_add
  memory_state
  memory_update
)
SMOKE_STEPS=25000
MAIN_STEPS=50000
BATCH_SIZE=64
EVAL_INTERVAL=1000
TRAIN_EVAL_BATCHES=4
QUAL_EVAL_BATCHES=64
MINIMUM_EXAMPLES=4096

run_distribution() {
  local distribution="$1"
  local train_steps="$2"
  for architecture in "${ARCHITECTURES[@]}"; do
    PILOT_PRESET=shortest_path_main \
    PILOT_ARCHITECTURE="${architecture}" \
    DEVICE="${DEVICE}" \
    SEED="${SEED}" \
    TRAIN_STEPS="${train_steps}" \
    EVAL_INTERVAL="${EVAL_INTERVAL}" \
    TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES}" \
    QUAL_EVAL_BATCHES="${QUAL_EVAL_BATCHES}" \
    DIAGNOSTIC_EVAL_BATCHES="${TRAIN_EVAL_BATCHES}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    RUN_DIAGNOSTICS=1 \
    RESULT_ROOT="${RESULT_ROOT}" \
      run_trace_pilot_variant \
        "${distribution}/${architecture}" \
        --shortest-path-distribution "${distribution}"
  done

  python scripts/check_shortest_path_mastery.py \
    --root "${RESULT_ROOT}" \
    --distribution "${distribution}" \
    --architectures "${ARCHITECTURES[@]}" \
    --seed "${SEED}" \
    --minimum-examples "${MINIMUM_EXAMPLES}"
}

printf 'Shortest-path mastery qualification\n'
printf 'device=%s result_root=%s\n' "${DEVICE}" "${RESULT_ROOT}"
run_distribution smoke "${SMOKE_STEPS}"
run_distribution main "${MAIN_STEPS}"
