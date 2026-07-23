#!/usr/bin/env bash
set -euo pipefail

# Exploratory local workflow: canonical scripts never consume these overrides.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
source scripts/lib/local_pilot.sh

DEVICE="$(local_pilot_device)"
SEEDS="${SEEDS:-${SEED:-1337}}"
TRAIN_STEPS="${TRAIN_STEPS:-250}"
EVAL_INTERVAL="${EVAL_INTERVAL:-${TRAIN_STEPS}}"
EVAL_BATCHES="${EVAL_BATCHES:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ARCHITECTURES="${ARCHITECTURES:-transformer memory_tape}"
BBH_TASKS="${BBH_TASKS:-pointer_chasing tracking permutation state_machine}"
TRACE_TASKS="${TRACE_TASKS:-random_graph_walk shortest_path}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-results/local_pilots/main_matrix/${RUN_ID}}"

for task in ${BBH_TASKS}; do
  for architecture in ${ARCHITECTURES}; do
    for seed in ${SEEDS}; do
      PILOT_TASK="${task}" \
      PILOT_PRESET="${task}_main" \
      PILOT_ARCHITECTURE="${architecture}" \
      DEVICE="${DEVICE}" \
      SEED="${seed}" \
      TRAIN_STEPS="${TRAIN_STEPS}" \
      EVAL_INTERVAL="${EVAL_INTERVAL}" \
      EVAL_BATCHES="${EVAL_BATCHES}" \
      BATCH_SIZE="${BATCH_SIZE}" \
      RESULT_ROOT="${RESULT_ROOT}" \
        run_bbh_pilot_variant "bbh/${task}/${architecture}"
    done
  done
done

for task in ${TRACE_TASKS}; do
  for architecture in ${ARCHITECTURES}; do
    for seed in ${SEEDS}; do
      PILOT_PRESET="${task}_main" \
      PILOT_ARCHITECTURE="${architecture}" \
      DEVICE="${DEVICE}" \
      SEED="${seed}" \
      TRAIN_STEPS="${TRAIN_STEPS}" \
      EVAL_INTERVAL="${EVAL_INTERVAL}" \
      EVAL_BATCHES="${EVAL_BATCHES}" \
      BATCH_SIZE="${BATCH_SIZE}" \
      RESULT_ROOT="${RESULT_ROOT}" \
        run_trace_pilot_variant "trace/${task}/${architecture}"
    done
  done
done

if [[ "${INCLUDE_OTHELLO:-1}" == "1" ]]; then
  for architecture in ${ARCHITECTURES}; do
    for seed in ${SEEDS}; do
      PILOT_PRESET=othello_main \
      PILOT_ARCHITECTURE="${architecture}" \
      DEVICE="${DEVICE}" \
      SEED="${seed}" \
      TRAIN_STEPS="${TRAIN_STEPS}" \
      EVAL_INTERVAL="${EVAL_INTERVAL}" \
      EVAL_BATCHES="${EVAL_BATCHES}" \
      BATCH_SIZE="${BATCH_SIZE}" \
      RUN_DIAGNOSTICS=0 \
      RESULT_ROOT="${RESULT_ROOT}" \
        run_trace_pilot_variant "trace/othello/${architecture}" \
          --othello-data-dir "${OTHELLO_DATA_DIR:-data/othello_local_pilot}" \
          --othello-train-games "${OTHELLO_TRAIN_GAMES:-2048}" \
          --othello-val-games "${OTHELLO_VAL_GAMES:-128}"
    done
  done
fi

python scripts/summarize_learning_runs.py --root "${RESULT_ROOT}"
