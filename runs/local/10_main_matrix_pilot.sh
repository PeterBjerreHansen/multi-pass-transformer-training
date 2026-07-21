#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
source scripts/lib/local_pilot.sh

DEVICE="$(local_pilot_device)"
SEED="${SEED:-1337}"
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
  TASK="${task}" \
  DEVICE="${DEVICE}" \
  SEED="${SEED}" \
  TRAIN_STEPS="${TRAIN_STEPS}" \
  EVAL_INTERVAL="${EVAL_INTERVAL}" \
  EVAL_BATCHES="${EVAL_BATCHES}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  ARCHITECTURES="${ARCHITECTURES}" \
  RESULT_ROOT="${RESULT_ROOT}/bbh/${task}" \
    bash runs/bbh/10_bbh_curriculum.sh
done

for task in ${TRACE_TASKS}; do
  DEVICE="${DEVICE}" \
  SEED="${SEED}" \
  TRAIN_STEPS="${TRAIN_STEPS}" \
  EVAL_INTERVAL="${EVAL_INTERVAL}" \
  EVAL_BATCHES="${EVAL_BATCHES}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  TOKEN_SELECTION=argmax \
  ARCHITECTURES="${ARCHITECTURES}" \
  RESULT_ROOT="${RESULT_ROOT}/trace/${task}" \
    bash "runs/trace/10_${task}_trace.sh"
done

if [[ "${INCLUDE_OTHELLO:-1}" == "1" ]]; then
  DEVICE="${DEVICE}" \
  SEED="${SEED}" \
  TRAIN_STEPS="${TRAIN_STEPS}" \
  EVAL_INTERVAL="${EVAL_INTERVAL}" \
  EVAL_BATCHES="${EVAL_BATCHES}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  TOKEN_SELECTION=argmax \
  ARCHITECTURES="${ARCHITECTURES}" \
  OTHELLO_DATA_DIR="${OTHELLO_DATA_DIR:-data/othello_local_pilot}" \
  OTHELLO_TRAIN_GAMES="${OTHELLO_TRAIN_GAMES:-2048}" \
  OTHELLO_VAL_GAMES="${OTHELLO_VAL_GAMES:-128}" \
  RESULT_ROOT="${RESULT_ROOT}/trace/othello" \
    bash runs/trace/10_othello_trace.sh
fi

python scripts/summarize_learning_runs.py --root "${RESULT_ROOT}"
