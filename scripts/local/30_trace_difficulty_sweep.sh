#!/usr/bin/env bash
set -euo pipefail

# Exploratory local workflow: use fixed ablation scripts for reported results.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
source scripts/lib/local_pilot.sh

DEVICE="$(local_pilot_device)"
SEEDS="${SEEDS:-1337}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-${EVAL_BATCHES:-2}}"
QUAL_EVAL_BATCHES="${QUAL_EVAL_BATCHES:-16}"
DIAGNOSTIC_EVAL_BATCHES="${DIAGNOSTIC_EVAL_BATCHES:-${TRAIN_EVAL_BATCHES}}"
MIN_QUAL_EXAMPLES="${MIN_QUAL_EXAMPLES:-256}"
ARCHITECTURES="${ARCHITECTURES:-transformer memory_tape}"
TASKS="${TASKS:-random_graph_walk shortest_path}"
RGW_LENGTHS="${RGW_LENGTHS:-8 16 32}"
SHORTEST_PATH_VARIANTS="${SHORTEST_PATH_VARIANTS:-easy:8:3:2:5 intermediate:16:4:3:20 main:24:6:3:40}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-0}"
MEMORY_GATE_INIT="${MEMORY_GATE_INIT:-0.1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-results/local_pilots/trace_difficulty/${RUN_ID}}"

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
    run_trace_pilot_variant "${variant}/${architecture}" \
      --memory-gate-init "${MEMORY_GATE_INIT}" "$@"
}

for task in ${TASKS}; do
  case "${task}" in
    random_graph_walk)
      for length in ${RGW_LENGTHS}; do
        for architecture in ${ARCHITECTURES}; do
          for seed in ${SEEDS}; do
            run_variant random_graph_walk_main \
              "random_graph_walk/length_${length}" "${architecture}" "${seed}" \
              --max-level "${length}"
          done
        done
      done
      ;;
    shortest_path)
      for specification in ${SHORTEST_PATH_VARIANTS}; do
        IFS=: read -r name num_nodes path_length branching_factor distractor_edges \
          <<<"${specification}"
        if [[ -z "${name}" || -z "${num_nodes}" || -z "${path_length}" || \
              -z "${branching_factor}" || -z "${distractor_edges}" ]]; then
          printf 'invalid shortest-path specification: %s\n' "${specification}" >&2
          exit 2
        fi
        for architecture in ${ARCHITECTURES}; do
          for seed in ${SEEDS}; do
            run_variant shortest_path_main \
              "shortest_path/${name}" "${architecture}" "${seed}" \
              --num-nodes "${num_nodes}" \
              --shortest-path-length "${path_length}" \
              --branching-factor "${branching_factor}" \
              --distractor-edges "${distractor_edges}"
          done
        done
      done
      ;;
    *)
      printf 'unsupported TASKS entry: %s\n' "${task}" >&2
      exit 2
      ;;
  esac
done

python scripts/summarize_learning_runs.py --root "${RESULT_ROOT}"
