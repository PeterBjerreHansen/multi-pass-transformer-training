#!/usr/bin/env bash
set -euo pipefail

# Fixed two-stage calibration. The only normal overrides are DEVICE, RUN_ID,
# and RESULT_ROOT; CALIBRATION_SMOKE=1 is reserved for code validation.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
source scripts/lib/local_pilot.sh

if command -v caffeinate >/dev/null 2>&1 && [[ "${MPT_CAFFEINATED:-0}" != "1" ]]; then
  exec caffeinate -dimsu env MPT_CAFFEINATED=1 bash "$0"
fi

DEVICE="$(local_pilot_device)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-results/local_pilots/shortest_path_calibration/${RUN_ID}}"
mkdir -p "${RESULT_ROOT}"

STAGE_ONE_SEED=1337
FINAL_SEEDS=(1337 2027 4099)
STAGE_ONE_STEPS=10000
FINAL_STEPS=25000
BATCH_SIZE=16
EVAL_INTERVAL=1000
TRAIN_EVAL_BATCHES=2
STAGE_ONE_EVAL_BATCHES=32
FINAL_EVAL_BATCHES=64
DIAGNOSTIC_EVAL_BATCHES=4
VARIANTS=(
  "easy:8:3:2:5"
  "nodes_10:10:3:2:5"
  "nodes_12:12:3:2:5"
  "distractors_8:8:3:2:8"
  "dense_9:9:3:2:10"
  "path_4:8:4:2:5"
  "path_4_nodes_10:10:4:2:5"
  "branching_3:8:3:3:10"
  "combined:10:4:2:8"
)

if [[ "${CALIBRATION_SMOKE:-0}" == "1" ]]; then
  FINAL_SEEDS=(1337)
  STAGE_ONE_STEPS=2
  FINAL_STEPS=3
  BATCH_SIZE=2
  EVAL_INTERVAL=1
  TRAIN_EVAL_BATCHES=1
  STAGE_ONE_EVAL_BATCHES=1
  FINAL_EVAL_BATCHES=1
  DIAGNOSTIC_EVAL_BATCHES=1
  VARIANTS=(
    "easy:8:3:2:5"
    "path_4:8:4:2:5"
    "combined:10:4:2:8"
  )
fi

checkpoint_step() {
  python -c \
    'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("step", 0)))' \
    "$1"
}

train_to_step() {
  local variant="$1"
  local architecture="$2"
  local seed="$3"
  local target_step="$4"
  local num_nodes="$5"
  local path_length="$6"
  local branching_factor="$7"
  local distractor_edges="$8"
  local run_dir="${RESULT_ROOT}/runs/${variant}/${architecture}/seed_${seed}"
  local current_step=0
  mkdir -p "${run_dir}"

  if [[ -f "${run_dir}/latest.pt" ]]; then
    current_step="$(checkpoint_step "${run_dir}/latest.pt")"
  fi
  if (( current_step >= target_step )); then
    printf 'skip training: %s/%s/seed_%s already at step %s\n' \
      "${variant}" "${architecture}" "${seed}" "${current_step}"
    return
  fi

  local remaining_steps=$((target_step - current_step))
  printf '\ntrain: variant=%s architecture=%s seed=%s step=%s->%s\n' \
    "${variant}" "${architecture}" "${seed}" "${current_step}" "${target_step}"
  if (( current_step > 0 )); then
    python -m experiments.train_trace \
      --preset shortest_path_main \
      --resume-from "${run_dir}" \
      --train-steps "${remaining_steps}" \
      --device "${DEVICE}" \
      --run-dir "${run_dir}"
  else
    python -m experiments.train_trace \
      --preset shortest_path_main \
      --architecture "${architecture}" \
      --token-selection argmax \
      --num-nodes "${num_nodes}" \
      --shortest-path-length "${path_length}" \
      --branching-factor "${branching_factor}" \
      --distractor-edges "${distractor_edges}" \
      --batch-size "${BATCH_SIZE}" \
      --train-steps "${remaining_steps}" \
      --eval-interval "${EVAL_INTERVAL}" \
      --eval-batches "${TRAIN_EVAL_BATCHES}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      --run-dir "${run_dir}"
  fi
}

evaluate_run() {
  local variant="$1"
  local architecture="$2"
  local seed="$3"
  local eval_batches="$4"
  local run_dir="${RESULT_ROOT}/runs/${variant}/${architecture}/seed_${seed}"
  local modes=(recompute append_recurrent)
  if [[ "${architecture}" == "transformer" ]]; then
    modes=(recompute)
  fi
  for mode in "${modes[@]}"; do
    python -m experiments.eval_trace_drift \
      --input-run-dir "${run_dir}" \
      --run-dir "${run_dir}/drift/${mode}" \
      --inference-mode "${mode}" \
      --token-selection argmax \
      --device "${DEVICE}" \
      --eval-batches "${eval_batches}" \
      --seed "${seed}"
  done
}

diagnose_run() {
  local variant="$1"
  local seed="$2"
  local run_dir="${RESULT_ROOT}/runs/${variant}/memory_tape/seed_${seed}"
  python -m experiments.eval_diagnostics \
    --input-run-dir "${run_dir}" \
    --output "${run_dir}/diagnostics.json" \
    --device "${DEVICE}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batches "${DIAGNOSTIC_EVAL_BATCHES}" \
    --seed "${seed}"
}

printf 'Shortest-path overnight calibration\n'
printf 'device=%s result_root=%s\n' "${DEVICE}" "${RESULT_ROOT}"

if [[ ! -s "${RESULT_ROOT}/calibration_selection.tsv" ]]; then
  printf '\n=== Stage one: isolate the difficulty cliff ===\n'
  for specification in "${VARIANTS[@]}"; do
    IFS=: read -r variant num_nodes path_length branching_factor distractor_edges \
      <<<"${specification}"
    train_to_step \
      "${variant}" transformer "${STAGE_ONE_SEED}" "${STAGE_ONE_STEPS}" \
      "${num_nodes}" "${path_length}" "${branching_factor}" "${distractor_edges}"
    evaluate_run \
      "${variant}" transformer "${STAGE_ONE_SEED}" "${STAGE_ONE_EVAL_BATCHES}"
  done

  python scripts/summarize_shortest_path_calibration.py select \
    --root "${RESULT_ROOT}" \
    --stage-one-seed "${STAGE_ONE_SEED}" \
    --final-seeds "${FINAL_SEEDS[@]}"
else
  printf 'reuse existing selection: %s\n' \
    "${RESULT_ROOT}/calibration_selection.tsv"
fi

printf '\n=== Stage two: three-seed architecture confirmation ===\n'
while IFS=$'\t' read -r variant num_nodes path_length branching_factor distractor_edges; do
  for architecture in transformer memory_tape; do
    for seed in "${FINAL_SEEDS[@]}"; do
      train_to_step \
        "${variant}" "${architecture}" "${seed}" "${FINAL_STEPS}" \
        "${num_nodes}" "${path_length}" "${branching_factor}" "${distractor_edges}"
      evaluate_run "${variant}" "${architecture}" "${seed}" "${FINAL_EVAL_BATCHES}"
      if [[ "${architecture}" == "memory_tape" ]]; then
        diagnose_run "${variant}" "${seed}"
      fi
    done
  done
done < "${RESULT_ROOT}/calibration_selection.tsv"

python scripts/summarize_learning_runs.py --root "${RESULT_ROOT}/runs"
python scripts/summarize_shortest_path_calibration.py report --root "${RESULT_ROOT}"

printf '\n=== Final report ===\n'
cat "${RESULT_ROOT}/calibration_report.md"
