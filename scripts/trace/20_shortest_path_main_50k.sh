#!/usr/bin/env bash
set -euo pipefail

# Fixed first long-range comparison on the main shortest-path distribution.
# Models run sequentially so the script is safe on a single MPS device.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
source scripts/lib/local_pilot.sh

if [[ "$(uname -s)" == "Darwin" ]] \
  && command -v caffeinate >/dev/null 2>&1 \
  && [[ -z "${MPT_CAFFEINATED:-}" ]]; then
  export MPT_CAFFEINATED=1
  exec caffeinate -dimsu "$0" "$@"
fi

DEVICE="$(local_pilot_device)"
RESULT_ROOT="${RESULT_ROOT:-results/trace/shortest_path_main_50k}"
SEED=1337
TARGET_STEPS=50000
QUALIFICATION_EVAL_BATCHES=64
DIAGNOSTIC_EVAL_BATCHES=4
DIAGNOSTIC_BATCH_SIZE=16
ARCHITECTURES=(
  transformer
  memory_tape
  memory_add
)

checkpoint_step() {
  python -c \
    'import sys; from experiments.common import load_checkpoint_payload; print(int(load_checkpoint_payload(sys.argv[1], device="cpu").get("step", 0)))' \
    "$1"
}

for architecture in "${ARCHITECTURES[@]}"; do
  run_dir="${RESULT_ROOT}/${architecture}/seed_${SEED}"
  checkpoint="${run_dir}/latest.pt"
  current_step=0
  if [[ -f "${checkpoint}" ]]; then
    current_step="$(checkpoint_step "${checkpoint}")"
  fi

  if (( current_step < TARGET_STEPS )); then
    remaining_steps=$((TARGET_STEPS - current_step))
    train_args=(
      --preset shortest_path_main
      --architecture "${architecture}"
      --train-steps "${remaining_steps}"
      --seed "${SEED}"
      --device "${DEVICE}"
      --run-dir "${run_dir}"
    )
    if (( current_step > 0 )); then
      train_args+=(--resume-from "${run_dir}")
    fi
    python -m experiments.train_trace "${train_args[@]}"
  else
    printf '%s already reached step %d; skipping training\n' \
      "${architecture}" "${current_step}"
  fi

  inference_modes=(recompute)
  if [[ "${architecture}" != "transformer" ]]; then
    inference_modes+=(append_recurrent)
  fi
  for inference_mode in "${inference_modes[@]}"; do
    python -m experiments.eval_trace_drift \
      --input-run-dir "${run_dir}" \
      --inference-mode "${inference_mode}" \
      --token-selection argmax \
      --device "${DEVICE}" \
      --eval-batches "${QUALIFICATION_EVAL_BATCHES}" \
      --seed "${SEED}" \
      --run-dir "${run_dir}/drift/${inference_mode}"
  done

  if [[ "${architecture}" != "transformer" ]]; then
    python -m experiments.eval_diagnostics \
      --input-run-dir "${run_dir}" \
      --device "${DEVICE}" \
      --batch-size "${DIAGNOSTIC_BATCH_SIZE}" \
      --eval-batches "${DIAGNOSTIC_EVAL_BATCHES}" \
      --seed "${SEED}" \
      --output "${run_dir}/diagnostics.json"
  fi
done

python scripts/summarize_learning_runs.py --root "${RESULT_ROOT}"
