#!/usr/bin/env bash

# Shared helpers for local learning and ablation pilots. Callers remain
# responsible for `set -euo pipefail` so this file can be sourced safely.

local_pilot_device() {
  if [[ -n "${DEVICE:-}" ]]; then
    printf '%s\n' "${DEVICE}"
    return
  fi
  python -c 'from experiments.common import auto_device; print(auto_device())'
}

run_bbh_pilot_variant() {
  if [[ $# -lt 1 ]]; then
    printf 'run_bbh_pilot_variant requires a variant name\n' >&2
    return 2
  fi

  local variant="$1"
  shift
  local task="${PILOT_TASK:-pointer_chasing}"
  local preset="${PILOT_PRESET:-${task}_main}"
  local architecture="${PILOT_ARCHITECTURE:-memory_tape}"
  local seed="${SEED:-1337}"
  local device
  device="$(local_pilot_device)"
  local train_steps="${TRAIN_STEPS:-250}"
  local eval_interval="${EVAL_INTERVAL:-${train_steps}}"
  local eval_batches="${EVAL_BATCHES:-1}"
  local batch_size="${BATCH_SIZE:-16}"
  local result_root="${RESULT_ROOT:-results/local_pilots}"
  local run_dir="${result_root}/${variant}/seed_${seed}"

  python -m experiments.train_bbh \
    --preset "${preset}" \
    --architecture "${architecture}" \
    --train-steps "${train_steps}" \
    --eval-interval "${eval_interval}" \
    --eval-batches "${eval_batches}" \
    --batch-size "${batch_size}" \
    --seed "${seed}" \
    --device "${device}" \
    --run-dir "${run_dir}" \
    "$@"
}

run_trace_pilot_variant() {
  if [[ $# -lt 1 ]]; then
    printf 'run_trace_pilot_variant requires a variant name\n' >&2
    return 2
  fi

  local variant="$1"
  shift
  local preset="${PILOT_PRESET:-random_graph_walk_main}"
  local architecture="${PILOT_ARCHITECTURE:-memory_tape}"
  local seed="${SEED:-1337}"
  local device
  device="$(local_pilot_device)"
  local train_steps="${TRAIN_STEPS:-250}"
  local eval_interval="${EVAL_INTERVAL:-${train_steps}}"
  # Keep repeated in-training evaluation cheap while allowing a larger,
  # statistically useful post-training qualification set. EVAL_BATCHES remains
  # a backwards-compatible fallback for older branch pilot scripts.
  local train_eval_batches="${TRAIN_EVAL_BATCHES:-${EVAL_BATCHES:-1}}"
  local qualification_eval_batches="${QUAL_EVAL_BATCHES:-${EVAL_BATCHES:-1}}"
  local diagnostic_eval_batches="${DIAGNOSTIC_EVAL_BATCHES:-${EVAL_BATCHES:-1}}"
  local batch_size="${BATCH_SIZE:-16}"
  local result_root="${RESULT_ROOT:-results/local_pilots}"
  local run_dir="${result_root}/${variant}/seed_${seed}"

  python -m experiments.train_trace \
    --preset "${preset}" \
    --architecture "${architecture}" \
    --token-selection argmax \
    --train-steps "${train_steps}" \
    --eval-interval "${eval_interval}" \
    --eval-batches "${train_eval_batches}" \
    --batch-size "${batch_size}" \
    --seed "${seed}" \
    --device "${device}" \
    --run-dir "${run_dir}" \
    "$@"

  local modes=(recompute append_recurrent)
  if [[ "${architecture}" == "transformer" ]]; then
    modes=(recompute)
  fi
  for inference_mode in "${modes[@]}"; do
    python -m experiments.eval_trace_drift \
      --input-run-dir "${run_dir}" \
      --inference-mode "${inference_mode}" \
      --token-selection argmax \
      --device "${device}" \
      --eval-batches "${qualification_eval_batches}" \
      --seed "${seed}" \
      --run-dir "${run_dir}/drift/${inference_mode}"
  done

  if [[ "${RUN_DIAGNOSTICS:-1}" == "1" && "${architecture}" != "transformer" ]]; then
    local diagnostics_batch_size="${batch_size}"
    if (( diagnostics_batch_size < 2 )); then
      diagnostics_batch_size=2
    fi
    python -m experiments.eval_diagnostics \
      --input-run-dir "${run_dir}" \
      --device "${device}" \
      --batch-size "${diagnostics_batch_size}" \
      --eval-batches "${diagnostic_eval_batches}" \
      --seed "${seed}" \
      --output "${run_dir}/diagnostics.json"
  fi
}
