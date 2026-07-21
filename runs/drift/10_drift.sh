#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <trace-run-directory> [seed]" >&2
  exit 2
fi

INPUT_RUN_DIR="$1"
SEED="${2:-1337}"
DEVICE="${DEVICE:-}"
EVAL_BATCHES="${EVAL_BATCHES:-}"
RESULT_ROOT="${RESULT_ROOT:-${INPUT_RUN_DIR}/drift}"

extra_args=()
[[ -n "${DEVICE}" ]] && extra_args+=(--device "${DEVICE}")
[[ -n "${EVAL_BATCHES}" ]] && extra_args+=(--eval-batches "${EVAL_BATCHES}")

for MODE in recompute append_recurrent; do
  python -m experiments.eval_trace_drift \
    --input-run-dir "${INPUT_RUN_DIR}" \
    --run-dir "${RESULT_ROOT}/${MODE}" \
    --inference-mode "${MODE}" \
    --token-selection argmax \
    --seed "${SEED}" \
    "${extra_args[@]}"
done
