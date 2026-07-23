#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <trace-run-directory> [seed]" >&2
  exit 2
fi

INPUT_RUN_DIR="$1"
SEED="${2:-1337}"
DEVICE="${DEVICE:-}"
RESULT_ROOT="${RESULT_ROOT:-${INPUT_RUN_DIR}/drift}"

runtime_args=()
[[ -n "${DEVICE}" ]] && runtime_args+=(--device "${DEVICE}")

for MODE in recompute append_recurrent; do
  python -m experiments.eval_trace_drift \
    --input-run-dir "${INPUT_RUN_DIR}" \
    --run-dir "${RESULT_ROOT}/${MODE}" \
    --inference-mode "${MODE}" \
    --token-selection argmax \
    --seed "${SEED}" \
    "${runtime_args[@]}"
done
