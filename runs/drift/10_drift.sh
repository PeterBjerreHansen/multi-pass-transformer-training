#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <trace-run-directory> [seed]" >&2
  exit 2
fi

INPUT_RUN_DIR="$1"
SEED="${2:-1337}"

for MODE in recompute append_recurrent; do
  python -m experiments.eval_trace_drift \
    --input-run-dir "${INPUT_RUN_DIR}" \
    --inference-mode "${MODE}" \
    --token-selection argmax \
    --seed "${SEED}"
done
