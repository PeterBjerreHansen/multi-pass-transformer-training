#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

# "results/trace/othello/memory_concat/20260429_120658"
# "results/trace/othello/memory_tape/20260424_134447"
# "results/trace/othello/memory_update/20260430_052605"
DEVICE="${DEVICE:-mps}" 
INPUT_RUN_DIR="${INPUT_RUN_DIR:-"results/trace/othello/memory_update/20260430_052605"}" 
EVAL_BATCHES="${EVAL_BATCHES:-4}"
TOKEN_SELECTION="${TOKEN_SELECTION:-sample}"
INFERENCE_MODE="${INFERENCE_MODE:-final_pass}" # recompute final_pass
CACHE_SOURCE="${CACHE_SOURCE:-last}" # penultimate last

if [[ -z "$INPUT_RUN_DIR" ]]; then
  echo "Set INPUT_RUN_DIR to a saved trace run directory." >&2
  exit 1
fi

python3 -m experiments.eval_trace_drift \
  --device "$DEVICE" \
  --eval-batches "$EVAL_BATCHES" \
  --token-selection "$TOKEN_SELECTION" \
  --inference-mode "$INFERENCE_MODE" \
  --cache-source "$CACHE_SOURCE" \
  --input-run-dir "$INPUT_RUN_DIR"
