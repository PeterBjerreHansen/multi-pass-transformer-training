#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

RESULTS_ROOT="${RESULTS_ROOT:-results}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
TRACE_RUN_DIR="$RESULTS_ROOT/trace/random_graph_walk/memory_tape/${RUN_ID}_smoke"

python3 -m experiments.train_trace \
  --preset random_graph_walk_smoke \
  --architecture memory_tape \
  --device cpu \
  --batch-size 1 \
  --train-steps 1 \
  --eval-interval 1 \
  --eval-batches 1 \
  --run-dir "$TRACE_RUN_DIR"

python3 -m experiments.eval_trace_drift \
  --input-run-dir "$TRACE_RUN_DIR" \
  --device cpu \
  --eval-batches 1 \
  --inference-mode final_pass \
  --token-selection argmax \
  --cache-source penultimate
