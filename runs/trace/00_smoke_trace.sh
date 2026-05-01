#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

DEVICE="${DEVICE:-cpu}"
PRESET="${PRESET:-random_graph_walk_smoke}"
ARCHITECTURE="${ARCHITECTURE:-transformer}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

case "$PRESET" in
  random_graph_walk_smoke) RESULT_GROUP="random_graph_walk" ;;
  othello_smoke) RESULT_GROUP="othello" ;;
  *)
    echo "Unsupported smoke preset: $PRESET" >&2
    exit 1
    ;;
esac

CMD=(
  python3 -m experiments.train_trace
  --preset "$PRESET"
  --architecture "$ARCHITECTURE"
  --device "$DEVICE"
  --batch-size 8
  --train-steps 20
  --eval-interval 10
  --eval-batches 1
  --run-dir "$RESULTS_ROOT/trace/$RESULT_GROUP/$ARCHITECTURE/$RUN_ID"
)

"${CMD[@]}"
