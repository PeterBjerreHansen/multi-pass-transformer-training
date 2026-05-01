#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

DEVICE="${DEVICE:-cpu}"
PRESET="${PRESET:-pointer_chasing_smoke}"
ARCHITECTURE="${ARCHITECTURE:-transformer}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

RESULT_GROUP="${PRESET%_smoke}"
RESULT_GROUP="${RESULT_GROUP%_main}"

CMD=(
  python3 -m experiments.train_bbh
  --preset "$PRESET"
  --architecture "$ARCHITECTURE"
  --device "$DEVICE"
  --run-dir "$RESULTS_ROOT/bbh/$RESULT_GROUP/$ARCHITECTURE/$RUN_ID"
)

"${CMD[@]}"
