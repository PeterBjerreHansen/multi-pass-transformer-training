#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

DEVICE="${DEVICE:-mps}"
PRESETS="${PRESETS:-pointer_chasing_main tracking_main permutation_main state_machine_main}"
ARCHES="${ARCHES:-transformer memory_tape}" # transformer memory_tape memory_update memory_concat
RESULTS_ROOT="${RESULTS_ROOT:-results}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

for PRESET in $PRESETS; do
  for ARCH in $ARCHES; do
    RESULT_GROUP="${PRESET%_main}"
    RESULT_GROUP="${RESULT_GROUP%_smoke}"

    CMD=(
      python3 -m experiments.train_bbh
      --preset "$PRESET"
      --architecture "$ARCH"
      --device "$DEVICE"
      --run-dir "$RESULTS_ROOT/bbh/$RESULT_GROUP/$ARCH/$RUN_ID"
    )

    "${CMD[@]}"
  done
done
