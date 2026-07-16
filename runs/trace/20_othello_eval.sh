#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:?Set RUN_DIR to a trained Othello run directory}"
DEVICE="${DEVICE:-mps}"
EXAMPLES="${EXAMPLES:-64}"
EVALUATION_MODE="${EVALUATION_MODE:-all}"
TOKEN_SELECTION="${TOKEN_SELECTION:-argmax}"

python -m experiments.eval_othello \
  --input-run-dir "${RUN_DIR}" \
  --device "${DEVICE}" \
  --examples "${EXAMPLES}" \
  --evaluation-mode "${EVALUATION_MODE}" \
  --inference-modes recompute append_recurrent \
  --token-selection "${TOKEN_SELECTION}"
