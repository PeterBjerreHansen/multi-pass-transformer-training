#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

RUN_DIR="${RUN_DIR:?Set RUN_DIR to a trained Othello run directory}"
DEVICE="${DEVICE:-mps}"

python -m experiments.eval_othello \
  --input-run-dir "${RUN_DIR}" \
  --device "${DEVICE}" \
  --examples 64 \
  --evaluation-mode all \
  --inference-modes recompute append_recurrent \
  --token-selection argmax
