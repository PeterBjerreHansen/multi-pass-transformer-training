#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cpu}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

python3 train_bbh_symbolic.py \
  --task walk \
  --supervision final \
  --architecture transformer \
  --model-size tiny \
  --device "$DEVICE" \
  --curriculum-start-level 1 \
  --max-level 4 \
  --batch-size 8 \
  --train-steps 20 \
  --eval-interval 10 \
  --eval-batches 1 \
  --curriculum-threshold 0.95 \
  --log-policy promotions \
  --log-jsonl runs_bbh/logs/00_smoke_walk_final_"$RUN_ID".jsonl
