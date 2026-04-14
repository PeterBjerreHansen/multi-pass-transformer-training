#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-mps}"
MODEL_SIZE="${MODEL_SIZE:-small}"
STEPS="${STEPS:-20000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
EVAL_BATCHES="${EVAL_BATCHES:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_LEVEL="${MAX_LEVEL:-64}"
THRESHOLD="${THRESHOLD:-0.99}"
REVIEW_EASIER_EVERY="${REVIEW_EASIER_EVERY:-2}"
LR="${LR:-0.0001}"
N_PASS="${N_PASS:-4}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

COMMON_ARGS=(
  --task walk
  --supervision final
  --model-size "$MODEL_SIZE"
  --device "$DEVICE"
  --max-level "$MAX_LEVEL"
  --batch-size "$BATCH_SIZE"
  --train-steps "$STEPS"
  --eval-interval "$EVAL_INTERVAL"
  --eval-batches "$EVAL_BATCHES"
  --curriculum-threshold "$THRESHOLD"
  --review-easier-every "$REVIEW_EASIER_EVERY"
  --lr "$LR"
  --log-policy promotions
)

python3 train_bbh_symbolic.py "${COMMON_ARGS[@]}" \
  --architecture transformer \
  --log-jsonl runs_bbh/logs/10_walk_final_transformer_"$MODEL_SIZE"_"$RUN_ID".jsonl

python3 train_bbh_symbolic.py "${COMMON_ARGS[@]}" \
  --architecture memory_tape \
  --n-pass "$N_PASS" \
  --log-jsonl runs_bbh/logs/10_walk_final_memory_tape_"$MODEL_SIZE"_"$RUN_ID".jsonl

python3 train_bbh_symbolic.py "${COMMON_ARGS[@]}" \
  --architecture memory_update \
  --n-pass "$N_PASS" \
  --memory-update-gate off \
  --log-jsonl runs_bbh/logs/10_walk_final_memory_update_"$MODEL_SIZE"_"$RUN_ID".jsonl
