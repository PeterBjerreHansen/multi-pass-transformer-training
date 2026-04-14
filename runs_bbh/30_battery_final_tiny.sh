#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-mps}"
ARCHITECTURE="${ARCHITECTURE:-memory_update}"
MODEL_SIZE="${MODEL_SIZE:-tiny}"
STEPS="${STEPS:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
EVAL_BATCHES="${EVAL_BATCHES:-3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
THRESHOLD="${THRESHOLD:-0.99}"
REVIEW_EASIER_EVERY="${REVIEW_EASIER_EVERY:-2}"
LR="${LR:-0.0001}"
N_PASS="${N_PASS:-4}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

if [[ "$#" -gt 0 ]]; then
  TASKS=("$@")
else
  TASKS=(permutation dyck boolean_rpn arithmetic tracking truth_graph order_deduction)
fi

ARCH_ARGS=(--architecture "$ARCHITECTURE")
if [[ "$ARCHITECTURE" != "transformer" ]]; then
  ARCH_ARGS+=(--n-pass "$N_PASS")
fi
if [[ "$ARCHITECTURE" == "memory_update" ]]; then
  ARCH_ARGS+=(--memory-update-gate on)
fi

for TASK in "${TASKS[@]}"; do
  python3 train_bbh_symbolic.py \
    --task "$TASK" \
    --supervision final \
    --model-size "$MODEL_SIZE" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --train-steps "$STEPS" \
    --eval-interval "$EVAL_INTERVAL" \
    --eval-batches "$EVAL_BATCHES" \
    --curriculum-threshold "$THRESHOLD" \
    --review-easier-every "$REVIEW_EASIER_EVERY" \
    --lr "$LR" \
    --log-policy promotions \
    "${ARCH_ARGS[@]}" \
    --log-jsonl runs_bbh/logs/30_"$TASK"_final_"$ARCHITECTURE"_"$MODEL_SIZE"_"$RUN_ID".jsonl
done
