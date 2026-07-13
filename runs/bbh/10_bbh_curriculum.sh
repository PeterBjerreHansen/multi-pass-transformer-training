#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-pointer_chasing}"
SEED="${SEED:-1337}"

for ARCH in transformer memory_tape memory_concat memory_update; do
  python -m experiments.train_bbh \
    --preset "${TASK}_main" \
    --architecture "${ARCH}" \
    --seed "${SEED}" \
    --run-dir "results/bbh/${TASK}/${ARCH}/seed_${SEED}"
done
