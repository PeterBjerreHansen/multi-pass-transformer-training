#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-1337}"

for ARCH in transformer memory_tape joint_memory_tape memory_concat memory_update; do
  python -m experiments.train_trace \
    --preset othello_main \
    --architecture "${ARCH}" \
    --seed "${SEED}" \
    --run-dir "results/trace/othello/${ARCH}/seed_${SEED}"
done
