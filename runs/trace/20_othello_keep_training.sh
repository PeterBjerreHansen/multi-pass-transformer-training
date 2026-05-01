#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

RESUME_FROM="${RESUME_FROM:-results/trace/othello/transformer/20260423_153518}"
STEPS="${STEPS:-100_000}"

python3 -m experiments.train_trace \
  --resume-from "$RESUME_FROM" \
  --train-steps "$STEPS"
