#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

RESULT_ROOT="${RESULT_ROOT:-$(mktemp -d /tmp/mpt-stale-memory-training.XXXXXX)}"
CPU_RUN="${RESULT_ROOT}/cpu"

pytest -q

python -m experiments.train_trace \
  --preset random_graph_walk_smoke \
  --architecture memory_tape \
  --stale-memory-prob 0.5 \
  --device cpu \
  --run-dir "${CPU_RUN}"

python -m experiments.train_trace \
  --preset random_graph_walk_smoke \
  --resume-from "${CPU_RUN}" \
  --train-steps 1 \
  --device cpu \
  --run-dir "${CPU_RUN}"

python -m experiments.eval_diagnostics \
  --input-run-dir "${CPU_RUN}" \
  --device cpu \
  --batch-size 2 \
  --eval-batches 1 \
  --extra-passes 2

python -m experiments.eval_trace_drift \
  --input-run-dir "${CPU_RUN}" \
  --inference-mode recompute \
  --token-selection argmax \
  --device cpu \
  --eval-batches 1 \
  --run-dir "${CPU_RUN}/drift/recompute"

python -m experiments.eval_trace_drift \
  --input-run-dir "${CPU_RUN}" \
  --inference-mode append_recurrent \
  --token-selection argmax \
  --device cpu \
  --eval-batches 1 \
  --run-dir "${CPU_RUN}/drift/append_recurrent"

if python -c 'import torch; raise SystemExit(0 if torch.backends.mps.is_available() else 1)'; then
  python -m experiments.train_trace \
    --preset random_graph_walk_smoke \
    --architecture memory_tape \
    --stale-memory-prob 0.25 \
    --device mps \
    --run-dir "${RESULT_ROOT}/mps"
fi

git diff --check
