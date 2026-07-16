#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

RESULT_ROOT="${RESULT_ROOT:-$(mktemp -d /tmp/mpt-generation-aligned.XXXXXX)}"
CPU_RUN="${RESULT_ROOT}/cpu"

pytest -q

python -m experiments.train_trace \
  --preset random_graph_walk_smoke \
  --architecture memory_tape \
  --append-train-prob 1 \
  --append-train-microbatch-size 1 \
  --append-train-horizon 2 \
  --append-train-warmup-steps 0 \
  --append-train-ramp-steps 0 \
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

for inference_mode in recompute append_recurrent; do
  python -m experiments.eval_trace_drift \
    --input-run-dir "${CPU_RUN}" \
    --inference-mode "${inference_mode}" \
    --token-selection argmax \
    --device cpu \
    --eval-batches 1 \
    --run-dir "${CPU_RUN}/drift/${inference_mode}"
done

if python -c 'import torch; raise SystemExit(0 if torch.backends.mps.is_available() else 1)'; then
  python -m experiments.train_trace \
    --preset random_graph_walk_smoke \
    --architecture memory_tape \
    --append-train-prob 1 \
    --append-train-microbatch-size 1 \
    --append-train-horizon 2 \
    --append-train-warmup-steps 0 \
    --append-train-ramp-steps 0 \
    --device mps \
    --run-dir "${RESULT_ROOT}/mps"
fi

git diff --check
