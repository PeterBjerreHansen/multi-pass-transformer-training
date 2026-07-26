#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT
cd "${ROOT}"

pytest -q
python -m experiments.train_trace --preset random_graph_walk_smoke --architecture memory_tape \
  --null-memory-slot on --device cpu --run-dir "${TMP_DIR}/cpu"
python -m experiments.eval_diagnostics --input-run-dir "${TMP_DIR}/cpu" \
  --output "${TMP_DIR}/cpu/diagnostics.json" --batch-size 2 --device cpu

if python -c 'import torch,sys; sys.exit(0 if torch.backends.mps.is_available() else 1)'; then
  python -m experiments.train_trace --preset random_graph_walk_smoke --architecture memory_tape \
    --null-memory-slot on --device mps --run-dir "${TMP_DIR}/mps"
fi

git diff --check
