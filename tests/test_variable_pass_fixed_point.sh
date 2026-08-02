#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT
cd "${ROOT}"

pytest -q

python -m experiments.train_trace --preset shortest_path_smoke \
  --architecture memory_tape --train-pass-mode fixed \
  --min-n-pass 2 --max-n-pass 3 --pass-loss-weights 0 0 1 \
  --device cpu --run-dir "${TMP_DIR}/fixed"
python -m experiments.train_trace --preset shortest_path_smoke \
  --architecture memory_tape --train-pass-mode uniform \
  --min-n-pass 2 --max-n-pass 4 \
  --device cpu --run-dir "${TMP_DIR}/uniform"
python -m experiments.train_trace --preset shortest_path_smoke \
  --architecture memory_tape --train-pass-mode fixed_point \
  --min-n-pass 2 --max-n-pass 4 \
  --fixed-point-residual-threshold 1000 --fixed-point-kl-threshold 1000 \
  --device cpu --run-dir "${TMP_DIR}/fixed_point"

python -m experiments.eval_trace --input-run-dir "${TMP_DIR}/fixed_point" \
  --output-dir "${TMP_DIR}/fixed_point/eval" --inference-mode append_recurrent \
  --eval-pass-mode fixed_point --min-n-pass 2 --max-n-pass 4 \
  --fixed-point-residual-threshold 1000 --fixed-point-kl-threshold 1000 \
  --token-selection argmax --eval-batches 1 --device cpu
python -m experiments.diagnose_memory --input-run-dir "${TMP_DIR}/fixed_point" \
  --output "${TMP_DIR}/fixed_point/diagnostics.json" \
  --batch-size 2 --eval-batches 1 --device cpu

if python -c 'import torch,sys; sys.exit(0 if torch.backends.mps.is_available() else 1)'; then
  python -m experiments.train_trace --preset shortest_path_smoke \
    --architecture memory_tape --train-pass-mode fixed_point \
    --min-n-pass 2 --max-n-pass 4 \
    --fixed-point-residual-threshold 1000 --fixed-point-kl-threshold 1000 \
    --device mps --run-dir "${TMP_DIR}/mps"
fi

git diff --check
