#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
TMP_DIR="$(mktemp -d /tmp/mpt-conditional-memory-gates.XXXXXX)"
trap 'rm -rf "${TMP_DIR}"' EXIT

pytest -q

run_dir="${TMP_DIR}/cpu/memory_tape"
python -m experiments.train_trace \
  --preset shortest_path_smoke \
  --architecture memory_tape \
  --conditional-memory-gate on \
  --device cpu \
  --run-dir "${run_dir}"

for inference_mode in recompute append_recurrent; do
  python -m experiments.eval_trace \
    --input-run-dir "${run_dir}" \
    --inference-mode "${inference_mode}" \
    --token-selection argmax \
    --device cpu \
    --eval-batches 1 \
    --output-dir "${run_dir}/drift/${inference_mode}"
done
python -m experiments.diagnose_memory \
  --input-run-dir "${run_dir}" \
  --device cpu \
  --batch-size 2 \
  --eval-batches 1 \
  --output "${run_dir}/diagnostics.json"

if python -c 'import torch; raise SystemExit(0 if torch.backends.mps.is_available() else 1)'; then
  python -m experiments.train_trace \
    --preset shortest_path_smoke \
    --architecture memory_tape \
    --conditional-memory-gate on \
    --device mps \
    --run-dir "${TMP_DIR}/mps/memory_tape"
fi

git diff --check
