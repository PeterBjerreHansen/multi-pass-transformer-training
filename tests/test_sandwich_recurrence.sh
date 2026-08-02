#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

RESULT_ROOT="${RESULT_ROOT:-$(mktemp -d /tmp/mpt-sandwich-recurrence.XXXXXX)}"

pytest -q

run_smoke() {
  local name="$1"
  local layout="$2"
  local persistent_input="$3"
  local iterations="$4"
  local run_dir="${RESULT_ROOT}/${name}"

  python -m experiments.train_trace \
    --preset shortest_path_smoke \
    --architecture looped_transformer \
    --loop-layout "${layout}" \
    --loop-persistent-input "${persistent_input}" \
    --n-layer 4 \
    --n-head 1 \
    --n-embd 16 \
    --n-pass "${iterations}" \
    --inference-mode recompute \
    --device cpu \
    --run-dir "${run_dir}"

  python -m experiments.eval_trace \
    --input-run-dir "${run_dir}" \
    --output-dir "${run_dir}/drift/recompute" \
    --inference-mode recompute \
    --token-selection argmax \
    --eval-batches 1 \
    --device cpu

  python -m experiments.diagnose_looped \
    --input-run-dir "${run_dir}" \
    --eval-batches 1 \
    --device cpu
}

run_smoke full_no_input full off 2
run_smoke sandwich_no_input sandwich off 3
run_smoke full_persistent_input full on 2
run_smoke sandwich_persistent_input sandwich on 3

if python -c 'import torch; raise SystemExit(0 if torch.backends.mps.is_available() else 1)'; then
  python -m experiments.train_trace \
    --preset shortest_path_smoke \
    --architecture looped_transformer \
    --loop-layout sandwich \
    --loop-persistent-input on \
    --n-layer 4 \
    --n-pass 3 \
    --inference-mode recompute \
    --device mps \
    --run-dir "${RESULT_ROOT}/mps"
fi

git diff --check
