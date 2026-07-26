#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

for ARCH in memory_tape joint_memory_tape memory_concat memory_add memory_state memory_update; do
  python -m experiments.train_bbh \
    --preset pointer_chasing_smoke \
    --architecture "${ARCH}" \
    --device cpu

  python -m experiments.train_trace \
    --preset shortest_path_smoke \
    --architecture "${ARCH}" \
    --device cpu
done
