#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

for ARCH in memory_tape joint_memory_tape; do
  python -m experiments.train_bbh \
    --preset pointer_chasing_smoke \
    --architecture "${ARCH}" \
    --device cpu

  python -m experiments.train_trace \
    --preset random_graph_walk_smoke \
    --architecture "${ARCH}" \
    --device cpu
done
