#!/usr/bin/env bash
set -euo pipefail

python -m experiments.train_bbh \
  --preset pointer_chasing_smoke \
  --architecture memory_tape \
  --device cpu

python -m experiments.train_trace \
  --preset random_graph_walk_smoke \
  --architecture memory_tape \
  --device cpu
