#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PRESET="${PRESET:-shortest_path_main}"
SEEDS="${SEEDS:-1337 2027 4099}"
DEVICE="${DEVICE:-mps}"
TRAIN_STEPS="${TRAIN_STEPS:-200000}"
EVAL_BATCHES="${EVAL_BATCHES:-64}"
RESULT_ROOT="${RESULT_ROOT:-results/trace/ablations/sandwich_recurrence}"

if (( TRAIN_STEPS < 2 )); then
  echo "TRAIN_STEPS must be at least 2" >&2
  exit 2
fi
WARMUP_STEPS="${WARMUP_STEPS:-4000}"
if (( WARMUP_STEPS >= TRAIN_STEPS )); then
  WARMUP_STEPS=$((TRAIN_STEPS / 10))
  (( WARMUP_STEPS < 1 )) && WARMUP_STEPS=1
fi

run_variant() {
  local variant="$1"
  local layout="$2"
  local persistent_input="$3"
  local iterations="$4"

  for seed in ${SEEDS}; do
    local run_dir="${RESULT_ROOT}/${variant}/seed_${seed}"
    python -m experiments.train_trace \
      --preset "${PRESET}" \
      --architecture looped_transformer \
      --loop-layout "${layout}" \
      --loop-persistent-input "${persistent_input}" \
      --n-layer 4 \
      --n-pass "${iterations}" \
      --inference-mode recompute \
      --train-steps "${TRAIN_STEPS}" \
      --lr-decay-steps "${TRAIN_STEPS}" \
      --lr-warmup-steps "${WARMUP_STEPS}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      --run-dir "${run_dir}"

    python -m experiments.eval_trace \
      --input-run-dir "${run_dir}" \
      --output-dir "${run_dir}/drift/recompute" \
      --checkpoint best \
      --inference-mode recompute \
      --token-selection argmax \
      --eval-batches "${EVAL_BATCHES}" \
      --device "${DEVICE}"

    python -m experiments.diagnose_looped \
      --input-run-dir "${run_dir}" \
      --output "${run_dir}/diagnostics.json" \
      --checkpoint best \
      --eval-batches 4 \
      --device "${DEVICE}"
  done
}

# Four physical layers are resident in every variant. The full control performs
# 4 layers x 4 iterations = 16 block applications. The sandwich performs
# 1 prelude + (2 core layers x 7 iterations) + 1 coda = 16 applications.
run_variant full_no_input full off 4
run_variant sandwich_no_input sandwich off 7
run_variant full_persistent_input full on 4
run_variant sandwich_persistent_input sandwich on 7

python -m experiments.summarize_ablation \
  --root "${RESULT_ROOT}" \
  --control full_no_input \
  --variants sandwich_no_input full_persistent_input sandwich_persistent_input \
  --quality-metric drift.recompute.optimal_path \
  --recommendation-mode pareto \
  --output-dir "${RESULT_ROOT}/summary_full_control"

# Isolate the persistent-input effect inside the sandwich topology as a second
# paired comparison rather than inferring it through the full-stack control.
python -m experiments.summarize_ablation \
  --root "${RESULT_ROOT}" \
  --control sandwich_no_input \
  --variants sandwich_persistent_input \
  --quality-metric drift.recompute.optimal_path \
  --recommendation-mode pareto \
  --output-dir "${RESULT_ROOT}/summary_sandwich_input"
