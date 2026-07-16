#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-mps}"
SEEDS="${SEEDS:-1337 2027 4099}"
TRAIN_STEPS="${TRAIN_STEPS:-50000}"
RESULT_ROOT="${RESULT_ROOT:-results/ablations/06_position_offsets}"

run_eval() {
  local run_dir="$1" seed="$2"
  for offset in 0 16 32 64; do
    for mode in recompute append_recurrent; do
      output_dir="${run_dir}/drift/offset_${offset}/${mode}"
      if [[ "${offset}" == 0 ]]; then
        output_dir="${run_dir}/drift/${mode}"
      fi
      python -m experiments.eval_trace_drift --input-run-dir "${run_dir}" \
        --run-dir "${output_dir}" --inference-mode "${mode}" --eval-position-offset "${offset}" \
        --token-selection argmax --device "${DEVICE}" --seed "${seed}"
    done
  done
  python -m experiments.eval_diagnostics --input-run-dir "${run_dir}" \
    --output "${run_dir}/diagnostics.json" --device "${DEVICE}" --seed "${seed}"
}

for seed in ${SEEDS}; do
  for variant in offset_zero offset_uniform_0_64; do
    run_dir="${RESULT_ROOT}/${variant}/seed_${seed}"
    extra_args=()
    if [[ "${variant}" == offset_uniform_0_64 ]]; then
      extra_args+=(--train-position-offset-max 64)
    fi
    python -m experiments.train_trace --preset random_graph_walk_main --architecture memory_tape \
      --max-position-embeddings 149 --token-selection argmax --train-steps "${TRAIN_STEPS}" \
      --device "${DEVICE}" --seed "${seed}" --run-dir "${run_dir}" "${extra_args[@]}"
    run_eval "${run_dir}" "${seed}"
  done
done

python -m experiments.summarize_ablation --root "${RESULT_ROOT}" --control offset_zero \
  --variants offset_uniform_0_64 --recommendation-mode position-offset
