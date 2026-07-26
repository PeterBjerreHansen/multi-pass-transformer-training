# Stale-memory training

## Experiment and hypothesis

This branch tests deployment-shaped memory corruption for multi-pass models. During training, `--stale-memory-prob P` makes passes three and later occasionally read a learned memory from an older refinement pass instead of the immediately preceding pass. Corruption is applied per example and token position, only on the teacher-forced answer suffix; prompt, separator, EOS, padding, evaluation, and generation keep the normal memory schedule.

The hypothesis is that exposure to mixed refinement levels makes the model less brittle when deployed memories were produced with a different effective number of passes. The benchmark compares `P = 0, 0.1, 0.25, 0.5` and measures ordinary quality, recurrent drift, and the loss incurred when the final pass is deliberately rerun against each earlier learned tape.

For a short local `P=0` versus `P=0.25` run with both inference modes and diagnostics, use `bash scripts/pilot_stale_memory_training.sh`. It defaults to one seed and 250 steps; `DEVICE`, `TRAIN_STEPS`, `TREATMENT_PROB`, `BATCH_SIZE`, and `RESULT_ROOT` are overrideable.

## Branch-specific code review

- `MultiPassTransformer.forward` accepts an optional per-position memory-source plan. The default path remains numerically identical, while the routed path validates that a pass can only read an available learned memory.
- `experiments/common.py` constructs suffix-only routes and records realized corruption rates and source-pass histograms. A dedicated CPU `torch.Generator` is independent of task sampling and is saved in checkpoints, giving deterministic resume behavior.
- Both training entry points expose `--stale-memory-prob`, log window and cumulative statistics, and restore the corruption RNG and counters.
- `refinement_robustness` adds a direct diagnostic for the hypothesis by comparing the final pass with memories from every earlier pass.
- The test coverage includes routing equivalence, causality, gradient flow, suffix masking, RNG determinism, checkpoint resume, CLI logging, and an end-to-end CPU smoke cycle.

The design is deliberately low-level and flexible, but the route tensor has shape `[passes, batch, sequence]` and is sampled on CPU before being copied to the training device. That is simple and reproducible; if profiling shows overhead at large batch sizes, it can be optimized by sampling only eligible suffix indices.

## Merging into `main`

This can be merged without an architectural rewrite because `P=0` preserves existing behavior and checkpoint shapes. The main conflict risk is textual: variable depth and position offsets also change `MultiPassTransformer.forward`, `forward_and_loss`, and both training loops.

The least-pain combined API is:

```python
forward(
    idx,
    *,
    n_pass: int | None = None,
    position_offset: int = 0,
    memory_source_passes: torch.Tensor | None = None,
)
```

Resolve the effective pass count first, then validate the route plan against that count. Keep route construction in the training layer rather than the model config, because it is stochastic training policy rather than architecture. If several training-time ablations are merged, a small shared helper returning forward keyword arguments and checkpointable sampler state would remove the duplicated wiring from `train_trace.py` and `train_bbh.py` without requiring a broader trainer rewrite.
