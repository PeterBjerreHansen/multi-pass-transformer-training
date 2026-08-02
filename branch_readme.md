# Variable-pass and fixed-point training

## Experiment

This branch compares three training policies without creating three separate
model implementations:

| `--train-pass-mode` | Training depth | Training loss |
| --- | --- | --- |
| `fixed` | exactly `max_n_pass` | configurable `pass_loss_weights` |
| `uniform` | integer sampled uniformly from `[min_n_pass, max_n_pass]` | final pass only |
| `fixed_point` | first jointly converged pass, capped by `max_n_pass` | final executed pass only |

`max_n_pass` replaces the ambiguous fixed-depth `n_pass` experiment knob. In
fixed mode it is simply the old fixed pass count. In uniform and fixed-point
modes it is a hard compute budget. `min_n_pass` is the lower end of the uniform
range and prevents premature fixed-point halting.

Evaluation has only two meaningful deterministic policies:
`--eval-pass-mode fixed` and `--eval-pass-mode fixed_point`. The same minimum,
maximum, and thresholds are shared with training. `experiments.eval_trace`
accepts evaluation-time overrides, allowing every checkpoint to be compared at
the same fixed depth and under the same adaptive stopping rule. Uniform
evaluation is deliberately omitted because evaluation noise would obscure the
training comparison.

## Fixed-point criterion

For example `b`, consecutive memory tapes are compared using

```
r_b = ||M_new[b] - M_old[b]||_inf / (||M_new[b]||_inf + 1e-8).
```

This is the per-example relative L-infinity residual used as a halting signal
by [Fixed-Point Reasoners](https://arxiv.org/pdf/2606.18206v1), adapted to the
memory tape because the tape—not the ordinary hidden stream—is the recurrent
state in these architectures. Padding is excluded.

The companion output criterion is the mean tokenwise
`KL(p_old || p_new)` over valid positions. An example halts only when both

```
r_b <= fixed_point_residual_threshold
logit_kl_b <= fixed_point_kl_threshold
```

hold after at least `min_n_pass` passes. Converged examples are frozen while
the remaining sub-batch continues, so different examples genuinely execute a
different number of passes. The hard stopping decision is detached; the final
loss backpropagates through every pass actually executed for that example.

This is intentionally not a complete reproduction of FPRM training. It does
not add damping, residual scaling, truncated-BPTT windows, or deep supervision.
Those changes would confound the first question: whether variable-pass or
fixed-point-oriented training improves the existing memory-tape architecture.

The initial thresholds are `0.1` for the tape residual, following the FPRM
default, and `1e-3` for logit KL. They are provisional experiment settings, not
claims of universal calibration.

## Why these two measurements

- Absolute L-infinity distance is equally sensitive to a localized failure but
  makes the tolerance depend directly on tape scale.
- Relative L2 or RMS residuals are smoother and less outlier-sensitive, but a
  small average can hide one tape coordinate that is still changing sharply.
- Cosine distance ignores scale, which is useful diagnostically but can call
  two states stable while their magnitudes continue to change.
- Hidden-state residuals measure a representation that is recomputed each pass;
  the memory tape is the state actually fed into the next pass.
- Logit KL alone can stop on an output plateau while the recurrent state keeps
  moving. Conversely, the tape criterion alone does not guarantee that the
  decoded prediction is stable. Requiring both is therefore a conservative
  first policy.

## Ablation

`scripts/trace/ablate_variable_pass_fixed_point.sh` runs three variants:

- `fixed_k4`: fixed four-pass training, final-pass loss;
- `uniform_k2_k6`: uniform depths 2–6, whose expected depth is four;
- `fixed_point_k2_k6`: fixed-point training with the same 2–6 bounds.

Every best checkpoint is evaluated under fixed `K=4` for a matched-compute
quality comparison and under the common fixed-point rule with a six-pass cap.
Both `recompute` and `append_recurrent` generation are retained. The adaptive
evaluation reports teacher-forced mean/max pass count, convergence rate, final
tape residual, and final logit KL. These are explicitly named
`teacher_forced_*`; they are not generation-policy statistics. The standard
offline memory diagnostic is also produced.

For a short sanity run use
`scripts/trace/pilot_variable_pass_fixed_point.sh`. Full branch validation is
`tests/test_variable_pass_fixed_point.sh`.

## Implementation notes

- `MultiPassTransformer.forward` takes keyword-only policy overrides. Fixed
  execution remains the default path.
- Fixed-point execution operates on the active sub-batch and freezes completed
  examples, rather than using a batch-average stopping decision.
- `TrainingPassController` centralizes sampling, final-only loss selection,
  cumulative pass/convergence summaries, and checkpoint restoration for both
  trace and BBH training loops.
- Threshold reductions use tensors already produced by a pass and run only in
  fixed-point mode. Fixed and uniform training do not pay for KL/residual
  calculations. Logging reuses the halting values and adds no extra forwards.
