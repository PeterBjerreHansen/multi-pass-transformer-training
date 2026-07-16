# Variable-depth training

## Experiment and hypothesis

This branch trains multi-pass models with a different pass depth on each optimizer step. `--train-pass-range MIN MAX` samples an inclusive integer depth, while `--sampled-tail-loss-weights EARLIER FINAL` applies loss only to the sampled run's final two passes. Evaluation, recurrent prefill, and generation continue to use the configured `n_pass`.

The hypothesis is that supervising the same recurrent computation at several depths encourages useful iterative refinement rather than specialization to one fixed pass count. The supplied benchmark compares fixed `K=4` with uniform `K in [2, 6]`; the latter also has an expected depth of four, making the compute comparison reasonably matched.

## Branch-specific code review

- `MultiPassTransformer.forward(idx, n_pass=None)` resolves an optional call-time depth without mutating the model config. The ordinary call path is unchanged.
- Shared helpers validate the sampled range, construct `[0, ..., EARLIER, FINAL]` weights, and sample depth with a dedicated Python RNG.
- Trace and BBH training save and restore the depth RNG plus the cumulative sampled-depth histogram, and log the chosen depth and effective loss weights.
- The fixed `--pass-loss-weights` interface remains separate, avoiding ambiguous reinterpretation of a config-sized list.
- Tests cover the forward override, config immutability, deterministic sampling, dynamic tail weights, and CLI metric output.

The implementation is compact and easy to follow. Its main weakness is duplicated sampler/checkpoint plumbing in the two training loops. Also, the model argument is currently positional-or-keyword; making `n_pass` keyword-only would better protect call sites as more experimental forward controls are added.

## Merging into `main`

This is one of the easier branches to merge. It adds no parameters or checkpoint tensors, and leaving `--train-pass-range` unset preserves fixed-depth behavior.

The likely conflicts are in `MultiPassTransformer.forward`, `forward_and_loss`, and the training scripts, because the position-offset and stale-memory branches touch the same seams. Resolve them by adopting a keyword-only combined forward signature and computing `effective_n_pass` before any other pass-dependent validation. If both variable depth and stale-memory routing are kept, construct the routing plan using the sampled depth, not `config.n_pass`.

For the cleanest long-term merge, introduce a small checkpointable training-sampler container holding optional depth, position-offset, and corruption RNGs. This is a localized refactor of training setup and checkpoint `extra_state`; the model architecture itself does not need restructuring.
