# Sandwich recurrence ablation

This exploratory branch tests two design claims from the living research report
[Towards Looped Models Done Right](https://ifm-research.notion.site/Towards-Looped-Models-Done-Right-3ade511912ec8128987dfeb7a5580043),
specifically its experiments on
[localized recurrence](https://ifm-research.notion.site/Towards-Looped-Models-Done-Right-3ade511912ec8128987dfeb7a5580043#3ade511912ec80bd9fa7d77206c17cbb)
and
[persistent input injection](https://ifm-research.notion.site/Towards-Looped-Models-Done-Right-3ade511912ec8128987dfeb7a5580043#3ade511912ec80e89c19eaac7d8864a9):

1. Localizing recurrence inside a one-shot prelude and coda may be better than
   repeatedly applying the complete Transformer stack.
2. Repeated access to a fixed, contextualized prelude representation may help
   the recurrent core retain the problem specification.

The claims are hypotheses for this much smaller codebase, not established
results here. The cited report uses substantially larger models and training
runs, and its public page describes ongoing work.

## Architecture and ablations

The branch adds one `looped_transformer` architecture with two orthogonal
settings:

- `--loop-layout full`: embed once and repeat every Transformer block.
- `--loop-layout sandwich`: run the first block once as a prelude, repeat the
  middle blocks as the recurrent core, and run the last block once as a coda.
- `--loop-persistent-input off`: recur only on the current hidden state.
- `--loop-persistent-input on`: before every recurrent iteration, combine the
  state with a fixed input using learned per-channel retention and input
  coefficients. This is the raw token-plus-position embedding in `full` and
  the contextualized prelude representation in `sandwich`.

The persistent write follows the report's form
`z <- alpha * z + delta * W_in(e)`, with
`delta = softplus(b_delta)` and
`alpha = exp(-delta * exp(retention_log_scale))`. It is not the old scalar
memory-attention gate. It starts gently at `delta=0.1`, `alpha=exp(-0.1)`, and
`W_in=I`.

All variants instantiate the same four Transformer blocks and the same input
write parameters. Disabled input-write parameters receive no gradients. The
benchmark also matches Transformer-block applications:

- full control: `4 blocks * 4 iterations = 16` applications;
- sandwich: `1 prelude + 2 core blocks * 7 iterations + 1 coda = 16`.

Only the final iteration computes a language-model readout or contributes
training loss. The offline loop diagnostic explicitly requests earlier
readouts, so diagnostic convenience does not burden or confound training
throughput.

## Branch-specific code review

- `models.py` contains `LoopedTransformerConfig` and `LoopedTransformer`. The
  model reuses the ordinary causal `Block`; no memory attention, random initial
  state, hierarchical state, or task-specific component is introduced.
- `model_factory.py` registers the architecture and explicitly classifies it as
  recompute-only. Depth recurrence over a complete prompt does not supply an
  append-aligned tape, so mapping it onto `append_recurrent` would be misleading.
- `experiments/diagnose_looped.py` reports loss, state-update size, and state
  cosine at each iteration, plus the learned input-write coefficients.
- `scripts/trace/ablate_sandwich_recurrence.sh` runs the four matched ablations
  on shortest path, evaluates the best checkpoint, and produces paired
  summaries against the full-stack control and within the sandwich topology.
- `tests/test_sandwich_recurrence.py` checks causality, topology call counts,
  parameter matching, gradients, generation mode, and factory integration.

Run the software and CPU/MPS smoke checks with:

```bash
bash tests/test_sandwich_recurrence.sh
```

Run the default three-seed benchmark with:

```bash
caffeinate -dimsu bash scripts/trace/ablate_sandwich_recurrence.sh
```

`DEVICE`, `SEEDS`, `TRAIN_STEPS`, `EVAL_BATCHES`, `WARMUP_STEPS`, `PRESET`, and
`RESULT_ROOT` may be overridden for pilots. Results remain under ignored
`results/`.

## Merge assessment

The experiment is deliberately isolated behind a new architecture name and is
easy to merge mechanically. It should only be merged into `main` as a supported
model if the sandwich layout improves shortest-path quality or efficiency in
paired seeds and remains useful on at least one non-graph task. Otherwise the
branch is still useful as an archived, reproducible negative result.

The main structural caveat is generation: this model supports recomputation,
not the repository's append-recurrent memory protocol. If it succeeds and later
needs deployment-aligned recurrence, the least painful path is a separate
follow-up that places an explicit aligned memory mechanism inside the recurrent
core. That should not be folded into this topology/input-injection ablation.
