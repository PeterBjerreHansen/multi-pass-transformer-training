# Multi-Pass Transformer Training

Research code for training causal Transformers through repeated parallel passes over the same sequence, with a learned per-token memory tape connecting consecutive passes.

The repository is deliberately small and self-contained. It contains:

- a causal Transformer baseline;
- a cross-attention `MemoryTapeTransformer`;
- memory-concatenation and memory-update variants;
- synthetic final-answer and trace-generation tasks;
- deterministic training/evaluation utilities;
- free-generation drift evaluation;
- standalone memory-use and pass-dynamics diagnostics.

## Core recurrence

For a token sequence `x` and `K` passes:

```text
M_0 = 0

for k = 1 ... K:
    memory_tape = shift_right(M_{k-1})
    H_k = TransformerPass(x, memory_tape)
    logits_k = LMHead(H_k)
    M_k = normalize(MemHead(H_k))
```

The right shift enforces the causal rule that position `t` may read memories written at positions `< t`, but never its own or future memory.

## MemoryTape architecture

Each block applies:

```text
X = X + CausalSelfAttention(LN(X))
X = X + gate * CausalMemoryCrossAttention(LN(X), Normalize(memory_tape))
X = X + MLP(LN(X))
```

The implementation uses:

- bias-free memory query/key/value/output projections;
- one learned scalar memory gate per layer, initialized to `0.1`;
- normalized emitted memory vectors;
- exact zero memory output when the readable tape is zero;
- structured per-pass outputs.

```python
output = model(tokens)

output.logits
output.hidden_states
output.final_memory
output.passes[0].logits
output.passes[0].memory_states
```

## Pass-weighted loss

Pass weights are always normalized internally:

```text
normalized_weight_k = weight_k / sum_j weight_j
loss = sum_k normalized_weight_k * cross_entropy_k
```

Consequently, these are equivalent:

```text
[0, 0, 1, 1]
[0, 0, 0.5, 0.5]
```

If weights are omitted, all passes receive equal weight.

## Inference modes

Multi-pass models expose exactly two inference modes.

### `recompute`

After every generated token, rerun all `K` passes over the current context. This is the direct inference procedure for the finite-pass training model.

### `append_recurrent`

1. Run all `K` passes over the prompt.
2. Cache the final prompt memory tape `M_K`.
3. Generate the first token from the final prompt logits.
4. For each subsequent prediction, run one pass over the current prefix using the persistent cache.
5. Append only the memory emitted for the newest token.
6. Never rewrite older cached memories.

```python
result = model.generate(
    prompt,
    max_new_tokens=32,
    inference_mode="append_recurrent",
    do_sample=False,
)
```

Persistent recurrent generation does not crop the memory context. Every context used to compute logits must fit within `block_size`. The returned sequence may be one token longer because the final sampled token does not need another recurrent update.

The causal Transformer baseline supports only `recompute`.

## Tasks

### Final-answer curricula

- pointer chasing;
- permutation updates;
- object tracking;
- finite-state-machine execution.

Training promotes the curriculum only on deterministic, fixed evaluation batches for each level.

### Trace generation

- random labelled graph walks;
- legal Othello move traces.

Othello traces are generated deterministically and cached as compact NumPy arrays. Dataset contents are invariant to the number of generation workers.

## Training

Install:

```bash
pip install -e ".[test]"
```

BBH-style curriculum smoke run:

```bash
python -m experiments.train_bbh \
  --preset pointer_chasing_smoke \
  --architecture memory_tape \
  --device cpu
```

Trace smoke run:

```bash
python -m experiments.train_trace \
  --preset random_graph_walk_smoke \
  --architecture memory_tape \
  --device cpu
```

Full presets can be inspected with:

```bash
python -m experiments.train_bbh --help
python -m experiments.train_trace --help
```

Checkpoints contain model, optimizer, Python/Torch/CUDA RNG state, and task-specific training RNG state. Evaluation sampling is isolated from training RNG state.

## Free-generation drift evaluation

Evaluate a saved trace checkpoint under either inference mode:

```bash
python -m experiments.eval_trace_drift \
  --input-run-dir results/trace/random_graph_walk/memory_tape/<run> \
  --inference-mode append_recurrent \
  --token-selection argmax
```

This writes:

- aggregate free-generation legality metrics;
- per-position probability of remaining legal.

## Standalone diagnostics

Diagnostics are intentionally separate from training:

```bash
python -m experiments.eval_diagnostics \
  --input-run-dir results/trace/random_graph_walk/memory_tape/<run> \
  --extra-passes 6
```

### Memory interventions

The final trained pass is reevaluated with:

- the correct previous-pass tape;
- zero memory;
- memory from another batch example;
- causally resampled memory positions, where position `j` draws only from positions `<= j`;
- causal prefix-mean memories;
- an additional one-position memory lag.

The output reports NLL changes relative to correct memory. Every position intervention preserves causality: no future-derived memory is moved into an earlier readable position. The interventions distinguish an ignored tape from sequence-specific, temporally aligned, or coarse-summary memory use.

### Pass and equilibrium dynamics

The diagnostics report:

- loss at every trained pass;
- consecutive-pass logit KL divergence;
- memory RMS change and cosine distance;
- hidden-state RMS change;
- memory norm, feature variation, and effective rank;
- trajectories from additional evaluation-only passes starting at `M_K`.

Additional passes test whether the learned update appears to converge, drift, or oscillate. They do not change model parameters.

## Tests

```bash
pytest -q
```

The tests cover:

- token and memory causality;
- exact memory shifting;
- zero-memory behavior;
- gradient flow through earlier memory writes;
- pass-weight normalization;
- normalized memory output;
- final-memory recurrent prefill;
- append-only cache immutability;
- recurrent context limits;
- deterministic evaluation;
- checkpoint continuation;
- task solvers and Othello legality;
- diagnostic execution.

## Repository layout

```text
models.py                         model components and architectures
model_factory.py                  architecture construction
experiments/common.py             shared training/evaluation/checkpoint code
experiments/train_bbh.py          final-answer curricula
experiments/train_trace.py        fixed trace training
experiments/eval_trace_drift.py   free-generation drift evaluation
experiments/eval_diagnostics.py   memory and pass-dynamics diagnostics
tasks/                            symbolic task definitions
tests/                            semantic and integration tests
```

The refreshed MemoryTape parameterization and inference semantics are intentionally incompatible with checkpoints produced by older implementations.
