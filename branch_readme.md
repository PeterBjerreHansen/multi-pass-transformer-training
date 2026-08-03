# Archived architectures

This branch preserves the exploratory architectures removed from `main`. The
current root `README.md` is kept synchronized with `main`; this file contains
the historical architecture descriptions and the notes needed to revive them.

## What is archived

All archived models use the shared `MultiPassTransformer` wrapper. A pass reads
the one-position-shifted tape written by the previous pass, produces logits,
and writes a new per-token memory tape. The differences below are only in how
the token stream and shifted tape are combined inside each decoder block.

### JointMemoryTape

JointMemoryTape replaces separate token self-attention and memory
cross-attention with one causal attention distribution over two source banks.
Token and memory keys/values have separate projections, but they compete in the
same softmax:

```text
Q = LN_q(H) W_q
K = concat(token_keys, memory_keys)
V = concat(token_values, memory_values)
H <- H + causal_attention(Q, K, V)
H <- H + MLP(LN(H))
```

The shifted tape is content-addressable, but its slots share probability mass
with token slots. A zero memory bank therefore still changes the first-pass
attention distribution. This makes JointMemoryTape a substantial architectural
variant, not a parameter-isolated MemoryTape ablation.

### MemoryConcat

MemoryConcat fuses the token stream and normalized shifted tape once before the
ordinary decoder stack:

```text
H = W_fuse(concat(X, LN_mem(R)))
H <- ordinary causal decoder blocks
```

It tests whether an aligned memory feature is useful at all without giving the
model a separate content-addressed reader. The fusion projection was
initialized close to a token identity with a small memory contribution.

### MemoryState

MemoryState makes the memory-derived stream the direct working representation,
while retaining ordinary causal decoder blocks:

```text
H = LN_mem_in(R) + W_token_to_mem LN_token_in(X)
H <- ordinary causal decoder blocks
```

Unlike MemoryAdd, the normalized memory path is active from initialization.
Unlike MemoryUpdate, it has no per-layer cross-attention back to a separate
token source.

### MemoryUpdate

MemoryUpdate uses the memory-derived stream as its state and updates it from
the token stream inside every block:

```text
S = LN_mem_in(R) + W_token_to_mem LN_token_in(X)
for each block:
    S <- S + causal_cross_attention(Q=LN_q(S), KV=LN_kv(X))
    S <- S + causal_self_attention(LN_self(S))
    S <- S + MLP(LN_mlp(S))
```

This is state-biased rather than a compact-state cell: both token attention and
state self-attention can still access the full causal prefix. Its purpose was
to test whether making memory the primary working stream was more effective
than treating it as an auxiliary read on a token decoder.

## Branch-specific review

The code is a historical snapshot, not part of the supported model registry on
`main`. It retains the old factory options, model configurations, CLI choices,
gradient-group handling, diagnostics, launch matrices, and tests required to
reproduce the old comparisons. Checkpoints and result artifacts are not kept
in Git.

The implementations are useful as reference experiments, but they should not
be merged wholesale: they add several model classes and widen shared seams
that are intentionally small on `main`. If one architecture is revived, copy
only its model/configuration, factory registration, targeted CLI fields, and
tests. Rebase that focused change onto current `main`, update the supported
architecture registry and plotting colors together, then run the canonical
BBH/trace suites before restoring any benchmark launcher.

The least painful revival order is MemoryUpdate or MemoryState first, because
their input-fusion code is relatively self-contained. JointMemoryTape changes
the attention source layout and requires the most careful causal-mask and
parameter-count validation. MemoryConcat is the smallest conceptual ablation,
but its fusion initialization must be rechecked against the current gateless
MemoryTape/MemoryAdd baselines.
