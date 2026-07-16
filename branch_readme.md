# Null memory slot

## Experiment and hypothesis

This branch adds an optional null source to each `MemoryTapeTransformer` reader. The null source has a learned key per attention head and a fixed zero value, so a query can explicitly allocate attention probability to "no useful memory here" without adding a learned content vector. The existing scalar memory gate remains after the attention projection.

The hypothesis is diagnostic as much as architectural: a null slot should help only when ordinary memory attention is compulsory and diffuse. The benchmark therefore compares null off versus on and requires high normalized attention entropy, a materially active scalar gate, non-trivial null mass, and a quality win before recommending a merge.

## Branch-specific code review

- `CausalCrossAttention` prepends the learned null key and zero value and uses an explicit boolean mask because query and source lengths no longer match.
- The mask always exposes the null column and preserves causal access to the already shifted memory tape. Manual attention and SDPA share the same allowed-position definition.
- `memory_attention_diagnostics` reports per-pass/layer normalized entropy, effective source count, null probability mass, and the control-side diagnostic precondition.
- The CLI and config restrict the option to `memory_tape`; the default `off` path keeps the previous attention implementation and checkpoint structure.
- Tests cover exact mask contents, SDPA/manual agreement, the absence of a trainable null value, null-key gradients, default-off behavior, finite diagnostics, checkpoint loading, and CPU/MPS smoke paths.

The attention implementation is sound, but the diagnostic method duplicates much of the model's pass/layer traversal. That creates maintenance risk if the normal forward path later gains read masks, position offsets, or other per-layer behavior.

## Merging into `main`

With the feature flag defaulting to off, the architecture can be merged without checkpoint migration. However, it should only be enabled by default or promoted in presets if the diagnostic precondition and benchmark rule are met.

For an elegant merge, refactor cross-attention to optionally return lightweight attention statistics from the same projections used in the real forward pass. Then diagnostics can attach a collector or hook instead of reimplementing `MemoryTapeTransformer` traversal. This is a contained change around `CausalCrossAttention` and `MemoryBlock`, not a full model rewrite.

This branch conflicts directly with memory width in the cross-attention constructor and shape validation. Merge memory-width generalization first, then add the independent null-slot flag. If memory-read masks are also merged, ensure disabled readers are excluded from entropy/precondition aggregation rather than being rerun solely for diagnostics.
