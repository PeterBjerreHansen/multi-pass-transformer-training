# Independent memory width

## Experiment and hypothesis

This branch gives `MemoryTapeTransformer` an independent tape width through `MemoryTapeConfig.n_memory_embd` and `--n-memory-embd`. The token backbone, queries, attention outputs, and logits stay at width `D`, while the writer emits `Dm` features and each reader projects `Dm -> 2D`.

The hypothesis is that the recurrent tape is over-wide relative to the information it must preserve. A narrower tape may retain task quality while reducing parameters, persistent recurrent-state size, and reader bandwidth. The benchmark compares legacy behavior, explicit `Dm=128`, `Dm=64`, and `Dm=32`, with the explicit full-width case serving as the normalization-matched control.

## Branch-specific code review

- Omitting `n_memory_embd` preserves the legacy writer shape and normalization order. Supplying it uses `D -> Dm` followed by `Dm`-wide normalization.
- `CausalCrossAttention` accepts a separate memory input width while keeping query and output dimensions unchanged.
- Initial tapes, recurrent placeholders, validation, and emitted memory states consistently use `Dm`.
- Benchmark metadata reports parameter counts, checkpoint size, bytes per tape token, persistent bytes per sequence, and theoretical reader input traffic.
- Tests cover several widths, manual versus SDPA attention, emitted and recurrent shapes, gradients, parameter reduction, invalid widths, saved config, and diagnostic smoke execution.

The implementation works, but architecture-specific memory-width logic currently lives in the shared `MultiPassTransformer` constructor through an `isinstance(MemoryTapeConfig)` check. That is expedient for an ablation but is the least elegant part of the branch: the base class now knows details that only `MemoryTapeTransformer` needs.

## Merging into `main`

The feature is worth merging if the benchmark finds a non-inferior narrower tape, but the shared-base leakage should ideally be cleaned up first. The least-pain refactor is to give `MultiPassTransformer` a small overridable memory specification—either a class method returning memory width and writer normalization order, or a dedicated `MemoryWriter` module. The base class can then allocate and validate recurrent memory through `self.memory_dim` without checking a concrete config type.

After that localized refactor, the remaining merge is straightforward: add the config/CLI option, generalize cross-attention input width, and retain `None` as the exact compatibility path.

If this branch is combined with null slots, merge memory width first and construct `CausalCrossAttention(config, memory_dim=..., use_null_memory_slot=...)`; the two changes are conceptually orthogonal but edit the same attention constructor and shape checks.
