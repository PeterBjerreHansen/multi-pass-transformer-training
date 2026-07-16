# Memory-reading layer placement

## Experiment and hypothesis

This branch tests whether every transformer layer needs to read the recurrent memory tape. `--memory-read-pattern` selects all layers or a single early, middle, or late layer while still instantiating every reader and scalar gate.

The hypothesis is that memory access may be useful at a particular representational stage rather than at every layer. Skipping redundant reads could improve throughput while preserving parameter-matched comparisons and task quality. The benchmark compares `all`, `early`, `middle`, and `late` on the four-layer small model.

## Branch-specific code review

- `resolve_memory_read_layers` maps the four CLI patterns to concrete layer indices, with `middle` defined as `n_layer // 2`.
- `MemoryTapeConfig` validates explicit indices and stores the resolved architecture in checkpoints.
- `MemoryTapeTransformer` builds a boolean mask but still constructs every `MemoryBlock`, reader, normalization, and gate. Disabled layers skip only memory-reader computation.
- Inactive reader parameters therefore remain parameter-matched and receive no gradients, while active readers train normally.
- Gate diagnostics include the resolved `read_enabled` mask. Tests cover pattern resolution, invalid indices, numerical equivalence for all-layer reads, equal parameter counts, active/inactive gradients, CLI persistence, and recurrent diagnostic smoke execution.

This is a small and clean implementation. The main semantic point is that a disabled layer skips both memory attention and its gate, but otherwise runs the same self-attention and MLP path. That makes the comparison easy to interpret.

## Merging into `main`

This is probably the easiest architecture branch to merge. The default `None`/`all` behavior is numerically equivalent, parameter names and shapes do not change, and the implementation is localized to config resolution plus one conditional in `MemoryBlock`.

Minor conflicts will occur in `MemoryTapeConfig`, `MemoryBlock.forward`, and diagnostics if memory width or null slots are merged first. Resolve them by keeping three independent concepts: memory source width, whether the reader has a null source, and whether the reader is executed for a given layer.

If several memory-reader experiments are merged, replace the bare boolean with a small immutable per-layer reader policy passed from `MemoryTapeTransformer`. That policy can carry `enabled` now and later diagnostic or local-attention settings, avoiding an expanding list of unrelated keyword arguments without restructuring the rest of the model.
