# Position-offset training

## Experiment and hypothesis

This branch separates sequence capacity (`block_size`) from the learned positional-table capacity (`max_position_embeddings`) and allows a sequence to be embedded at a non-zero absolute position. Training can sample offsets uniformly with `--train-position-offset-max`; evaluation can select a fixed coordinate origin with `--eval-position-offset`.

The hypothesis is that training across absolute offsets prevents the model from overfitting task roles to the first positions in the table and improves robustness when generation or memory reuse begins later in a larger coordinate system. The benchmark parameter-matches control and treatment with 149 position embeddings, trains at offset zero versus offsets 0–64, and evaluates offsets 0, 16, 32, and 64 in both inference modes.

## Branch-specific code review

- `TransformerConfig.max_position_embeddings` defaults to `block_size`, preserving the original table shape for old configs.
- Token embedding, forward, generation, and recurrent prefill accept keyword-only offsets for the baseline transformer and every multi-pass architecture.
- Recompute generation advances the offset when a long context is cropped. Recurrent state stores the starting offset so appended steps remain in the same coordinate system.
- Training uses a dedicated checkpointed RNG, and drift evaluation records the requested offset. The ablation summarizer separately enforces offset-zero non-inferiority.
- Tests cover bounds, offset-zero equivalence, shifted embeddings, cropped recomputation, recurrent-state consistency, deterministic sampling, old config loading, and CLI logging.

This is a broad but coherent change. The present recurrent path does not implement a sliding recurrent window—it rejects contexts beyond `block_size`—so retaining a constant base offset is correct for current behavior. If recurrent windows are added later, the state must advance its offset by the number of discarded tokens.

## Merging into `main`

The change is mergeable, but more invasive than the other ablations because positional coordinates are a cross-cutting model API. Merge the config/table change first, then thread `position_offset` through forward and generation call sites, and finally add stochastic training offsets. Keeping those as separate commits will make regression diagnosis and checkpoint compatibility easier.

Several experiment branches modify the same forward and training functions. The least-pain resolution is a keyword-only forward API with independent `position_offset`, `n_pass`, and optional memory-routing arguments. Centralize range validation in one embedding helper rather than repeating it in architecture subclasses.

If `main` is expected to support future sliding-window recurrence, introduce a small position-state abstraction containing `base_offset` and retained-token length. That restructuring is not necessary for this experiment, but it is the cleanest way to avoid ad hoc offset arithmetic once tokens can be discarded from recurrent state.
