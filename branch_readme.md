# Conditional memory gates

## Experiment and hypothesis

This branch compares gateless `memory_tape` on `main` with an optional learned
gate on each memory-read residual. The gate is scalar per batch item, sequence
position, and layer, but unlike the former MemoryTape scalar it is conditional:

```text
sigmoid(w · [LN(current_state), LN(proposed_memory_delta)] + b)
```

The hypothesis is that memory should be admitted selectively: a token may need a strong read or update in one state and almost none in another. A global learned scalar cannot express that distinction and is scale-nonidentifiable with the preceding output projection.

The gate begins at exactly 0.5. Its residual projection is scaled so the gated treatment and gateless control have the same initial function and every shared parameter has the same seeded initialization. The comparison therefore tests learned conditioning rather than a quieter or louder starting residual.

## Branch-specific code review

- `ConditionalResidualGate` uses two bias-free layer normalizations and a `2D -> 1` projection. Its projection is represented by directly zero-initialized parameters, so adding the gate consumes no initialization RNG.
- `MemoryBlock` conditions on the post-self-attention token state and its proposed tape-read delta.
- `--conditional-memory-gate {off,on}` is accepted only by `memory_tape`; `off` preserves the exact architecture and checkpoint shapes on `main`.
- Gate parameters are reported as a distinct gradient-norm group.
- Diagnostics report gate mean, variation, saturation, and per-layer values. They also compare learned gates with forced-open, forced-half, and cross-example gates. The cross-example intervention preserves causal positions while asking whether example-specific conditioning matters.
- The benchmark recommends a merge only for a quality win in at least two seeds, nontrivial gate variation in at least two seeds, and a positive cross-example loss penalty in at least two seeds.

Run the local paired pilot with:

```bash
bash scripts/trace/pilot_conditional_memory_gates.sh
```

Run the full paired experiment with:

```bash
bash scripts/trace/ablate_conditional_memory_gates.sh
```

The full experiment defaults to three seeds and 200,000 steps per variant,
with cosine decay ending at training completion. Final quality uses 4,096
fresh examples from the best validation-loss checkpoint. The 250-step pilot
uses a five-step warmup and completes its decay within the pilot.

Both scripts accept `DEVICE`, `SEEDS`/`SEED`, `TRAIN_STEPS`, `BATCH_SIZE`,
`TRAIN_EVAL_BATCHES`, `FINAL_EVAL_BATCHES`, `DIAGNOSTIC_BATCHES`, and
`RESULT_ROOT` overrides as applicable.

## Merging into `main`

The implementation is localized and the `off` path is exact, so merging the feature flag would be mechanically easy. Enabling it in canonical presets should require stronger evidence: the gate adds a small parameter and compute cost, and its average scale can still partially trade off against the residual projection.

The intervention diagnostics are therefore important. If quality improves but
gate variation stays near zero or cross-example substitution is harmless, the
treatment has rediscovered an expensive global scale and should not be merged.

No restructuring is needed for a successful single-architecture result. If conditional routing later expands to more residual branches, the least-pain cleanup is a shared `ResidualPolicy` interface returning a multiplicative coefficient plus optional diagnostics; that avoids adding independent gate flags to every block type.
