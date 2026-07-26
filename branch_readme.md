# Generation-aligned training

## Experiment and hypothesis

This branch adds a mixed teacher-forced objective that follows the deployment-time `append_recurrent` memory schedule. Every optimizer step still trains the ordinary full-sequence K-pass objective. With probability `--append-train-prob`, a small subset of that same batch also receives a truncated recurrent loss.

The aligned rollout starts after at least one generated token. Earlier answer tokens are teacher-forced without gradients, the recurrent state is detached, and a short window is unrolled with gradients. The hypothesis is that directly training on frozen historical memories plus online memory appends will reduce the teacher-forced schedule gap and improve append-recurrent legality without sacrificing recompute quality.

For a short local `P=0` versus generation-aligned `P=0.25` run with both inference modes and diagnostics, use `bash scripts/pilot_generation_aligned_training.sh`. It defaults to one seed, 250 steps, and shortened 50-step warm-up/ramp periods; the device, step count, probability, microbatch, horizon, schedule, batch size, and result root are overrideable through their corresponding uppercase environment variables.

## Computational design

- The ordinary full-sequence output is reused as an exact detached prompt prefill. Causality makes its prompt-prefix memories and logits identical to a separate prompt-only K-pass forward, avoiding that extra computation.
- Standard loss is backpropagated first, freeing its saved activations. The aligned graph is then built only for the recurrent window and accumulated into the same optimizer update.
- A shared window start keeps the aligned microbatch vectorized. Rows are grouped by prompt length and the requested horizon is shortened only when the task's entire answer is shorter.
- Warm-up and linear probability ramping are built in. Defaults are a microbatch of 8, horizon 4, 5,000 standard-only steps, and a 5,000-step ramp.
- Sampling uses a dedicated checkpointed RNG, independent of task-data generation. Evaluation and public generation APIs remain inference-only and deterministic under their existing controls.

## Branch-specific code review

- `MultiPassTransformer` now exposes grad-enabled internal recurrent primitives while retaining `torch.no_grad()` public wrappers. `detach_recurrent_state` makes the truncation boundary explicit.
- `generation_aligned_loss` handles scheduling, compatible-row selection, detached burn-in, differentiable unrolling, and aligned-loss metadata.
- Trace and BBH trainers expose the append-training flags, accumulate the second backward pass before the optimizer step, and checkpoint RNG/statistics.
- Metrics distinguish the standard loss from the combined objective and report applied updates, trained target tokens, mean window start, horizon counts, and throughput overhead.
- The ablation recommendation accepts either an append-quality win or a reduced schedule gap with append and recompute quality retained.
- Tests cover inference/gradient API separation, prefix reuse, writer/reader gradients across recurrent steps, probability scheduling, deterministic sampling, checkpoint resume, CLI logging, and recommendation logic.

The principal limitation is computational: each recurrent step still reruns a full pass over the growing prefix. Truncation bounds activation memory, but long no-gradient burn-ins remain time-consuming. The current implementation is therefore intended to establish whether the objective works before investing in cache-aware recurrent training.

## Merging into `main`

The change is backward compatible when `--append-train-prob` is zero and adds no parameters or checkpoint tensor shapes. The internal/public recurrent split is independently useful and can be merged as a small preparatory commit.

The training-loop wiring is duplicated between trace and BBH. If this method is retained, the least-pain cleanup is a shared optional auxiliary-objective runner that owns its RNG, scheduling, metrics, and checkpoint state. The normal task trainers would provide the batch and standard output, then backpropagate the returned auxiliary loss.

If profiling shows promising quality but unacceptable cost, the next restructuring should be an exact cached append path. Cache self-attention K/V and projected memory K/V after prefill, then process one new position per recurrent step. That is a meaningful model change, but it can be isolated behind the same internal recurrent API introduced here, leaving the training objective and public generation interface intact.
