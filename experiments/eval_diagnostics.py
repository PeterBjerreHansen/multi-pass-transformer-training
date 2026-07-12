from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from experiments.common import (
    load_checkpoint_payload,
    load_json_if_exists,
    resolve_device_arg,
    restore_checkpoint_state,
    stable_seed,
    validate_model_args,
    validate_training_args,
    write_json,
)
from experiments.train_bbh import (
    BBH_TASKS,
    build_fixed_eval_batches as build_bbh_eval_batches,
    build_training_objects as build_bbh_training_objects,
    validate_task_args as validate_bbh_task_args,
)
from experiments.train_trace import (
    TRACE_TASKS,
    build_fixed_eval_batches as build_trace_eval_batches,
    build_training_objects as build_trace_training_objects,
    validate_task_args as validate_trace_task_args,
)
from model_factory import is_multi_pass_architecture


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run memory-intervention and pass-dynamics diagnostics on a saved checkpoint.",
        allow_abbrev=False,
    )
    parser.add_argument("--input-run-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--extra-passes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args(argv)


def _load_args(cli_args) -> tuple[SimpleNamespace, Path]:
    run_dir = Path(cli_args.input_run_dir).resolve()
    config = load_json_if_exists(run_dir / "config.json")
    if config is None or "args" not in config:
        raise FileNotFoundError(f"missing config.json with saved args in {run_dir}")
    saved = dict(config["args"])
    if cli_args.device is not None:
        saved["device"] = cli_args.device
    saved["batch_size"] = cli_args.batch_size or max(2, min(int(saved.get("batch_size", 2)), 16))
    if int(saved["batch_size"]) < 2:
        raise ValueError("diagnostics require --batch-size >= 2 for cross-example interventions")
    saved["eval_batches"] = cli_args.eval_batches
    saved["seed"] = cli_args.seed
    saved["run_dir"] = str(run_dir)
    saved["resume_from"] = str(run_dir)
    args = SimpleNamespace(**saved)
    resolve_device_arg(args)
    validate_model_args(args)
    validate_training_args(args)
    return args, run_dir


def _build_model_and_batches(args, *, bbh_level: int | None = None):
    if args.task in BBH_TASKS:
        validate_bbh_task_args(args)
        _task, _block_size, _vocab, stoi, _itos, model, _optimizer = build_bbh_training_objects(args)
        level = args.curriculum_start_level if bbh_level is None else bbh_level
        batches = build_bbh_eval_batches(args, BBH_TASKS[args.task], stoi, level)
        return model, batches
    if args.task in TRACE_TASKS:
        validate_trace_task_args(args)
        _block_size, _vocab, stoi, _itos, model, _optimizer = build_trace_training_objects(args)
        batches = build_trace_eval_batches(args, stoi)
        return model, batches
    raise ValueError(f"unsupported saved task: {args.task}")


def _nll(model, logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float(model.calc_loss(logits, targets).item())


def _memory_stats(memory: torch.Tensor) -> dict[str, float]:
    flat = memory.detach().float().reshape(-1, memory.shape[-1])
    return {
        "rms_norm": float(memory.detach().float().square().mean().sqrt().item()),
        "feature_std": float(flat.std(dim=0, unbiased=False).mean().item()),
        "effective_rank": _effective_rank(flat),
    }


def _effective_rank(flat: torch.Tensor) -> float:
    if flat.shape[0] < 2 or flat.shape[1] < 1:
        return 0.0
    centered = flat - flat.mean(dim=0, keepdim=True)
    # Cap rows to keep diagnostics inexpensive on long batches.
    if centered.shape[0] > 4096:
        centered = centered[:4096]
    singular = torch.linalg.svdvals(centered.cpu())
    energy = singular.square()
    total = energy.sum()
    if not torch.isfinite(total) or total <= 0:
        return 0.0
    probabilities = energy / total
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return float(entropy.exp().item())


def _memory_distance(previous: torch.Tensor, current: torch.Tensor) -> dict[str, float]:
    prev = previous.detach().float()
    curr = current.detach().float()
    cosine = F.cosine_similarity(prev, curr, dim=-1)
    return {
        "rms_delta": float((curr - prev).square().mean().sqrt().item()),
        "mean_cosine_distance": float((1.0 - cosine).mean().item()),
    }


def _logit_kl(previous: torch.Tensor, current: torch.Tensor) -> float:
    prev_log = F.log_softmax(previous.detach().float(), dim=-1)
    curr_log = F.log_softmax(current.detach().float(), dim=-1)
    prev_prob = prev_log.exp()
    value = float((prev_prob * (prev_log - curr_log)).sum(dim=-1).mean().item())
    # Roundoff can produce tiny negative values for a theoretically non-negative KL.
    return max(0.0, value)


@torch.no_grad()
def memory_interventions(model, batch, *, seed: int) -> dict:
    output = model(batch.idx)
    if len(output.passes) < 2:
        raise ValueError("memory interventions require at least two passes")
    previous_memory = output.passes[-2].memory_states
    if previous_memory is None:
        raise ValueError("model did not emit memory states")
    token_stream = model.embed_tokens(batch.idx)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    seq_len = previous_memory.shape[1]

    # Each destination position j samples only from source positions <= j.
    # This disrupts position correspondence without moving future-derived
    # memories into an earlier readable location.
    causal_sources = torch.tensor(
        [int(torch.randint(0, position + 1, (1,), generator=generator).item()) for position in range(seq_len)],
        device=previous_memory.device,
        dtype=torch.long,
    )
    causal_resample = previous_memory[:, causal_sources, :]

    counts = torch.arange(1, seq_len + 1, device=previous_memory.device, dtype=previous_memory.dtype)
    causal_prefix_mean = previous_memory.cumsum(dim=1) / counts[None, :, None]

    extra_lag = torch.zeros_like(previous_memory)
    if seq_len > 1:
        extra_lag[:, 1:, :] = previous_memory[:, :-1, :]

    interventions: dict[str, torch.Tensor] = {
        "correct": previous_memory,
        "zero": torch.zeros_like(previous_memory),
        "cross_example": previous_memory.roll(1, dims=0),
        "causal_position_resample": causal_resample,
        "causal_prefix_mean": causal_prefix_mean,
        "extra_lag": extra_lag,
    }

    losses: dict[str, float] = {}
    for name, memory in interventions.items():
        pass_output = model.forward_pass(token_stream, memory)
        losses[name] = _nll(model, pass_output.logits, batch.targets)

    baseline = losses["correct"]
    return {
        "losses": losses,
        "loss_deltas": {name: value - baseline for name, value in losses.items()},
        "source_memory": _memory_stats(previous_memory),
    }


@torch.no_grad()
def pass_dynamics(model, batch, *, extra_passes: int) -> dict:
    output = model(batch.idx)
    per_pass = []
    for index, item in enumerate(output.passes, start=1):
        if item.memory_states is None:
            raise ValueError("multi-pass diagnostic requires memory states")
        entry = {
            "pass": index,
            "loss": _nll(model, item.logits, batch.targets),
            "memory": _memory_stats(item.memory_states),
        }
        if index > 1:
            previous = output.passes[index - 2]
            if previous.memory_states is None:
                raise RuntimeError("previous pass did not emit memory states")
            entry["memory_change"] = _memory_distance(previous.memory_states, item.memory_states)
            entry["logit_kl_from_previous"] = _logit_kl(previous.logits, item.logits)
            entry["hidden_rms_delta"] = float(
                (item.hidden_states.detach().float() - previous.hidden_states.detach().float())
                .square()
                .mean()
                .sqrt()
                .item()
            )
        per_pass.append(entry)

    token_stream = model.embed_tokens(batch.idx)
    memory = output.final_memory
    previous_logits = output.logits
    previous_hidden = output.hidden_states
    continuation = []
    for offset in range(1, extra_passes + 1):
        item = model.forward_pass(token_stream, memory)
        if item.memory_states is None:
            raise RuntimeError("extra pass did not emit memory states")
        continuation.append(
            {
                "pass": len(output.passes) + offset,
                "loss": _nll(model, item.logits, batch.targets),
                "memory": _memory_stats(item.memory_states),
                "memory_change": _memory_distance(memory, item.memory_states),
                "logit_kl_from_previous": _logit_kl(previous_logits, item.logits),
                "hidden_rms_delta": float(
                    (item.hidden_states.detach().float() - previous_hidden.detach().float())
                    .square()
                    .mean()
                    .sqrt()
                    .item()
                ),
            }
        )
        memory = item.memory_states
        previous_logits = item.logits
        previous_hidden = item.hidden_states

    return {
        "trained_passes": per_pass,
        "extra_passes": continuation,
        "memory_gate_stats": model.memory_gate_stats(),
    }


def _mean_numbers(items: list[dict]) -> dict:
    """Average identically structured diagnostic dictionaries."""
    if len(items) == 1:
        return items[0]

    def merge(values):
        first = values[0]
        if isinstance(first, dict):
            return {key: merge([value[key] for value in values]) for key in first}
        if isinstance(first, list):
            return [merge([value[index] for value in values]) for index in range(len(first))]
        if isinstance(first, (int, float)) and all(isinstance(value, (int, float)) for value in values):
            return sum(float(value) for value in values) / len(values)
        return first

    return merge(items)


def evaluate_diagnostics(cli_args) -> Path:
    if cli_args.extra_passes < 0:
        raise ValueError("--extra-passes must be non-negative")
    args, run_dir = _load_args(cli_args)
    if not is_multi_pass_architecture(args.architecture):
        raise ValueError("memory diagnostics require a multi-pass architecture")
    checkpoint = load_checkpoint_payload(run_dir / "latest.pt", device="cpu")
    bbh_level = None
    if args.task in BBH_TASKS:
        bbh_level = int(checkpoint.get("extra_state", {}).get("current_level", args.curriculum_start_level))
    model, batches = _build_model_and_batches(args, bbh_level=bbh_level)
    restore_checkpoint_state(checkpoint, model=model, optimizer=None, device=args.device)

    was_training = model.training
    model.eval()
    try:
        intervention_results = []
        dynamics_results = []
        for batch_index, batch in enumerate(batches):
            intervention_results.append(
                memory_interventions(
                    model,
                    batch,
                    seed=stable_seed(cli_args.seed, "memory_intervention", batch_index),
                )
            )
            dynamics_results.append(pass_dynamics(model, batch, extra_passes=cli_args.extra_passes))
    finally:
        model.train(was_training)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_run_dir": str(run_dir),
        "task": args.task,
        "architecture": args.architecture,
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "batch_size": args.batch_size,
        "eval_batches": args.eval_batches,
        "memory_interventions": _mean_numbers(intervention_results),
        "pass_dynamics": _mean_numbers(dynamics_results),
    }
    output = Path(cli_args.output).resolve() if cli_args.output else run_dir / "diagnostics.json"
    write_json(output, payload)
    print(f"wrote {output}")
    return output


def main(argv: list[str] | None = None) -> None:
    evaluate_diagnostics(parse_args(argv))


if __name__ == "__main__":
    main()
