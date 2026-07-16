from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time
from typing import Callable, Iterator, Sequence

import torch

from model_factory import ARCHITECTURES, build_model, is_multi_pass_architecture


MODEL_SIZE_PRESETS = {
    "tiny": {"n_layer": 2, "n_head": 2, "n_embd": 64},
    "small": {"n_layer": 4, "n_head": 4, "n_embd": 128},
    "medium": {"n_layer": 6, "n_head": 6, "n_embd": 192},
    "large": {"n_layer": 8, "n_head": 8, "n_embd": 256},
}
CHECKPOINT_LABEL_WIDTH = 14


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    config_path: Path
    metrics_path: Path
    checkpoint_path: Path


def add_shared_model_args(parser, *, default_inference_mode: str) -> None:
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="transformer")
    parser.add_argument("--model-size", choices=sorted(MODEL_SIZE_PRESETS), default="small")
    parser.add_argument(
        "--inference-mode",
        choices=["recompute", "append_recurrent"],
        default=default_inference_mode,
    )
    parser.add_argument("--token-selection", choices=["sample", "argmax"], default="sample")
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--n-embd", type=int, default=None)
    parser.add_argument("--n-pass", type=int, default=4)
    parser.add_argument("--memory-update-gate", choices=["on", "off"], default="off")
    parser.add_argument("--memory-gate-bias", type=float, default=-1.0)
    parser.add_argument("--pass-loss-weights", type=float, nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--block-size", type=int, default=None)


def add_shared_training_args(parser) -> None:
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=50_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume-from", default=None)


def apply_model_size_preset(args) -> None:
    preset = MODEL_SIZE_PRESETS[args.model_size]
    for key, value in preset.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)


def validate_model_args(args) -> None:
    apply_model_size_preset(args)
    if args.n_layer < 1 or args.n_head < 1 or args.n_embd < 1:
        raise ValueError("model dimensions must be positive")
    if args.n_embd % args.n_head != 0:
        raise ValueError("--n-embd must be divisible by --n-head")

    if args.architecture == "transformer":
        args.pass_loss_weights = None
        args.inference_mode = "recompute"
        return

    if args.n_pass < 2:
        raise ValueError("--n-pass must be at least 2 for multi-pass architectures")
    if args.pass_loss_weights is None:
        args.pass_loss_weights = [1.0] * args.n_pass
    if len(args.pass_loss_weights) != args.n_pass:
        raise ValueError("--pass-loss-weights must match --n-pass")
    weights = torch.tensor(args.pass_loss_weights, dtype=torch.float64)
    if not torch.isfinite(weights).all():
        raise ValueError("--pass-loss-weights must be finite")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("--pass-loss-weights must be non-negative with a positive sum")


def validate_training_args(args) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.train_steps < 1:
        raise ValueError("--train-steps must be at least 1")
    if args.eval_interval < 1 or args.eval_batches < 1:
        raise ValueError("--eval-interval and --eval-batches must be at least 1")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative")


def build_model_and_optimizer(args, *, vocab_size: int, block_size: int):
    validate_model_args(args)
    model = build_model(args, vocab_size=vocab_size, block_size=block_size, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def resolve_device_arg(args) -> None:
    if getattr(args, "device", None) is None:
        args.device = auto_device()


def synchronize_device(device: str | None) -> None:
    if not device:
        return
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif str(device).startswith("mps") and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def stable_seed(base_seed: int, *parts: object) -> int:
    text = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


@contextmanager
def isolated_torch_rng(seed: int) -> Iterator[None]:
    """Use deterministic sampling during evaluation without perturbing training RNG."""
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps_available = (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
        and hasattr(torch.mps, "set_rng_state")
        and getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )
    mps_state = torch.mps.get_rng_state() if mps_available else None
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if mps_available and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def token_selection_is_sampling(args) -> bool:
    return getattr(args, "token_selection", "sample") == "sample"


def effective_inference_mode(args, requested_mode: str | None = None) -> str:
    if args.architecture == "transformer":
        return "recompute"
    return requested_mode or args.inference_mode


def forward_and_loss(model, batch, args):
    output = model(batch.idx)
    if not is_multi_pass_architecture(args.architecture):
        loss = model.calc_loss(output.logits, batch.targets)
        return loss, output, (loss.detach(),)

    loss_output = model.calc_total_loss(
        output,
        batch.targets,
        loss_weights=args.pass_loss_weights,
    )
    return loss_output.loss, output, tuple(item.detach() for item in loss_output.pass_losses)


def gradient_norms(model) -> dict[str, float]:
    """Return L2 gradient norms for the model and its memory-specific subsystems.

    Parameter groups are disjoint so their norms can be compared to the global
    norm without double-counting.  Groups that have no gradient on a step are
    omitted rather than reported as a misleading zero.
    """
    squared_sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        squared_norm = parameter.grad.detach().float().square().sum()

        if "memory_gate" in name or "token_gate" in name:
            group = "memory_gate"
        elif "cross_attn" in name or "token_attn" in name or "joint_attn" in name:
            group = "memory_attention"
        elif name.startswith(("mem_head", "ln_mem")):
            group = "memory_writer"
        else:
            group = "backbone"

        for group_name in ("global", group):
            squared_sums[group_name] = squared_sums.get(group_name, squared_norm.new_zeros(())) + squared_norm
            counts[group_name] = counts.get(group_name, 0) + 1

    return {
        group: math.sqrt(float(value.item()))
        for group, value in squared_sums.items()
        if counts.get(group, 0) > 0
    }


def update_gradient_norm_window(window: dict[str, dict[str, float]], norms: dict[str, float]) -> None:
    """Accumulate mean and maximum gradient norms over a training interval."""
    for group, value in norms.items():
        entry = window.setdefault(group, {"sum": 0.0, "max": 0.0, "count": 0.0})
        entry["sum"] += value
        entry["max"] = max(entry["max"], value)
        entry["count"] += 1


def summarize_gradient_norm_window(window: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        group: {
            "mean": entry["sum"] / entry["count"],
            "max": entry["max"],
        }
        for group, entry in window.items()
        if entry["count"] > 0
    }


def format_gradient_norms(summary: dict[str, dict[str, float]]) -> str | None:
    global_norm = summary.get("global")
    if global_norm is None:
        return None
    return f"grad_norm mean/max {global_norm['mean']:.3g}/{global_norm['max']:.3g}"


@torch.no_grad()
def basic_generation_metrics(
    model,
    batch,
    args,
    *,
    inference_mode: str | None = None,
) -> dict[str, float]:
    exact_matches = []
    token_accuracies = []
    mode = effective_inference_mode(args, inference_mode)
    do_sample = token_selection_is_sampling(args)

    for row in range(batch.idx.shape[0]):
        prompt_len = int(batch.prompt_lengths[row].item())
        output_len = int(batch.output_lengths[row].item())
        prompt = batch.idx[row : row + 1, :prompt_len]
        target_suffix = batch.targets[
            row : row + 1,
            prompt_len - 1 : prompt_len - 1 + output_len,
        ]
        generated = model.generate(
            prompt,
            max_new_tokens=output_len,
            do_sample=do_sample,
            inference_mode=mode,
        )
        generated_suffix = generated[:, prompt_len : prompt_len + output_len]
        correct = generated_suffix == target_suffix
        exact_matches.append(correct.all(dim=1).float().mean())
        token_accuracies.append(correct.float().mean())

    return {
        "exact_match": torch.stack(exact_matches).mean().item(),
        "token_accuracy": torch.stack(token_accuracies).mean().item(),
    }


@torch.no_grad()
def evaluate_prebuilt_batches(
    model,
    args,
    batches: Sequence[object],
    *,
    generation_metrics_fn: Callable = basic_generation_metrics,
    inference_mode: str | None = None,
    generation_seed: int | None = None,
) -> dict[str, float]:
    if not batches:
        raise ValueError("evaluate_prebuilt_batches requires at least one batch")

    was_training = model.training
    model.eval()
    synchronize_device(getattr(args, "device", None))
    start = time.perf_counter()
    total_loss = 0.0
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    sequence_count = 0
    input_token_count = 0
    output_token_count = 0

    seed = generation_seed if generation_seed is not None else stable_seed(args.seed, "generation")
    try:
        with isolated_torch_rng(seed):
            for batch in batches:
                sequence_count += int(batch.idx.shape[0])
                input_token_count += int(batch.idx.numel())
                output_token_count += int(batch.output_lengths.sum().item())
                loss, _output, _pass_losses = forward_and_loss(model, batch, args)
                metrics = generation_metrics_fn(
                    model,
                    batch,
                    args,
                    inference_mode=inference_mode,
                )
                total_loss += float(loss.item())
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + float(value)
                    counts[key] = counts.get(key, 0) + 1
    finally:
        model.train(was_training)

    synchronize_device(getattr(args, "device", None))
    elapsed = time.perf_counter() - start
    result = {"loss": total_loss / len(batches)}
    result.update({key: total / counts[key] for key, total in totals.items()})
    if elapsed > 0:
        result.update(
            {
                "eval_time_s": elapsed,
                "eval_seq_per_s": sequence_count / elapsed,
                "eval_input_tok_per_s": input_token_count / elapsed,
                "eval_output_tok_per_s": output_token_count / elapsed,
            }
        )
    return result


def format_default_eval_metrics(metrics: dict[str, float]) -> str:
    return f"seq_acc {metrics['exact_match']:.3f} | token_acc {metrics['token_accuracy']:.3f}"


def format_checkpoint_line(prefix: str, fields: Sequence[str | None]) -> str:
    return " | ".join([f"{prefix:<{CHECKPOINT_LABEL_WIDTH}}", *(field for field in fields if field)])


def format_pass_losses(pass_losses: Sequence[torch.Tensor]) -> str:
    if len(pass_losses) == 1:
        return f"{pass_losses[0].item():.4f}"
    return "[" + ", ".join(f"{loss.item():.4f}" for loss in pass_losses) + "]"


def memory_gate_stats(model) -> dict | None:
    method = getattr(model, "memory_gate_stats", None)
    return None if method is None else method()


def format_memory_gate_stats(stats: dict) -> str:
    values = stats.get("effective")
    if not isinstance(values, list):
        raise TypeError("memory gate stats must contain an effective list")
    text = "[" + ", ".join(f"{float(value):.4f}" for value in values) + "]"
    return (
        f"effective {text} | mean_abs {float(stats['mean_abs_effective']):.4f} | "
        f"max_abs {float(stats['max_abs_effective']):.4f}"
    )


def append_jsonl(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=_json_default) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_resume_artifacts(resume_from: str | Path) -> RunArtifacts:
    path = Path(resume_from).resolve()
    run_dir = path if path.is_dir() else path.parent
    checkpoint_path = run_dir / "latest.pt" if path.is_dir() else path
    return RunArtifacts(
        run_dir=run_dir,
        config_path=run_dir / "config.json",
        metrics_path=run_dir / "metrics.jsonl",
        checkpoint_path=checkpoint_path,
    )


def load_checkpoint_payload(path: str | Path, *, device: str | None = None) -> dict:
    checkpoint = Path(path)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "latest.pt"
    try:
        return torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint, map_location=device)


def restore_checkpoint_state(checkpoint: dict, *, model, optimizer=None, device: str | None = None) -> dict:
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if device is not None:
            for state in optimizer.state.values():
                for key, value in list(state.items()):
                    state[key] = _move_to_device(value, device)
    if "python_random_state" in checkpoint:
        random.setstate(checkpoint["python_random_state"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if "cuda_rng_state_all" in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    return checkpoint


def save_latest_checkpoint(
    artifacts: RunArtifacts | None,
    *,
    model,
    optimizer,
    args,
    step: int,
    extra_state: dict | None = None,
) -> Path | None:
    if artifacts is None:
        return None
    payload = {
        "step": int(step),
        "args": dict(vars(args)),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "model_config": model.config.to_dict(),
        "extra_state": dict(extra_state or {}),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    temp = artifacts.checkpoint_path.with_suffix(".pt.tmp")
    torch.save(payload, temp)
    temp.replace(artifacts.checkpoint_path)
    return artifacts.checkpoint_path


def prepare_run_artifacts(
    args,
    *,
    model,
    default_root_parts: tuple[str, ...],
    extra_config: dict | None = None,
) -> RunArtifacts:
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    elif args.resume_from:
        run_dir = resolve_resume_artifacts(args.resume_from).run_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("results").joinpath(*default_root_parts, timestamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = str(run_dir)
    artifacts = RunArtifacts(
        run_dir=run_dir,
        config_path=run_dir / "config.json",
        metrics_path=run_dir / "metrics.jsonl",
        checkpoint_path=run_dir / "latest.pt",
    )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cwd": str(Path.cwd()),
        "argv": list(sys.argv),
        "command": " ".join(shlex.quote(item) for item in sys.argv),
        "git": _git_metadata(),
        "args": dict(vars(args)),
        "model_config": model.config.to_dict(),
    }
    if extra_config:
        payload.update(extra_config)
    write_json(artifacts.config_path, payload)
    return artifacts


def saved_args_from_run(run_dir: str | Path) -> dict:
    payload = load_json_if_exists(Path(run_dir).resolve() / "config.json")
    if payload is None or "args" not in payload:
        raise FileNotFoundError(f"missing config.json with saved args in {run_dir}")
    return dict(payload["args"])


def _move_to_device(value, device: str):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _git_metadata() -> dict[str, str | None]:
    def run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip() or None

    return {"branch": run_git("branch", "--show-current"), "commit": run_git("rev-parse", "HEAD")}


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
