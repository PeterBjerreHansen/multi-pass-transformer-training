import json
import random
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch

from model_factory import ARCHITECTURES, build_model, is_multi_pass_architecture
from models import MemoryTapeConfig


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


def add_shared_model_args(parser, *, default_inference_mode: str):
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="transformer")
    parser.add_argument("--model-size", choices=sorted(MODEL_SIZE_PRESETS), default="small")
    parser.add_argument("--inference-mode", choices=["recompute", "final_pass"], default=default_inference_mode)
    parser.add_argument("--token-selection", choices=["sample", "argmax"], default="sample")
    parser.add_argument(
        "--cache-source",
        choices=["penultimate", "last"],
        default="penultimate",
        help="For final-pass decoding, warm the recurrent cache from M_{K-1} or M_K.",
    )
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--n-embd", type=int, default=None)
    parser.add_argument("--n-pass", type=int, default=4)
    parser.add_argument(
        "--memory-tape-gate",
        default="scalar",
        help="MemoryTape read gate: none, tanh, or scalar.",
    )
    parser.add_argument("--memory-update-gate", choices=["on", "off"], default="off")
    parser.add_argument("--memory-gate-bias", type=float, default=-1.0)
    parser.add_argument("--pass-loss-weights", type=float, nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--block-size", type=int, default=None)


def add_shared_training_args(parser):
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=50_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional directory for config.json, metrics.jsonl, and latest.pt.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Optional path to a prior run directory or latest.pt checkpoint.",
    )


def apply_model_size_preset(args):
    preset = MODEL_SIZE_PRESETS[args.model_size]
    for key, value in preset.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)


def validate_model_args(args):
    apply_model_size_preset(args)

    if args.architecture == "transformer":
        args.pass_loss_weights = None
        args.inference_mode = "recompute"
        return

    if getattr(args, "n_pass", None) is None:
        args.n_pass = 4
    if args.n_pass < 2:
        raise ValueError("--n-pass must be at least 2 for multi-pass architectures")
    if args.architecture == "memory_tape":
        args.memory_tape_gate = MemoryTapeConfig.normalize_gate(args.memory_tape_gate)
    if getattr(args, "pass_loss_weights", None) is None:
        args.pass_loss_weights = [1.0] * args.n_pass
    if len(args.pass_loss_weights) != args.n_pass:
        raise ValueError("--pass-loss-weights must match --n-pass")


def build_model_and_optimizer(args, *, vocab_size: int, block_size: int):
    validate_model_args(args)
    model = build_model(args, vocab_size=vocab_size, block_size=block_size, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def resolve_device_arg(args):
    if getattr(args, "device", None) is None:
        args.device = auto_device()


def build_run_rngs(seed: int) -> tuple[random.Random, random.Random, random.Random]:
    return (
        random.Random(seed + 10),
        random.Random(seed + 20),
        random.Random(seed + 30),
    )


def synchronize_device(device: str | None):
    if device is None:
        return
    device_name = str(device)
    if device_name.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device_name == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def token_selection_is_sampling(args) -> bool:
    return getattr(args, "token_selection", "sample") == "sample"


def effective_inference_mode(args, requested_mode: str | None = None) -> str:
    requested = requested_mode or args.inference_mode
    if args.architecture == "transformer":
        return "recompute"
    return requested


def forward_and_loss(model, batch, args):
    if not is_multi_pass_architecture(args.architecture):
        logits = model(batch.idx)
        loss = model.calc_loss(logits, batch.targets)
        return loss, logits, [loss.detach()]

    logits_per_pass = model(batch.idx, return_all_logits=True)
    loss, losses = model.calc_total_loss(
        logits_per_pass,
        batch.targets,
        loss_weights=args.pass_loss_weights,
    )
    return loss, logits_per_pass[-1], [item.detach() for item in losses]


@torch.no_grad()
def basic_generation_metrics(
    model,
    batch,
    args,
    *,
    inference_mode: str | None = None,
) -> dict[str, float | None]:
    exact_matches = []
    token_accuracies = []
    mode = effective_inference_mode(args, inference_mode)
    do_sample = token_selection_is_sampling(args)

    for row in range(batch.idx.size(0)):
        prompt_len = int(batch.prompt_lengths[row].item())
        output_len = int(batch.output_lengths[row].item())
        prompt = batch.idx[row:row + 1, :prompt_len]
        target_suffix = batch.targets[row:row + 1, prompt_len - 1:prompt_len - 1 + output_len]

        generated = model.generate(
            prompt,
            max_new_tokens=output_len,
            do_sample=do_sample,
            inference_mode=mode,
            cache_source=getattr(args, "cache_source", "penultimate"),
        )
        generated_suffix = generated[:, prompt_len:prompt_len + output_len]
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
    batches: list[object],
    *,
    generation_metrics_fn: Callable,
    inference_mode: str | None = None,
) -> dict[str, float]:
    model.eval()
    synchronize_device(getattr(args, "device", None))
    start_time = time.perf_counter()
    total_loss = 0.0
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    sequence_count = 0
    input_token_count = 0
    output_token_count = 0

    if not batches:
        raise ValueError("evaluate_prebuilt_batches requires at least one batch")

    for batch in batches:
        sequence_count += int(batch.idx.size(0))
        input_token_count += int(batch.idx.numel())
        output_token_count += int(batch.output_lengths.sum().item())
        loss, _, _ = forward_and_loss(model, batch, args)
        batch_metrics = generation_metrics_fn(
            model,
            batch,
            args,
            inference_mode=inference_mode,
        )
        total_loss += float(loss.detach().item())
        for key, value in batch_metrics.items():
            if value is None:
                continue
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1

    synchronize_device(getattr(args, "device", None))
    elapsed_s = time.perf_counter() - start_time
    metrics = {"loss": total_loss / float(len(batches))}
    metrics.update({key: total / counts[key] for key, total in totals.items()})
    if elapsed_s > 0.0:
        metrics.update(
            {
                "eval_time_s": elapsed_s,
                "eval_seq_per_s": sequence_count / elapsed_s,
                "eval_input_tok_per_s": input_token_count / elapsed_s,
                "eval_output_tok_per_s": output_token_count / elapsed_s,
            }
        )
    return metrics


@torch.no_grad()
def evaluate_batches(
    model,
    args,
    batch_builder: Callable[[random.Random], object],
    rng: random.Random,
    *,
    generation_metrics_fn: Callable = basic_generation_metrics,
    inference_mode: str | None = None,
) -> dict[str, float]:
    batches = [batch_builder(rng) for _ in range(args.eval_batches)]
    return evaluate_prebuilt_batches(
        model,
        args,
        batches,
        generation_metrics_fn=generation_metrics_fn,
        inference_mode=inference_mode,
    )


def format_default_eval_metrics(metrics: dict[str, float]) -> str:
    return (
        f"seq_acc {float(metrics['exact_match']):.3f} | "
        f"token_acc {float(metrics['token_accuracy']):.3f}"
    )


def format_checkpoint_line(prefix: str, fields: list[str | None]) -> str:
    parts = [f"{prefix:<{CHECKPOINT_LABEL_WIDTH}}"]
    parts.extend(field for field in fields if field)
    return " | ".join(parts)


def format_pass_losses(pass_losses: list[torch.Tensor]) -> str:
    if len(pass_losses) == 1:
        return f"{pass_losses[0].item():.4f}"
    return "[" + ", ".join(f"{loss.item():.4f}" for loss in pass_losses) + "]"


def memory_gate_stats(model) -> dict[str, float | str | list[float]] | None:
    model_stats = getattr(model, "memory_gate_stats", None)
    if model_stats is None:
        return None
    return model_stats()


def format_memory_gate_stats(stats: dict[str, float | str | list[float]]) -> str:
    effective_values = stats.get("effective")
    if not isinstance(effective_values, list):
        raise TypeError("memory gate stats must contain a list under 'effective'")
    mean_abs = stats.get("mean_abs_effective")
    max_abs = stats.get("max_abs_effective")
    if mean_abs is None or max_abs is None:
        raise TypeError("memory gate stats must contain mean/max effective magnitudes")
    effective_text = "[" + ", ".join(f"{value:.4f}" for value in effective_values) + "]"
    return (
        f"mode {stats.get('mode', 'unknown')} | "
        f"effective {effective_text} | "
        f"mean_abs {float(mean_abs):.4f} | "
        f"max_abs {float(max_abs):.4f}"
    )


def maybe_report_memory_gates(model, artifacts: RunArtifacts | None, step: int):
    stats = memory_gate_stats(model)
    if stats is None:
        return None
    print(f"  memory_gates | {format_memory_gate_stats(stats)}")
    if artifacts is not None:
        append_jsonl(
            artifacts.metrics_path,
            {
                "event": "memory_gate_stats",
                "step": step,
                "stats": stats,
            },
        )
    return stats


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, default=_json_default)
        handle.write("\n")


def append_jsonl(path: Path, event: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=_json_default) + "\n")


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def apply_mapping_args(args, mapping: dict | None, *, preserve: set[str] | None = None):
    if not mapping:
        return
    preserved = {"resume_from", "run_dir", "train_steps"}
    if getattr(args, "device", None) is not None:
        preserved.add("device")
    if preserve is not None:
        preserved.update(preserve)
    for key, value in mapping.items():
        if key in preserved:
            continue
        setattr(args, key, value)


def resolve_resume_artifacts(resume_from: str | Path) -> RunArtifacts:
    resume_path = Path(resume_from).resolve()
    if resume_path.is_dir():
        run_dir = resume_path
        checkpoint_path = run_dir / "latest.pt"
    else:
        run_dir = resume_path.parent
        checkpoint_path = resume_path
    return RunArtifacts(
        run_dir=run_dir,
        config_path=run_dir / "config.json",
        metrics_path=run_dir / "metrics.jsonl",
        checkpoint_path=checkpoint_path,
    )


def load_checkpoint_payload(checkpoint_path: str | Path, *, device: str | None = None) -> dict:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "latest.pt"
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


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


def restore_checkpoint_state(checkpoint: dict, *, model, optimizer, device: str | None = None) -> dict:
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if device is not None:
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                state[key] = _move_to_device(value, device)

    python_random_state = checkpoint.get("python_random_state")
    if python_random_state is not None:
        random.setstate(python_random_state)
    torch_rng_state = checkpoint.get("torch_rng_state")
    if torch_rng_state is not None:
        torch.set_rng_state(torch_rng_state)
    cuda_rng_state_all = checkpoint.get("cuda_rng_state_all")
    if cuda_rng_state_all is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng_state_all)
    return checkpoint


def save_latest_checkpoint(artifacts: RunArtifacts | None, *, model, optimizer, args, step: int, extra_state=None):
    if artifacts is None:
        return None

    payload = {
        "step": int(step),
        "args": dict(vars(args)),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    model_config = getattr(model, "config", None)
    if model_config is not None and hasattr(model_config, "to_dict"):
        payload["model_config"] = model_config.to_dict()
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if extra_state:
        payload["extra_state"] = dict(extra_state)

    tmp_path = artifacts.checkpoint_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(artifacts.checkpoint_path)
    return artifacts.checkpoint_path


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

    return {
        "branch": run_git("branch", "--show-current"),
        "commit": run_git("rev-parse", "HEAD"),
    }


def _default_run_dir(*parts: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results").joinpath(*parts, timestamp).resolve()


def build_run_config(args, *, model=None, extra: dict | None = None) -> dict:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cwd": str(Path.cwd()),
        "argv": list(sys.argv),
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "git": _git_metadata(),
        "args": dict(vars(args)),
    }
    model_config = getattr(model, "config", None)
    if model_config is not None and hasattr(model_config, "to_dict"):
        payload["model_config"] = model_config.to_dict()
    if extra:
        payload.update(extra)
    return payload


def prepare_run_artifacts(
    args,
    *,
    model=None,
    default_root_parts: tuple[str, ...],
    extra_config: dict | None = None,
) -> RunArtifacts:
    if getattr(args, "run_dir", None):
        run_dir = Path(args.run_dir).resolve()
    elif getattr(args, "resume_from", None):
        run_dir = resolve_resume_artifacts(args.resume_from).run_dir
    else:
        run_dir = _default_run_dir(*default_root_parts)

    run_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = str(run_dir)

    artifacts = RunArtifacts(
        run_dir=run_dir,
        config_path=run_dir / "config.json",
        metrics_path=run_dir / "metrics.jsonl",
        checkpoint_path=run_dir / "latest.pt",
    )
    write_json(artifacts.config_path, build_run_config(args, model=model, extra=extra_config))
    return artifacts
