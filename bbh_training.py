import argparse
import json
import random
from pathlib import Path
from typing import Callable

import torch

from model_factory import ARCHITECTURES, build_model, is_multi_pass_architecture
from tasks.bbh_symbolic_registry import TASKS


MODEL_SIZE_PRESETS = {
    "tiny": {"n_layer": 2, "n_head": 2, "n_embd": 64},
    "small": {"n_layer": 4, "n_head": 4, "n_embd": 128},
    "medium": {"n_layer": 6, "n_head": 6, "n_embd": 192},
    "large": {"n_layer": 8, "n_head": 8, "n_embd": 256},
}


def add_common_args(parser: argparse.ArgumentParser, *, default_log_policy: str):
    parser.add_argument("--task", choices=sorted(TASKS), default="walk")
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="transformer")
    parser.add_argument("--model-size", choices=sorted(MODEL_SIZE_PRESETS), default="small")
    parser.add_argument("--generation-mode", choices=["recompute", "greedy"], default="greedy")
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--n-embd", type=int, default=None)
    parser.add_argument("--n-pass", type=int, default=4)
    parser.add_argument("--memory-update-gate", choices=["on", "off"], default="off")
    parser.add_argument("--memory-gate-bias", type=float, default=-1.0)
    parser.add_argument("--pass-loss-weights", type=float, nargs="*", default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument(
        "--num-objects",
        type=int,
        default=4,
        help="Number of objects for the permutation task.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=50_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-level", type=int, default=None)
    parser.add_argument(
        "--compare-generation-modes",
        action="store_true",
        help="For multi-pass architectures, evaluate both recompute and greedy generation.",
    )
    parser.add_argument(
        "--log-jsonl",
        default=None,
        help="Optional path for structured training/eval events.",
    )
    parser.add_argument(
        "--log-policy",
        choices=["promotions", "evals"],
        default=default_log_policy,
        help="JSONL logging detail. 'promotions' keeps logs compact; 'evals' records every eval event.",
    )


def finalize_args(args, *, supervision: str):
    args.supervision = supervision
    apply_model_size_preset(args)
    apply_task_defaults(args)
    return args


def apply_model_size_preset(args):
    preset = MODEL_SIZE_PRESETS[args.model_size]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def apply_task_defaults(args):
    task = TASKS[args.task]
    if not hasattr(args, "curriculum_start_level"):
        args.curriculum_start_level = None
    if not hasattr(args, "curriculum_threshold"):
        args.curriculum_threshold = None
    if not hasattr(args, "review_easier_every"):
        args.review_easier_every = 0
    if args.curriculum_start_level is None:
        args.curriculum_start_level = task.default_start_level
    if args.max_level is None:
        args.max_level = task.default_max_level


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def validate_model_args(args):
    if args.architecture == "transformer":
        if args.pass_loss_weights is not None:
            raise ValueError("--pass-loss-weights is only used for multi-pass architectures")
        args.generation_mode = "recompute"
        return

    if args.n_pass < 2:
        raise ValueError("--n-pass must be at least 2 for multi-pass architectures")
    if args.pass_loss_weights is None:
        args.pass_loss_weights = [1.0] * args.n_pass
    if len(args.pass_loss_weights) != args.n_pass:
        raise ValueError("--pass-loss-weights must match --n-pass")


def validate_task_args(args):
    task = TASKS[args.task]
    if args.curriculum_start_level < task.min_level:
        raise ValueError(f"--curriculum-start-level must be >= {task.min_level} for task {args.task}")
    if args.max_level < args.curriculum_start_level:
        raise ValueError("--max-level must be >= --curriculum-start-level")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.train_steps < 1:
        raise ValueError("--train-steps must be at least 1")
    if args.eval_batches < 1:
        raise ValueError("--eval-batches must be at least 1")
    if args.curriculum_threshold is not None and not 0.0 <= args.curriculum_threshold <= 1.0:
        raise ValueError("--curriculum-threshold must be in [0, 1]")
    if args.review_easier_every < 0:
        raise ValueError("--review-easier-every must be >= 0")
    if args.block_size is not None and args.block_size < 8:
        raise ValueError("--block-size must be at least 8")
    if args.task == "permutation" and args.num_objects < 2:
        raise ValueError("--num-objects must be at least 2 for task permutation")


def task_kwargs(args) -> dict[str, int]:
    if args.task == "permutation":
        return {"num_objects": args.num_objects}
    return {}


def build_training_objects(args, vocab_level: int | None = None):
    task = TASKS[args.task]
    level = args.max_level if vocab_level is None else vocab_level
    kwargs = task_kwargs(args)
    block_size = args.block_size or task.required_block_size(level, args.supervision, **kwargs)
    vocab, stoi, _ = task.build_vocab(level, **kwargs)
    model = build_model(args, vocab_size=len(vocab), block_size=block_size, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return task, block_size, vocab, stoi, model, optimizer


def build_task_batch(args, stoi, level: int, rng: random.Random):
    task = TASKS[args.task]
    return task.build_batch(
        args.batch_size,
        level,
        stoi,
        args.supervision,
        args.device,
        rng,
        **task_kwargs(args),
    )


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


def generation_mode_for(args, generation_mode: str | None = None) -> str:
    if args.architecture == "transformer":
        return "recompute"
    return generation_mode or args.generation_mode


@torch.no_grad()
def generation_metrics(
    model,
    batch,
    args,
    *,
    generation_mode: str | None = None,
    final_token_id: int | None = None,
) -> dict[str, float | None]:
    exact_matches = []
    token_accuracies = []
    final_exact_matches = []
    final_token_accuracies = []
    trace_token_accuracies = []
    mode = generation_mode_for(args, generation_mode)

    for row in range(batch.idx.size(0)):
        prompt_len = int(batch.prompt_lengths[row].item())
        output_len = int(batch.output_lengths[row].item())
        prompt = batch.idx[row:row + 1, :prompt_len]
        target_suffix = batch.targets[row:row + 1, prompt_len - 1:prompt_len - 1 + output_len]

        generated = model.generate(
            prompt,
            max_new_tokens=output_len,
            do_sample=False,
            generation_mode=mode,
        )
        generated_suffix = generated[:, prompt_len:prompt_len + output_len]
        correct = generated_suffix == target_suffix
        exact_matches.append(correct.all(dim=1).float().mean())
        token_accuracies.append(correct.float().mean())

        if final_token_id is not None:
            final_positions = (target_suffix[0] == final_token_id).nonzero(as_tuple=False)
            if final_positions.numel() > 0:
                final_start = int(final_positions[0].item())
                final_correct = correct[:, final_start:output_len]
                final_exact_matches.append(final_correct.all(dim=1).float().mean())
                final_token_accuracies.append(final_correct.float().mean())
                if final_start > 0:
                    trace_token_accuracies.append(correct[:, :final_start].float().mean())

    metrics: dict[str, float | None] = {
        "exact_match": torch.stack(exact_matches).mean().item(),
        "token_accuracy": torch.stack(token_accuracies).mean().item(),
    }
    if final_exact_matches:
        metrics["final_exact_match"] = torch.stack(final_exact_matches).mean().item()
        metrics["final_token_accuracy"] = torch.stack(final_token_accuracies).mean().item()
    if trace_token_accuracies:
        metrics["trace_token_accuracy"] = torch.stack(trace_token_accuracies).mean().item()
    return metrics


@torch.no_grad()
def evaluate_batches(
    model,
    args,
    batch_builder: Callable[[random.Random], object],
    rng: random.Random,
    eval_batches: int | None = None,
    *,
    generation_mode: str | None = None,
    final_token_id: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    n_batches = args.eval_batches if eval_batches is None else eval_batches

    for _ in range(n_batches):
        batch = batch_builder(rng)
        loss, _, _ = forward_and_loss(model, batch, args)
        batch_metrics = generation_metrics(
            model,
            batch,
            args,
            generation_mode=generation_mode,
            final_token_id=final_token_id,
        )
        total_loss += float(loss.detach().item())
        for key, value in batch_metrics.items():
            if value is None:
                continue
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1

    metrics = {"loss": total_loss / float(n_batches)}
    metrics.update({key: total / counts[key] for key, total in totals.items()})
    return metrics


def evaluate_level(model, args, stoi, level: int, rng: random.Random, *, generation_mode: str | None = None):
    return evaluate_batches(
        model,
        args,
        batch_builder=lambda batch_rng: build_task_batch(args, stoi, level, batch_rng),
        rng=rng,
        generation_mode=generation_mode,
        final_token_id=stoi.get("<final>"),
    )


def format_eval_metrics(metrics: dict[str, float]) -> str:
    text = (
        f"seq_acc {float(metrics['exact_match']):.3f} | "
        f"token_acc {float(metrics['token_accuracy']):.3f}"
    )
    if "final_exact_match" in metrics:
        text += f" | final_acc {float(metrics['final_exact_match']):.3f}"
    if "trace_token_accuracy" in metrics:
        text += f" | trace_tok {float(metrics['trace_token_accuracy']):.3f}"
    return text


def format_pass_losses(pass_losses: list[torch.Tensor]) -> str:
    if len(pass_losses) == 1:
        return f"{pass_losses[0].item():.4f}"
    return "[" + ", ".join(f"{loss.item():.4f}" for loss in pass_losses) + "]"


def log_jsonl(path: str | None, event: dict):
    if path is None:
        return
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def should_log_eval(args) -> bool:
    return args.log_policy == "evals"


def print_common_run_header(args, block_size: int, vocab_size: int, model, *, training_mode: str):
    print(f"device: {args.device}")
    print("task_family: bbh_symbolic")
    print(f"training_mode: {training_mode}")
    print(f"task: {args.task}")
    if args.task == "permutation":
        print(f"num_objects: {args.num_objects}")
    print(f"supervision: {args.supervision}")
    print(f"architecture: {args.architecture}")
    print(f"model_size: {args.model_size}")
    print(f"generation_mode: {args.generation_mode}")
    if is_multi_pass_architecture(args.architecture):
        print(f"n_pass: {args.n_pass}")
        if args.architecture == "memory_update":
            print(f"memory_update_gate: {args.memory_update_gate}")
            if args.memory_update_gate == "on":
                print(f"memory_gate_bias: {args.memory_gate_bias}")
        print(f"pass_loss_weights: {args.pass_loss_weights}")
    print(f"max_level: {args.max_level}")
    print(f"log_policy: {args.log_policy}")
    print(f"block_size: {block_size}")
    print(f"vocab_size: {vocab_size}")
    print("number of parameters: %.2fM" % (model.get_num_params() / 1e6,))
