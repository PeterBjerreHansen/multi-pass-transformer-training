import argparse
import json
import random
from pathlib import Path
from typing import Callable

import torch

from tasks.permutation import (
    build_permutation_batch,
    build_permutation_vocab,
    required_block_size,
)
from model_factory import ARCHITECTURES, build_model, is_multi_pass_architecture


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a transformer architecture on permutation composition with a curriculum.",
        allow_abbrev=False,
    )
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="transformer")
    parser.add_argument("--generation-mode", choices=["recompute", "greedy"], default="greedy")
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-pass", type=int, default=4)
    parser.add_argument("--memory-update-gate", choices=["on", "off"], default="off")
    parser.add_argument("--memory-gate-bias", type=float, default=-1.0)
    parser.add_argument("--pass-loss-weights", type=float, nargs="*", default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--log-jsonl",
        default=None,
        help="Optional path for structured training/eval events.",
    )
    parser.add_argument("--num-objects", type=int, default=4)
    parser.add_argument("--max-num-swaps", type=int, default=64)
    parser.add_argument("--curriculum-start-swaps", type=int, default=1)
    parser.add_argument("--curriculum-threshold", type=float, default=0.99)
    parser.add_argument(
        "--review-easier-every",
        type=int,
        default=2,
        help="Sample an easier task every Nth batch. Use 0 to disable.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=50_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


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
    if args.num_objects < 2:
        raise ValueError("--num-objects must be at least 2")
    if args.max_num_swaps < 0:
        raise ValueError("--max-num-swaps must be non-negative")
    if args.curriculum_start_swaps < 1:
        raise ValueError("--curriculum-start-swaps must be at least 1")
    if args.curriculum_start_swaps > args.max_num_swaps:
        raise ValueError("--curriculum-start-swaps must be <= --max-num-swaps")
    if not 0.0 <= args.curriculum_threshold <= 1.0:
        raise ValueError("--curriculum-threshold must be in [0, 1]")
    if args.review_easier_every < 0:
        raise ValueError("--review-easier-every must be >= 0")


def build_batch(args, stoi, num_swaps: int, rng: random.Random):
    return build_permutation_batch(
        batch_size=args.batch_size,
        num_objects=args.num_objects,
        num_swaps=num_swaps,
        stoi=stoi,
        device=args.device,
        rng=rng,
    )


def forward_and_loss(model, batch, args):
    if not is_multi_pass_architecture(args.architecture):
        logits = model(batch.idx)
        loss = model.calc_loss(logits, batch.targets)
        return loss, logits, [loss.detach()]

    logits_per_pass = model(
        batch.idx,
        return_all_logits=True,
    )
    loss, losses = model.calc_total_loss(
        logits_per_pass,
        batch.targets,
        loss_weights=args.pass_loss_weights,
    )
    return loss, logits_per_pass[-1], [item.detach() for item in losses]


@torch.no_grad()
def generation_metrics(model, batch, args) -> tuple[float, float]:
    exact_matches = []
    token_accuracies = []
    generation_mode = "recompute" if args.architecture == "transformer" else args.generation_mode

    for row in range(batch.idx.size(0)):
        prompt_len = int(batch.prompt_lengths[row].item())
        output_len = int(batch.output_lengths[row].item())
        prompt = batch.idx[row:row + 1, :prompt_len]
        target_suffix = batch.targets[row:row + 1, prompt_len - 1:prompt_len - 1 + output_len]

        generated = model.generate(
            prompt,
            max_new_tokens=output_len,
            do_sample=False,
            generation_mode=generation_mode,
        )
        generated_suffix = generated[:, prompt_len:prompt_len + output_len]
        exact_matches.append((generated_suffix == target_suffix).all(dim=1).float().mean())
        token_accuracies.append((generated_suffix == target_suffix).float().mean())

    return torch.stack(exact_matches).mean().item(), torch.stack(token_accuracies).mean().item()


@torch.no_grad()
def evaluate_batches(
    model,
    args,
    batch_builder: Callable[[random.Random], object],
    rng: random.Random,
    eval_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_exact_match = 0.0
    total_token_accuracy = 0.0
    n_batches = args.eval_batches if eval_batches is None else eval_batches

    for _ in range(n_batches):
        batch = batch_builder(rng)
        loss, _, _ = forward_and_loss(model, batch, args)
        exact_match, token_accuracy = generation_metrics(model, batch, args)
        total_loss += float(loss.detach().item())
        total_exact_match += exact_match
        total_token_accuracy += token_accuracy

    denom = float(n_batches)
    return {
        "loss": total_loss / denom,
        "exact_match": total_exact_match / denom,
        "token_accuracy": total_token_accuracy / denom,
    }


def evaluate_num_swaps(model, args, stoi, num_swaps: int, rng: random.Random):
    return evaluate_batches(
        model,
        args,
        batch_builder=lambda batch_rng: build_batch(args, stoi, num_swaps, batch_rng),
        rng=rng,
    )


def format_eval_metrics(metrics: dict[str, float]) -> str:
    return (
        f"seq_acc {float(metrics['exact_match']):.3f} | "
        f"token_acc {float(metrics['token_accuracy']):.3f}"
    )


def format_pass_losses(pass_losses: list[torch.Tensor]) -> str:
    if len(pass_losses) == 1:
        return f"{pass_losses[0].item():.4f}"
    return "[" + ", ".join(f"{loss.item():.4f}" for loss in pass_losses) + "]"


def choose_curriculum_train_level(
    step: int,
    current_level: int,
    start_level: int,
    review_easier_every: int,
    rng: random.Random,
) -> int:
    if review_easier_every > 0 and current_level > start_level and step % review_easier_every == 0:
        return rng.randint(0, current_level - 1)
    return current_level


def choose_easier_eval_level(current_level: int, rng: random.Random) -> int:
    return rng.randint(0, current_level - 1)


def log_jsonl(path: str | None, event: dict):
    if path is None:
        return
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def print_run_header(args, block_size: int, model):
    print(f"device: {args.device}")
    print("task: permutation")
    print("training_mode: curriculum")
    print(f"architecture: {args.architecture}")
    print(f"generation_mode: {args.generation_mode}")
    print(f"num_objects: {args.num_objects}")
    if is_multi_pass_architecture(args.architecture):
        print(f"n_pass: {args.n_pass}")
        if args.architecture == "memory_update":
            print(f"memory_update_gate: {args.memory_update_gate}")
            if args.memory_update_gate == "on":
                print(f"memory_gate_bias: {args.memory_gate_bias}")
        print(f"pass_loss_weights: {args.pass_loss_weights}")
    print(f"curriculum_start_swaps: {args.curriculum_start_swaps}")
    print(f"max_num_swaps: {args.max_num_swaps}")
    print(f"curriculum_threshold: {args.curriculum_threshold}")
    print(f"review_easier_every: {args.review_easier_every}")
    print(f"block_size: {block_size}")
    print("number of parameters: %.2fM" % (model.get_num_params() / 1e6,))


def main():
    args = parse_args()
    set_seed(args.seed)
    validate_model_args(args)
    validate_task_args(args)

    block_size = required_block_size(args.num_objects, args.max_num_swaps)
    vocab, stoi, _ = build_permutation_vocab(args.num_objects)
    model = build_model(args, vocab_size=len(vocab), block_size=block_size, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_rng = random.Random(args.seed + 10)
    eval_rng = random.Random(args.seed + 20)
    log_rng = random.Random(args.seed + 30)
    current_level = args.curriculum_start_swaps
    promotion_history: list[tuple[int, int, float]] = []

    print_run_header(args, block_size, model)
    log_jsonl(args.log_jsonl, {"event": "run_start", "task": "permutation", "config": vars(args), "block_size": block_size})

    for step in range(1, args.train_steps + 1):
        model.train()
        sampled_num_swaps = choose_curriculum_train_level(
            step,
            current_level,
            args.curriculum_start_swaps,
            args.review_easier_every,
            train_rng,
        )

        batch = build_batch(args, stoi, sampled_num_swaps, train_rng)
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = forward_and_loss(model, batch, args)
        loss.backward()
        optimizer.step()

        should_log = step == 1 or step % args.eval_interval == 0 or step == args.train_steps
        if not should_log:
            continue

        model.eval()
        with torch.no_grad():
            log_batch = build_batch(args, stoi, current_level, log_rng)
            log_loss, _, log_pass_losses = forward_and_loss(model, log_batch, args)

        suffix = f"train_swaps {current_level:3d} | curriculum_level {current_level:3d}"
        print(
            f"step {step:5d} | train_loss {log_loss.item():.4f} | "
            f"pass_losses {format_pass_losses(log_pass_losses)} | {suffix}"
        )
        log_jsonl(
            args.log_jsonl,
            {
                "event": "train_step",
                "step": step,
                "train_loss": float(log_loss.detach().item()),
                "train_swaps": current_level,
                "sampled_train_swaps": sampled_num_swaps,
                "curriculum_level": current_level,
            },
        )

        current_metrics = evaluate_num_swaps(model, args, stoi, current_level, eval_rng)
        print(
            f"  eval_current swaps {current_level:3d} | "
            f"loss {float(current_metrics['loss']):.4f} | "
            f"{format_eval_metrics(current_metrics)}"
        )
        log_jsonl(args.log_jsonl, {"event": "eval", "step": step, "level": current_level, "metrics": current_metrics})

        if current_level > args.curriculum_start_swaps:
            easier_level = choose_easier_eval_level(current_level, eval_rng)
            easier_metrics = evaluate_num_swaps(model, args, stoi, easier_level, eval_rng)
            print(
                f"  eval_easier swaps  {easier_level:3d} | "
                f"loss {float(easier_metrics['loss']):.4f} | "
                f"{format_eval_metrics(easier_metrics)}"
            )
            log_jsonl(args.log_jsonl, {"event": "eval_easier", "step": step, "level": easier_level, "metrics": easier_metrics})

        metric_value = float(current_metrics["exact_match"])
        if metric_value >= args.curriculum_threshold and current_level < args.max_num_swaps:
            promotion_history.append((current_level, step, metric_value))
            current_level += 1
            print(
                f"  curriculum_promote -> swaps {current_level:3d} | "
                f"seq_acc {metric_value:.3f}"
            )
            log_jsonl(args.log_jsonl, {"event": "curriculum_promote", "step": step, "next_level": current_level, "seq_acc": metric_value})

    if promotion_history:
        print("promotion_history:")
        for solved_swaps, step, metric_value in promotion_history:
            print(
                f"  solved_swaps {solved_swaps:3d} | "
                f"promoted_at_step {step:5d} | "
                f"seq_acc {metric_value:.3f}"
            )
    else:
        print("promotion_history: none")


if __name__ == "__main__":
    main()
