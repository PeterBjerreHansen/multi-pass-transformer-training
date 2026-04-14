import argparse
import random

import torch

from tasks.permutation import (
    build_permutation_batch,
    build_permutation_vocab,
    required_block_size,
)
from model_factory import build_model, is_multi_pass_architecture
from train_utils import (
    add_model_args,
    add_training_args,
    choose_curriculum_train_level,
    choose_easier_eval_level,
    evaluate_batches,
    format_eval_metrics,
    format_pass_losses,
    forward_and_loss,
    log_jsonl,
    set_seed,
    validate_model_args,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a transformer architecture on permutation composition with a curriculum.",
        allow_abbrev=False,
    )
    add_model_args(parser, default_n_embd=128)
    parser.add_argument("--num-objects", type=int, default=4)
    parser.add_argument("--max-num-swaps", type=int, default=32)
    parser.add_argument("--curriculum-start-swaps", type=int, default=1)
    parser.add_argument("--curriculum-threshold", type=float, default=0.98)
    parser.add_argument(
        "--review-easier-every",
        type=int,
        default=2,
        help="Sample an easier task every Nth batch. Use 0 to disable.",
    )
    add_training_args(parser, default_train_steps=50_000, default_lr=1e-4)
    return parser.parse_args()


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


def evaluate_num_swaps(model, args, stoi, num_swaps: int, rng: random.Random):
    return evaluate_batches(
        model,
        args,
        batch_builder=lambda batch_rng: build_batch(args, stoi, num_swaps, batch_rng),
        rng=rng,
    )


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
