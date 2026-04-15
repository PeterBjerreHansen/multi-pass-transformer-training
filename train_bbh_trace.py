import argparse
import random

import torch

from bbh_training import (
    add_common_args,
    build_task_batch,
    build_training_objects,
    evaluate_level,
    finalize_args,
    format_eval_metrics,
    format_pass_losses,
    forward_and_loss,
    generation_mode_for,
    log_jsonl,
    print_common_run_header,
    set_seed,
    should_log_eval,
    validate_model_args,
    validate_task_args,
)
from model_factory import is_multi_pass_architecture


def parse_level_list(value: str) -> list[int]:
    if not value:
        return []
    try:
        levels = [int(item) for item in value.split(",") if item]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--eval-levels must be a comma-separated list of integers") from exc
    if not levels:
        raise argparse.ArgumentTypeError("--eval-levels must contain at least one level")
    return levels


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Train trace-supervised BBH-symbolic tasks at one fixed level.",
        allow_abbrev=False,
    )
    add_common_args(parser, default_log_policy="evals")
    parser.add_argument(
        "--eval-levels",
        type=parse_level_list,
        default=None,
        help="Comma-separated eval levels. Defaults to the fixed training level.",
    )
    args = parser.parse_args(argv)
    args = finalize_args(args, supervision="trace")
    args.curriculum_start_level = args.max_level
    args.curriculum_threshold = None
    args.review_easier_every = 0
    if args.eval_levels is None:
        args.eval_levels = [args.max_level]
    return args


def print_run_header(args, block_size: int, vocab_size: int, model):
    print_common_run_header(args, block_size, vocab_size, model, training_mode="trace_fixed")
    print(f"fixed_level: {args.max_level}")
    print("eval_levels: " + ",".join(str(level) for level in args.eval_levels))


def run_trace_fixed(args):
    set_seed(args.seed)
    validate_model_args(args)
    validate_task_args(args)

    vocab_level = max([args.max_level, *args.eval_levels])
    task, block_size, vocab, stoi, model, optimizer = build_training_objects(args, vocab_level=vocab_level)
    for level in args.eval_levels:
        if level < task.min_level:
            raise ValueError(f"--eval-levels entries must be >= {task.min_level} for task {args.task}")

    train_rng = random.Random(args.seed + 10)
    eval_rng = random.Random(args.seed + 20)
    log_rng = random.Random(args.seed + 30)

    print_run_header(args, block_size, len(vocab), model)
    log_jsonl(
        args.log_jsonl,
        {
            "event": "run_start",
            "task_family": "bbh_symbolic",
            "task": args.task,
            "training_mode": "trace_fixed",
            "fixed_level": args.max_level,
            "eval_levels": args.eval_levels,
            "config": vars(args),
            "block_size": block_size,
            "vocab_size": len(vocab),
        },
    )

    for step in range(1, args.train_steps + 1):
        model.train()
        batch = build_task_batch(args, stoi, args.max_level, train_rng)
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = forward_and_loss(model, batch, args)
        loss.backward()
        optimizer.step()

        should_log = step == 1 or step % args.eval_interval == 0 or step == args.train_steps
        if not should_log:
            continue

        model.eval()
        with torch.no_grad():
            log_batch = build_task_batch(args, stoi, args.max_level, log_rng)
            log_loss, _, log_pass_losses = forward_and_loss(model, log_batch, args)

        print(
            f"step {step:5d} | train_loss {log_loss.item():.4f} | "
            f"pass_losses {format_pass_losses(log_pass_losses)} | "
            f"fixed_level {args.max_level:3d}"
        )
        if should_log_eval(args):
            log_jsonl(
                args.log_jsonl,
                {
                    "event": "train_step",
                    "step": step,
                    "train_loss": float(log_loss.detach().item()),
                    "fixed_level": args.max_level,
                },
            )

        for eval_level in args.eval_levels:
            metrics = evaluate_level(model, args, stoi, eval_level, eval_rng)
            print(
                f"  eval_trace level {eval_level:3d} | "
                f"loss {float(metrics['loss']):.4f} | "
                f"{format_eval_metrics(metrics)}"
            )
            if should_log_eval(args):
                log_jsonl(
                    args.log_jsonl,
                    {
                        "event": "eval",
                        "step": step,
                        "level": eval_level,
                        "generation_mode": generation_mode_for(args),
                        "metrics": metrics,
                    },
                )

            if args.compare_generation_modes and is_multi_pass_architecture(args.architecture):
                for mode in ("recompute", "greedy"):
                    if mode == args.generation_mode:
                        continue
                    mode_metrics = evaluate_level(model, args, stoi, eval_level, eval_rng, generation_mode=mode)
                    print(
                        f"  eval_{mode:9s} level {eval_level:3d} | "
                        f"loss {float(mode_metrics['loss']):.4f} | "
                        f"{format_eval_metrics(mode_metrics)}"
                    )
                    if should_log_eval(args):
                        log_jsonl(
                            args.log_jsonl,
                            {
                                "event": "eval",
                                "step": step,
                                "level": eval_level,
                                "generation_mode": mode,
                                "metrics": mode_metrics,
                            },
                        )

    log_jsonl(
        args.log_jsonl,
        {
            "event": "run_end",
            "fixed_level": args.max_level,
            "eval_levels": args.eval_levels,
        },
    )


def main(argv: list[str] | None = None):
    run_trace_fixed(parse_args(argv))


if __name__ == "__main__":
    main()
