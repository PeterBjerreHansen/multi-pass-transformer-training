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


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Train answer-only BBH-symbolic tasks with a curriculum.",
        allow_abbrev=False,
    )
    add_common_args(parser, default_log_policy="promotions")
    parser.add_argument("--curriculum-start-level", type=int, default=None)
    parser.add_argument("--curriculum-threshold", type=float, default=0.95)
    parser.add_argument(
        "--review-easier-every",
        type=int,
        default=2,
        help="Sample an easier task every Nth batch. Use 0 to disable.",
    )
    args = parser.parse_args(argv)
    return finalize_args(args, supervision="final")


def choose_curriculum_train_level(
    step: int,
    current_level: int,
    review_min_level: int,
    curriculum_start_level: int,
    review_easier_every: int,
    rng: random.Random,
) -> int:
    if review_easier_every > 0 and current_level > curriculum_start_level and step % review_easier_every == 0:
        return rng.randint(review_min_level, current_level - 1)
    return current_level


def choose_easier_eval_level(current_level: int, min_level: int, rng: random.Random) -> int:
    return rng.randint(min_level, current_level - 1)


def print_run_header(args, block_size: int, vocab_size: int, model):
    print_common_run_header(args, block_size, vocab_size, model, training_mode="answer_curriculum")
    print(f"curriculum_start_level: {args.curriculum_start_level}")
    print(f"curriculum_threshold: {args.curriculum_threshold}")
    print(f"review_easier_every: {args.review_easier_every}")


def run_answer_curriculum(args):
    set_seed(args.seed)
    validate_model_args(args)
    validate_task_args(args)

    task, block_size, vocab, stoi, model, optimizer = build_training_objects(args)
    train_rng = random.Random(args.seed + 10)
    eval_rng = random.Random(args.seed + 20)
    log_rng = random.Random(args.seed + 30)
    current_level = args.curriculum_start_level
    promotion_history: list[tuple[int, int, float]] = []

    print_run_header(args, block_size, len(vocab), model)
    log_jsonl(
        args.log_jsonl,
        {
            "event": "run_start",
            "task_family": "bbh_symbolic",
            "task": args.task,
            "training_mode": "answer_curriculum",
            "config": vars(args),
            "block_size": block_size,
            "vocab_size": len(vocab),
        },
    )

    for step in range(1, args.train_steps + 1):
        model.train()
        sampled_level = choose_curriculum_train_level(
            step,
            current_level,
            task.min_level,
            args.curriculum_start_level,
            args.review_easier_every,
            train_rng,
        )

        batch = build_task_batch(args, stoi, sampled_level, train_rng)
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = forward_and_loss(model, batch, args)
        loss.backward()
        optimizer.step()

        should_log = step == 1 or step % args.eval_interval == 0 or step == args.train_steps
        if not should_log:
            continue

        model.eval()
        with torch.no_grad():
            log_batch = build_task_batch(args, stoi, current_level, log_rng)
            log_loss, _, log_pass_losses = forward_and_loss(model, log_batch, args)

        print(
            f"step {step:5d} | train_loss {log_loss.item():.4f} | "
            f"pass_losses {format_pass_losses(log_pass_losses)} | "
            f"train_level {current_level:3d}"
        )
        if should_log_eval(args):
            log_jsonl(
                args.log_jsonl,
                {
                    "event": "train_step",
                    "step": step,
                    "train_loss": float(log_loss.detach().item()),
                    "train_level": current_level,
                    "sampled_train_level": sampled_level,
                    "curriculum_level": current_level,
                },
            )

        current_metrics = evaluate_level(model, args, stoi, current_level, eval_rng)
        print(
            f"  eval_current level {current_level:3d} | "
            f"loss {float(current_metrics['loss']):.4f} | "
            f"{format_eval_metrics(current_metrics)}"
        )
        if should_log_eval(args):
            log_jsonl(
                args.log_jsonl,
                {
                    "event": "eval",
                    "step": step,
                    "level": current_level,
                    "generation_mode": generation_mode_for(args),
                    "metrics": current_metrics,
                },
            )

        if args.compare_generation_modes and is_multi_pass_architecture(args.architecture):
            for mode in ("recompute", "greedy"):
                if mode == args.generation_mode:
                    continue
                mode_metrics = evaluate_level(model, args, stoi, current_level, eval_rng, generation_mode=mode)
                print(
                    f"  eval_{mode:9s} level {current_level:3d} | "
                    f"loss {float(mode_metrics['loss']):.4f} | "
                    f"{format_eval_metrics(mode_metrics)}"
                )
                if should_log_eval(args):
                    log_jsonl(
                        args.log_jsonl,
                        {
                            "event": "eval",
                            "step": step,
                            "level": current_level,
                            "generation_mode": mode,
                            "metrics": mode_metrics,
                        },
                    )

        if current_level > args.curriculum_start_level:
            easier_level = choose_easier_eval_level(current_level, task.min_level, eval_rng)
            easier_metrics = evaluate_level(model, args, stoi, easier_level, eval_rng)
            print(
                f"  eval_easier level {easier_level:3d} | "
                f"loss {float(easier_metrics['loss']):.4f} | "
                f"{format_eval_metrics(easier_metrics)}"
            )
            if should_log_eval(args):
                log_jsonl(
                    args.log_jsonl,
                    {"event": "eval_easier", "step": step, "level": easier_level, "metrics": easier_metrics},
                )

        metric_value = float(current_metrics["exact_match"])
        if metric_value >= args.curriculum_threshold and current_level < args.max_level:
            promotion_history.append((current_level, step, metric_value))
            current_level += 1
            print(f"  curriculum_promote -> level {current_level:3d} | seq_acc {metric_value:.3f}")
            log_jsonl(
                args.log_jsonl,
                {
                    "event": "curriculum_promote",
                    "step": step,
                    "solved_level": current_level - 1,
                    "next_level": current_level,
                    "seq_acc": metric_value,
                },
            )

    log_jsonl(
        args.log_jsonl,
        {
            "event": "run_end",
            "final_level": current_level,
            "promotion_history": [
                {"solved_level": level, "step": step, "seq_acc": metric}
                for level, step, metric in promotion_history
            ],
        },
    )

    if promotion_history:
        print("promotion_history:")
        for solved_level, step, metric_value in promotion_history:
            print(f"  solved_level {solved_level:3d} | promoted_at_step {step:5d} | seq_acc {metric_value:.3f}")
    else:
        print("promotion_history: none")


def main(argv: list[str] | None = None):
    run_answer_curriculum(parse_args(argv))


if __name__ == "__main__":
    main()
