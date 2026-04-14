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


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Train transformer architectures on procedural BBH-inspired symbolic tasks.",
        allow_abbrev=False,
    )
    parser.add_argument("--task", choices=sorted(TASKS), default="walk")
    parser.add_argument("--supervision", choices=["final", "trace"], default="final")
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
    parser.add_argument("--curriculum-start-level", type=int, default=None)
    parser.add_argument("--max-level", type=int, default=None)
    parser.add_argument("--curriculum-threshold", type=float, default=0.99)
    parser.add_argument(
        "--review-easier-every",
        type=int,
        default=2,
        help="Sample an easier task every Nth batch. Use 0 to disable.",
    )
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
        default="promotions",
        help="JSONL logging detail. 'promotions' keeps logs compact; 'evals' records every eval event.",
    )
    args = parser.parse_args(argv)
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
    if not 0.0 <= args.curriculum_threshold <= 1.0:
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
def generation_metrics(model, batch, args, *, generation_mode: str | None = None) -> tuple[float, float]:
    exact_matches = []
    token_accuracies = []
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
    *,
    generation_mode: str | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_exact_match = 0.0
    total_token_accuracy = 0.0
    n_batches = args.eval_batches if eval_batches is None else eval_batches

    for _ in range(n_batches):
        batch = batch_builder(rng)
        loss, _, _ = forward_and_loss(model, batch, args)
        exact_match, token_accuracy = generation_metrics(model, batch, args, generation_mode=generation_mode)
        total_loss += float(loss.detach().item())
        total_exact_match += exact_match
        total_token_accuracy += token_accuracy

    denom = float(n_batches)
    return {
        "loss": total_loss / denom,
        "exact_match": total_exact_match / denom,
        "token_accuracy": total_token_accuracy / denom,
    }


def evaluate_level(model, args, stoi, level: int, rng: random.Random, *, generation_mode: str | None = None):
    return evaluate_batches(
        model,
        args,
        batch_builder=lambda batch_rng: build_task_batch(args, stoi, level, batch_rng),
        rng=rng,
        generation_mode=generation_mode,
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


def log_jsonl(path: str | None, event: dict):
    if path is None:
        return
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def should_log_eval(args) -> bool:
    return args.log_policy == "evals"


def print_run_header(args, block_size: int, vocab_size: int, model):
    print(f"device: {args.device}")
    print("task_family: bbh_symbolic")
    print("training_mode: curriculum")
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
    print(f"curriculum_start_level: {args.curriculum_start_level}")
    print(f"max_level: {args.max_level}")
    print(f"curriculum_threshold: {args.curriculum_threshold}")
    print(f"review_easier_every: {args.review_easier_every}")
    print(f"log_policy: {args.log_policy}")
    print(f"block_size: {block_size}")
    print(f"vocab_size: {vocab_size}")
    print("number of parameters: %.2fM" % (model.get_num_params() / 1e6,))


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    set_seed(args.seed)
    validate_model_args(args)
    validate_task_args(args)

    task = TASKS[args.task]
    kwargs = task_kwargs(args)
    block_size = args.block_size or task.required_block_size(args.max_level, args.supervision, **kwargs)
    vocab, stoi, _ = task.build_vocab(args.max_level, **kwargs)
    model = build_model(args, vocab_size=len(vocab), block_size=block_size, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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


if __name__ == "__main__":
    main()
