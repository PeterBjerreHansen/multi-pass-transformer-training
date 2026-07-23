from __future__ import annotations

import argparse
from dataclasses import dataclass
import random
import time
from typing import Callable

from tasks.bbh import permutation, pointer_chasing, state_machine, tracking
from experiments.common import (
    append_jsonl,
    build_model_and_optimizer,
    effective_inference_mode,
    evaluate_prebuilt_batches,
    generation_aligned_loss,
    format_checkpoint_line,
    format_default_eval_metrics,
    format_gradient_norms,
    format_memory_gate_stats,
    format_pass_losses,
    forward_and_loss,
    gradient_norms,
    load_checkpoint_payload,
    memory_gate_stats,
    model_benchmark_stats,
    new_append_train_stats,
    prepare_run_artifacts,
    resolve_device_arg,
    resolve_resume_artifacts,
    restore_checkpoint_state,
    runtime_resource_stats,
    save_latest_checkpoint,
    set_seed,
    stable_seed,
    synchronize_device,
    summarize_gradient_norm_window,
    update_append_train_stats,
    update_gradient_norm_window,
    validate_model_args,
    validate_training_args,
)
from experiments.presets import BBH_PRESETS, preset_help_text, resolve_preset_args
from model_factory import ARCHITECTURES


@dataclass(frozen=True)
class BBHTask:
    name: str
    min_level: int
    shape_arg_names: tuple[str, ...]
    vocab_builder: Callable
    block_size_builder: Callable
    batch_builder: Callable

    def shape_kwargs(self, args) -> dict[str, int]:
        return {name: getattr(args, name) for name in self.shape_arg_names}

    def build_vocab(self, args):
        return self.vocab_builder(**self.shape_kwargs(args))

    def required_block_size(self, args, level: int) -> int:
        kwargs = self.shape_kwargs(args)
        level_name = {
            "pointer_chasing": "num_hops",
            "permutation": "num_swaps",
            "tracking": "num_ops",
            "state_machine": "num_steps",
        }[self.name]
        return self.block_size_builder(**kwargs, **{level_name: level})

    def build_batch(self, args, *, batch_size: int, level: int, stoi, rng: random.Random):
        kwargs = self.shape_kwargs(args)
        level_name = {
            "pointer_chasing": "num_hops",
            "permutation": "num_swaps",
            "tracking": "num_ops",
            "state_machine": "num_steps",
        }[self.name]
        return self.batch_builder(
            batch_size=batch_size,
            stoi=stoi,
            device=args.device,
            rng=rng,
            **kwargs,
            **{level_name: level},
        )


BBH_TASKS = {
    "pointer_chasing": BBHTask(
        "pointer_chasing", 0, ("num_nodes",),
        pointer_chasing.build_pointer_chasing_vocab,
        pointer_chasing.required_block_size,
        pointer_chasing.build_pointer_chasing_batch,
    ),
    "permutation": BBHTask(
        "permutation", 0, ("num_objects",),
        permutation.build_permutation_vocab,
        permutation.required_block_size,
        permutation.build_permutation_batch,
    ),
    "tracking": BBHTask(
        "tracking", 1, ("num_objects",),
        tracking.build_tracking_vocab,
        tracking.required_block_size,
        tracking.build_tracking_batch,
    ),
    "state_machine": BBHTask(
        "state_machine", 0, ("num_states", "alphabet_size"),
        state_machine.build_state_machine_vocab,
        state_machine.required_block_size,
        state_machine.build_state_machine_batch,
    ),
}


def _add_override(parser, *names, **kwargs) -> None:
    parser.add_argument(*names, default=argparse.SUPPRESS, **kwargs)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Train BBH-style final-answer curricula.",
        allow_abbrev=False,
    )
    _add_override(parser, "--preset", choices=sorted(BBH_PRESETS), help=preset_help_text(BBH_PRESETS))
    _add_override(parser, "--task", choices=sorted(BBH_TASKS))
    _add_override(parser, "--architecture", choices=ARCHITECTURES)
    _add_override(parser, "--model-size", choices=["tiny", "small", "medium", "large"])
    _add_override(parser, "--inference-mode", choices=["recompute", "append_recurrent"])
    _add_override(parser, "--token-selection", choices=["sample", "argmax"])
    _add_override(parser, "--n-layer", type=int)
    _add_override(parser, "--n-head", type=int)
    _add_override(parser, "--n-embd", type=int)
    _add_override(parser, "--n-pass", type=int)
    _add_override(parser, "--pass-loss-weights", type=float, nargs="*")
    _add_override(parser, "--memory-update-gate", choices=["on", "off"])
    _add_override(parser, "--memory-gate-bias", type=float)
    _add_override(parser, "--append-train-prob", type=float)
    _add_override(parser, "--append-train-microbatch-size", type=int)
    _add_override(parser, "--append-train-horizon", type=int)
    _add_override(parser, "--append-train-loss-weight", type=float)
    _add_override(parser, "--append-train-warmup-steps", type=int)
    _add_override(parser, "--append-train-ramp-steps", type=int)
    _add_override(parser, "--memory-gate-init", type=float)
    _add_override(parser, "--num-nodes", type=int)
    _add_override(parser, "--num-objects", type=int)
    _add_override(parser, "--num-states", type=int)
    _add_override(parser, "--alphabet-size", type=int)
    _add_override(parser, "--curriculum-start-level", type=int)
    _add_override(parser, "--curriculum-threshold", type=float)
    _add_override(parser, "--review-easier-every", type=int)
    _add_override(parser, "--max-level", type=int)
    _add_override(parser, "--batch-size", type=int)
    _add_override(parser, "--train-steps", type=int)
    _add_override(parser, "--lr", type=float)
    _add_override(parser, "--weight-decay", type=float)
    _add_override(parser, "--eval-interval", type=int)
    _add_override(parser, "--eval-batches", type=int)
    _add_override(parser, "--seed", type=int)
    _add_override(parser, "--device")
    _add_override(parser, "--block-size", type=int)
    _add_override(parser, "--run-dir")
    _add_override(parser, "--resume-from")
    raw = parser.parse_args(argv)
    return resolve_preset_args(
        raw,
        BBH_PRESETS,
        default_preset="pointer_chasing_main",
        parser=parser,
    )


def validate_task_args(args) -> None:
    if args.task not in BBH_TASKS:
        raise ValueError(f"unsupported BBH task: {args.task}")
    task = BBH_TASKS[args.task]
    if args.curriculum_start_level < task.min_level:
        raise ValueError(f"--curriculum-start-level must be >= {task.min_level}")
    if args.max_level < args.curriculum_start_level:
        raise ValueError("--max-level must be >= --curriculum-start-level")
    if not 0 <= args.curriculum_threshold <= 1:
        raise ValueError("--curriculum-threshold must be in [0, 1]")
    if args.review_easier_every < 0:
        raise ValueError("--review-easier-every must be non-negative")
    task.build_vocab(args)
    required = task.required_block_size(args, args.max_level)
    if args.block_size is not None and args.block_size < required:
        raise ValueError(f"--block-size must be at least {required} for level {args.max_level}")


def build_training_objects(args):
    task = BBH_TASKS[args.task]
    block_size = args.block_size or task.required_block_size(args, args.max_level)
    vocab, stoi, itos = task.build_vocab(args)
    model, optimizer = build_model_and_optimizer(args, vocab_size=len(vocab), block_size=block_size)
    return task, block_size, vocab, stoi, itos, model, optimizer


def build_fixed_eval_batches(args, task: BBHTask, stoi, level: int):
    rng = random.Random(stable_seed(args.seed, "bbh", args.task, "eval", level))
    return [
        task.build_batch(args, batch_size=args.batch_size, level=level, stoi=stoi, rng=rng)
        for _ in range(args.eval_batches)
    ]


def choose_train_level(args, task: BBHTask, current_level: int, step: int, rng: random.Random) -> int:
    if (
        args.review_easier_every > 0
        and current_level > args.curriculum_start_level
        and step % args.review_easier_every == 0
    ):
        return rng.randint(task.min_level, current_level - 1)
    return current_level


def _apply_resume_args(args, checkpoint: dict) -> None:
    saved = checkpoint.get("args", {})
    preserve = {
        "resume_from": args.resume_from,
        "run_dir": args.run_dir,
        "train_steps": args.train_steps,
        "device": args.device,
    }
    for key, value in saved.items():
        setattr(args, key, value)
    for key, value in preserve.items():
        if value is not None:
            setattr(args, key, value)


def run_answer_curriculum(args) -> None:
    checkpoint = None
    resume_step = 0
    if args.resume_from:
        artifacts = resolve_resume_artifacts(args.resume_from)
        checkpoint = load_checkpoint_payload(artifacts.checkpoint_path, device="cpu")
        resume_step = int(checkpoint.get("step", 0))
        _apply_resume_args(args, checkpoint)

    resolve_device_arg(args)
    set_seed(args.seed)
    validate_model_args(args)
    validate_training_args(args)
    validate_task_args(args)
    task, block_size, vocab, stoi, _itos, model, optimizer = build_training_objects(args)
    artifacts = prepare_run_artifacts(
        args,
        model=model,
        default_root_parts=("bbh", args.task, args.architecture),
        extra_config={"script": "experiments.train_bbh"},
    )

    train_rng = random.Random(stable_seed(args.seed, "bbh", args.task, "train"))
    append_train_rng = random.Random(
        stable_seed(args.seed, "bbh", args.task, "append_train")
    )
    append_train_stats = new_append_train_stats(args.append_train_prob)
    current_level = args.curriculum_start_level
    promotion_history: list[tuple[int, int, float]] = []
    if checkpoint is not None:
        restore_checkpoint_state(checkpoint, model=model, optimizer=optimizer, device=args.device)
        extra = checkpoint.get("extra_state", {})
        current_level = int(extra.get("current_level", current_level))
        promotion_history = [tuple(item) for item in extra.get("promotion_history", [])]
        if "train_rng_state" in extra:
            train_rng.setstate(extra["train_rng_state"])
        if "append_train_rng_state" in extra:
            append_train_rng.setstate(extra["append_train_rng_state"])
        if "append_train_stats" in extra:
            append_train_stats = extra["append_train_stats"]

    print(f"device: {args.device}")
    print(f"task: {args.task}")
    print(f"architecture: {args.architecture}")
    print(f"inference_mode: {effective_inference_mode(args)}")
    print(f"block_size: {block_size}")
    print(f"parameters: {model.get_num_params():,}")
    if args.architecture != "transformer":
        normalized = [weight / sum(args.pass_loss_weights) for weight in args.pass_loss_weights]
        print(f"n_pass: {args.n_pass}")
        print(f"pass_loss_weights_normalized: {normalized}")
        print(f"append_train_prob: {args.append_train_prob}")
        if args.append_train_prob > 0:
            print(
                "append_train: "
                f"microbatch={args.append_train_microbatch_size} "
                f"horizon={args.append_train_horizon} "
                f"weight={args.append_train_loss_weight} "
                f"warmup={args.append_train_warmup_steps} "
                f"ramp={args.append_train_ramp_steps}"
            )
    gates = memory_gate_stats(model)
    if gates is not None:
        print(f"memory_gates: {format_memory_gate_stats(gates)}")

    append_jsonl(
        artifacts.metrics_path,
        {
            "event": "run_start" if checkpoint is None else "run_resume",
            "step": resume_step,
            "task": args.task,
            "architecture": args.architecture,
            "config": vars(args),
            "model_stats": model_benchmark_stats(model),
        },
    )

    start_step = resume_step + 1
    final_step = resume_step + args.train_steps if checkpoint is not None else args.train_steps
    window_start = time.perf_counter()
    window_steps = 0
    window_tokens = 0
    gradient_norm_window: dict[str, dict[str, float]] = {}
    append_train_window_stats = new_append_train_stats(args.append_train_prob)

    for step in range(start_step, final_step + 1):
        model.train()
        sampled_level = choose_train_level(args, task, current_level, step, train_rng)
        batch = task.build_batch(
            args,
            batch_size=args.batch_size,
            level=sampled_level,
            stoi=stoi,
            rng=train_rng,
        )
        optimizer.zero_grad(set_to_none=True)
        loss, output, pass_losses = forward_and_loss(model, batch, args)
        loss.backward()
        append_loss, append_step_stats = generation_aligned_loss(
            model,
            batch,
            output,
            args,
            step=step,
            rng=append_train_rng,
        )
        if append_loss is not None:
            append_loss.backward()
        update_append_train_stats(append_train_stats, append_step_stats)
        update_append_train_stats(append_train_window_stats, append_step_stats)
        update_gradient_norm_window(gradient_norm_window, gradient_norms(model))
        optimizer.step()
        window_steps += 1
        window_tokens += int(batch.idx.numel())

        should_eval = step == 1 or step % args.eval_interval == 0 or step == final_step
        if not should_eval:
            continue

        synchronize_device(args.device)
        elapsed = time.perf_counter() - window_start
        tok_per_s = window_tokens / elapsed if elapsed > 0 else 0.0
        fields = [f"loss {loss.item():.4f}", f"tok/s {tok_per_s:.1f}", f"level {current_level}"]
        gradient_summary = summarize_gradient_norm_window(gradient_norm_window)
        fields.append(format_gradient_norms(gradient_summary))
        if args.architecture != "transformer":
            fields.append(f"pass_losses {format_pass_losses(pass_losses)}")
            if append_train_window_stats["applied_updates"]:
                fields.append(
                    "append_loss "
                    f"{append_train_window_stats['mean_raw_loss']:.4f} "
                    f"({append_train_window_stats['applied_updates']} updates)"
                )
        print(format_checkpoint_line(f"step {step}", fields))

        eval_batches = build_fixed_eval_batches(args, task, stoi, current_level)
        metrics = evaluate_prebuilt_batches(
            model,
            args,
            eval_batches,
            inference_mode=args.inference_mode,
            generation_seed=stable_seed(args.seed, "bbh", args.task, "generation", current_level),
        )
        print(
            format_checkpoint_line(
                "eval",
                [f"loss {metrics['loss']:.4f}", f"level {current_level}", format_default_eval_metrics(metrics)],
            )
        )
        append_jsonl(
            artifacts.metrics_path,
            {
                "event": "eval",
                "step": step,
                "level": current_level,
                "sampled_train_level": sampled_level,
                "train_loss": float(loss.item()),
                "train_objective_loss": float(loss.item())
                + (float(append_step_stats["weighted_loss"]) if append_step_stats["applied"] else 0.0),
                "pass_losses": [float(item.item()) for item in pass_losses],
                "append_train_step": append_step_stats,
                "append_train_window_stats": append_train_window_stats,
                "append_train_stats": append_train_stats,
                "metrics": metrics,
                "gradient_norms": gradient_summary,
                "memory_gate_stats": memory_gate_stats(model),
                "train_tok_per_s": tok_per_s,
                "resource_stats": runtime_resource_stats(args.device),
            },
        )

        exact_match = float(metrics["exact_match"])
        if exact_match >= args.curriculum_threshold and current_level < args.max_level:
            promotion_history.append((current_level, step, exact_match))
            current_level += 1
            print(f"curriculum_promote -> level {current_level}")

        save_latest_checkpoint(
            artifacts,
            model=model,
            optimizer=optimizer,
            args=args,
            step=step,
            extra_state={
                "current_level": current_level,
                "promotion_history": promotion_history,
                "train_rng_state": train_rng.getstate(),
                "append_train_rng_state": append_train_rng.getstate(),
                "append_train_stats": append_train_stats,
            },
        )
        synchronize_device(args.device)
        window_start = time.perf_counter()
        window_steps = 0
        window_tokens = 0
        gradient_norm_window = {}
        append_train_window_stats = new_append_train_stats(args.append_train_prob)

    append_jsonl(
        artifacts.metrics_path,
        {"event": "run_end", "final_level": current_level, "promotion_history": promotion_history},
    )
    print(f"run_dir: {artifacts.run_dir}")


def main(argv: list[str] | None = None) -> None:
    run_answer_curriculum(parse_args(argv))


if __name__ == "__main__":
    main()
