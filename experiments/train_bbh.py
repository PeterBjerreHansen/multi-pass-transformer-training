import argparse
import random
import time
from dataclasses import dataclass
from typing import Callable

from tasks.bbh import permutation, pointer_chasing, state_machine, tracking
from experiments.common import (
    append_jsonl,
    basic_generation_metrics,
    build_model_and_optimizer,
    build_run_rngs,
    effective_inference_mode,
    evaluate_prebuilt_batches,
    format_checkpoint_line,
    format_default_eval_metrics,
    format_memory_gate_stats,
    format_pass_losses,
    forward_and_loss,
    load_checkpoint_payload,
    maybe_report_memory_gates,
    memory_gate_stats,
    prepare_run_artifacts,
    resolve_resume_artifacts,
    resolve_device_arg,
    restore_checkpoint_state,
    save_latest_checkpoint,
    set_seed,
    synchronize_device,
    validate_model_args,
)
from experiments.presets import BBH_PRESETS, preset_help_text, resolve_preset_args


@dataclass(frozen=True)
class BBHTask:
    name: str
    min_level: int
    default_start_level: int
    default_max_level: int
    shape_arg_names: tuple[str, ...]
    vocab_builder: Callable
    block_size_builder: Callable
    batch_builder: Callable

    def vocab_kwargs(self, args) -> dict[str, int]:
        return {name: getattr(args, name) for name in self.shape_arg_names}

    def build_vocab(self, args):
        return self.vocab_builder(**self.vocab_kwargs(args))

    def required_block_size(self, args, level: int) -> int:
        kwargs = dict(self.vocab_kwargs(args))
        if self.name == "pointer_chasing":
            return self.block_size_builder(num_hops=level, **kwargs)
        if self.name == "permutation":
            return self.block_size_builder(num_swaps=level, **kwargs)
        if self.name == "tracking":
            return self.block_size_builder(num_ops=level, **kwargs)
        if self.name == "state_machine":
            return self.block_size_builder(num_steps=level, **kwargs)
        raise ValueError(f"Unsupported BBH task: {self.name}")

    def build_batch(self, args, batch_size: int, level: int, stoi, device: str, rng: random.Random):
        kwargs = dict(self.vocab_kwargs(args))
        if self.name == "pointer_chasing":
            return self.batch_builder(batch_size=batch_size, num_hops=level, stoi=stoi, device=device, rng=rng, **kwargs)
        if self.name == "permutation":
            return self.batch_builder(batch_size=batch_size, num_swaps=level, stoi=stoi, device=device, rng=rng, **kwargs)
        if self.name == "tracking":
            return self.batch_builder(batch_size=batch_size, num_ops=level, stoi=stoi, device=device, rng=rng, **kwargs)
        if self.name == "state_machine":
            return self.batch_builder(batch_size=batch_size, num_steps=level, stoi=stoi, device=device, rng=rng, **kwargs)
        raise ValueError(f"Unsupported BBH task: {self.name}")


BBH_TASKS = {
    "permutation": BBHTask(
        name="permutation",
        min_level=0,
        default_start_level=1,
        default_max_level=64,
        shape_arg_names=("num_objects",),
        vocab_builder=permutation.build_permutation_vocab,
        block_size_builder=permutation.required_block_size,
        batch_builder=permutation.build_permutation_batch,
    ),
    "tracking": BBHTask(
        name="tracking",
        min_level=1,
        default_start_level=1,
        default_max_level=64,
        shape_arg_names=("num_objects",),
        vocab_builder=tracking.build_tracking_vocab,
        block_size_builder=tracking.required_block_size,
        batch_builder=tracking.build_tracking_batch,
    ),
    "pointer_chasing": BBHTask(
        name="pointer_chasing",
        min_level=0,
        default_start_level=0,
        default_max_level=64,
        shape_arg_names=("num_nodes",),
        vocab_builder=pointer_chasing.build_pointer_chasing_vocab,
        block_size_builder=pointer_chasing.required_block_size,
        batch_builder=pointer_chasing.build_pointer_chasing_batch,
    ),
    "state_machine": BBHTask(
        name="state_machine",
        min_level=0,
        default_start_level=0,
        default_max_level=64,
        shape_arg_names=("num_states", "alphabet_size"),
        vocab_builder=state_machine.build_state_machine_vocab,
        block_size_builder=state_machine.required_block_size,
        batch_builder=state_machine.build_state_machine_batch,
    ),
}


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Train BBH-style final-answer tasks from named experiment presets.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--preset",
        choices=sorted(BBH_PRESETS),
        default=argparse.SUPPRESS,
        help=f"Named BBH experiment preset. {preset_help_text(BBH_PRESETS)}",
    )
    parser.add_argument(
        "--architecture",
        choices=["transformer", "memory_tape", "memory_concat", "memory_update"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--token-selection", choices=["sample", "argmax"], default=argparse.SUPPRESS)
    parser.add_argument("--train-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--device", default=argparse.SUPPRESS)
    parser.add_argument("--run-dir", default=argparse.SUPPRESS)
    parser.add_argument("--resume-from", default=argparse.SUPPRESS)
    return resolve_preset_args(parser.parse_args(argv), BBH_PRESETS, parser=parser)


def validate_task_args(args):
    task = BBH_TASKS[args.task]
    if args.curriculum_start_level < task.min_level:
        raise ValueError(f"--curriculum-start-level must be >= {task.min_level} for task {args.task}")
    if args.max_level < args.curriculum_start_level:
        raise ValueError("--max-level must be >= --curriculum-start-level")
    if not 0.0 <= args.curriculum_threshold <= 1.0:
        raise ValueError("--curriculum-threshold must be in [0, 1]")
    if args.review_easier_every < 0:
        raise ValueError("--review-easier-every must be >= 0")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.train_steps < 1:
        raise ValueError("--train-steps must be at least 1")
    if args.eval_batches < 1:
        raise ValueError("--eval-batches must be at least 1")
    if args.block_size is not None and args.block_size < 8:
        raise ValueError("--block-size must be at least 8")
    task.build_vocab(args)
    required_block_size = task.required_block_size(args, args.max_level)
    if args.block_size is not None and args.block_size < required_block_size:
        raise ValueError(
            f"--block-size ({args.block_size}) must be >= required block size "
            f"({required_block_size}) for task {args.task} at level {args.max_level}"
        )


def build_training_objects(args):
    task = BBH_TASKS[args.task]
    required_block_size = task.required_block_size(args, args.max_level)
    block_size = args.block_size or required_block_size
    vocab, stoi, _ = task.build_vocab(args)
    model, optimizer = build_model_and_optimizer(args, vocab_size=len(vocab), block_size=block_size)
    return task, block_size, vocab, stoi, model, optimizer


def build_task_batch(args, task: BBHTask, stoi, level: int, rng: random.Random):
    return task.build_batch(args, args.batch_size, level, stoi, args.device, rng)


def build_eval_batches(args, task: BBHTask, stoi, level: int, rng: random.Random):
    return [build_task_batch(args, task, stoi, level, rng) for _ in range(args.eval_batches)]


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


def print_run_header(args, task: BBHTask, block_size: int, vocab_size: int, model):
    if getattr(args, "preset", None):
        print(f"preset: {args.preset}")
    print(f"device: {args.device}")
    print("task_family: symbolic_tasks")
    print("training_mode: answer_curriculum")
    print(f"task: {args.task}")
    for arg_name in task.shape_arg_names:
        print(f"{arg_name}: {getattr(args, arg_name)}")
    print(f"architecture: {args.architecture}")
    print(f"model_size: {args.model_size}")
    print(f"inference_mode: {effective_inference_mode(args)}")
    print(f"token_selection: {args.token_selection}")
    if args.resume_from:
        print(f"resume_from: {args.resume_from}")
    if args.run_dir:
        print(f"run_dir: {args.run_dir}")
    if args.architecture != "transformer":
        print(f"n_pass: {args.n_pass}")
        print(f"cache_source: {args.cache_source}")
        print(f"pass_loss_weights: {args.pass_loss_weights}")
        if args.architecture == "memory_tape":
            print(f"memory_tape_gate: {args.memory_tape_gate}")
        if args.architecture == "memory_update":
            print(f"memory_update_gate: {args.memory_update_gate}")
            if args.memory_update_gate == "on":
                print(f"memory_gate_bias: {args.memory_gate_bias}")
    print(f"curriculum_start_level: {args.curriculum_start_level}")
    print(f"curriculum_threshold: {args.curriculum_threshold}")
    print(f"review_easier_every: {args.review_easier_every}")
    print(f"max_level: {args.max_level}")
    print(f"block_size: {block_size}")
    print(f"vocab_size: {vocab_size}")
    print("number of parameters: %.2fM" % (model.get_num_params() / 1e6,))


def run_answer_curriculum(args):
    resume_checkpoint = None
    resume_step = 0
    if args.resume_from:
        resume_artifacts = resolve_resume_artifacts(args.resume_from)
        resume_checkpoint = load_checkpoint_payload(resume_artifacts.checkpoint_path, device="cpu")
        resume_step = int(resume_checkpoint.get("step", 0))

    resolve_device_arg(args)
    set_seed(args.seed)
    validate_model_args(args)
    validate_task_args(args)

    task, block_size, vocab, stoi, model, optimizer = build_training_objects(args)
    artifacts = prepare_run_artifacts(
        args,
        model=model,
        default_root_parts=("bbh", args.task, args.architecture),
        extra_config={"script": "experiments.train_bbh"},
    )
    if resume_checkpoint is not None:
        restore_checkpoint_state(resume_checkpoint, model=model, optimizer=optimizer, device=args.device)

    train_rng, eval_rng, _ = build_run_rngs(args.seed)
    current_level = args.curriculum_start_level
    promotion_history: list[tuple[int, int, float]] = []
    if resume_checkpoint is not None:
        extra_state = resume_checkpoint.get("extra_state", {})
        if "current_level" in extra_state:
            current_level = int(extra_state["current_level"])
        if "promotion_history" in extra_state:
            promotion_history = [tuple(item) for item in extra_state["promotion_history"]]
        train_rng_state = extra_state.get("train_rng_state")
        eval_rng_state = extra_state.get("eval_rng_state")
        if train_rng_state is not None:
            train_rng.setstate(train_rng_state)
        if eval_rng_state is not None:
            eval_rng.setstate(eval_rng_state)
    initial_gate_stats = memory_gate_stats(model)

    print_run_header(args, task, block_size, len(vocab), model)
    if resume_checkpoint is not None:
        print(f"resume_step: {resume_step}")
    if initial_gate_stats is not None:
        print(f"memory_gates_init: {format_memory_gate_stats(initial_gate_stats)}")

    append_jsonl(
        artifacts.metrics_path,
        {
            "event": "run_start",
            "task_family": "symbolic_tasks",
            "training_mode": "answer_curriculum",
            "task": args.task,
            "architecture": args.architecture,
            "config": vars(args),
            "block_size": block_size,
            "vocab_size": len(vocab),
            "memory_gate_stats": initial_gate_stats,
        },
    )
    if resume_checkpoint is not None:
        append_jsonl(
            artifacts.metrics_path,
            {
                "event": "run_resume",
                "step": resume_step,
                "resume_from": str(args.resume_from),
                "current_level": current_level,
            },
        )

    synchronize_device(args.device)
    train_window_start = time.perf_counter()
    train_window_steps = 0
    train_window_sequences = 0
    train_window_tokens = 0

    start_step = resume_step + 1 if resume_checkpoint is not None else 1
    final_step = resume_step + args.train_steps if resume_checkpoint is not None else args.train_steps

    for step in range(start_step, final_step + 1):
        model.train()
        sampled_level = choose_curriculum_train_level(
            step,
            current_level,
            task.min_level,
            args.curriculum_start_level,
            args.review_easier_every,
            train_rng,
        )

        batch = build_task_batch(args, task, stoi, sampled_level, train_rng)
        optimizer.zero_grad(set_to_none=True)
        loss, _, pass_losses = forward_and_loss(model, batch, args)
        loss.backward()
        optimizer.step()
        train_window_steps += 1
        train_window_sequences += int(batch.idx.size(0))
        train_window_tokens += int(batch.idx.numel())

        should_log = step == 1 or step % args.eval_interval == 0 or step == final_step
        if not should_log:
            continue

        synchronize_device(args.device)
        elapsed_s = time.perf_counter() - train_window_start
        train_timing = None
        if train_window_steps > 0 and elapsed_s > 0.0:
            train_timing = {
                "time_s": elapsed_s,
                "step_ms": 1000.0 * elapsed_s / float(train_window_steps),
                "seq_per_s": train_window_sequences / elapsed_s,
                "tok_per_s": train_window_tokens / elapsed_s,
            }

        train_fields = [f"loss {float(loss.detach().item()):.4f}"]
        if train_timing is not None:
            train_fields.extend([f"{train_timing['time_s']:.2f}s", f"tok/s {train_timing['tok_per_s']:.1f}"])
        train_fields.append(f"level {current_level:3d}")
        if args.architecture != "transformer":
            train_fields.append(f"pass_losses {format_pass_losses(pass_losses)}")
        print(format_checkpoint_line(f"step {step:5d}", train_fields))
        maybe_report_memory_gates(model, artifacts, step)

        train_payload = {
            "event": "train_step",
            "step": step,
            "train_loss": float(loss.detach().item()),
            "last_pass_loss": float(pass_losses[-1].detach().item()),
            "pass_losses": [float(pass_loss.detach().item()) for pass_loss in pass_losses],
            "train_level": current_level,
            "sampled_train_level": sampled_level,
            "curriculum_level": current_level,
        }
        if train_timing is not None:
            train_payload.update(train_timing)
        append_jsonl(artifacts.metrics_path, train_payload)

        current_eval_batches = build_eval_batches(args, task, stoi, current_level, eval_rng)
        current_metrics = evaluate_prebuilt_batches(
            model,
            args,
            current_eval_batches,
            generation_metrics_fn=basic_generation_metrics,
            inference_mode=args.inference_mode,
        )
        eval_fields = [f"loss {float(current_metrics['loss']):.4f}"]
        if "eval_time_s" in current_metrics:
            eval_fields.extend(
                [f"{float(current_metrics['eval_time_s']):.2f}s", f"tok/s {float(current_metrics['eval_output_tok_per_s']):.1f}"]
            )
        eval_fields.append(f"level {current_level:3d}")
        eval_fields.append(format_default_eval_metrics(current_metrics))
        print(format_checkpoint_line("eval", eval_fields))
        append_jsonl(
            artifacts.metrics_path,
            {
                "event": "eval",
                "step": step,
                "level": current_level,
                "requested_inference_mode": args.inference_mode,
                "effective_inference_mode": effective_inference_mode(args),
                "token_selection": args.token_selection,
                "metrics": current_metrics,
            },
        )

        if current_level > args.curriculum_start_level:
            easier_level = choose_easier_eval_level(current_level, task.min_level, eval_rng)
            easier_eval_batches = build_eval_batches(args, task, stoi, easier_level, eval_rng)
            easier_metrics = evaluate_prebuilt_batches(
                model,
                args,
                easier_eval_batches,
                generation_metrics_fn=basic_generation_metrics,
                inference_mode=args.inference_mode,
            )
            easier_fields = [f"loss {float(easier_metrics['loss']):.4f}"]
            if "eval_time_s" in easier_metrics:
                easier_fields.extend(
                    [f"{float(easier_metrics['eval_time_s']):.2f}s", f"tok/s {float(easier_metrics['eval_output_tok_per_s']):.1f}"]
                )
            easier_fields.append(f"level {easier_level:3d}")
            easier_fields.append(format_default_eval_metrics(easier_metrics))
            print(format_checkpoint_line("eval_easier", easier_fields))
            append_jsonl(
                artifacts.metrics_path,
                {
                    "event": "eval_easier",
                    "step": step,
                    "level": easier_level,
                    "requested_inference_mode": args.inference_mode,
                    "effective_inference_mode": effective_inference_mode(args),
                    "token_selection": args.token_selection,
                    "metrics": easier_metrics,
                },
            )

        metric_value = float(current_metrics["exact_match"])
        if metric_value >= args.curriculum_threshold and current_level < args.max_level:
            promotion_history.append((current_level, step, metric_value))
            current_level += 1
            print(f"  curriculum_promote -> level {current_level:3d} | seq_acc {metric_value:.3f}")
            append_jsonl(
                artifacts.metrics_path,
                {
                    "event": "curriculum_promote",
                    "step": step,
                    "solved_level": current_level - 1,
                    "next_level": current_level,
                    "seq_acc": metric_value,
                },
            )

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
                "eval_rng_state": eval_rng.getstate(),
            },
        )
        if step < final_step:
            print()

        synchronize_device(args.device)
        train_window_start = time.perf_counter()
        train_window_steps = 0
        train_window_sequences = 0
        train_window_tokens = 0

    append_jsonl(
        artifacts.metrics_path,
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
