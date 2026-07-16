from __future__ import annotations

import argparse
import random
import time

from tasks.trace import othello, random_graph_walk
from experiments.common import (
    append_jsonl,
    build_model_and_optimizer,
    effective_inference_mode,
    evaluate_prebuilt_batches,
    format_checkpoint_line,
    format_gradient_norms,
    format_memory_gate_stats,
    format_pass_losses,
    forward_and_loss,
    gradient_norms,
    load_checkpoint_payload,
    memory_gate_stats,
    model_benchmark_stats,
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
    update_gradient_norm_window,
    validate_model_args,
    validate_training_args,
)
from experiments.presets import TRACE_PRESETS, preset_help_text, resolve_preset_args
from model_factory import ARCHITECTURES


TRACE_TASKS = ("random_graph_walk", "othello")


def _add_override(parser, *names, **kwargs) -> None:
    parser.add_argument(*names, default=argparse.SUPPRESS, **kwargs)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train fixed-length trace tasks.", allow_abbrev=False)
    _add_override(parser, "--preset", choices=sorted(TRACE_PRESETS), help=preset_help_text(TRACE_PRESETS))
    _add_override(parser, "--task", choices=TRACE_TASKS)
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
    _add_override(parser, "--null-memory-slot", choices=["off", "on"])
    _add_override(parser, "--num-states", type=int)
    _add_override(parser, "--label-pool-size", type=int)
    _add_override(parser, "--max-level", type=int)
    _add_override(parser, "--othello-data-dir")
    _add_override(parser, "--othello-train-games", type=int)
    _add_override(parser, "--othello-val-games", type=int)
    _add_override(parser, "--othello-dataset-seed", type=int)
    _add_override(parser, "--othello-prepend-opening", action="store_true")
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
    return resolve_preset_args(
        parser.parse_args(argv),
        TRACE_PRESETS,
        default_preset="random_graph_walk_main",
        parser=parser,
    )


def validate_task_args(args) -> None:
    if args.task == "random_graph_walk":
        random_graph_walk.build_random_graph_walk_vocab(args.num_states, args.label_pool_size)
        required = random_graph_walk.required_block_size(args.num_states, args.label_pool_size, args.max_level)
    elif args.task == "othello":
        othello.build_othello_vocab(
            othello_train_games=args.othello_train_games,
            othello_val_games=args.othello_val_games,
        )
        required = othello.required_block_size(
            othello_prepend_opening=args.othello_prepend_opening,
            othello_train_games=args.othello_train_games,
            othello_val_games=args.othello_val_games,
        )
    else:
        raise ValueError(f"unsupported trace task: {args.task}")
    if args.block_size is not None and args.block_size < required:
        raise ValueError(f"--block-size must be at least {required} for {args.task}")


def build_training_objects(args):
    if args.task == "random_graph_walk":
        block_size = args.block_size or random_graph_walk.required_block_size(
            args.num_states,
            args.label_pool_size,
            args.max_level,
        )
        vocab, stoi, itos = random_graph_walk.build_random_graph_walk_vocab(
            args.num_states,
            args.label_pool_size,
        )
    elif args.task == "othello":
        block_size = args.block_size or othello.required_block_size(
            othello_prepend_opening=args.othello_prepend_opening,
            othello_train_games=args.othello_train_games,
            othello_val_games=args.othello_val_games,
        )
        vocab, stoi, itos = othello.build_othello_vocab(
            othello_train_games=args.othello_train_games,
            othello_val_games=args.othello_val_games,
        )
    else:
        raise ValueError(f"unsupported trace task: {args.task}")
    model, optimizer = build_model_and_optimizer(args, vocab_size=len(vocab), block_size=block_size)
    return block_size, vocab, stoi, itos, model, optimizer


def build_task_batch(args, stoi, rng: random.Random, *, split: str):
    if args.task == "random_graph_walk":
        return random_graph_walk.build_random_graph_walk_batch(
            batch_size=args.batch_size,
            num_states=args.num_states,
            label_pool_size=args.label_pool_size,
            num_steps=args.max_level,
            stoi=stoi,
            device=args.device,
            rng=rng,
        )
    if args.task == "othello":
        return othello.build_othello_batch(
            batch_size=args.batch_size,
            stoi=stoi,
            device=args.device,
            rng=rng,
            split=split,
            othello_data_dir=args.othello_data_dir,
            othello_train_games=args.othello_train_games,
            othello_val_games=args.othello_val_games,
            othello_dataset_seed=args.othello_dataset_seed,
            othello_prepend_opening=args.othello_prepend_opening,
        )
    raise ValueError(f"unsupported trace task: {args.task}")


def build_fixed_eval_batches(args, stoi):
    rng = random.Random(stable_seed(args.seed, "trace", args.task, "eval"))
    return [build_task_batch(args, stoi, rng, split="val") for _ in range(args.eval_batches)]


def trace_generation_metrics(model, batch, args, *, inference_mode: str | None = None):
    if args.task == "random_graph_walk":
        return random_graph_walk.random_graph_walk_generation_metrics(
            model,
            batch,
            args,
            inference_mode=inference_mode,
            num_states=args.num_states,
            label_pool_size=args.label_pool_size,
        )
    if args.task == "othello":
        return othello.othello_generation_metrics(
            model,
            batch,
            args,
            inference_mode=inference_mode,
        )
    raise ValueError(f"unsupported trace task: {args.task}")


def format_trace_metrics(args, metrics: dict[str, float]) -> str:
    if args.task == "random_graph_walk":
        return random_graph_walk.format_random_graph_walk_eval_metrics(metrics)
    return othello.format_othello_eval_metrics(metrics)


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


def run_trace_training(args) -> None:
    checkpoint = None
    resume_step = 0
    if args.resume_from:
        resume_artifacts = resolve_resume_artifacts(args.resume_from)
        checkpoint = load_checkpoint_payload(resume_artifacts.checkpoint_path, device="cpu")
        resume_step = int(checkpoint.get("step", 0))
        _apply_resume_args(args, checkpoint)

    resolve_device_arg(args)
    set_seed(args.seed)
    validate_model_args(args)
    validate_training_args(args)
    validate_task_args(args)
    block_size, vocab, stoi, _itos, model, optimizer = build_training_objects(args)
    artifacts = prepare_run_artifacts(
        args,
        model=model,
        default_root_parts=("trace", args.task, args.architecture),
        extra_config={"script": "experiments.train_trace"},
    )

    train_rng = random.Random(stable_seed(args.seed, "trace", args.task, "train"))
    if checkpoint is not None:
        restore_checkpoint_state(checkpoint, model=model, optimizer=optimizer, device=args.device)
        extra = checkpoint.get("extra_state", {})
        if "train_rng_state" in extra:
            train_rng.setstate(extra["train_rng_state"])

    print(f"device: {args.device}")
    print(f"task: {args.task}")
    print(f"architecture: {args.architecture}")
    print(f"inference_mode: {effective_inference_mode(args)}")
    print(f"block_size: {block_size}")
    print(f"parameters: {model.get_num_params():,}")
    if args.architecture != "transformer":
        total_weight = sum(args.pass_loss_weights)
        print(f"n_pass: {args.n_pass}")
        print(f"pass_loss_weights_normalized: {[weight / total_weight for weight in args.pass_loss_weights]}")
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

    fixed_eval_batches = build_fixed_eval_batches(args, stoi)
    start_step = resume_step + 1
    final_step = resume_step + args.train_steps if checkpoint is not None else args.train_steps
    window_start = time.perf_counter()
    window_tokens = 0
    gradient_norm_window: dict[str, dict[str, float]] = {}

    for step in range(start_step, final_step + 1):
        model.train()
        batch = build_task_batch(args, stoi, train_rng, split="train")
        optimizer.zero_grad(set_to_none=True)
        loss, _output, pass_losses = forward_and_loss(model, batch, args)
        loss.backward()
        update_gradient_norm_window(gradient_norm_window, gradient_norms(model))
        optimizer.step()
        window_tokens += int(batch.idx.numel())

        should_eval = step == 1 or step % args.eval_interval == 0 or step == final_step
        if not should_eval:
            continue

        synchronize_device(args.device)
        elapsed = time.perf_counter() - window_start
        tok_per_s = window_tokens / elapsed if elapsed > 0 else 0.0
        fields = [f"loss {loss.item():.4f}", f"tok/s {tok_per_s:.1f}"]
        gradient_summary = summarize_gradient_norm_window(gradient_norm_window)
        fields.append(format_gradient_norms(gradient_summary))
        if args.architecture != "transformer":
            fields.append(f"pass_losses {format_pass_losses(pass_losses)}")
        print(format_checkpoint_line(f"step {step}", fields))

        metrics = evaluate_prebuilt_batches(
            model,
            args,
            fixed_eval_batches,
            generation_metrics_fn=trace_generation_metrics,
            inference_mode=args.inference_mode,
            generation_seed=stable_seed(args.seed, "trace", args.task, "generation"),
        )
        print(
            format_checkpoint_line(
                "eval",
                [f"loss {metrics['loss']:.4f}", format_trace_metrics(args, metrics)],
            )
        )
        append_jsonl(
            artifacts.metrics_path,
            {
                "event": "eval",
                "step": step,
                "train_loss": float(loss.item()),
                "pass_losses": [float(item.item()) for item in pass_losses],
                "metrics": metrics,
                "gradient_norms": gradient_summary,
                "memory_gate_stats": memory_gate_stats(model),
                "train_tok_per_s": tok_per_s,
                "resource_stats": runtime_resource_stats(args.device),
            },
        )
        save_latest_checkpoint(
            artifacts,
            model=model,
            optimizer=optimizer,
            args=args,
            step=step,
            extra_state={"train_rng_state": train_rng.getstate()},
        )
        synchronize_device(args.device)
        window_start = time.perf_counter()
        window_tokens = 0
        gradient_norm_window = {}

    append_jsonl(artifacts.metrics_path, {"event": "run_end", "task": args.task})
    print(f"run_dir: {artifacts.run_dir}")


def main(argv: list[str] | None = None) -> None:
    run_trace_training(parse_args(argv))


if __name__ == "__main__":
    main()
