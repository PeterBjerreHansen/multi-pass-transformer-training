import argparse
import time

from tasks.trace import othello, random_graph_walk
from experiments.common import (
    append_jsonl,
    build_model_and_optimizer,
    build_run_rngs,
    effective_inference_mode,
    evaluate_prebuilt_batches,
    format_checkpoint_line,
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
from experiments.presets import TRACE_PRESETS, preset_help_text, resolve_preset_args


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Train trace tasks from named experiment presets.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--preset",
        choices=sorted(TRACE_PRESETS),
        default=argparse.SUPPRESS,
        help=f"Named trace experiment preset. {preset_help_text(TRACE_PRESETS)}",
    )
    parser.add_argument(
        "--architecture",
        choices=["transformer", "memory_tape", "memory_concat", "memory_update"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--token-selection", choices=["sample", "argmax"], default=argparse.SUPPRESS)
    parser.add_argument("--cache-source", choices=["penultimate", "last"], default=argparse.SUPPRESS)
    parser.add_argument("--memory-tape-gate", default=argparse.SUPPRESS)
    parser.add_argument("--memory-update-gate", choices=["on", "off"], default=argparse.SUPPRESS)
    parser.add_argument("--memory-gate-bias", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--block-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--othello-data-dir", default=argparse.SUPPRESS)
    parser.add_argument("--othello-train-games", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--othello-val-games", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--othello-dataset-seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--othello-prepend-opening", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--train-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--weight-decay", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--eval-interval", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--eval-batches", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--device", default=argparse.SUPPRESS)
    parser.add_argument("--run-dir", default=argparse.SUPPRESS)
    parser.add_argument("--resume-from", default=argparse.SUPPRESS)
    raw_args = parser.parse_args(argv)
    return resolve_preset_args(raw_args, TRACE_PRESETS, parser=parser)


def validate_task_args(args):
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.train_steps < 1:
        raise ValueError("--train-steps must be at least 1")
    if args.eval_batches < 1:
        raise ValueError("--eval-batches must be at least 1")
    if args.block_size is not None and args.block_size < 8:
        raise ValueError("--block-size must be at least 8")

    if args.task == "random_graph_walk":
        if args.max_level < 1:
            raise ValueError("--max-level must be at least 1 for random_graph_walk")
        random_graph_walk.build_random_graph_walk_vocab(args.num_states, args.label_pool_size)
        required_block_size = random_graph_walk.required_block_size(args.num_states, args.label_pool_size, args.max_level)
    elif args.task == "othello":
        othello.build_othello_vocab(
            othello_train_games=args.othello_train_games,
            othello_val_games=args.othello_val_games,
        )
        required_block_size = othello.required_block_size(
            othello_prepend_opening=args.othello_prepend_opening,
            othello_train_games=args.othello_train_games,
            othello_val_games=args.othello_val_games,
        )
    else:
        raise ValueError(f"Unsupported trace task: {args.task}")

    if args.block_size is not None and args.block_size < required_block_size:
        raise ValueError(
            f"--block-size ({args.block_size}) must be >= required block size "
            f"({required_block_size}) for task {args.task}"
        )


def build_training_objects(args):
    if args.task == "random_graph_walk":
        block_size = args.block_size or random_graph_walk.required_block_size(
            args.num_states,
            args.label_pool_size,
            args.max_level,
        )
        vocab, stoi, _ = random_graph_walk.build_random_graph_walk_vocab(args.num_states, args.label_pool_size)
    elif args.task == "othello":
        block_size = args.block_size or othello.required_block_size(
            othello_prepend_opening=args.othello_prepend_opening,
            othello_train_games=args.othello_train_games,
            othello_val_games=args.othello_val_games,
        )
        vocab, stoi, _ = othello.build_othello_vocab(
            othello_train_games=args.othello_train_games,
            othello_val_games=args.othello_val_games,
        )
    else:
        raise ValueError(f"Unsupported trace task: {args.task}")

    model, optimizer = build_model_and_optimizer(args, vocab_size=len(vocab), block_size=block_size)
    return block_size, vocab, stoi, model, optimizer


def build_task_batch(args, stoi, rng, *, split: str):
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
    raise ValueError(f"Unsupported trace task: {args.task}")


def build_eval_batches(args, stoi, rng):
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
    raise ValueError(f"Unsupported trace task: {args.task}")


def format_eval_metrics(args, metrics: dict[str, float]) -> str:
    if args.task == "random_graph_walk":
        return random_graph_walk.format_random_graph_walk_eval_metrics(metrics)
    if args.task == "othello":
        return othello.format_othello_eval_metrics(metrics)
    raise ValueError(f"Unsupported trace task: {args.task}")


def print_run_header(args, block_size: int, vocab_size: int, model):
    if getattr(args, "preset", None):
        print(f"preset: {args.preset}")
    print(f"device: {args.device}")
    print("task_family: symbolic_tasks")
    print("training_mode: trace_fixed")
    print(f"task: {args.task}")
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
    if args.task == "random_graph_walk":
        print(f"num_states: {args.num_states}")
        print(f"label_pool_size: {args.label_pool_size}")
        print(f"level: {args.max_level}")
    else:
        print(f"othello_data_dir: {args.othello_data_dir}")
        print(f"othello_train_games: {args.othello_train_games}")
        print(f"othello_val_games: {args.othello_val_games}")
        print(f"othello_dataset_seed: {args.othello_dataset_seed}")
        print(f"othello_prepend_opening: {args.othello_prepend_opening}")
    print(f"block_size: {block_size}")
    print(f"vocab_size: {vocab_size}")
    print("number of parameters: %.2fM" % (model.get_num_params() / 1e6,))


def run_trace_training(args):
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

    block_size, vocab, stoi, model, optimizer = build_training_objects(args)
    artifacts = prepare_run_artifacts(
        args,
        model=model,
        default_root_parts=("trace", args.task, args.architecture),
        extra_config={"script": "experiments.train_trace"},
    )
    if resume_checkpoint is not None:
        restore_checkpoint_state(resume_checkpoint, model=model, optimizer=optimizer, device=args.device)

    train_rng, eval_rng, initial_log_rng = build_run_rngs(args.seed)
    if resume_checkpoint is not None:
        extra_state = resume_checkpoint.get("extra_state", {})
        train_rng_state = extra_state.get("train_rng_state")
        eval_rng_state = extra_state.get("eval_rng_state")
        if train_rng_state is not None:
            train_rng.setstate(train_rng_state)
        if eval_rng_state is not None:
            eval_rng.setstate(eval_rng_state)
    initial_gate_stats = memory_gate_stats(model)

    print_run_header(args, block_size, len(vocab), model)
    if resume_checkpoint is not None:
        print(f"resume_step: {resume_step}")
    if initial_gate_stats is not None:
        print(f"memory_gates_init: {format_memory_gate_stats(initial_gate_stats)}")

    append_jsonl(
        artifacts.metrics_path,
        {
            "event": "run_start",
            "task_family": "symbolic_tasks",
            "training_mode": "trace_fixed",
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
            },
        )

    synchronize_device(args.device)
    train_window_start = time.perf_counter()
    train_window_steps = 0
    train_window_sequences = 0
    train_window_tokens = 0
    final_step = resume_step + args.train_steps if resume_checkpoint is not None else args.train_steps

    def log_eval_checkpoint(
        step: int,
        *,
        log_loss: float,
        log_pass_losses,
        train_timing: dict[str, float] | None = None,
    ):
        train_fields = [f"loss {log_loss:.4f}"]
        if train_timing is None:
            train_fields.extend(["0.00s", "tok/s 0.0"])
        else:
            train_fields.extend([f"{train_timing['time_s']:.2f}s", f"tok/s {train_timing['tok_per_s']:.1f}"])
        if args.task == "random_graph_walk":
            train_fields.append(f"level {args.max_level:3d}")
        if args.architecture != "transformer":
            train_fields.append(f"pass_losses {format_pass_losses(log_pass_losses)}")
        print(format_checkpoint_line(f"step {step:5d}", train_fields))
        maybe_report_memory_gates(model, artifacts, step)

        train_payload = {
            "event": "train_step",
            "step": step,
            "train_loss": log_loss,
            "last_pass_loss": float(log_pass_losses[-1].detach().item()),
            "pass_losses": [float(loss.detach().item()) for loss in log_pass_losses],
        }
        if args.task == "random_graph_walk":
            train_payload["level"] = args.max_level
        if train_timing is not None:
            train_payload.update(train_timing)
        append_jsonl(artifacts.metrics_path, train_payload)

        eval_batches = build_eval_batches(args, stoi, eval_rng)
        metrics = evaluate_prebuilt_batches(
            model,
            args,
            eval_batches,
            generation_metrics_fn=trace_generation_metrics,
            inference_mode=args.inference_mode,
        )
        eval_fields = [f"loss {float(metrics['loss']):.4f}"]
        if "eval_time_s" in metrics:
            eval_fields.extend([f"{float(metrics['eval_time_s']):.2f}s", f"tok/s {float(metrics['eval_output_tok_per_s']):.1f}"])
        if args.task == "random_graph_walk":
            eval_fields.append(f"level {args.max_level:3d}")
        eval_fields.append(format_eval_metrics(args, metrics))
        print(format_checkpoint_line("eval", eval_fields))
        append_jsonl(
            artifacts.metrics_path,
            {
                "event": "eval",
                "step": step,
                "requested_inference_mode": args.inference_mode,
                "effective_inference_mode": effective_inference_mode(args),
                "token_selection": args.token_selection,
                "metrics": metrics,
                "level": args.max_level if args.task == "random_graph_walk" else None,
            },
        )
        save_latest_checkpoint(
            artifacts,
            model=model,
            optimizer=optimizer,
            args=args,
            step=step,
            extra_state={
                "train_rng_state": train_rng.getstate(),
                "eval_rng_state": eval_rng.getstate(),
            },
        )
        if step < final_step:
            print()

    start_step = 1
    if resume_checkpoint is None:
        initial_batch = build_task_batch(args, stoi, initial_log_rng, split="train")
        initial_loss, _, initial_pass_losses = forward_and_loss(model, initial_batch, args)
        log_eval_checkpoint(
            0,
            log_loss=float(initial_loss.detach().item()),
            log_pass_losses=initial_pass_losses,
        )
        print()
    else:
        start_step = resume_step + 1

    for step in range(start_step, final_step + 1):
        model.train()
        batch = build_task_batch(args, stoi, train_rng, split="train")
        optimizer.zero_grad(set_to_none=True)
        loss, _, pass_losses = forward_and_loss(model, batch, args)
        loss.backward()
        optimizer.step()
        train_window_steps += 1
        train_window_sequences += int(batch.idx.size(0))
        train_window_tokens += int(batch.idx.numel())

        should_log = step % args.eval_interval == 0 or step == final_step
        if should_log:
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
            log_eval_checkpoint(
                step,
                log_loss=float(loss.detach().item()),
                log_pass_losses=pass_losses,
                train_timing=train_timing,
            )
            synchronize_device(args.device)
            train_window_start = time.perf_counter()
            train_window_steps = 0
            train_window_sequences = 0
            train_window_tokens = 0

    append_jsonl(
        artifacts.metrics_path,
        {
            "event": "run_end",
            "task": args.task,
            "level": args.max_level if args.task == "random_graph_walk" else None,
        },
    )


def main(argv: list[str] | None = None):
    run_trace_training(parse_args(argv))


if __name__ == "__main__":
    main()
