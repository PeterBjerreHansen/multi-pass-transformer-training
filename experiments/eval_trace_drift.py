import argparse
from datetime import datetime
from pathlib import Path
import random
from types import SimpleNamespace

from tasks.trace import othello, random_graph_walk
from experiments.common import (
    append_jsonl,
    effective_inference_mode,
    evaluate_prebuilt_batches,
    load_checkpoint_payload,
    load_json_if_exists,
    restore_checkpoint_state,
    set_seed,
    validate_model_args,
    write_json,
)
from experiments.train_trace import (
    build_eval_batches,
    build_training_objects,
    trace_generation_metrics,
    validate_task_args,
)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Evaluate one saved trace run under one inference configuration.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input-run-dir",
        required=True,
        help="Trace training run directory containing config.json and latest.pt.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional output directory for config.json, summary.jsonl, and per_position.jsonl.",
    )
    parser.add_argument("--device", default=None, help="Optional override for the saved training device.")
    parser.add_argument("--eval-batches", type=int, default=None, help="Optional override for validation batches.")
    parser.add_argument("--token-selection", choices=["sample", "argmax"], default="sample")
    parser.add_argument("--inference-mode", choices=["recompute", "final_pass"], required=True)
    parser.add_argument(
        "--cache-source",
        choices=["penultimate", "last"],
        default=None,
        help="Optional override for the saved final-pass cache source.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args(argv)


def _saved_args_for_run(input_run_dir: str) -> tuple[dict[str, object], Path]:
    run_dir = Path(input_run_dir).resolve()
    config_payload = load_json_if_exists(run_dir / "config.json")
    if config_payload is None or "args" not in config_payload:
        raise FileNotFoundError(f"Missing config.json with saved args in {run_dir}")
    return dict(config_payload["args"]), run_dir


def _default_run_dir(args) -> Path:
    saved_args, input_run_dir = _saved_args_for_run(args.input_run_dir)
    task = str(saved_args.get("task", "unknown"))
    architecture = str(saved_args.get("architecture", "unknown"))
    cache_source = str(args.cache_source or saved_args.get("cache_source", "penultimate"))
    variant = f"{architecture}_{cache_source}_{args.inference_mode}"
    return Path("results").joinpath("drift", task, variant, input_run_dir.name).resolve()


def _prepare_eval_dir(args) -> tuple[Path, Path, Path]:
    run_dir = Path(args.run_dir).resolve() if args.run_dir else _default_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.jsonl"
    per_position_path = run_dir / "per_position.jsonl"
    for path in (summary_path, per_position_path):
        if path.exists():
            path.unlink()
    write_json(
        config_path,
        {
            "created_at": datetime.now().isoformat(),
            "args": vars(args),
            "input_run_dir": str(Path(args.input_run_dir).resolve()),
        },
    )
    return run_dir, summary_path, per_position_path


def _load_eval_args(input_run_dir: str, cli_args) -> tuple[SimpleNamespace, Path]:
    saved_args, run_dir = _saved_args_for_run(input_run_dir)
    if cli_args.device is not None:
        saved_args["device"] = cli_args.device
    if cli_args.eval_batches is not None:
        saved_args["eval_batches"] = cli_args.eval_batches
    saved_args["token_selection"] = cli_args.token_selection
    if cli_args.cache_source is not None:
        saved_args["cache_source"] = cli_args.cache_source
    saved_args["run_dir"] = str(run_dir)
    saved_args["resume_from"] = str(run_dir)

    args = SimpleNamespace(**saved_args)
    validate_model_args(args)
    validate_task_args(args)
    return args, run_dir


def _trace_legality_prefix(args, prompt_tokens: list[int], generated_tokens: list[int]) -> tuple[int, bool]:
    if args.task == "random_graph_walk":
        return random_graph_walk.legal_prefix_length(
            prompt_tokens,
            generated_tokens,
            num_states=args.num_states,
            label_pool_size=args.label_pool_size,
        )
    if args.task == "othello":
        return othello.legal_prefix_length(generated_tokens)
    raise ValueError(f"Unsupported trace task: {args.task}")


def _valid_target_mask(args, target_tokens) -> list[bool]:
    if args.task == "othello":
        return [int(token) != 0 for token in target_tokens]
    return [True] * len(target_tokens)


def collect_per_position_metrics(model, args, batches: list[object], *, requested_inference_mode: str):
    effective_mode = effective_inference_mode(args, requested_inference_mode)
    do_sample = args.token_selection == "sample"
    position_token_legality: dict[int, float] = {}
    position_counts: dict[int, int] = {}

    for batch in batches:
        for row in range(batch.idx.size(0)):
            prompt_len = int(batch.prompt_lengths[row].item())
            output_len = int(batch.output_lengths[row].item())
            answer_len = output_len - 1
            prompt = batch.idx[row:row + 1, :prompt_len]
            target_suffix = batch.targets[row, prompt_len - 1:prompt_len - 1 + answer_len].tolist()
            generated = model.generate(
                prompt,
                max_new_tokens=output_len,
                do_sample=do_sample,
                inference_mode=effective_mode,
                cache_source=getattr(args, "cache_source", "penultimate"),
            )
            generated_suffix = generated[0, prompt_len:prompt_len + answer_len].tolist()
            prompt_tokens = batch.idx[row, 1:prompt_len - 1].tolist()
            legal_prefix_len, _ = _trace_legality_prefix(args, prompt_tokens, generated_suffix)
            valid_mask = _valid_target_mask(args, target_suffix)

            for position, is_valid in enumerate(valid_mask):
                if not is_valid:
                    continue
                position_counts[position] = position_counts.get(position, 0) + 1
                position_token_legality[position] = position_token_legality.get(position, 0.0) + float(
                    legal_prefix_len >= position + 1
                )

    return {
        position: {
            "count": position_counts[position],
            "token_legality": position_token_legality[position] / position_counts[position],
        }
        for position in sorted(position_counts)
    }


def evaluate_run(cli_args, summary_path: Path, per_position_path: Path):
    args, run_dir = _load_eval_args(cli_args.input_run_dir, cli_args)
    checkpoint = load_checkpoint_payload(run_dir / "latest.pt", device="cpu")
    block_size, vocab, stoi, model, optimizer = build_training_objects(args)
    restore_checkpoint_state(checkpoint, model=model, optimizer=optimizer, device=args.device)

    rng = random.Random(cli_args.seed + hash(str(run_dir)) % 10_000)
    batches = build_eval_batches(args, stoi, rng)

    metrics = evaluate_prebuilt_batches(
        model,
        args,
        batches,
        generation_metrics_fn=trace_generation_metrics,
        inference_mode=cli_args.inference_mode,
    )
    per_position_metrics = collect_per_position_metrics(
        model,
        args,
        batches,
        requested_inference_mode=cli_args.inference_mode,
    )
    effective_mode = effective_inference_mode(args, cli_args.inference_mode)

    summary_payload = {
        "event": "trace_drift_summary",
        "input_run_dir": str(run_dir),
        "task": args.task,
        "architecture": args.architecture,
        "cache_source": getattr(args, "cache_source", "penultimate"),
        "inference_mode": cli_args.inference_mode,
        "effective_inference_mode": effective_mode,
        "token_selection": args.token_selection,
        "block_size": block_size,
        "vocab_size": len(vocab),
        "metrics": metrics,
    }
    append_jsonl(summary_path, summary_payload)

    for position, position_metrics in per_position_metrics.items():
        append_jsonl(
            per_position_path,
            {
                "event": "per_position",
                "input_run_dir": str(run_dir),
                "task": args.task,
                "architecture": args.architecture,
                "cache_source": getattr(args, "cache_source", "penultimate"),
                "inference_mode": cli_args.inference_mode,
                "effective_inference_mode": effective_mode,
                "position": position,
                "count": int(position_metrics["count"]),
                "token_legality": float(position_metrics["token_legality"]),
            },
        )

    print(
        f"{run_dir.name}: {args.task} | {args.architecture} | {cli_args.inference_mode} | "
        f"token_legality {metrics['token_legality']:.3f} | "
        f"sequence_legality {metrics.get('sequence_legality', 0.0):.3f}"
    )


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    set_seed(args.seed)
    run_dir, summary_path, per_position_path = _prepare_eval_dir(args)
    evaluate_run(args, summary_path, per_position_path)
    print(f"wrote {summary_path}")
    print(f"wrote {per_position_path}")
    print(f"run_dir: {run_dir}")


if __name__ == "__main__":
    main()
