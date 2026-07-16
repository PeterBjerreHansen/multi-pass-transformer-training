from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from experiments.common import (
    append_jsonl,
    effective_inference_mode,
    evaluate_prebuilt_batches,
    load_checkpoint_payload,
    saved_args_from_run,
    set_seed,
    stable_seed,
    write_json,
)
from experiments.train_trace import (
    TRACE_TASKS,
    build_fixed_eval_batches,
    build_training_objects,
    trace_generation_metrics,
    validate_task_args,
)
from experiments.common import resolve_device_arg, restore_checkpoint_state, validate_model_args, validate_training_args


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Evaluate free-generation drift for a saved trace run.",
        allow_abbrev=False,
    )
    parser.add_argument("--input-run-dir", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--token-selection", choices=["sample", "argmax"], default="argmax")
    parser.add_argument("--inference-mode", choices=["recompute", "append_recurrent"], required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval-position-offset", type=int, default=None)
    return parser.parse_args(argv)


def _load_eval_args(cli_args) -> tuple[SimpleNamespace, Path]:
    input_dir = Path(cli_args.input_run_dir).resolve()
    saved = saved_args_from_run(input_dir)
    if cli_args.device is not None:
        saved["device"] = cli_args.device
    if cli_args.eval_batches is not None:
        saved["eval_batches"] = cli_args.eval_batches
    saved["token_selection"] = cli_args.token_selection
    saved["inference_mode"] = cli_args.inference_mode
    saved["seed"] = cli_args.seed
    if cli_args.eval_position_offset is not None:
        saved["eval_position_offset"] = cli_args.eval_position_offset
    saved["run_dir"] = str(input_dir)
    saved["resume_from"] = str(input_dir)
    args = SimpleNamespace(**saved)
    resolve_device_arg(args)
    validate_model_args(args)
    validate_training_args(args)
    validate_task_args(args)
    return args, input_dir


def _default_output_dir(cli_args, args, input_dir: Path) -> Path:
    if cli_args.run_dir:
        return Path(cli_args.run_dir).resolve()
    name = f"{args.architecture}_{cli_args.inference_mode}_{cli_args.token_selection}"
    return Path("results", "drift", args.task, name, input_dir.name).resolve()


def _legality_prefix(args, prompt_tokens: list[int], generated_tokens: list[int]) -> tuple[int, bool]:
    return TRACE_TASKS[args.task].legality_prefix(args, prompt_tokens, generated_tokens)


def _valid_target_mask(args, target_tokens: list[int]) -> list[bool]:
    return TRACE_TASKS[args.task].valid_target_mask(args, target_tokens)


def collect_per_position_metrics(model, args, batches, *, inference_mode: str) -> dict[int, dict[str, float]]:
    do_sample = args.token_selection == "sample"
    legal_counts: dict[int, int] = {}
    totals: dict[int, int] = {}

    # Use a fixed global sampling stream while preserving the caller's RNG state.
    from experiments.common import isolated_torch_rng

    with isolated_torch_rng(stable_seed(args.seed, "drift", args.task, "paired_generation")):
        for batch in batches:
            for row in range(batch.idx.shape[0]):
                prompt_len = int(batch.prompt_lengths[row].item())
                output_len = int(batch.output_lengths[row].item())
                trace_len = output_len - 1
                prompt = batch.idx[row : row + 1, :prompt_len]
                target_trace = batch.targets[
                    row,
                    prompt_len - 1 : prompt_len - 1 + trace_len,
                ].tolist()
                generated = model.generate(
                    prompt,
                    max_new_tokens=output_len,
                    do_sample=do_sample,
                    inference_mode=inference_mode,
                    position_offset=args.eval_position_offset,
                )
                generated_trace = generated[0, prompt_len : prompt_len + trace_len].tolist()
                prompt_tokens = batch.idx[row, 1 : prompt_len - 1].tolist()
                legal_prefix_len, _all_legal = _legality_prefix(args, prompt_tokens, generated_trace)
                valid = _valid_target_mask(args, target_trace)
                for position, is_valid in enumerate(valid):
                    if not is_valid:
                        continue
                    totals[position] = totals.get(position, 0) + 1
                    legal_counts[position] = legal_counts.get(position, 0) + int(
                        legal_prefix_len >= position + 1
                    )

    return {
        position: {
            "count": float(totals[position]),
            "token_legality": legal_counts[position] / totals[position],
        }
        for position in sorted(totals)
    }


def evaluate_run(cli_args) -> Path:
    args, input_dir = _load_eval_args(cli_args)
    output_dir = _default_output_dir(cli_args, args, input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    per_position_path = output_dir / "per_position.jsonl"
    if per_position_path.exists():
        per_position_path.unlink()

    checkpoint = load_checkpoint_payload(input_dir / "latest.pt", device="cpu")
    block_size, vocab, stoi, _itos, model, _optimizer = build_training_objects(args)
    restore_checkpoint_state(checkpoint, model=model, optimizer=None, device=args.device)
    batches = build_fixed_eval_batches(args, stoi)

    set_seed(cli_args.seed)
    metrics = evaluate_prebuilt_batches(
        model,
        args,
        batches,
        generation_metrics_fn=trace_generation_metrics,
        inference_mode=cli_args.inference_mode,
        generation_seed=stable_seed(args.seed, "drift", args.task, "paired_generation"),
    )
    per_position = collect_per_position_metrics(
        model,
        args,
        batches,
        inference_mode=effective_inference_mode(args, cli_args.inference_mode),
    )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_run_dir": str(input_dir),
        "task": args.task,
        "architecture": args.architecture,
        "inference_mode": cli_args.inference_mode,
        "effective_inference_mode": effective_inference_mode(args, cli_args.inference_mode),
        "token_selection": args.token_selection,
        "block_size": block_size,
        "eval_position_offset": args.eval_position_offset,
        "vocab_size": len(vocab),
        "metrics": metrics,
    }
    write_json(summary_path, summary)
    for position, values in per_position.items():
        append_jsonl(
            per_position_path,
            {
                "position": position,
                "count": int(values["count"]),
                "token_legality": values["token_legality"],
            },
        )

    print(
        f"{input_dir.name}: {args.task} | {args.architecture} | {cli_args.inference_mode} | "
        f"token_legality {metrics['token_legality']:.3f} | "
        f"sequence_legality {metrics['sequence_legality']:.3f}"
    )
    print(f"output_dir: {output_dir}")
    return output_dir


def main(argv: list[str] | None = None) -> None:
    evaluate_run(parse_args(argv))


if __name__ == "__main__":
    main()
