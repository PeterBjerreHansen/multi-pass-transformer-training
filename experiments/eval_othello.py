from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Iterable

import torch
import torch.nn.functional as F

from experiments.common import (
    append_jsonl,
    isolated_torch_rng,
    load_checkpoint_payload,
    resolve_device_arg,
    restore_checkpoint_state,
    saved_args_from_run,
    stable_seed,
    validate_model_args,
    validate_training_args,
    write_json,
)
from experiments.train_trace import build_training_objects, validate_task_args
from model_factory import is_multi_pass_architecture
from tasks.common import BOS_TOKEN, EOS_TOKEN, SEP_TOKEN
from tasks.trace import othello


EVALUATION_MODES = ("full-game", "random-prefix", "prefix-grid", "all")
INFERENCE_MODES = ("recompute", "append_recurrent")
DEFAULT_PREFIX_FRACTIONS = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class OthelloEvalExample:
    example_index: int
    protocol: str
    trace_move_ids: tuple[int, ...]
    cut: int

    @property
    def prefix_move_ids(self) -> tuple[int, ...]:
        return self.trace_move_ids[: self.cut]

    @property
    def suffix_move_ids(self) -> tuple[int, ...]:
        return self.trace_move_ids[self.cut :]


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Evaluate Othello continuation legality from deterministic prefix cuts.",
        allow_abbrev=False,
    )
    parser.add_argument("--input-run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument("--evaluation-mode", choices=EVALUATION_MODES, default="all")
    parser.add_argument(
        "--inference-modes",
        nargs="+",
        choices=INFERENCE_MODES,
        default=list(INFERENCE_MODES),
    )
    parser.add_argument(
        "--prefix-fractions",
        nargs="+",
        type=float,
        default=list(DEFAULT_PREFIX_FRACTIONS),
    )
    parser.add_argument("--token-selection", choices=["argmax", "sample"], default="argmax")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args(argv)


def validate_eval_args(args) -> None:
    if args.examples < 1:
        raise ValueError("--examples must be positive")
    if not args.prefix_fractions:
        raise ValueError("--prefix-fractions must not be empty")
    if any(not 0.0 < fraction < 1.0 for fraction in args.prefix_fractions):
        raise ValueError("--prefix-fractions values must be strictly between 0 and 1")
    if len(set(args.inference_modes)) != len(args.inference_modes):
        raise ValueError("--inference-modes must not contain duplicates")


def build_eval_examples(
    traces: Iterable[list[int]],
    *,
    stoi: dict[str, int],
    evaluation_mode: str,
    prefix_fractions: Iterable[float],
    rng: random.Random,
) -> list[OthelloEvalExample]:
    examples: list[OthelloEvalExample] = []
    fractions = tuple(prefix_fractions)
    for example_index, square_trace in enumerate(traces):
        move_ids = tuple(stoi[othello.move_token(square)] for square in square_trace)
        if not move_ids:
            raise ValueError("Othello evaluation trace must contain at least one move")

        cuts: list[tuple[str, int]] = []
        if evaluation_mode in {"full-game", "all"}:
            cuts.append(("full-game", 0))
        if evaluation_mode in {"random-prefix", "all"}:
            cuts.append(("random-prefix", rng.randint(1, len(move_ids) - 1)))
        if evaluation_mode in {"prefix-grid", "all"}:
            for fraction in fractions:
                cut = min(len(move_ids) - 1, max(1, round(len(move_ids) * fraction)))
                cuts.append((f"prefix-grid-{fraction:g}", cut))

        seen: set[tuple[str, int]] = set()
        for protocol, cut in cuts:
            key = (protocol, cut)
            if key in seen:
                continue
            seen.add(key)
            examples.append(
                OthelloEvalExample(
                    example_index=example_index,
                    protocol=protocol,
                    trace_move_ids=move_ids,
                    cut=cut,
                )
            )
    return examples


def legal_set_step_metrics(logits: torch.Tensor, legal_token_ids: tuple[int, ...], gold_token_id: int) -> dict:
    if logits.ndim != 1:
        raise ValueError("logits must have shape [vocab]")
    if not legal_token_ids:
        raise ValueError("legal_token_ids must not be empty at an active move position")
    if gold_token_id not in legal_token_ids:
        raise ValueError("gold Othello move is not in the legal set")
    legal_index = torch.tensor(legal_token_ids, dtype=torch.long, device=logits.device)
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    legal_log_mass = torch.logsumexp(log_probabilities.index_select(0, legal_index), dim=0)
    return {
        "legal_set_nll": float((-legal_log_mass).item()),
        "gold_move_nll": float((-log_probabilities[gold_token_id]).item()),
        "legal_probability_mass": float(legal_log_mass.exp().item()),
        "top1_legal": float(int(logits.argmax().item()) in legal_token_ids),
        "legal_set_size": float(len(legal_token_ids)),
    }


def _load_eval_args(cli_args) -> tuple[SimpleNamespace, Path]:
    run_dir = Path(cli_args.input_run_dir).resolve()
    saved = saved_args_from_run(run_dir)
    if saved.get("task") != "othello":
        raise ValueError("experiments.eval_othello requires an Othello checkpoint")
    if cli_args.device is not None:
        saved["device"] = cli_args.device
    saved["run_dir"] = str(run_dir)
    saved["resume_from"] = str(run_dir)
    args = SimpleNamespace(**saved)
    resolve_device_arg(args)
    validate_model_args(args)
    validate_training_args(args)
    validate_task_args(args)
    return args, run_dir


def _sample_validation_traces(args, *, count: int, rng: random.Random) -> list[list[int]]:
    dataset = othello.load_othello_dataset(
        split="val",
        othello_data_dir=args.othello_data_dir,
        othello_train_games=args.othello_train_games,
        othello_val_games=args.othello_val_games,
        othello_dataset_seed=args.othello_dataset_seed,
    )
    return [dataset.sample_trace(rng) for _ in range(count)]


def _serialized_prompt(args, stoi: dict[str, int], prefix_move_ids: tuple[int, ...]) -> list[int]:
    prompt = [stoi[BOS_TOKEN]]
    if args.othello_prepend_opening:
        prompt.extend(stoi[othello.move_token(square)] for square in othello.OPENING_PREFIX)
    prompt.append(stoi[SEP_TOKEN])
    prompt.extend(prefix_move_ids)
    return prompt


def _score_generated_continuation(
    prefix_move_ids: tuple[int, ...],
    generated_token_ids: list[int],
    *,
    eos_id: int,
    reference_suffix: tuple[int, ...],
) -> dict:
    eos_position = next(
        (position for position, token_id in enumerate(generated_token_ids) if token_id == eos_id),
        None,
    )
    attempted_moves = (
        generated_token_ids if eos_position is None else generated_token_ids[:eos_position]
    )
    accepted = list(prefix_move_ids)
    legal_prefix_length = 0
    terminal_reached = False
    for token_id in attempted_moves:
        legal_ids = othello.legal_move_token_ids_after_prefix(accepted)
        if not legal_ids or token_id not in legal_ids:
            break
        accepted.append(token_id)
        legal_prefix_length += 1
    else:
        terminal_reached = not othello.legal_move_token_ids_after_prefix(accepted)

    sequence_legality = float(
        eos_position is not None
        and legal_prefix_length == len(attempted_moves)
        and terminal_reached
    )
    denominator = len(attempted_moves)
    legal_move_fraction = (
        float(legal_prefix_length) / denominator
        if denominator
        else float(not reference_suffix and sequence_legality)
    )
    exact_tokens = [*reference_suffix, eos_id]
    exact_suffix = float(
        eos_position is not None
        and generated_token_ids[: eos_position + 1] == exact_tokens
    )
    return {
        "sequence_legality": sequence_legality,
        "exact_suffix": exact_suffix,
        "legal_move_fraction": legal_move_fraction,
        "legal_prefix_length": float(legal_prefix_length),
        "generated_move_count": float(len(attempted_moves)),
        "terminal_reached": float(terminal_reached),
        "eos_emitted": float(eos_position is not None),
    }


@torch.no_grad()
def _teacher_forced_metrics(
    model,
    args,
    stoi: dict[str, int],
    example: OthelloEvalExample,
    *,
    inference_mode: str,
    recompute_cache: dict[tuple[int, ...], torch.Tensor],
) -> dict:
    trace = example.trace_move_ids
    eos_id = stoi[EOS_TOKEN]
    prompt_tokens = _serialized_prompt(args, stoi, example.prefix_move_ids)
    step_metrics = []

    if inference_mode == "recompute":
        logits_by_position = recompute_cache.get(trace)
        if logits_by_position is None:
            full_input = _serialized_prompt(args, stoi, trace)
            tensor = torch.tensor([full_input], dtype=torch.long, device=args.device)
            logits_by_position = model(tensor).logits[0].detach()
            recompute_cache[trace] = logits_by_position
        base_length = len(_serialized_prompt(args, stoi, ()))
        for move_index in range(example.cut, len(trace)):
            legal_ids = othello.legal_move_token_ids_after_prefix(trace[:move_index])
            step_metrics.append(
                legal_set_step_metrics(
                    logits_by_position[base_length - 1 + move_index],
                    legal_ids,
                    trace[move_index],
                )
            )
        eos_logits = logits_by_position[base_length - 1 + len(trace)]
    elif inference_mode == "append_recurrent":
        state = model.prefill_recurrent(
            torch.tensor([prompt_tokens], dtype=torch.long, device=args.device)
        )
        for move_index in range(example.cut, len(trace)):
            legal_ids = othello.legal_move_token_ids_after_prefix(trace[:move_index])
            step_metrics.append(
                legal_set_step_metrics(
                    state.next_token_logits[0],
                    legal_ids,
                    trace[move_index],
                )
            )
            state = model.recurrent_step(
                state,
                torch.tensor([[trace[move_index]]], dtype=torch.long, device=args.device),
            )
        eos_logits = state.next_token_logits[0]
    else:
        raise ValueError(f"unsupported inference mode: {inference_mode}")

    if not step_metrics:
        raise ValueError("Othello prefix cut must leave at least one move to evaluate")
    move_count = len(step_metrics)
    result = {
        key: sum(item[key] for item in step_metrics) / move_count
        for key in step_metrics[0]
    }
    result["move_count"] = float(move_count)
    result["eos_nll"] = float(-F.log_softmax(eos_logits.float(), dim=-1)[eos_id].item())
    return result


@torch.no_grad()
def _free_generation_metrics(
    model,
    args,
    stoi: dict[str, int],
    example: OthelloEvalExample,
    *,
    inference_mode: str,
    do_sample: bool,
    generation_seed: int,
) -> dict:
    prompt_tokens = _serialized_prompt(args, stoi, example.prefix_move_ids)
    prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=args.device)
    # Othello can take at most 60 moves. Generate enough room for any legal
    # continuation from the cut plus EOS, not merely the sampled reference path.
    max_new_tokens = othello.MAX_MOVES - example.cut + 1
    with isolated_torch_rng(generation_seed):
        generated = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            inference_mode=inference_mode,
        )
    generated_tokens = generated[0, len(prompt_tokens) :].tolist()
    return _score_generated_continuation(
        example.prefix_move_ids,
        generated_tokens,
        eos_id=stoi[EOS_TOKEN],
        reference_suffix=example.suffix_move_ids,
    )


def _length_bucket(length: int) -> str:
    if length == 0:
        return "0"
    if length <= 15:
        return "1-15"
    if length <= 30:
        return "16-30"
    if length <= 45:
        return "31-45"
    return "46+"


def _summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0, "teacher_move_count": 0}
    free_fields = tuple(rows[0]["free_generation"])
    teacher_fields = tuple(
        field for field in rows[0]["teacher_forced"] if field != "move_count"
    )
    teacher_move_count = sum(int(row["teacher_forced"]["move_count"]) for row in rows)
    summary = {
        "count": len(rows),
        "teacher_move_count": teacher_move_count,
        "free_generation": {
            field: sum(float(row["free_generation"][field]) for row in rows) / len(rows)
            for field in free_fields
        },
        "teacher_forced": {},
    }
    for field in teacher_fields:
        if field == "eos_nll":
            summary["teacher_forced"][field] = (
                sum(float(row["teacher_forced"][field]) for row in rows) / len(rows)
            )
        else:
            summary["teacher_forced"][field] = sum(
                float(row["teacher_forced"][field])
                * int(row["teacher_forced"]["move_count"])
                for row in rows
            ) / teacher_move_count
    return summary


def _group_summaries(rows: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {group: _summarize_rows(items) for group, items in sorted(groups.items())}


def evaluate_othello(cli_args) -> Path:
    validate_eval_args(cli_args)
    args, run_dir = _load_eval_args(cli_args)
    output_dir = (
        Path(cli_args.output_dir).resolve()
        if cli_args.output_dir
        else run_dir / "othello_eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    per_example_path = output_dir / "per_example.jsonl"
    if per_example_path.exists():
        per_example_path.unlink()

    checkpoint = load_checkpoint_payload(run_dir / "latest.pt", device="cpu")
    _block_size, _vocab, stoi, _itos, model, _optimizer = build_training_objects(args)
    restore_checkpoint_state(checkpoint, model=model, optimizer=None, device=args.device)

    example_rng = random.Random(stable_seed(cli_args.seed, "othello_eval", "examples"))
    traces = _sample_validation_traces(args, count=cli_args.examples, rng=example_rng)
    examples = build_eval_examples(
        traces,
        stoi=stoi,
        evaluation_mode=cli_args.evaluation_mode,
        prefix_fractions=cli_args.prefix_fractions,
        rng=example_rng,
    )
    evaluated_modes = list(cli_args.inference_modes)
    if not is_multi_pass_architecture(args.architecture):
        evaluated_modes = [mode for mode in evaluated_modes if mode == "recompute"]
    if not evaluated_modes:
        raise ValueError("the selected architecture has no requested supported inference mode")

    rows = []
    recompute_cache: dict[tuple[int, ...], torch.Tensor] = {}
    was_training = model.training
    model.eval()
    try:
        for inference_mode in evaluated_modes:
            for example in examples:
                generation_seed = stable_seed(
                    cli_args.seed,
                    "othello_eval",
                    "generation",
                    example.example_index,
                    example.protocol,
                )
                row = {
                    "example_index": example.example_index,
                    "protocol": example.protocol,
                    "inference_mode": inference_mode,
                    "prompt_moves": example.cut,
                    "prompt_bucket": _length_bucket(example.cut),
                    "reference_suffix_moves": len(example.suffix_move_ids),
                    "suffix_bucket": _length_bucket(len(example.suffix_move_ids)),
                    "free_generation": _free_generation_metrics(
                        model,
                        args,
                        stoi,
                        example,
                        inference_mode=inference_mode,
                        do_sample=cli_args.token_selection == "sample",
                        generation_seed=generation_seed,
                    ),
                    "teacher_forced": _teacher_forced_metrics(
                        model,
                        args,
                        stoi,
                        example,
                        inference_mode=inference_mode,
                        recompute_cache=recompute_cache,
                    ),
                }
                rows.append(row)
                append_jsonl(per_example_path, row)
    finally:
        model.train(was_training)

    mode_summaries = {}
    for inference_mode in evaluated_modes:
        mode_rows = [row for row in rows if row["inference_mode"] == inference_mode]
        mode_summaries[inference_mode] = {
            "overall": _summarize_rows(mode_rows),
            "by_protocol": _group_summaries(mode_rows, "protocol"),
            "by_prompt_bucket": _group_summaries(mode_rows, "prompt_bucket"),
            "by_suffix_bucket": _group_summaries(mode_rows, "suffix_bucket"),
        }

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_run_dir": str(run_dir),
        "task": args.task,
        "architecture": args.architecture,
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "evaluation_mode": cli_args.evaluation_mode,
        "token_selection": cli_args.token_selection,
        "base_trace_count": len(traces),
        "continuation_example_count": len(examples),
        "prefix_fractions": list(cli_args.prefix_fractions),
        "requested_inference_modes": list(cli_args.inference_modes),
        "evaluated_inference_modes": evaluated_modes,
        "modes": mode_summaries,
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, payload)
    print(f"wrote {summary_path}")
    print(f"wrote {per_example_path}")
    return output_dir


def main(argv: list[str] | None = None) -> None:
    evaluate_othello(parse_args(argv))


if __name__ == "__main__":
    main()
