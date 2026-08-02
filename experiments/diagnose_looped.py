"""Inspect recurrent-state dynamics in a saved looped-transformer run."""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from experiments.common import (
    EVALUATION_CHECKPOINTS,
    load_checkpoint_payload,
    resolve_device_arg,
    resolve_evaluation_checkpoint,
    restore_checkpoint_state,
    saved_args_from_run,
    validate_model_args,
    validate_training_args,
    write_json,
)
from experiments.train_trace import (
    build_fixed_eval_batches,
    build_training_objects,
    validate_task_args,
)
from models import LoopedTransformer


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Diagnose pass dynamics for a looped-transformer checkpoint.",
        allow_abbrev=False,
    )
    parser.add_argument("--input-run-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint", choices=EVALUATION_CHECKPOINTS, default="best")
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-batches", type=int, default=4)
    return parser.parse_args(argv)


def _mean_state_metrics(state: torch.Tensor, previous: torch.Tensor) -> tuple[float, float]:
    state_flat = state.float().reshape(-1, state.shape[-1])
    previous_flat = previous.float().reshape(-1, previous.shape[-1])
    update = state_flat - previous_flat
    ratio = update.norm(dim=-1) / previous_flat.norm(dim=-1).clamp_min(1e-12)
    cosine = F.cosine_similarity(state_flat, previous_flat, dim=-1)
    return float(ratio.mean().item()), float(cosine.mean().item())


@torch.no_grad()
def diagnose(cli_args) -> Path:
    run_dir = Path(cli_args.input_run_dir).resolve()
    saved = saved_args_from_run(run_dir)
    saved["run_dir"] = str(run_dir)
    saved["resume_from"] = str(run_dir)
    saved["inference_mode"] = "recompute"
    saved["eval_batches"] = cli_args.eval_batches
    if cli_args.device is not None:
        saved["device"] = cli_args.device
    args = SimpleNamespace(**saved)
    resolve_device_arg(args)
    validate_model_args(args)
    validate_training_args(args)
    validate_task_args(args)
    if args.architecture != "looped_transformer":
        raise ValueError("loop diagnostics require architecture=looped_transformer")

    checkpoint_path = resolve_evaluation_checkpoint(run_dir, cli_args.checkpoint)
    checkpoint = load_checkpoint_payload(checkpoint_path, device="cpu")
    _block_size, _vocab, stoi, _itos, model, _optimizer = build_training_objects(args)
    if not isinstance(model, LoopedTransformer):
        raise TypeError("model factory did not build a LoopedTransformer")
    restore_checkpoint_state(checkpoint, model=model, optimizer=None, device=args.device)
    model.eval()

    batches = build_fixed_eval_batches(args, stoi)
    losses = [0.0 for _ in range(args.n_pass)]
    update_ratios = [0.0 for _ in range(args.n_pass)]
    state_cosines = [0.0 for _ in range(args.n_pass)]
    for batch in batches:
        fixed_input = model.fixed_recurrent_input(batch.idx)
        output = model.forward_iterations(batch.idx)
        previous = fixed_input
        for iteration, pass_output in enumerate(output.passes):
            losses[iteration] += float(model.calc_loss(pass_output.logits, batch.targets).item())
            state = pass_output.memory_states
            if state is None:
                raise RuntimeError("looped model did not expose its recurrent state")
            ratio, cosine = _mean_state_metrics(state, previous)
            update_ratios[iteration] += ratio
            state_cosines[iteration] += cosine
            previous = state

    count = len(batches)
    alpha, delta = model.input_write_coefficients()
    if args.loop_layout == "full":
        block_applications = args.n_layer * args.n_pass
    else:
        block_applications = 2 + (args.n_layer - 2) * args.n_pass
    payload = {
        "architecture": args.architecture,
        "checkpoint": cli_args.checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "loop_layout": args.loop_layout,
        "persistent_input": args.loop_persistent_input == "on",
        "physical_layers": args.n_layer,
        "loop_iterations": args.n_pass,
        "block_applications": block_applications,
        "input_write": {
            "alpha_mean": float(alpha.mean().item()),
            "alpha_min": float(alpha.min().item()),
            "alpha_max": float(alpha.max().item()),
            "delta_mean": float(delta.mean().item()),
            "delta_min": float(delta.min().item()),
            "delta_max": float(delta.max().item()),
        },
        "iterations": [
            {
                "iteration": iteration + 1,
                "teacher_forced_loss": losses[iteration] / count,
                "state_update_ratio": update_ratios[iteration] / count,
                "state_cosine_to_previous": state_cosines[iteration] / count,
                "final_readout_uses_coda": (
                    args.loop_layout == "sandwich" and iteration == args.n_pass - 1
                ),
            }
            for iteration in range(args.n_pass)
        ],
    }
    output_path = Path(cli_args.output).resolve() if cli_args.output else run_dir / "diagnostics.json"
    write_json(output_path, payload)
    print(f"wrote {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> None:
    diagnose(parse_args(argv))


if __name__ == "__main__":
    main()
