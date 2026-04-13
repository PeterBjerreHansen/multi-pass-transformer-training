import json
import random
from pathlib import Path
from typing import Callable

import torch

from model_factory import ARCHITECTURES, is_multi_pass_architecture


def add_model_args(parser, *, default_n_embd: int):
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="transformer")
    parser.add_argument("--generation-mode", choices=["recompute", "greedy"], default="greedy")
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=default_n_embd)
    parser.add_argument("--n-pass", type=int, default=2)
    parser.add_argument("--memory-update-gate", choices=["on", "off"], default="off")
    parser.add_argument("--memory-gate-bias", type=float, default=-1.0)
    parser.add_argument("--pass-loss-weights", type=float, nargs="*", default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--log-jsonl",
        default=None,
        help="Optional path for structured training/eval events.",
    )


def add_training_args(
    parser,
    *,
    default_train_steps: int,
    default_lr: float,
    default_eval_interval: int = 200,
    default_eval_batches: int = 2,
):
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=default_train_steps)
    parser.add_argument("--lr", type=float, default=default_lr)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-interval", type=int, default=default_eval_interval)
    parser.add_argument("--eval-batches", type=int, default=default_eval_batches)
    parser.add_argument("--seed", type=int, default=1337)


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


def forward_and_loss(model, batch, args):
    if not is_multi_pass_architecture(args.architecture):
        logits = model(batch.idx)
        loss = model.calc_loss(logits, batch.targets)
        return loss, logits, [loss.detach()]

    logits_per_pass = model(
        batch.idx,
        return_all_logits=True,
    )
    loss, losses = model.calc_total_loss(
        logits_per_pass,
        batch.targets,
        loss_weights=args.pass_loss_weights,
    )
    return loss, logits_per_pass[-1], [item.detach() for item in losses]


@torch.no_grad()
def teacher_forced_metrics(logits: torch.Tensor, batch) -> tuple[float, float]:
    preds = logits.argmax(dim=-1)
    mask = batch.metric_mask
    correct = preds == batch.targets
    token_accuracy = correct[mask].float().mean().item()
    exact_match = (correct | ~mask).all(dim=1).float().mean().item()
    return exact_match, token_accuracy


@torch.no_grad()
def generation_metrics(model, batch, args) -> tuple[float, float]:
    exact_matches = []
    token_accuracies = []
    generation_mode = "recompute" if args.architecture == "transformer" else args.generation_mode

    for row in range(batch.idx.size(0)):
        prompt_len = int(batch.prompt_lengths[row].item())
        output_len = int(batch.output_lengths[row].item())
        prompt = batch.idx[row:row + 1, :prompt_len]
        target_suffix = batch.targets[row:row + 1, prompt_len - 1:prompt_len - 1 + output_len]

        generated = model.generate(
            prompt,
            max_new_tokens=output_len,
            do_sample=False,
            generation_mode=generation_mode,
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
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_exact_match = 0.0
    total_token_accuracy = 0.0
    n_batches = args.eval_batches if eval_batches is None else eval_batches

    for _ in range(n_batches):
        batch = batch_builder(rng)
        loss, _, _ = forward_and_loss(model, batch, args)
        exact_match, token_accuracy = generation_metrics(model, batch, args)
        total_loss += float(loss.detach().item())
        total_exact_match += exact_match
        total_token_accuracy += token_accuracy

    denom = float(n_batches)
    return {
        "loss": total_loss / denom,
        "exact_match": total_exact_match / denom,
        "token_accuracy": total_token_accuracy / denom,
    }


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
    start_level: int,
    review_easier_every: int,
    rng: random.Random,
) -> int:
    if review_easier_every > 0 and current_level > start_level and step % review_easier_every == 0:
        return rng.randint(0, current_level - 1)
    return current_level


def choose_easier_eval_level(current_level: int, rng: random.Random) -> int:
    return rng.randint(0, current_level - 1)


def log_jsonl(path: str | None, event: dict):
    if path is None:
        return
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")
