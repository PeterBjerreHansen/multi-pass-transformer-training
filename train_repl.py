import argparse
import random

import torch

from tasks.repl import (
    BOS_TOKEN,
    build_repl_batch,
    build_repl_vocab,
    decode_ids,
    parse_operand_patterns,
    parse_operations,
    required_block_size,
    token_ids_to_trace_lines,
)
from model_factory import build_model, is_multi_pass_architecture
from train_utils import (
    add_model_args,
    add_training_args,
    choose_curriculum_train_level,
    choose_easier_eval_level,
    evaluate_batches,
    format_eval_metrics,
    format_pass_losses,
    forward_and_loss,
    log_jsonl,
    set_seed,
    teacher_forced_metrics,
    validate_model_args,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a transformer architecture on REPL-style traces with a curriculum.",
        allow_abbrev=False,
    )
    add_model_args(parser, default_n_embd=128)
    parser.add_argument("--max-num-vars", type=int, default=4)
    parser.add_argument("--randomize-num-vars", action="store_true")
    parser.add_argument("--value-mod", type=int, default=10)
    parser.add_argument("--operations", type=str, default="add,sub")
    parser.add_argument("--operand-patterns", type=str, default="num_num,var_num,num_var,var_var")
    parser.add_argument("--print-prob", type=float, default=0.25)
    parser.add_argument("--train-program-length", type=int, default=6)
    parser.add_argument("--loss-on", choices=["suffix", "full"], default="suffix")
    parser.add_argument("--curriculum-start-program-length", type=int, default=1)
    parser.add_argument("--curriculum-threshold", type=float, default=0.99)
    parser.add_argument(
        "--review-easier-every",
        type=int,
        default=2,
        help="Sample an easier task every Nth batch. Use 0 to disable.",
    )
    parser.add_argument(
        "--inspect-eval-errors",
        type=int,
        default=0,
        help="Print up to this many current-level eval failures at each curriculum eval.",
    )
    add_training_args(parser, default_train_steps=20_000, default_lr=1e-4)
    return parser.parse_args()


def validate_task_args(args):
    if args.max_num_vars < 2:
        raise ValueError("--max-num-vars must be at least 2")
    if args.train_program_length < 0:
        raise ValueError("--train-program-length must be non-negative")
    if not 0.0 <= args.print_prob <= 1.0:
        raise ValueError("--print-prob must be in [0, 1]")
    if args.curriculum_start_program_length < 1:
        raise ValueError("--curriculum-start-program-length must be at least 1")
    if args.curriculum_start_program_length > args.train_program_length:
        raise ValueError("--curriculum-start-program-length must be <= --train-program-length")
    if not 0.0 <= args.curriculum_threshold <= 1.0:
        raise ValueError("--curriculum-threshold must be in [0, 1]")
    if args.review_easier_every < 0:
        raise ValueError("--review-easier-every must be >= 0")


def build_batch(args, stoi, program_length: int, rng: random.Random):
    return build_repl_batch(
        batch_size=args.batch_size,
        max_num_vars=args.max_num_vars,
        program_length=program_length,
        stoi=stoi,
        operations=args.operations,
        operand_patterns=args.operand_patterns,
        value_mod=args.value_mod,
        randomize_num_vars=args.randomize_num_vars,
        print_prob=args.print_prob,
        loss_on=args.loss_on,
        device=args.device,
        rng=rng,
    )


def evaluate_program_length(model, args, stoi, program_length: int, rng: random.Random):
    return evaluate_batches(
        model,
        args,
        batch_builder=lambda batch_rng: build_batch(args, stoi, program_length, batch_rng),
        rng=rng,
    )


def _suffix_ids_to_lines(suffix_ids: torch.Tensor, stoi, itos) -> list[str]:
    bos_id = stoi[BOS_TOKEN]
    ids_list = [bos_id] + suffix_ids.detach().cpu().tolist()
    try:
        return token_ids_to_trace_lines(ids_list, itos)
    except ValueError:
        tokens = decode_ids(ids_list, itos)
        return ["<raw> " + " ".join(tokens[1:])]


@torch.no_grad()
def inspect_eval_examples(model, args, stoi, itos, program_length: int, rng: random.Random):
    if args.inspect_eval_errors <= 0:
        return

    shown = 0
    attempts = 0
    max_attempts = max(args.inspect_eval_errors * 10, args.inspect_eval_errors)

    while shown < args.inspect_eval_errors and attempts < max_attempts:
        attempts += 1
        batch = build_batch(args, stoi, program_length, rng)
        loss, _, _ = forward_and_loss(model, batch, args)
        prompt_len = int(batch.prompt_lengths[0].item())
        output_len = int(batch.output_lengths[0].item())
        prompt_ids = batch.idx[0, :prompt_len].detach().cpu().tolist()
        target_suffix = batch.targets[0, prompt_len - 1:prompt_len - 1 + output_len]

        prompt = batch.idx[:, :prompt_len]
        generated = model.generate(
            prompt,
            max_new_tokens=output_len,
            do_sample=False,
            generation_mode=args.generation_mode,
        )
        predicted_suffix = generated[0, prompt_len:prompt_len + output_len]

        if torch.equal(predicted_suffix, target_suffix):
            continue

        prompt_lines = token_ids_to_trace_lines(prompt_ids, itos)
        target_lines = _suffix_ids_to_lines(target_suffix, stoi, itos)
        predicted_lines = _suffix_ids_to_lines(predicted_suffix, stoi, itos)

        shown += 1
        print(f"  inspect_error {shown} | program_length {program_length:3d} | loss {loss.item():.4f}")
        print("    prompt:")
        for line in prompt_lines:
            print(f"      {line}")
        print("    target_suffix:")
        for line in target_lines:
            print(f"      {line}")
        print("    predicted_suffix:")
        for line in predicted_lines:
            print(f"      {line}")


def print_run_header(args, block_size: int, model):
    print(f"device: {args.device}")
    print("task: repl")
    print("training_mode: curriculum")
    print(f"architecture: {args.architecture}")
    print(f"generation_mode: {args.generation_mode}")
    print(f"max_num_vars: {args.max_num_vars}")
    print(f"randomize_num_vars: {args.randomize_num_vars}")
    print(f"value_mod: {args.value_mod}")
    print(f"operations: {args.operations}")
    print(f"operand_patterns: {args.operand_patterns}")
    print(f"print_prob: {args.print_prob}")
    print(f"loss_on: {args.loss_on}")
    if is_multi_pass_architecture(args.architecture):
        print(f"n_pass: {args.n_pass}")
        if args.architecture == "memory_update":
            print(f"memory_update_gate: {args.memory_update_gate}")
            if args.memory_update_gate == "on":
                print(f"memory_gate_bias: {args.memory_gate_bias}")
        print(f"pass_loss_weights: {args.pass_loss_weights}")
    print(f"curriculum_start_program_length: {args.curriculum_start_program_length}")
    print(f"curriculum_max_program_length: {args.train_program_length}")
    print(f"curriculum_threshold: {args.curriculum_threshold}")
    print(f"review_easier_every: {args.review_easier_every}")
    print(f"block_size: {block_size}")
    print("number of parameters: %.2fM" % (model.get_num_params() / 1e6,))


def main():
    args = parse_args()
    set_seed(args.seed)
    args.operations = parse_operations(args.operations)
    args.operand_patterns = parse_operand_patterns(args.operand_patterns)
    validate_model_args(args)
    validate_task_args(args)

    block_size = required_block_size(
        max_num_vars=args.max_num_vars,
        program_length=args.train_program_length,
    )
    vocab, stoi, itos = build_repl_vocab(value_mod=args.value_mod, max_num_vars=args.max_num_vars)
    model = build_model(args, vocab_size=len(vocab), block_size=block_size, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_rng = random.Random(args.seed + 10)
    eval_rng = random.Random(args.seed + 20)
    log_rng = random.Random(args.seed + 30)
    current_level = args.curriculum_start_program_length
    promotion_history: list[tuple[int, int, float]] = []

    print_run_header(args, block_size, model)
    log_jsonl(args.log_jsonl, {"event": "run_start", "task": "repl", "config": vars(args), "block_size": block_size})

    for step in range(1, args.train_steps + 1):
        model.train()
        train_program_length = choose_curriculum_train_level(
            step,
            current_level,
            args.curriculum_start_program_length,
            args.review_easier_every,
            train_rng,
        )

        batch = build_batch(args, stoi, train_program_length, train_rng)
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = forward_and_loss(model, batch, args)
        loss.backward()
        optimizer.step()

        should_log = step == 1 or step % args.eval_interval == 0 or step == args.train_steps
        if not should_log:
            continue

        model.eval()
        with torch.no_grad():
            log_batch = build_batch(args, stoi, current_level, log_rng)
            log_loss, log_logits, log_pass_losses = forward_and_loss(model, log_batch, args)
            train_exact, train_token_acc = teacher_forced_metrics(log_logits, log_batch)

        suffix = f"train_program_length {current_level:3d} | curriculum_level {current_level:3d}"
        print(
            f"step {step:5d} | train_loss {log_loss.item():.4f} | "
            f"pass_losses {format_pass_losses(log_pass_losses)} | "
            f"{suffix} | train_trace_acc {train_exact:.3f} | train_token_acc {train_token_acc:.3f}"
        )
        log_jsonl(
            args.log_jsonl,
            {
                "event": "train_step",
                "step": step,
                "train_loss": float(log_loss.detach().item()),
                "train_program_length": current_level,
                "sampled_train_program_length": train_program_length,
                "curriculum_level": current_level,
                "train_exact_match": train_exact,
                "train_token_accuracy": train_token_acc,
            },
        )

        current_metrics = evaluate_program_length(model, args, stoi, current_level, eval_rng)
        print(
            f"  eval_current program_length {current_level:3d} | "
            f"loss {float(current_metrics['loss']):.4f} | "
            f"{format_eval_metrics(current_metrics)}"
        )
        log_jsonl(args.log_jsonl, {"event": "eval", "step": step, "level": current_level, "metrics": current_metrics})
        inspect_eval_examples(model, args, stoi, itos, current_level, eval_rng)

        if current_level > args.curriculum_start_program_length:
            easier_level = choose_easier_eval_level(current_level, eval_rng)
            easier_metrics = evaluate_program_length(model, args, stoi, easier_level, eval_rng)
            print(
                f"  eval_easier program_length  {easier_level:3d} | "
                f"loss {float(easier_metrics['loss']):.4f} | "
                f"{format_eval_metrics(easier_metrics)}"
            )
            log_jsonl(args.log_jsonl, {"event": "eval_easier", "step": step, "level": easier_level, "metrics": easier_metrics})

        metric_value = float(current_metrics["exact_match"])
        if metric_value >= args.curriculum_threshold and current_level < args.train_program_length:
            promotion_history.append((current_level, step, metric_value))
            current_level += 1
            print(
                f"  curriculum_promote -> program_length {current_level:3d} | "
                f"seq_acc {metric_value:.3f}"
            )
            log_jsonl(args.log_jsonl, {"event": "curriculum_promote", "step": step, "next_level": current_level, "seq_acc": metric_value})

    if promotion_history:
        print("promotion_history:")
        for solved_program_length, step, metric_value in promotion_history:
            print(
                f"  solved_program_length {solved_program_length:3d} | "
                f"promoted_at_step {step:5d} | "
                f"seq_acc {metric_value:.3f}"
            )
    else:
        print("promotion_history: none")


if __name__ == "__main__":
    main()
