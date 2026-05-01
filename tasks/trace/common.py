from typing import Callable, Sequence

import numpy as np
import torch


def trace_generation_metrics(
    model,
    batch,
    args,
    *,
    legality_check: Callable[[Sequence[int], Sequence[int]], tuple[int, bool]],
    inference_mode: str | None = None,
) -> dict[str, float | None]:
    token_legalities = []
    sequence_legalities = []
    legal_lengths = []

    if args.architecture == "transformer":
        mode = "recompute"
    else:
        mode = inference_mode or args.inference_mode
    do_sample = getattr(args, "token_selection", "sample") == "sample"

    for row in range(batch.idx.size(0)):
        prompt_len = int(batch.prompt_lengths[row].item())
        output_len = int(batch.output_lengths[row].item())
        prompt = batch.idx[row : row + 1, :prompt_len]
        target_suffix = batch.targets[row : row + 1, prompt_len - 1 : prompt_len - 1 + output_len]

        generated = model.generate(
            prompt,
            max_new_tokens=output_len,
            do_sample=do_sample,
            inference_mode=mode,
            cache_source=getattr(args, "cache_source", "penultimate"),
        )
        generated_suffix = generated[:, prompt_len : prompt_len + output_len]

        prompt_tokens = batch.idx[row, 1 : prompt_len - 1].tolist()
        generated_tokens = generated_suffix[0].tolist()
        target_tokens = target_suffix[0, :-1].tolist()
        valid_token_count = sum(int(token != 0) for token in target_tokens)
        if valid_token_count == 0:
            valid_token_count = len(target_tokens)
        legal_prefix_len, all_legal = legality_check(prompt_tokens, generated_tokens[:-1])
        legal_lengths.append(float(legal_prefix_len))
        token_legalities.append(float(legal_prefix_len) / float(valid_token_count))

        eos_ok = int(generated_tokens[-1] == int(target_suffix[0, -1].item()))
        sequence_legalities.append(float(all_legal and eos_ok == 1))

    return {
        "token_legality": float(np.mean(token_legalities)),
        "sequence_legality": float(np.mean(sequence_legalities)),
        "mean_legal_len": float(np.mean(legal_lengths)),
    }


def format_legal_generation_metrics(metrics: dict[str, float]) -> str:
    return (
        f"token_legality {float(metrics['token_legality']):.3f} | "
        f"sequence_legality {float(metrics['sequence_legality']):.3f} | "
        f"mean_legal_len {float(metrics['mean_legal_len']):.2f}"
    )
