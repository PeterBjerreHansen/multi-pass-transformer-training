from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import random
import torch


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
SEP_TOKEN = "<sep>"
EOS_TOKEN = "<eos>"
SWAP_TOKEN = "swap"
TRACE_TOKEN = "<trace>"
FINAL_TOKEN = "<final>"


@dataclass
class TaskBatch:
    idx: torch.Tensor
    targets: torch.Tensor
    metric_mask: torch.Tensor
    prompt_lengths: torch.Tensor
    output_lengths: torch.Tensor


def required_block_size(num_objects: int, num_swaps: int, supervision: str = "final") -> int:
    _validate_supervision(supervision)
    # Full sequence is:
    # BOS, initial permutation, swap p_i p_j repeated num_swaps times, SEP, final permutation, EOS.
    # The autoregressive input drops the final token.
    final_only_size = 2 * num_objects + 3 * num_swaps + 2
    if supervision == "final":
        return final_only_size
    return final_only_size + num_swaps * (num_objects + 1) + 1


def build_permutation_vocab(num_objects: int) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    if num_objects < 2:
        raise ValueError("num_objects must be at least 2")
    tokens = [PAD_TOKEN, BOS_TOKEN, SEP_TOKEN, EOS_TOKEN, SWAP_TOKEN, TRACE_TOKEN, FINAL_TOKEN]
    tokens.extend(f"o{i}" for i in range(num_objects))
    tokens.extend(f"p{i}" for i in range(num_objects))
    stoi = {token: i for i, token in enumerate(tokens)}
    itos = {i: token for token, i in stoi.items()}
    return tokens, stoi, itos


def decode_ids(ids: Sequence[int], itos: Dict[int, str]) -> List[str]:
    return [itos[int(i)] for i in ids]


def build_permutation_batch(
    batch_size: int,
    num_objects: int,
    num_swaps: int,
    stoi: Dict[str, int],
    device=None,
    rng: random.Random | None = None,
    supervision: str = "final",
) -> TaskBatch:
    _validate_supervision(supervision)
    rng = rng or random.Random()
    initial_state = [stoi[f"o{i}"] for i in range(num_objects)]
    prompt_len = num_objects + 3 * num_swaps + 2
    output_len = num_objects + 1 if supervision == "final" else num_swaps * (num_objects + 1) + num_objects + 2

    idx_rows = []
    target_rows = []
    metric_rows = []
    for _ in range(batch_size):
        final_state, swap_tokens, trace_states = _sample_permutation_example(num_objects, num_swaps, stoi, rng)
        if supervision == "final":
            answer_tokens = final_state
        else:
            answer_tokens = []
            for state in trace_states:
                answer_tokens.extend([stoi[TRACE_TOKEN], *state])
            answer_tokens.extend([stoi[FINAL_TOKEN], *final_state])
        full = [stoi[BOS_TOKEN], *initial_state, *swap_tokens, stoi[SEP_TOKEN], *answer_tokens, stoi[EOS_TOKEN]]
        idx_row = full[:-1]
        target_row = full[1:]
        metric_row = [False] * len(target_row)
        for pos in range(prompt_len - 1):
            target_row[pos] = -1
        for pos in range(prompt_len - 1, prompt_len - 1 + output_len):
            metric_row[pos] = True
        idx_rows.append(idx_row)
        target_rows.append(target_row)
        metric_rows.append(metric_row)

    idx = torch.tensor(idx_rows, dtype=torch.long, device=device)
    targets = torch.tensor(target_rows, dtype=torch.long, device=device)
    metric_mask = torch.tensor(metric_rows, dtype=torch.bool, device=device)
    prompt_lengths = torch.full((batch_size,), prompt_len, dtype=torch.long, device=device)
    output_lengths = torch.full((batch_size,), output_len, dtype=torch.long, device=device)
    return TaskBatch(
        idx=idx,
        targets=targets,
        metric_mask=metric_mask,
        prompt_lengths=prompt_lengths,
        output_lengths=output_lengths,
    )


def _sample_permutation_example(
    num_objects: int,
    num_swaps: int,
    stoi: Dict[str, int],
    rng: random.Random,
) -> Tuple[List[int], List[int], List[List[int]]]:
    state = list(range(num_objects))
    swap_tokens: List[int] = []
    trace_states: List[List[int]] = []

    for _ in range(num_swaps):
        i, j = rng.sample(range(num_objects), k=2)
        state[i], state[j] = state[j], state[i]
        swap_tokens.extend([stoi[SWAP_TOKEN], stoi[f"p{i}"], stoi[f"p{j}"]])
        trace_states.append([stoi[f"o{obj}"] for obj in state])

    final_state = [stoi[f"o{obj}"] for obj in state]
    return final_state, swap_tokens, trace_states


def _validate_supervision(supervision: str):
    if supervision not in {"final", "trace"}:
        raise ValueError("supervision must be either 'final' or 'trace'")
