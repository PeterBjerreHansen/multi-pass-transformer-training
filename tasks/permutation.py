from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import random
import torch


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
SEP_TOKEN = "<sep>"
EOS_TOKEN = "<eos>"
SWAP_TOKEN = "swap"


@dataclass
class TaskBatch:
    idx: torch.Tensor
    targets: torch.Tensor
    metric_mask: torch.Tensor
    prompt_lengths: torch.Tensor
    output_lengths: torch.Tensor


def required_block_size(num_objects: int, num_swaps: int) -> int:
    # Full sequence is:
    # BOS, initial permutation, swap p_i p_j repeated num_swaps times, SEP, final permutation, EOS.
    # The autoregressive input drops the final token.
    return 2 * num_objects + 3 * num_swaps + 2


def build_permutation_vocab(num_objects: int) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    if num_objects < 2:
        raise ValueError("num_objects must be at least 2")
    tokens = [PAD_TOKEN, BOS_TOKEN, SEP_TOKEN, EOS_TOKEN, SWAP_TOKEN]
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
) -> TaskBatch:
    rng = rng or random.Random()
    initial_state = [stoi[f"o{i}"] for i in range(num_objects)]
    prompt_len = num_objects + 3 * num_swaps + 2
    output_len = num_objects + 1

    idx_rows = []
    target_rows = []
    metric_rows = []
    for _ in range(batch_size):
        final_state, swap_tokens = _sample_permutation_example(num_objects, num_swaps, stoi, rng)
        full = [stoi[BOS_TOKEN], *initial_state, *swap_tokens, stoi[SEP_TOKEN], *final_state, stoi[EOS_TOKEN]]
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
) -> Tuple[List[int], List[int]]:
    state = list(range(num_objects))
    swap_tokens: List[int] = []

    for _ in range(num_swaps):
        i, j = rng.sample(range(num_objects), k=2)
        state[i], state[j] = state[j], state[i]
        swap_tokens.extend([stoi[SWAP_TOKEN], stoi[f"p{i}"], stoi[f"p{j}"]])

    final_state = [stoi[f"o{obj}"] for obj in state]
    return final_state, swap_tokens
