from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import torch


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
SEP_TOKEN = "<sep>"
EOS_TOKEN = "<eos>"
QUERY_TOKEN = "<query>"


@dataclass
class SymbolicBatch:
    idx: torch.Tensor
    targets: torch.Tensor
    prompt_lengths: torch.Tensor
    output_lengths: torch.Tensor


def build_vocab(tokens: Iterable[str]) -> Tuple[list[str], Dict[str, int], Dict[int, str]]:
    seen = set()
    ordered_tokens = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        ordered_tokens.append(token)
    stoi = {token: i for i, token in enumerate(ordered_tokens)}
    itos = {i: token for token, i in stoi.items()}
    return ordered_tokens, stoi, itos


def make_sequence(prompt_tokens: list[int], answer_tokens: list[int], stoi: Dict[str, int]) -> tuple[list[int], int, int]:
    full = [
        stoi[BOS_TOKEN],
        *prompt_tokens,
        stoi[SEP_TOKEN],
        *answer_tokens,
        stoi[EOS_TOKEN],
    ]
    prompt_len = len(prompt_tokens) + 2
    output_len = len(answer_tokens) + 1
    return full, prompt_len, output_len


def build_batch_from_sequences(
    rows: Sequence[tuple[list[int], int, int]],
    *,
    pad_id: int,
    device=None,
) -> SymbolicBatch:
    if not rows:
        raise ValueError("rows must not be empty")

    idx_rows = []
    target_rows = []
    prompt_lengths = []
    output_lengths = []

    for full, prompt_len, output_len in rows:
        idx_row = full[:-1]
        target_row = full[1:]
        for pos in range(prompt_len - 1):
            target_row[pos] = -1
        idx_rows.append(idx_row)
        target_rows.append(target_row)
        prompt_lengths.append(prompt_len)
        output_lengths.append(output_len)

    max_len = max(len(row) for row in idx_rows)
    padded_idx_rows = [row + [pad_id] * (max_len - len(row)) for row in idx_rows]
    padded_target_rows = [row + [-1] * (max_len - len(row)) for row in target_rows]
    return SymbolicBatch(
        idx=torch.tensor(padded_idx_rows, dtype=torch.long, device=device),
        targets=torch.tensor(padded_target_rows, dtype=torch.long, device=device),
        prompt_lengths=torch.tensor(prompt_lengths, dtype=torch.long, device=device),
        output_lengths=torch.tensor(output_lengths, dtype=torch.long, device=device),
    )
