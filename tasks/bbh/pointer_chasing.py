"""Pointer-chasing task: read a directed graph where each node points to one next node,
start from a queried node, follow the pointer chain for a fixed number of hops, and predict the destination.

Example sequence:
<bos> n0 -> n2 n1 -> n0 n2 -> n1 <query> n0 hop hop <sep> n1 <eos>
"""

from typing import Dict, List, Sequence, Tuple

import random

from tasks.common import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    QUERY_TOKEN,
    SEP_TOKEN,
    SymbolicBatch,
    build_batch_from_sequences,
    build_vocab,
    decode_ids,
    make_sequence,
)


EDGE_TOKEN = "->"
STEP_TOKEN = "hop"
DEFAULT_NUM_NODES = 32


def node_token(index: int) -> str:
    return f"n{index}"


def required_block_size(num_nodes: int, num_hops: int) -> int:
    if num_nodes < 2:
        raise ValueError("num_nodes must be at least 2")
    if num_hops < 0:
        raise ValueError("num_hops must be non-negative")
    prompt_tokens = 3 * num_nodes + 2 + num_hops
    answer_len = 1
    return 2 + prompt_tokens + answer_len


def build_pointer_chasing_vocab(num_nodes: int) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    if num_nodes < 2:
        raise ValueError("num_nodes must be at least 2")
    tokens = [PAD_TOKEN, BOS_TOKEN, SEP_TOKEN, EOS_TOKEN, QUERY_TOKEN, EDGE_TOKEN, STEP_TOKEN]
    tokens.extend(node_token(index) for index in range(num_nodes))
    return build_vocab(tokens)


def solve_pointer_chasing(pointers: Sequence[int], start_node: int, num_hops: int) -> tuple[list[int], int]:
    if not pointers:
        raise ValueError("pointers must not be empty")
    if not 0 <= start_node < len(pointers):
        raise ValueError("start_node must index into pointers")
    if num_hops < 1:
        raise ValueError("num_hops must be at least 1")
    for target in pointers:
        if target < 0 or target >= len(pointers):
            raise ValueError("pointer target must index into pointers")

    current = start_node
    trace: list[int] = []
    for _ in range(num_hops):
        current = pointers[current]
        trace.append(current)
    return trace, current


def sample_pointer_chasing_example(
    num_nodes: int,
    num_hops: int,
    stoi: Dict[str, int],
    rng: random.Random,
) -> tuple[list[int], list[int], list[int], int, int]:
    if num_nodes < 2:
        raise ValueError("num_nodes must be at least 2")
    if num_hops < 0:
        raise ValueError("num_hops must be non-negative")

    cycle_order = list(range(num_nodes))
    rng.shuffle(cycle_order)
    pointers = [0] * num_nodes
    for index, source in enumerate(cycle_order):
        pointers[source] = cycle_order[(index + 1) % num_nodes]
    start_node = rng.randrange(num_nodes)
    edge_order = list(range(num_nodes))
    if num_hops > 0:
        rng.shuffle(edge_order)

    prompt: list[int] = []
    for source in edge_order:
        prompt.extend([stoi[node_token(source)], stoi[EDGE_TOKEN], stoi[node_token(pointers[source])]])
    prompt.extend([stoi[QUERY_TOKEN], stoi[node_token(start_node)]])
    prompt.extend(stoi[STEP_TOKEN] for _ in range(num_hops))

    if num_hops == 0:
        final_node = pointers[start_node]
    else:
        _, final_node = solve_pointer_chasing(pointers, start_node, num_hops)
    answer = [stoi[node_token(final_node)]]
    return prompt, answer, pointers, start_node, final_node


def build_pointer_chasing_batch(
    batch_size: int,
    num_nodes: int,
    num_hops: int,
    stoi: Dict[str, int],
    device=None,
    rng: random.Random | None = None,
) -> SymbolicBatch:
    rng = rng or random.Random()
    rows = []
    for _ in range(batch_size):
        prompt, answer, _, _, _ = sample_pointer_chasing_example(num_nodes, num_hops, stoi, rng)
        rows.append(make_sequence(prompt, answer, stoi))
    return build_batch_from_sequences(rows, pad_id=stoi[PAD_TOKEN], device=device)


__all__ = [
    "build_pointer_chasing_batch",
    "build_pointer_chasing_vocab",
    "decode_ids",
    "node_token",
    "required_block_size",
    "sample_pointer_chasing_example",
    "solve_pointer_chasing",
]
