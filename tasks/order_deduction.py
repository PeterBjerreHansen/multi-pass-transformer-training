from typing import Dict, List, Tuple

import random

from tasks.bbh_symbolic_common import (
    BOS_TOKEN,
    EOS_TOKEN,
    FINAL_TOKEN,
    PAD_TOKEN,
    QUERY_TOKEN,
    SEP_TOKEN,
    TRACE_TOKEN,
    SymbolicBatch,
    build_batch_from_sequences,
    build_vocab,
    decode_ids,
    make_sequence,
    validate_supervision,
)


TASK_TOKEN = "ORDER"
LT_TOKEN = "LT"
SEMI_TOKEN = ";"
TRUE_TOKEN = "T"
FALSE_TOKEN = "F"


def obj_token(index: int) -> str:
    return f"o{index}"


def required_block_size(num_objects: int, supervision: str = "final") -> int:
    validate_supervision(supervision)
    if num_objects < 2:
        raise ValueError("num_objects must be at least 2")
    prompt_tokens = 1 + (num_objects - 1) * 4 + 4
    answer_len = 2 if supervision == "final" else 4 * (num_objects * (num_objects - 1) // 2) + 2
    return 2 + prompt_tokens + answer_len


def build_order_vocab(max_num_objects: int) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    if max_num_objects < 2:
        raise ValueError("max_num_objects must be at least 2")
    tokens = [
        PAD_TOKEN,
        BOS_TOKEN,
        SEP_TOKEN,
        EOS_TOKEN,
        TASK_TOKEN,
        TRACE_TOKEN,
        FINAL_TOKEN,
        QUERY_TOKEN,
        LT_TOKEN,
        SEMI_TOKEN,
        TRUE_TOKEN,
        FALSE_TOKEN,
    ]
    tokens.extend(obj_token(index) for index in range(max_num_objects))
    return build_vocab(tokens)


def sample_order_example(
    num_objects: int,
    stoi: Dict[str, int],
    rng: random.Random,
    supervision: str = "final",
) -> tuple[list[int], list[int], list[int], tuple[int, int], bool]:
    if num_objects < 2:
        raise ValueError("num_objects must be at least 2")
    validate_supervision(supervision)

    order = list(range(num_objects))
    rng.shuffle(order)
    rank = {obj: index for index, obj in enumerate(order)}

    prompt = [stoi[TASK_TOKEN]]
    for left, right in zip(order, order[1:]):
        prompt.extend([stoi[obj_token(left)], stoi[LT_TOKEN], stoi[obj_token(right)], stoi[SEMI_TOKEN]])

    a, b = rng.sample(range(num_objects), k=2)
    final = rank[a] < rank[b]
    prompt.extend([stoi[QUERY_TOKEN], stoi[obj_token(a)], stoi[LT_TOKEN], stoi[obj_token(b)]])

    answer: list[int] = []
    if supervision == "trace":
        for distance in range(1, num_objects):
            for left_index in range(0, num_objects - distance):
                left = order[left_index]
                right = order[left_index + distance]
                answer.extend([stoi[TRACE_TOKEN], stoi[obj_token(left)], stoi[LT_TOKEN], stoi[obj_token(right)]])
    answer.extend([stoi[FINAL_TOKEN], stoi[TRUE_TOKEN if final else FALSE_TOKEN]])
    return prompt, answer, order, (a, b), final


def build_order_batch(
    batch_size: int,
    num_objects: int,
    stoi: Dict[str, int],
    supervision: str = "final",
    device=None,
    rng: random.Random | None = None,
) -> SymbolicBatch:
    validate_supervision(supervision)
    rng = rng or random.Random()
    rows = []
    for _ in range(batch_size):
        prompt, answer, _, _, _ = sample_order_example(num_objects, stoi, rng, supervision=supervision)
        rows.append(make_sequence(prompt, answer, stoi))
    return build_batch_from_sequences(rows, pad_id=stoi[PAD_TOKEN], device=device)


__all__ = [
    "build_order_batch",
    "build_order_vocab",
    "decode_ids",
    "required_block_size",
    "sample_order_example",
]
