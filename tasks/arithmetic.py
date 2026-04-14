from typing import Dict, List, Sequence, Tuple

import random

from tasks.bbh_symbolic_common import (
    BOS_TOKEN,
    EOS_TOKEN,
    FINAL_TOKEN,
    PAD_TOKEN,
    SEP_TOKEN,
    STATE_TOKEN,
    SymbolicBatch,
    build_batch_from_sequences,
    build_vocab,
    decode_ids,
    make_sequence,
    validate_supervision,
)


TASK_TOKEN = "ARITH_MOD"
ADD_TOKEN = "+"
SUB_TOKEN = "-"
MUL_TOKEN = "*"
MOD_PREFIX = "mod"
OPS = (ADD_TOKEN, SUB_TOKEN, MUL_TOKEN)


def required_block_size(num_steps: int, supervision: str = "final") -> int:
    validate_supervision(supervision)
    prompt_tokens = 3 + 2 * num_steps  # task, modulus, initial, op/value pairs.
    answer_len = 2 if supervision == "final" else 2 * num_steps + 2
    return 2 + prompt_tokens + answer_len


def build_arithmetic_vocab(max_modulus: int = 10) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    if max_modulus < 2:
        raise ValueError("max_modulus must be at least 2")
    tokens = [
        PAD_TOKEN,
        BOS_TOKEN,
        SEP_TOKEN,
        EOS_TOKEN,
        TASK_TOKEN,
        STATE_TOKEN,
        FINAL_TOKEN,
        ADD_TOKEN,
        SUB_TOKEN,
        MUL_TOKEN,
    ]
    tokens.extend(f"{MOD_PREFIX}{modulus}" for modulus in range(2, max_modulus + 1))
    tokens.extend(str(value) for value in range(max_modulus))
    return build_vocab(tokens)


def eval_modular_arithmetic(initial: int, operations: Sequence[tuple[str, int]], modulus: int) -> tuple[list[int], int]:
    value = initial % modulus
    states: list[int] = []
    for op, rhs in operations:
        if op == ADD_TOKEN:
            value = (value + rhs) % modulus
        elif op == SUB_TOKEN:
            value = (value - rhs) % modulus
        elif op == MUL_TOKEN:
            value = (value * rhs) % modulus
        else:
            raise ValueError(f"Unsupported arithmetic op: {op}")
        states.append(value)
    return states, value


def sample_arithmetic_example(
    num_steps: int,
    modulus: int,
    stoi: Dict[str, int],
    rng: random.Random,
    supervision: str = "final",
) -> tuple[list[int], list[int], int, list[tuple[str, int]], int]:
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    if modulus < 2:
        raise ValueError("modulus must be at least 2")
    validate_supervision(supervision)

    initial = rng.randrange(modulus)
    operations = [(rng.choice(OPS), rng.randrange(modulus)) for _ in range(num_steps)]
    states, final = eval_modular_arithmetic(initial, operations, modulus)
    prompt = [stoi[TASK_TOKEN], stoi[f"{MOD_PREFIX}{modulus}"], stoi[str(initial)]]
    for op, rhs in operations:
        prompt.extend([stoi[op], stoi[str(rhs)]])

    answer: list[int] = []
    if supervision == "trace":
        for state in states:
            answer.extend([stoi[STATE_TOKEN], stoi[str(state)]])
    answer.extend([stoi[FINAL_TOKEN], stoi[str(final)]])
    return prompt, answer, initial, operations, final


def build_arithmetic_batch(
    batch_size: int,
    num_steps: int,
    modulus: int,
    stoi: Dict[str, int],
    supervision: str = "final",
    device=None,
    rng: random.Random | None = None,
) -> SymbolicBatch:
    validate_supervision(supervision)
    rng = rng or random.Random()
    rows = []
    for _ in range(batch_size):
        prompt, answer, _, _, _ = sample_arithmetic_example(num_steps, modulus, stoi, rng, supervision=supervision)
        rows.append(make_sequence(prompt, answer, stoi))
    return build_batch_from_sequences(rows, pad_id=stoi[PAD_TOKEN], device=device)


__all__ = [
    "build_arithmetic_batch",
    "build_arithmetic_vocab",
    "decode_ids",
    "eval_modular_arithmetic",
    "required_block_size",
    "sample_arithmetic_example",
]
