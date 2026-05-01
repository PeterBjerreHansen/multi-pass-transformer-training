"""Hidden-register task: start from two hidden register values, apply a sequence of update ops,
emit one observation after each update, and predict the observation trace.

Example sequence:
<bos> r0 v3 r1 v5 op0 op1 <sep> v18 v17 <eos>
"""

from typing import Dict, List, Sequence, Tuple

import random

from tasks.common import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SEP_TOKEN,
    SymbolicBatch,
    build_batch_from_sequences,
    build_vocab,
    decode_ids,
    make_sequence,
)


REGISTER_TOKENS = ("r0", "r1")
OP_TOKENS = ("op0", "op1")
MODULUS = 61
OBSERVATION_DIM = 1


def value_token(value: int) -> str:
    return f"v{value}"


def required_block_size(num_steps: int) -> int:
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    prompt_tokens = 2 * len(REGISTER_TOKENS) + num_steps
    answer_len = num_steps * OBSERVATION_DIM
    return 2 + prompt_tokens + answer_len


def build_hidden_register_vocab(max_steps: int) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    tokens = [
        PAD_TOKEN,
        BOS_TOKEN,
        SEP_TOKEN,
        EOS_TOKEN,
    ]
    tokens.extend(REGISTER_TOKENS)
    tokens.extend(OP_TOKENS)
    tokens.extend(value_token(value) for value in range(MODULUS))
    return build_vocab(tokens)


def apply_update_rule(state: Sequence[int], op_index: int) -> list[int]:
    r0, r1 = state
    if op_index == 0:
        next_state = [
            r1,
            r0 + r1 + 1,
        ]
    elif op_index == 1:
        next_state = [
            r1,
            r0 + r1 + 1 + r0 * r1,
        ]
    else:
        raise ValueError(f"Unsupported hidden-register op index: {op_index}")
    return [value % MODULUS for value in next_state]


def observe_state(state: Sequence[int]) -> int:
    r0, r1 = state
    return (r0 + 2 * r1) % MODULUS


def solve_hidden_register(
    initial_state: Sequence[int],
    op_indices: Sequence[int],
) -> tuple[list[list[int]], list[int]]:
    state = [value % MODULUS for value in initial_state]
    states = []
    observations = []
    for op_index in op_indices:
        state = apply_update_rule(state, op_index)
        states.append(list(state))
        observations.append(observe_state(state))
    return states, observations


def sample_hidden_register_example(
    num_steps: int,
    stoi: Dict[str, int],
    rng: random.Random,
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")

    initial_state = [rng.randrange(MODULUS) for _ in REGISTER_TOKENS]
    op_indices = [rng.randrange(len(OP_TOKENS)) for _ in range(num_steps)]
    _, observations = solve_hidden_register(initial_state, op_indices)

    prompt: list[int] = []
    for register_token, value in zip(REGISTER_TOKENS, initial_state):
        prompt.extend([stoi[register_token], stoi[value_token(value)]])
    prompt.extend(stoi[OP_TOKENS[index]] for index in op_indices)

    answer = [stoi[value_token(observation)] for observation in observations]

    return prompt, answer, initial_state, op_indices, observations


def build_hidden_register_batch(
    batch_size: int,
    num_steps: int,
    stoi: Dict[str, int],
    device=None,
    rng: random.Random | None = None,
) -> SymbolicBatch:
    rng = rng or random.Random()
    rows = []
    for _ in range(batch_size):
        prompt, answer, _, _, _ = sample_hidden_register_example(
            num_steps,
            stoi,
            rng,
        )
        rows.append(make_sequence(prompt, answer, stoi))
    return build_batch_from_sequences(rows, pad_id=stoi[PAD_TOKEN], device=device)


__all__ = [
    "MODULUS",
    "OBSERVATION_DIM",
    "OP_TOKENS",
    "REGISTER_TOKENS",
    "apply_update_rule",
    "build_hidden_register_batch",
    "build_hidden_register_vocab",
    "decode_ids",
    "observe_state",
    "required_block_size",
    "sample_hidden_register_example",
    "solve_hidden_register",
    "value_token",
]
