"""NLFSR task: start from a binary register seed, repeatedly shift in a nonlinear feedback bit,
emit one observed bit after each step, and predict the observation trace.

Example full sequence:
<bos> steps t3 seed 1 0 1 1 0 0 1 0 0 1 1 0 <sep> 1 0 1 <eos>
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


STEPS_TOKEN = "steps"
SEED_TOKEN = "seed"
ZERO_TOKEN = "0"
ONE_TOKEN = "1"
BIT_TOKENS = (ZERO_TOKEN, ONE_TOKEN)
DEFAULT_NUM_REGISTERS = 12
MIN_NUM_REGISTERS = 4
OBSERVATION_DIM = 1
FEEDBACK_XOR_TAPS = (0, 2, 5, 9)
FEEDBACK_AND_TAPS = ((1, 4), (3, 7), (6, 10))
FEEDBACK_MIX_TAPS = ((0, 8, 11),)
OBSERVATION_XOR_TAPS = (1, 4, 8)
OBSERVATION_AND_TAPS = ((0, 6), (2, 9))
OBSERVATION_MIX_TAPS = ((3, 7, 10),)


def bit_token(value: int | bool) -> str:
    return ONE_TOKEN if int(value) else ZERO_TOKEN


def step_token(num_steps: int) -> str:
    return f"t{num_steps}"


def _validate_num_registers(num_registers: int) -> int:
    if num_registers < MIN_NUM_REGISTERS:
        raise ValueError(f"num_registers must be at least {MIN_NUM_REGISTERS}")
    return num_registers


def _scaled_tap(num_registers: int, canonical_index: int) -> int:
    if num_registers == DEFAULT_NUM_REGISTERS:
        return canonical_index
    return round(canonical_index * (num_registers - 1) / (DEFAULT_NUM_REGISTERS - 1))


def required_block_size(num_steps: int, num_registers: int = DEFAULT_NUM_REGISTERS) -> int:
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    num_registers = _validate_num_registers(num_registers)
    prompt_tokens = 2 + 1 + num_registers  # steps tN seed bits...
    answer_len = num_steps * OBSERVATION_DIM
    return 2 + prompt_tokens + answer_len


def build_nlfsr_vocab(
    max_steps: int,
    num_registers: int = DEFAULT_NUM_REGISTERS,
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    _validate_num_registers(num_registers)
    tokens = [
        PAD_TOKEN,
        BOS_TOKEN,
        SEP_TOKEN,
        EOS_TOKEN,
        STEPS_TOKEN,
        SEED_TOKEN,
        ZERO_TOKEN,
        ONE_TOKEN,
    ]
    tokens.extend(step_token(step) for step in range(1, max_steps + 1))
    return build_vocab(tokens)


def _validate_state(
    state: Sequence[int],
    num_registers: int | None = None,
) -> list[int]:
    expected_bits = _validate_num_registers(len(state)) if num_registers is None else _validate_num_registers(num_registers)
    if len(state) != expected_bits:
        raise ValueError(f"state must contain {expected_bits} bits")
    bits = [int(bit) for bit in state]
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("state values must be 0 or 1")
    return bits


def feedback_bit(state: Sequence[int]) -> int:
    s = _validate_state(state)
    num_registers = len(s)
    feedback = 0
    for tap in FEEDBACK_XOR_TAPS:
        feedback ^= s[_scaled_tap(num_registers, tap)]
    for left_tap, right_tap in FEEDBACK_AND_TAPS:
        feedback ^= s[_scaled_tap(num_registers, left_tap)] & s[_scaled_tap(num_registers, right_tap)]
    for left_tap, right_tap, gate_tap in FEEDBACK_MIX_TAPS:
        feedback ^= (
            s[_scaled_tap(num_registers, left_tap)] ^ s[_scaled_tap(num_registers, right_tap)]
        ) & s[_scaled_tap(num_registers, gate_tap)]
    return feedback


def observe_state(state: Sequence[int]) -> int:
    s = _validate_state(state)
    num_registers = len(s)
    observation = 0
    for tap in OBSERVATION_XOR_TAPS:
        observation ^= s[_scaled_tap(num_registers, tap)]
    for left_tap, right_tap in OBSERVATION_AND_TAPS:
        observation ^= s[_scaled_tap(num_registers, left_tap)] & s[_scaled_tap(num_registers, right_tap)]
    for left_tap, right_tap, gate_tap in OBSERVATION_MIX_TAPS:
        observation ^= (
            s[_scaled_tap(num_registers, left_tap)] ^ s[_scaled_tap(num_registers, right_tap)]
        ) & s[_scaled_tap(num_registers, gate_tap)]
    return observation


def apply_nlfsr_step(state: Sequence[int]) -> list[int]:
    s = _validate_state(state)
    return [*s[1:], feedback_bit(s)]


def solve_nlfsr(initial_state: Sequence[int], num_steps: int) -> tuple[list[list[int]], list[int]]:
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    state = _validate_state(initial_state)
    states = []
    observations = []
    for _ in range(num_steps):
        state = apply_nlfsr_step(state)
        states.append(list(state))
        observations.append(observe_state(state))
    return states, observations


def sample_nlfsr_example(
    num_steps: int,
    stoi: Dict[str, int],
    rng: random.Random,
    num_registers: int = DEFAULT_NUM_REGISTERS,
) -> tuple[list[int], list[int], list[int], list[int]]:
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    num_registers = _validate_num_registers(num_registers)

    initial_state = [rng.randrange(2) for _ in range(num_registers)]
    _, observations = solve_nlfsr(initial_state, num_steps)

    prompt = [
        stoi[STEPS_TOKEN],
        stoi[step_token(num_steps)],
        stoi[SEED_TOKEN],
        *(stoi[bit_token(bit)] for bit in initial_state),
    ]

    answer = [stoi[bit_token(observation)] for observation in observations]

    return prompt, answer, initial_state, observations


def build_nlfsr_batch(
    batch_size: int,
    num_steps: int,
    stoi: Dict[str, int],
    device=None,
    rng: random.Random | None = None,
    num_registers: int = DEFAULT_NUM_REGISTERS,
) -> SymbolicBatch:
    rng = rng or random.Random()
    rows = []
    for _ in range(batch_size):
        prompt, answer, _, _ = sample_nlfsr_example(
            num_steps,
            stoi,
            rng,
            num_registers=num_registers,
        )
        rows.append(make_sequence(prompt, answer, stoi))
    return build_batch_from_sequences(rows, pad_id=stoi[PAD_TOKEN], device=device)


__all__ = [
    "BIT_TOKENS",
    "DEFAULT_NUM_REGISTERS",
    "MIN_NUM_REGISTERS",
    "OBSERVATION_DIM",
    "apply_nlfsr_step",
    "bit_token",
    "build_nlfsr_batch",
    "build_nlfsr_vocab",
    "decode_ids",
    "feedback_bit",
    "observe_state",
    "required_block_size",
    "sample_nlfsr_example",
    "solve_nlfsr",
    "step_token",
]
