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
    numeric_tokens,
    validate_supervision,
)


TASK_TOKEN = "WALK"
UP_TOKEN = "U"
DOWN_TOKEN = "D"
LEFT_TOKEN = "L"
RIGHT_TOKEN = "R"

DIRECTIONS = {
    UP_TOKEN: (0, 1),
    DOWN_TOKEN: (0, -1),
    LEFT_TOKEN: (-1, 0),
    RIGHT_TOKEN: (1, 0),
}


def required_block_size(num_steps: int, supervision: str = "final") -> int:
    validate_supervision(supervision)
    prompt_tokens = 1 + num_steps
    if supervision == "final":
        answer_len = 3  # <final> x y
    else:
        answer_len = 3 * num_steps + 3  # <state> x y after each move, then final.
    return 2 + prompt_tokens + answer_len


def build_walk_vocab(max_steps: int) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    tokens = [
        PAD_TOKEN,
        BOS_TOKEN,
        SEP_TOKEN,
        EOS_TOKEN,
        TASK_TOKEN,
        STATE_TOKEN,
        FINAL_TOKEN,
        UP_TOKEN,
        DOWN_TOKEN,
        LEFT_TOKEN,
        RIGHT_TOKEN,
    ]
    tokens.extend(numeric_tokens(-max_steps, max_steps))
    return build_vocab(tokens)


def solve_walk(moves: Sequence[str]) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    x = 0
    y = 0
    states = []
    for move in moves:
        dx, dy = DIRECTIONS[move]
        x += dx
        y += dy
        states.append((x, y))
    return states, (x, y)


def sample_walk_example(
    num_steps: int,
    stoi: Dict[str, int],
    rng: random.Random,
    supervision: str = "final",
) -> tuple[list[int], list[int], list[str], tuple[int, int]]:
    validate_supervision(supervision)
    moves = [rng.choice(tuple(DIRECTIONS)) for _ in range(num_steps)]
    states, final_state = solve_walk(moves)
    prompt = [stoi[TASK_TOKEN], *(stoi[move] for move in moves)]
    answer: list[int] = []
    if supervision == "trace":
        for x, y in states:
            answer.extend([stoi[STATE_TOKEN], stoi[str(x)], stoi[str(y)]])
    answer.extend([stoi[FINAL_TOKEN], stoi[str(final_state[0])], stoi[str(final_state[1])]])
    return prompt, answer, moves, final_state


def build_walk_batch(
    batch_size: int,
    num_steps: int,
    stoi: Dict[str, int],
    supervision: str = "final",
    device=None,
    rng: random.Random | None = None,
) -> SymbolicBatch:
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    validate_supervision(supervision)
    rng = rng or random.Random()
    rows = []
    for _ in range(batch_size):
        prompt, answer, _, _ = sample_walk_example(num_steps, stoi, rng, supervision=supervision)
        rows.append(make_sequence(prompt, answer, stoi))
    return build_batch_from_sequences(rows, pad_id=stoi[PAD_TOKEN], device=device)


__all__ = [
    "build_walk_batch",
    "build_walk_vocab",
    "decode_ids",
    "required_block_size",
    "sample_walk_example",
    "solve_walk",
]
