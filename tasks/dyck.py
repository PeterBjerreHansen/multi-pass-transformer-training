from typing import Dict, List, Sequence, Tuple

import random

from tasks.bbh_symbolic_common import (
    BOS_TOKEN,
    EMPTY_TOKEN,
    EOS_TOKEN,
    FINAL_TOKEN,
    PAD_TOKEN,
    SEP_TOKEN,
    STACK_TOKEN,
    SymbolicBatch,
    build_batch_from_sequences,
    build_vocab,
    decode_ids,
    make_sequence,
    validate_supervision,
)


TASK_TOKEN = "DYCK"
OPEN_TOKENS = ("(", "[", "{", "<")
CLOSE_TOKENS = (")", "]", "}", ">")
CLOSE_FOR_OPEN = dict(zip(OPEN_TOKENS, CLOSE_TOKENS))


def required_block_size(prefix_length: int, max_depth: int, supervision: str = "final") -> int:
    validate_supervision(supervision)
    prompt_tokens = 1 + prefix_length
    if supervision == "final":
        answer_len = 1 + max_depth  # <final> plus closing suffix.
    else:
        answer_len = prefix_length * (2 + max_depth) + 1 + max_depth
    return 2 + prompt_tokens + answer_len


def build_dyck_vocab(max_bracket_types: int = 4) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    if not 1 <= max_bracket_types <= len(OPEN_TOKENS):
        raise ValueError(f"max_bracket_types must be in [1, {len(OPEN_TOKENS)}]")
    tokens = [
        PAD_TOKEN,
        BOS_TOKEN,
        SEP_TOKEN,
        EOS_TOKEN,
        TASK_TOKEN,
        STACK_TOKEN,
        FINAL_TOKEN,
        EMPTY_TOKEN,
    ]
    tokens.extend(OPEN_TOKENS[:max_bracket_types])
    tokens.extend(CLOSE_TOKENS[:max_bracket_types])
    return build_vocab(tokens)


def solve_dyck_prefix(prefix: Sequence[str]) -> tuple[list[list[str]], list[str]]:
    stack: list[str] = []
    stack_trace: list[list[str]] = []
    for token in prefix:
        if token in OPEN_TOKENS:
            stack.append(token)
        elif token in CLOSE_TOKENS:
            if not stack:
                raise ValueError("Invalid Dyck prefix: closing token with empty stack")
            opener = stack.pop()
            if CLOSE_FOR_OPEN[opener] != token:
                raise ValueError("Invalid Dyck prefix: mismatched closing token")
        else:
            raise ValueError(f"Unsupported Dyck token: {token}")
        stack_trace.append(list(stack))
    completion = [CLOSE_FOR_OPEN[opener] for opener in reversed(stack)]
    return stack_trace, completion


def sample_dyck_example(
    prefix_length: int,
    max_depth: int,
    bracket_types: int,
    stoi: Dict[str, int],
    rng: random.Random,
    supervision: str = "final",
) -> tuple[list[int], list[int], list[str], list[str]]:
    if prefix_length < 1:
        raise ValueError("prefix_length must be at least 1")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    if not 1 <= bracket_types <= len(OPEN_TOKENS):
        raise ValueError(f"bracket_types must be in [1, {len(OPEN_TOKENS)}]")
    validate_supervision(supervision)

    prefix: list[str] = []
    stack: list[str] = []
    opens = OPEN_TOKENS[:bracket_types]
    closes = CLOSE_TOKENS[:bracket_types]
    for step in range(prefix_length):
        can_open = len(stack) < max_depth
        can_close = bool(stack)
        must_finish_nonempty = step == prefix_length - 1 and not stack
        if must_finish_nonempty or not can_close or (can_open and rng.random() < 0.6):
            token = rng.choice(opens)
            stack.append(token)
            prefix.append(token)
        else:
            opener = stack.pop()
            prefix.append(CLOSE_FOR_OPEN[opener])

    stack_trace, completion = solve_dyck_prefix(prefix)
    prompt = [stoi[TASK_TOKEN], *(stoi[token] for token in prefix)]
    answer: list[int] = []
    if supervision == "trace":
        for state in stack_trace:
            answer.append(stoi[STACK_TOKEN])
            if state:
                answer.extend(stoi[token] for token in state)
            else:
                answer.append(stoi[EMPTY_TOKEN])
    answer.append(stoi[FINAL_TOKEN])
    answer.extend(stoi[token] for token in completion)
    return prompt, answer, prefix, completion


def build_dyck_batch(
    batch_size: int,
    prefix_length: int,
    max_depth: int,
    bracket_types: int,
    stoi: Dict[str, int],
    supervision: str = "final",
    device=None,
    rng: random.Random | None = None,
) -> SymbolicBatch:
    validate_supervision(supervision)
    rng = rng or random.Random()
    rows = []
    for _ in range(batch_size):
        prompt, answer, _, _ = sample_dyck_example(
            prefix_length,
            max_depth,
            bracket_types,
            stoi,
            rng,
            supervision=supervision,
        )
        rows.append(make_sequence(prompt, answer, stoi))
    return build_batch_from_sequences(rows, pad_id=stoi[PAD_TOKEN], device=device)


__all__ = [
    "build_dyck_batch",
    "build_dyck_vocab",
    "decode_ids",
    "required_block_size",
    "sample_dyck_example",
    "solve_dyck_prefix",
]
