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


TASK_TOKEN = "TRUTH"
STATE_TOKEN = "<state>"
FINAL_TOKEN = "<final>"
ASSIGN_TOKEN = "="
SEMI_TOKEN = ";"
TRUE_TOKEN = "T"
FALSE_TOKEN = "F"
NOT_TOKEN = "NOT"
AND_TOKEN = "AND"
OR_TOKEN = "OR"
XOR_TOKEN = "XOR"
BINARY_OPS = (AND_TOKEN, OR_TOKEN, XOR_TOKEN)
SUPERVISION_MODES = ("final", "trace")


def validate_supervision(supervision: str):
    if supervision not in SUPERVISION_MODES:
        raise ValueError(f"supervision must be one of {SUPERVISION_MODES}")


def var_token(index: int) -> str:
    return f"x{index}"


def required_block_size(num_vars: int, supervision: str = "final") -> int:
    validate_supervision(supervision)
    if num_vars < 2:
        raise ValueError("num_vars must be at least 2")
    prompt_tokens = 1 + 4 + (num_vars - 1) * 6 + 2
    answer_len = 1 if supervision == "final" else 3 * num_vars + 2
    return 2 + prompt_tokens + answer_len


def build_truth_vocab(max_num_vars: int) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    if max_num_vars < 2:
        raise ValueError("max_num_vars must be at least 2")
    tokens = [
        PAD_TOKEN,
        BOS_TOKEN,
        SEP_TOKEN,
        EOS_TOKEN,
        TASK_TOKEN,
        STATE_TOKEN,
        FINAL_TOKEN,
        QUERY_TOKEN,
        ASSIGN_TOKEN,
        SEMI_TOKEN,
        TRUE_TOKEN,
        FALSE_TOKEN,
        NOT_TOKEN,
        AND_TOKEN,
        OR_TOKEN,
        XOR_TOKEN,
    ]
    tokens.extend(var_token(index) for index in range(max_num_vars))
    return build_vocab(tokens)


def bool_token(value: bool) -> str:
    return TRUE_TOKEN if value else FALSE_TOKEN


def eval_truth_program(definitions: Sequence[tuple[str, tuple]]) -> list[bool]:
    values: list[bool] = []
    for kind, args in definitions:
        if kind == "const":
            values.append(bool(args[0]))
        elif kind == NOT_TOKEN:
            values.append(not values[int(args[0])])
        elif kind in BINARY_OPS:
            left = values[int(args[0])]
            right = values[int(args[1])]
            if kind == AND_TOKEN:
                values.append(left and right)
            elif kind == OR_TOKEN:
                values.append(left or right)
            elif kind == XOR_TOKEN:
                values.append(left != right)
        else:
            raise ValueError(f"Unsupported truth definition: {kind}")
    return values


def sample_truth_example(
    num_vars: int,
    stoi: Dict[str, int],
    rng: random.Random,
    supervision: str = "final",
) -> tuple[list[int], list[int], list[tuple[str, tuple]], int, bool]:
    if num_vars < 2:
        raise ValueError("num_vars must be at least 2")
    validate_supervision(supervision)

    definitions: list[tuple[str, tuple]] = [("const", (rng.choice((False, True)),))]
    prompt = [
        stoi[TASK_TOKEN],
        stoi[var_token(0)],
        stoi[ASSIGN_TOKEN],
        stoi[bool_token(definitions[0][1][0])],
        stoi[SEMI_TOKEN],
    ]
    for index in range(1, num_vars):
        if rng.random() < 0.35:
            source = rng.randrange(index)
            definitions.append((NOT_TOKEN, (source,)))
            prompt.extend([
                stoi[var_token(index)],
                stoi[ASSIGN_TOKEN],
                stoi[NOT_TOKEN],
                stoi[var_token(source)],
                stoi[SEMI_TOKEN],
            ])
        else:
            left = rng.randrange(index)
            right = rng.randrange(index)
            op = rng.choice(BINARY_OPS)
            definitions.append((op, (left, right)))
            prompt.extend([
                stoi[var_token(index)],
                stoi[ASSIGN_TOKEN],
                stoi[var_token(left)],
                stoi[op],
                stoi[var_token(right)],
                stoi[SEMI_TOKEN],
            ])

    values = eval_truth_program(definitions)
    query_index = rng.randrange(num_vars)
    final = values[query_index]
    prompt.extend([stoi[QUERY_TOKEN], stoi[var_token(query_index)]])

    answer: list[int] = []
    if supervision == "trace":
        for index, value in enumerate(values):
            answer.extend([stoi[STATE_TOKEN], stoi[var_token(index)], stoi[bool_token(value)]])
        answer.append(stoi[FINAL_TOKEN])
    answer.append(stoi[bool_token(final)])
    return prompt, answer, definitions, query_index, final


def build_truth_batch(
    batch_size: int,
    num_vars: int,
    stoi: Dict[str, int],
    supervision: str = "final",
    device=None,
    rng: random.Random | None = None,
) -> SymbolicBatch:
    validate_supervision(supervision)
    rng = rng or random.Random()
    rows = []
    for _ in range(batch_size):
        prompt, answer, _, _, _ = sample_truth_example(num_vars, stoi, rng, supervision=supervision)
        rows.append(make_sequence(prompt, answer, stoi))
    return build_batch_from_sequences(rows, pad_id=stoi[PAD_TOKEN], device=device)


__all__ = [
    "build_truth_batch",
    "build_truth_vocab",
    "decode_ids",
    "eval_truth_program",
    "required_block_size",
    "sample_truth_example",
]
