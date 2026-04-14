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


TASK_TOKEN = "BOOL_RPN"
TRUE_TOKEN = "T"
FALSE_TOKEN = "F"
NOT_TOKEN = "NOT"
AND_TOKEN = "AND"
OR_TOKEN = "OR"
XOR_TOKEN = "XOR"
BINARY_OPS = (AND_TOKEN, OR_TOKEN, XOR_TOKEN)


def required_block_size(num_binary_ops: int, supervision: str = "final") -> int:
    validate_supervision(supervision)
    if num_binary_ops < 1:
        raise ValueError("num_binary_ops must be at least 1")
    max_expr_len = 4 * num_binary_ops + 2  # values + binary ops + possible NOT after each value/op.
    prompt_tokens = 1 + max_expr_len
    if supervision == "final":
        answer_len = 2
    else:
        answer_len = max_expr_len * (num_binary_ops + 2) + 2
    return 2 + prompt_tokens + answer_len


def build_boolean_rpn_vocab() -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    tokens = [
        PAD_TOKEN,
        BOS_TOKEN,
        SEP_TOKEN,
        EOS_TOKEN,
        TASK_TOKEN,
        STACK_TOKEN,
        FINAL_TOKEN,
        EMPTY_TOKEN,
        TRUE_TOKEN,
        FALSE_TOKEN,
        NOT_TOKEN,
        AND_TOKEN,
        OR_TOKEN,
        XOR_TOKEN,
    ]
    return build_vocab(tokens)


def eval_bool_rpn(tokens: Sequence[str]) -> tuple[list[list[bool]], bool]:
    stack: list[bool] = []
    trace: list[list[bool]] = []
    for token in tokens:
        if token == TRUE_TOKEN:
            stack.append(True)
        elif token == FALSE_TOKEN:
            stack.append(False)
        elif token == NOT_TOKEN:
            if not stack:
                raise ValueError("NOT requires one stack value")
            stack[-1] = not stack[-1]
        elif token in BINARY_OPS:
            if len(stack) < 2:
                raise ValueError(f"{token} requires two stack values")
            right = stack.pop()
            left = stack.pop()
            if token == AND_TOKEN:
                stack.append(left and right)
            elif token == OR_TOKEN:
                stack.append(left or right)
            elif token == XOR_TOKEN:
                stack.append(left != right)
        else:
            raise ValueError(f"Unsupported Boolean RPN token: {token}")
        trace.append(list(stack))
    if len(stack) != 1:
        raise ValueError("Boolean RPN expression must leave exactly one stack value")
    return trace, stack[0]


def bool_token(value: bool) -> str:
    return TRUE_TOKEN if value else FALSE_TOKEN


def sample_boolean_rpn_example(
    num_binary_ops: int,
    stoi: Dict[str, int],
    rng: random.Random,
    supervision: str = "final",
    not_prob: float = 0.25,
) -> tuple[list[int], list[int], list[str], bool]:
    if num_binary_ops < 1:
        raise ValueError("num_binary_ops must be at least 1")
    if not 0.0 <= not_prob <= 1.0:
        raise ValueError("not_prob must be in [0, 1]")
    validate_supervision(supervision)

    expr: list[str] = []
    expr.append(rng.choice((TRUE_TOKEN, FALSE_TOKEN)))
    if rng.random() < not_prob:
        expr.append(NOT_TOKEN)
    for _ in range(num_binary_ops):
        expr.append(rng.choice((TRUE_TOKEN, FALSE_TOKEN)))
        if rng.random() < not_prob:
            expr.append(NOT_TOKEN)
        expr.append(rng.choice(BINARY_OPS))
        if rng.random() < not_prob:
            expr.append(NOT_TOKEN)

    trace, result = eval_bool_rpn(expr)
    prompt = [stoi[TASK_TOKEN], *(stoi[token] for token in expr)]
    answer: list[int] = []
    if supervision == "trace":
        for state in trace:
            answer.append(stoi[STACK_TOKEN])
            if state:
                answer.extend(stoi[bool_token(value)] for value in state)
            else:
                answer.append(stoi[EMPTY_TOKEN])
    answer.extend([stoi[FINAL_TOKEN], stoi[bool_token(result)]])
    return prompt, answer, expr, result


def build_boolean_rpn_batch(
    batch_size: int,
    num_binary_ops: int,
    stoi: Dict[str, int],
    supervision: str = "final",
    device=None,
    rng: random.Random | None = None,
) -> SymbolicBatch:
    validate_supervision(supervision)
    rng = rng or random.Random()
    rows = []
    for _ in range(batch_size):
        prompt, answer, _, _ = sample_boolean_rpn_example(num_binary_ops, stoi, rng, supervision=supervision)
        rows.append(make_sequence(prompt, answer, stoi))
    return build_batch_from_sequences(rows, pad_id=stoi[PAD_TOKEN], device=device)


__all__ = [
    "build_boolean_rpn_batch",
    "build_boolean_rpn_vocab",
    "decode_ids",
    "eval_bool_rpn",
    "required_block_size",
    "sample_boolean_rpn_example",
]
