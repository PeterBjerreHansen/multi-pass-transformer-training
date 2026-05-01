"""Random-graph-walk task: each state has a small set of outgoing labeled edges,
so the model must track the current state to know which actions are legal and where they lead.

Example sequence:
<bos> <states> s0 s1 s2 <alphabet> t0 t1 t2 <table> s0 t0 s1 s0 t1 s2 s1 t1 s0 s1 t2 s2 s2 t0 s1 s2 t2 s0 <start> s0 <sep> t0 t2 <eos>
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
from tasks.trace.common import format_legal_generation_metrics, trace_generation_metrics


STATES_TOKEN = "<states>"
ALPHABET_TOKEN = "<alphabet>"
TABLE_TOKEN = "<table>"
START_TOKEN = "<start>"
OUT_DEGREE = 2
DEFAULT_NUM_STATES = 6
DEFAULT_LABEL_POOL_SIZE = 4
TRANSITION_LABELS = (
    "t0",
    "t1",
    "t2",
    "t3",
    "t4",
    "t5",
    "t6",
    "t7",
    "t8",
    "t9",
)


def state_token(index: int) -> str:
    return f"s{index}"


def label_token(index: int) -> str:
    return TRANSITION_LABELS[index]


def required_block_size(num_states: int, label_pool_size: int, num_steps: int) -> int:
    _validate_sizes(num_states, label_pool_size, num_steps)
    prompt_tokens = 5 + num_states + label_pool_size + 3 * num_states * OUT_DEGREE
    answer_len = num_steps
    return 2 + prompt_tokens + answer_len


def build_random_graph_walk_vocab(
    num_states: int,
    label_pool_size: int,
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    _validate_sizes(num_states, label_pool_size, num_steps=1)
    tokens = [
        PAD_TOKEN,
        BOS_TOKEN,
        SEP_TOKEN,
        EOS_TOKEN,
        STATES_TOKEN,
        ALPHABET_TOKEN,
        TABLE_TOKEN,
        START_TOKEN,
    ]
    tokens.extend(state_token(index) for index in range(num_states))
    tokens.extend(label_token(index) for index in range(label_pool_size))
    return build_vocab(tokens)


def solve_random_graph_walk(
    transition_table: Sequence[Sequence[tuple[int, int]]],
    start_state: int,
    actions: Sequence[int],
) -> tuple[list[int], int]:
    if not transition_table:
        raise ValueError("transition_table must not be empty")
    num_states = len(transition_table)
    if not 0 <= start_state < num_states:
        raise ValueError("start_state must index into transition_table")
    for row in transition_table:
        if not row:
            raise ValueError("transition rows must not be empty")
        seen_labels = set()
        for label, target in row:
            if label in seen_labels:
                raise ValueError("state rows must not reuse action labels")
            if not 0 <= target < num_states:
                raise ValueError("transition target must index into transition_table")
            seen_labels.add(label)

    state = start_state
    trace: list[int] = []
    for action in actions:
        state = _lookup_target(transition_table, state, action)
        trace.append(state)
    return trace, state


def sample_random_graph_walk_table(
    num_states: int,
    label_pool_size: int,
    rng: random.Random,
) -> list[list[tuple[int, int]]]:
    _validate_sizes(num_states, label_pool_size, num_steps=1)

    source_ranks = list(range(num_states))
    label_indices = list(range(label_pool_size))
    rng.shuffle(source_ranks)
    rng.shuffle(label_indices)

    targets_by_slot: list[list[int]] = [[] for _ in range(OUT_DEGREE)]
    for slot in range(OUT_DEGREE):
        for _attempt in range(1_000):
            permutation = list(range(num_states))
            rng.shuffle(permutation)
            if all(
                permutation[source] != targets_by_slot[other_slot][source]
                for other_slot in range(slot)
                for source in range(num_states)
            ):
                targets_by_slot[slot] = permutation
                break
        else:
            raise RuntimeError("failed to sample non-degenerate target permutations")

    table: list[list[tuple[int, int]]] = []
    for source in range(num_states):
        labels = [
            label_indices[(source_ranks[source] + offset) % label_pool_size]
            for offset in range(OUT_DEGREE)
        ]
        table.append(
            [
                (labels[slot], targets_by_slot[slot][source])
                for slot in range(OUT_DEGREE)
            ]
        )
    return table


def sample_random_graph_walk_example(
    num_states: int,
    label_pool_size: int,
    num_steps: int,
    stoi: Dict[str, int],
    rng: random.Random,
) -> tuple[list[int], list[int], list[list[tuple[int, int]]], int, list[int], list[int], int]:
    _validate_sizes(num_states, label_pool_size, num_steps)

    transition_table = sample_random_graph_walk_table(num_states, label_pool_size, rng)
    start_state = rng.randrange(num_states)
    actions, trace, final_state = _sample_legal_walk(transition_table, start_state, num_steps, rng)

    prompt = [stoi[STATES_TOKEN]]
    prompt.extend(stoi[state_token(index)] for index in range(num_states))
    prompt.append(stoi[ALPHABET_TOKEN])
    prompt.extend(stoi[label_token(index)] for index in range(label_pool_size))
    prompt.append(stoi[TABLE_TOKEN])
    _append_transition_table(prompt, transition_table, stoi, rng)
    prompt.extend([stoi[START_TOKEN], stoi[state_token(start_state)]])

    answer = [stoi[label_token(action)] for action in actions]
    return prompt, answer, transition_table, start_state, actions, trace, final_state


def build_random_graph_walk_batch(
    batch_size: int,
    num_states: int,
    label_pool_size: int,
    num_steps: int,
    stoi: Dict[str, int],
    device=None,
    rng: random.Random | None = None,
) -> SymbolicBatch:
    _validate_sizes(num_states, label_pool_size, num_steps)
    rng = rng or random.Random()
    rows = []
    for _ in range(batch_size):
        prompt, answer, _, _, _, _, _ = sample_random_graph_walk_example(
            num_states,
            label_pool_size,
            num_steps,
            stoi,
            rng,
        )
        rows.append(make_sequence(prompt, answer, stoi))
    return build_batch_from_sequences(rows, pad_id=stoi[PAD_TOKEN], device=device)


def _append_transition_table(
    prompt: list[int],
    transition_table: Sequence[Sequence[tuple[int, int]]],
    stoi: Dict[str, int],
    rng: random.Random,
) -> None:
    triples = [
        (source, label, target)
        for source, edges in enumerate(transition_table)
        for label, target in edges
    ]
    rng.shuffle(triples)
    for source, label, target in triples:
        prompt.extend(
            [
                stoi[state_token(source)],
                stoi[label_token(label)],
                stoi[state_token(target)],
            ]
        )


def _lookup_target(
    transition_table: Sequence[Sequence[tuple[int, int]]],
    state: int,
    action: int,
) -> int:
    for label, target in transition_table[state]:
        if label == action:
            return target
    raise ValueError("action must be legal for the current state")


def _sample_legal_walk(
    transition_table: Sequence[Sequence[tuple[int, int]]],
    start_state: int,
    num_steps: int,
    rng: random.Random,
) -> tuple[list[int], list[int], int]:
    state = start_state
    actions: list[int] = []
    trace: list[int] = []
    for _ in range(num_steps):
        label, target = rng.choice(transition_table[state])
        actions.append(label)
        state = target
        trace.append(state)
    return actions, trace, state


def random_graph_walk_generation_metrics(
    model,
    batch,
    args,
    *,
    inference_mode: str | None = None,
    num_states: int,
    label_pool_size: int,
    **_unused,
) -> dict[str, float | None]:
    return trace_generation_metrics(
        model,
        batch,
        args,
        legality_check=lambda prompt_tokens, generated_tokens: legal_prefix_length(
            prompt_tokens,
            generated_tokens,
            num_states=num_states,
            label_pool_size=label_pool_size,
        ),
        inference_mode=inference_mode,
    )


def format_random_graph_walk_eval_metrics(metrics: dict[str, float]) -> str:
    return format_legal_generation_metrics(metrics)


def legal_prefix_length(
    prompt_tokens: Sequence[int],
    label_token_ids: Sequence[int],
    *,
    num_states: int,
    label_pool_size: int,
) -> tuple[int, bool]:
    transition_table, start_state = parse_prompt_metadata(
        prompt_tokens,
        num_states=num_states,
        label_pool_size=label_pool_size,
    )
    state = start_state
    legal_steps = 0
    for token_id in label_token_ids:
        action = token_id_to_label(token_id, num_states=num_states, label_pool_size=label_pool_size)
        if action is None:
            return legal_steps, False
        try:
            state = _lookup_target(transition_table, state, action)
        except ValueError:
            return legal_steps, False
        legal_steps += 1
    return legal_steps, True


def parse_prompt_metadata(
    prompt_tokens: Sequence[int],
    *,
    num_states: int,
    label_pool_size: int,
) -> tuple[list[list[tuple[int, int]]], int]:
    expected_len = 5 + num_states + label_pool_size + 3 * num_states * OUT_DEGREE
    if len(prompt_tokens) != expected_len:
        raise ValueError("prompt_tokens has unexpected length")
    if prompt_tokens[0] != _states_token_id():
        raise ValueError("prompt must begin with <states>")
    if prompt_tokens[1 + num_states] != _alphabet_token_id():
        raise ValueError("prompt must contain <alphabet> after the state list")
    if prompt_tokens[2 + num_states + label_pool_size] != _table_token_id():
        raise ValueError("prompt must contain <table> before the transition triples")

    table_start = 3 + num_states + label_pool_size
    table_end = table_start + 3 * num_states * OUT_DEGREE
    if prompt_tokens[table_end] != _start_token_id():
        raise ValueError("prompt must contain <start> before the start state")
    start_state = token_id_to_state(prompt_tokens[table_end + 1], num_states=num_states)
    if start_state is None:
        raise ValueError("prompt start-state token is invalid")

    transition_table: list[list[tuple[int, int]]] = [[] for _ in range(num_states)]
    for offset in range(table_start, table_end, 3):
        source = token_id_to_state(prompt_tokens[offset], num_states=num_states)
        label = token_id_to_label(prompt_tokens[offset + 1], num_states=num_states, label_pool_size=label_pool_size)
        target = token_id_to_state(prompt_tokens[offset + 2], num_states=num_states)
        if source is None or label is None or target is None:
            raise ValueError("prompt transition triple contains invalid token ids")
        transition_table[source].append((label, target))

    for row in transition_table:
        if len(row) != OUT_DEGREE:
            raise ValueError("prompt transition table is incomplete")
    return transition_table, start_state


def token_id_to_state(token_id: int, *, num_states: int) -> int | None:
    state_index = int(token_id) - _state_token_start()
    if 0 <= state_index < num_states:
        return state_index
    return None


def token_id_to_label(token_id: int, *, num_states: int, label_pool_size: int) -> int | None:
    label_index = int(token_id) - _label_token_start(num_states)
    if 0 <= label_index < label_pool_size:
        return label_index
    return None


def _states_token_id() -> int:
    return 4


def _alphabet_token_id() -> int:
    return 5


def _table_token_id() -> int:
    return 6


def _start_token_id() -> int:
    return 7


def _state_token_start() -> int:
    return 8


def _label_token_start(num_states: int) -> int:
    return _state_token_start() + num_states


def _validate_sizes(num_states: int, label_pool_size: int, num_steps: int):
    if num_states < 2:
        raise ValueError("num_states must be at least 2")
    if label_pool_size < OUT_DEGREE + 1:
        raise ValueError("label_pool_size must be at least 3")
    if label_pool_size > len(TRANSITION_LABELS):
        raise ValueError(f"label_pool_size must be <= {len(TRANSITION_LABELS)}")
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")


__all__ = [
    "ALPHABET_TOKEN",
    "DEFAULT_LABEL_POOL_SIZE",
    "DEFAULT_NUM_STATES",
    "OUT_DEGREE",
    "START_TOKEN",
    "STATES_TOKEN",
    "TABLE_TOKEN",
    "build_random_graph_walk_batch",
    "build_random_graph_walk_vocab",
    "decode_ids",
    "format_random_graph_walk_eval_metrics",
    "label_token",
    "legal_prefix_length",
    "parse_prompt_metadata",
    "random_graph_walk_generation_metrics",
    "required_block_size",
    "sample_random_graph_walk_example",
    "sample_random_graph_walk_table",
    "solve_random_graph_walk",
    "state_token",
]
