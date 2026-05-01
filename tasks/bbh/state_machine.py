"""State-machine task: read a transition table, start in a given state, apply an action sequence,
and predict the final state.

Example sequence: 
<bos> <states> s0 s1 s2 <alphabet> a0 a1 <table> s0 a0 s1 s0 a1 s2 s1 a0 s2 s1 a1 s0 s2 a0 s0 s2 a1 s1 <start> s0 <actions> a1 <sep> s2 <eos>
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


STATES_TOKEN = "<states>"
ALPHABET_TOKEN = "<alphabet>"
TABLE_TOKEN = "<table>"
START_TOKEN = "<start>"
ACTIONS_TOKEN = "<actions>"
DEFAULT_NUM_STATES = 4
DEFAULT_ALPHABET_SIZE = 2
LEVEL_ZERO_PART_WEIGHTS = (
    ("copy_no_action", 20),
    ("source_only_full_table", 40),
    ("action_lookup_fixed_source", 40),
)


def state_token(index: int) -> str:
    return f"s{index}"


def action_token(index: int) -> str:
    return f"a{index}"


def required_block_size(
    num_states: int,
    alphabet_size: int,
    num_steps: int,
) -> int:
    _validate_sizes(num_states, alphabet_size, num_steps)
    effective_steps = max(1, num_steps)
    prompt_tokens = 6 + num_states + alphabet_size + 3 * num_states * alphabet_size + effective_steps
    answer_len = 1
    return 2 + prompt_tokens + answer_len


def build_state_machine_vocab(
    num_states: int,
    alphabet_size: int,
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    _validate_sizes(num_states, alphabet_size, num_steps=1)
    tokens = [
        PAD_TOKEN,
        BOS_TOKEN,
        SEP_TOKEN,
        EOS_TOKEN,
        STATES_TOKEN,
        ALPHABET_TOKEN,
        TABLE_TOKEN,
        START_TOKEN,
        ACTIONS_TOKEN,
    ]
    tokens.extend(state_token(index) for index in range(num_states))
    tokens.extend(action_token(index) for index in range(alphabet_size))
    return build_vocab(tokens)


def solve_state_machine(
    transition_table: Sequence[Sequence[int]],
    start_state: int,
    actions: Sequence[int],
) -> tuple[list[int], int]:
    if not transition_table:
        raise ValueError("transition_table must not be empty")
    num_states = len(transition_table)
    alphabet_size = len(transition_table[0])
    if alphabet_size < 1:
        raise ValueError("transition rows must not be empty")
    if not 0 <= start_state < num_states:
        raise ValueError("start_state must index into transition_table")

    for row in transition_table:
        if len(row) != alphabet_size:
            raise ValueError("transition_table rows must share alphabet size")
        if any(target < 0 or target >= num_states for target in row):
            raise ValueError("transition target must index into transition_table")

    state = start_state
    trace: list[int] = []
    for action in actions:
        if not 0 <= action < alphabet_size:
            raise ValueError("action must index into transition row")
        state = transition_table[state][action]
        trace.append(state)
    return trace, state


def sample_transition_table(
    num_states: int,
    alphabet_size: int,
    rng: random.Random,
) -> list[list[int]]:
    """Sample a balanced DFA table with no one-step majority shortcut."""
    _validate_sizes(num_states, alphabet_size, num_steps=1)
    rows: list[list[int]] = [[] for _ in range(num_states)]
    for _ in range(alphabet_size):
        for _attempt in range(1_000):
            permutation = list(range(num_states))
            rng.shuffle(permutation)
            if all(target not in rows[source] for source, target in enumerate(permutation)):
                for source, target in enumerate(permutation):
                    rows[source].append(target)
                break
        else:
            return _sample_cyclic_transition_table(num_states, alphabet_size, rng)
    return rows


def _sample_cyclic_transition_table(
    num_states: int,
    alphabet_size: int,
    rng: random.Random,
) -> list[list[int]]:
    source_ranks = list(range(num_states))
    target_labels = list(range(num_states))
    offsets = rng.sample(range(num_states), k=alphabet_size)
    rng.shuffle(source_ranks)
    rng.shuffle(target_labels)
    rng.shuffle(offsets)
    return [
        [
            target_labels[(source_ranks[source] + offset) % num_states]
            for offset in offsets
        ]
        for source in range(num_states)
    ]


def _sample_source_only_transition_table(
    num_states: int,
    alphabet_size: int,
    rng: random.Random,
) -> list[list[int]]:
    targets = list(range(num_states))
    rng.shuffle(targets)
    return [[targets[source]] * alphabet_size for source in range(num_states)]


def _sample_fixed_source_action_lookup_table(
    num_states: int,
    alphabet_size: int,
    rng: random.Random,
) -> list[list[int]]:
    active_row = rng.sample(range(num_states), k=alphabet_size)
    rows = [list(active_row)]
    for _ in range(1, num_states):
        row = list(range(num_states))
        rng.shuffle(row)
        rows.append(row[:alphabet_size])
    return rows


def _append_transition_table(
    prompt: list[int],
    transition_table: Sequence[Sequence[int]],
    stoi: Dict[str, int],
    rng: random.Random,
    *,
    shuffle: bool = True,
) -> None:
    transition_triples = [
        (source, action)
        for source in range(len(transition_table))
        for action in range(len(transition_table[source]))
    ]
    if shuffle:
        rng.shuffle(transition_triples)
    for source, action in transition_triples:
        prompt.extend(
            [
                stoi[state_token(source)],
                stoi[action_token(action)],
                stoi[state_token(transition_table[source][action])],
            ]
        )


def sample_state_machine_example(
    num_states: int,
    alphabet_size: int,
    num_steps: int,
    stoi: Dict[str, int],
    rng: random.Random,
) -> tuple[list[int], list[int], list[list[int]], int, list[int], list[int], int]:
    _validate_sizes(num_states, alphabet_size, num_steps)
    if num_steps == 0:
        return _sample_level_zero_example(num_states, alphabet_size, stoi, rng)

    transition_table = sample_transition_table(num_states, alphabet_size, rng)
    start_state = rng.randrange(num_states)
    actions = [rng.randrange(alphabet_size) for _ in range(num_steps)]
    trace, final_state = solve_state_machine(transition_table, start_state, actions)

    prompt = [stoi[STATES_TOKEN]]
    prompt.extend(stoi[state_token(index)] for index in range(num_states))
    prompt.append(stoi[ALPHABET_TOKEN])
    prompt.extend(stoi[action_token(index)] for index in range(alphabet_size))
    prompt.append(stoi[TABLE_TOKEN])
    _append_transition_table(prompt, transition_table, stoi, rng)
    prompt.extend([stoi[START_TOKEN], stoi[state_token(start_state)], stoi[ACTIONS_TOKEN]])
    prompt.extend(stoi[action_token(action)] for action in actions)

    answer = [stoi[state_token(final_state)]]
    return prompt, answer, transition_table, start_state, actions, trace, final_state


def _sample_level_zero_example(
    num_states: int,
    alphabet_size: int,
    stoi: Dict[str, int],
    rng: random.Random,
) -> tuple[list[int], list[int], list[list[int]], int, list[int], list[int], int]:
    part = rng.choices(
        [name for name, _ in LEVEL_ZERO_PART_WEIGHTS],
        weights=[weight for _, weight in LEVEL_ZERO_PART_WEIGHTS],
        k=1,
    )[0]
    if part == "source_only_full_table":
        transition_table = _sample_source_only_transition_table(num_states, alphabet_size, rng)
        start_state = rng.randrange(num_states)
        actions = [rng.randrange(alphabet_size)]
    elif part == "action_lookup_fixed_source":
        transition_table = _sample_fixed_source_action_lookup_table(num_states, alphabet_size, rng)
        start_state = 0
        actions = [rng.randrange(alphabet_size)]
    else:
        transition_table = sample_transition_table(num_states, alphabet_size, rng)
        start_state = rng.randrange(num_states)
        actions = []

    prompt = [stoi[STATES_TOKEN]]
    prompt.extend(stoi[state_token(state)] for state in range(num_states))
    prompt.append(stoi[ALPHABET_TOKEN])
    prompt.extend(stoi[action_token(action)] for action in range(alphabet_size))
    prompt.append(stoi[TABLE_TOKEN])
    _append_transition_table(prompt, transition_table, stoi, rng, shuffle=False)

    if not actions:
        trace: list[int] = []
        final_state = start_state
    else:
        trace, final_state = solve_state_machine(transition_table, start_state, actions)
    prompt.extend([stoi[START_TOKEN], stoi[state_token(start_state)], stoi[ACTIONS_TOKEN]])
    prompt.extend(stoi[action_token(action)] for action in actions)
    answer = [stoi[state_token(final_state)]]
    return prompt, answer, transition_table, start_state, actions, trace, final_state


def build_state_machine_batch(
    batch_size: int,
    num_states: int,
    alphabet_size: int,
    num_steps: int,
    stoi: Dict[str, int],
    device=None,
    rng: random.Random | None = None,
) -> SymbolicBatch:
    _validate_sizes(num_states, alphabet_size, num_steps)
    rng = rng or random.Random()
    rows = []
    for _ in range(batch_size):
        prompt, answer, _, _, _, _, _ = sample_state_machine_example(
            num_states,
            alphabet_size,
            num_steps,
            stoi,
            rng,
        )
        rows.append(make_sequence(prompt, answer, stoi))
    return build_batch_from_sequences(rows, pad_id=stoi[PAD_TOKEN], device=device)


def _validate_sizes(num_states: int, alphabet_size: int, num_steps: int):
    if num_states < 2:
        raise ValueError("num_states must be at least 2")
    if alphabet_size < 1:
        raise ValueError("alphabet_size must be at least 1")
    if alphabet_size > num_states:
        raise ValueError("alphabet_size must be no larger than num_states")
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")


__all__ = [
    "ACTIONS_TOKEN",
    "ALPHABET_TOKEN",
    "START_TOKEN",
    "STATES_TOKEN",
    "TABLE_TOKEN",
    "action_token",
    "build_state_machine_batch",
    "build_state_machine_vocab",
    "decode_ids",
    "required_block_size",
    "sample_state_machine_example",
    "sample_transition_table",
    "solve_state_machine",
    "state_token",
]
