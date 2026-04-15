from typing import Dict, List, Tuple

import random

from tasks.bbh_symbolic_common import (
    BOS_TOKEN,
    EOS_TOKEN,
    FINAL_TOKEN,
    PAD_TOKEN,
    SEP_TOKEN,
    TRACE_TOKEN,
    SymbolicBatch,
    build_batch_from_sequences,
    build_vocab,
    decode_ids,
    make_sequence,
    validate_supervision,
)


SWAP_TOKEN = "swap"


def required_block_size(num_objects: int, num_swaps: int, supervision: str = "final") -> int:
    validate_supervision(supervision)
    # Full sequence is:
    # BOS, initial permutation, swap p_i p_j repeated num_swaps times, SEP, final permutation, EOS.
    # The autoregressive input drops the final token.
    final_only_size = 2 * num_objects + 3 * num_swaps + 2
    if supervision == "final":
        return final_only_size
    return final_only_size + num_swaps * (num_objects + 1) + 1


def build_permutation_vocab(num_objects: int) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    if num_objects < 2:
        raise ValueError("num_objects must be at least 2")
    tokens = [PAD_TOKEN, BOS_TOKEN, SEP_TOKEN, EOS_TOKEN, SWAP_TOKEN, TRACE_TOKEN, FINAL_TOKEN]
    tokens.extend(f"o{i}" for i in range(num_objects))
    tokens.extend(f"p{i}" for i in range(num_objects))
    return build_vocab(tokens)


def build_permutation_batch(
    batch_size: int,
    num_objects: int,
    num_swaps: int,
    stoi: Dict[str, int],
    device=None,
    rng: random.Random | None = None,
    supervision: str = "final",
) -> SymbolicBatch:
    validate_supervision(supervision)
    rng = rng or random.Random()
    initial_state = [stoi[f"o{i}"] for i in range(num_objects)]

    rows = []
    for _ in range(batch_size):
        final_state, swap_tokens, trace_states = _sample_permutation_example(num_objects, num_swaps, stoi, rng)
        if supervision == "final":
            answer_tokens = final_state
        else:
            answer_tokens = []
            for state in trace_states:
                answer_tokens.extend([stoi[TRACE_TOKEN], *state])
            answer_tokens.extend([stoi[FINAL_TOKEN], *final_state])
        rows.append(make_sequence([*initial_state, *swap_tokens], answer_tokens, stoi))

    return build_batch_from_sequences(rows, pad_id=stoi[PAD_TOKEN], device=device)


def _sample_permutation_example(
    num_objects: int,
    num_swaps: int,
    stoi: Dict[str, int],
    rng: random.Random,
) -> Tuple[List[int], List[int], List[List[int]]]:
    state = list(range(num_objects))
    swap_tokens: List[int] = []
    trace_states: List[List[int]] = []

    for _ in range(num_swaps):
        i, j = rng.sample(range(num_objects), k=2)
        state[i], state[j] = state[j], state[i]
        swap_tokens.extend([stoi[SWAP_TOKEN], stoi[f"p{i}"], stoi[f"p{j}"]])
        trace_states.append([stoi[f"o{obj}"] for obj in state])

    final_state = [stoi[f"o{obj}"] for obj in state]
    return final_state, swap_tokens, trace_states
