from __future__ import annotations

import random

import pytest

from tasks.bbh import permutation, pointer_chasing, state_machine, tracking
from tasks.trace import othello, random_graph_walk


def test_bbh_task_solvers_match_sampled_answers():
    rng = random.Random(3)

    _vocab, stoi, _ = pointer_chasing.build_pointer_chasing_vocab(5)
    _prompt, _answer, pointers, start, final = pointer_chasing.sample_pointer_chasing_example(5, 4, stoi, rng)
    assert pointer_chasing.solve_pointer_chasing(pointers, start, 4)[1] == final

    _vocab, stoi, _ = permutation.build_permutation_vocab(4)
    _prompt, _answer, swaps, final_state = permutation.sample_permutation_example(4, 5, stoi, rng)
    assert permutation.solve_permutation(4, swaps) == final_state

    _vocab, stoi, _ = tracking.build_tracking_vocab(4)
    _prompt, _answer, ops, query, final_object = tracking.sample_tracking_example(4, 5, stoi, rng)
    assert tracking.solve_tracking(4, ops)[1][query] == final_object

    _vocab, stoi, _ = state_machine.build_state_machine_vocab(4, 2)
    sample = state_machine.sample_state_machine_example(4, 2, 5, stoi, rng)
    _prompt, _answer, table, start, actions, _trace, final = sample
    assert state_machine.solve_state_machine(table, start, actions)[1] == final


def test_random_graph_walk_prompt_parsing_and_legality():
    rng = random.Random(11)
    _vocab, stoi, _ = random_graph_walk.build_random_graph_walk_vocab(5, 4)
    prompt, _answer, table, start, actions, _trace, _final = random_graph_walk.sample_random_graph_walk_example(
        5, 4, 8, stoi, rng
    )
    parsed_table, parsed_start = random_graph_walk.parse_prompt_metadata(prompt, num_states=5, label_pool_size=4)
    assert [sorted(row) for row in parsed_table] == [sorted(row) for row in table]
    assert parsed_start == start
    action_ids = [stoi[random_graph_walk.label_token(action)] for action in actions]
    assert random_graph_walk.legal_prefix_length(
        prompt, action_ids, num_states=5, label_pool_size=4
    ) == (8, True)


def test_othello_generated_games_are_legal_and_dataset_cache_is_deterministic(tmp_path):
    for seed in [0, 1, 2, 3, 309]:
        trace = othello.random_game_trace64(seed=seed)
        ids = [square + othello.MOVE_TOKEN_OFFSET for square in trace]
        assert othello.legal_prefix_length(ids) == (len(trace), True)
        cut = len(ids) // 2
        assert othello.legal_prefix_length(
            ids[cut:],
            prefix_move_token_ids=ids[:cut],
        ) == (len(ids) - cut, True)
        assert ids[cut] in othello.legal_move_token_ids_after_prefix(ids[:cut])
        padded = ids + [0] * (othello.MAX_MOVES - len(ids))
        assert othello.legal_prefix_length(padded) == (othello.MAX_MOVES, True)

    with pytest.raises(ValueError, match="illegal move"):
        othello.legal_move_token_ids_after_prefix([othello.MOVE_TOKEN_OFFSET])

    kwargs = dict(
        othello_data_dir=str(tmp_path),
        othello_train_games=8,
        othello_val_games=4,
        othello_dataset_seed=19,
    )
    othello.ensure_othello_datasets(**kwargs)
    first = othello.load_othello_dataset(split="train", **kwargs)
    trace_a = first.sample_trace(random.Random(7))
    othello._DATASET_CACHE.clear()
    second = othello.load_othello_dataset(split="train", **kwargs)
    trace_b = second.sample_trace(random.Random(7))
    assert trace_a == trace_b


def test_othello_generation_is_partition_invariant():
    seeds = othello.np.random.SeedSequence(123).generate_state(12, dtype=othello.np.uint64)
    whole = othello._generate_trace_dataset_arrays_from_seeds(seeds)
    left = othello._generate_trace_dataset_arrays_from_seeds(seeds[:5])
    right = othello._generate_trace_dataset_arrays_from_seeds(seeds[5:])
    partitioned_traces = othello.np.concatenate((left[0], right[0]), axis=0)
    partitioned_lengths = othello.np.concatenate((left[1], right[1]), axis=0)
    assert othello.np.array_equal(whole[0], partitioned_traces)
    assert othello.np.array_equal(whole[1], partitioned_lengths)
