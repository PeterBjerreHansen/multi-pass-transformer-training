import json
import random

import numpy as np
import pytest
import torch

from tasks.bbh import permutation
from tasks.bbh import pointer_chasing
from tasks.bbh import state_machine
from tasks.bbh import tracking
from tasks.trace import random_graph_walk
from tasks.trace import othello


def _decoded_answer(batch, row, itos):
    prompt_len = int(batch.prompt_lengths[row].item())
    output_len = int(batch.output_lengths[row].item())
    suffix = batch.targets[row, prompt_len - 1 : prompt_len - 1 + output_len]
    return [itos[int(token)] for token in suffix.tolist()]


def _assert_prompt_mask_and_metric(batch):
    for row in range(batch.idx.size(0)):
        prompt_len = int(batch.prompt_lengths[row].item())
        output_len = int(batch.output_lengths[row].item())
        assert torch.equal(batch.targets[row, : prompt_len - 1], torch.full((prompt_len - 1,), -1))
        assert not batch.metric_mask[row, : prompt_len - 1].any()
        assert batch.metric_mask[row, prompt_len - 1 : prompt_len - 1 + output_len].all()


def test_permutation_generation_and_batching():
    _, stoi, itos = permutation.build_permutation_vocab(num_objects=4)
    swaps = [(0, 2), (1, 3), (0, 1)]
    final_state = permutation.solve_permutation(4, swaps)
    assert final_state == [3, 2, 0, 1]

    with pytest.raises(ValueError, match="distinct"):
        permutation.solve_permutation(4, [(0, 0)])
    with pytest.raises(ValueError, match="index into"):
        permutation.solve_permutation(4, [(0, 4)])

    rng = random.Random(8)
    prompt, answer, sampled_swaps, sampled_final = permutation.sample_permutation_example(
        num_objects=4,
        num_swaps=5,
        stoi=stoi,
        rng=rng,
    )
    assert permutation.solve_permutation(4, sampled_swaps) == sampled_final
    assert [itos[token] for token in prompt[:4]] == [permutation.obj_token(i) for i in range(4)]
    assert [itos[token] for token in answer] == [permutation.obj_token(obj) for obj in sampled_final]

    batch = permutation.build_permutation_batch(
        batch_size=3,
        num_objects=4,
        num_swaps=5,
        stoi=stoi,
        rng=random.Random(7),
    )
    _assert_prompt_mask_and_metric(batch)
    assert batch.idx.shape == (3, permutation.required_block_size(4, 5))

    for row in range(batch.idx.size(0)):
        prompt_len = int(batch.prompt_lengths[row].item())
        output_len = int(batch.output_lengths[row].item())
        ids = batch.idx[row].tolist()
        targets = batch.targets[row]

        state = list(range(4))
        pos = 4 + 1
        while pos < prompt_len - 1:
            assert ids[pos] == stoi[permutation.SWAP_TOKEN]
            i = int(itos[ids[pos + 1]][1:])
            j = int(itos[ids[pos + 2]][1:])
            state[i], state[j] = state[j], state[i]
            pos += 3

        expected_suffix = [*(stoi[permutation.obj_token(obj)] for obj in state), stoi[permutation.EOS_TOKEN]]
        actual_suffix = targets[prompt_len - 1 : prompt_len - 1 + output_len].tolist()
        assert actual_suffix == expected_suffix


def test_pointer_chasing_generation_and_batching():
    _, stoi, itos = pointer_chasing.build_pointer_chasing_vocab(num_nodes=5)
    trace, final = pointer_chasing.solve_pointer_chasing(
        pointers=[1, 3, 4, 2, 0],
        start_node=0,
        num_hops=6,
    )
    assert trace == [1, 3, 2, 4, 0, 1]
    assert final == 1

    with pytest.raises(ValueError, match="pointer target"):
        pointer_chasing.solve_pointer_chasing([1, -1], start_node=0, num_hops=2)
    with pytest.raises(ValueError, match="pointer target"):
        pointer_chasing.solve_pointer_chasing([1, 2], start_node=0, num_hops=2)

    rng = random.Random(1)
    prompt, answer, pointers, start_node, final_node = pointer_chasing.sample_pointer_chasing_example(
        5,
        6,
        stoi,
        rng,
    )
    _, solved_final = pointer_chasing.solve_pointer_chasing(pointers, start_node, 6)
    assert solved_final == final_node
    assert sorted(pointers) == list(range(5))
    visited = []
    node = start_node
    for _ in range(5):
        visited.append(node)
        node = pointers[node]
    assert len(set(visited)) == 5
    assert node == start_node
    assert [itos[token] for token in prompt][0].startswith("n")
    assert [itos[token] for token in prompt][1] == pointer_chasing.EDGE_TOKEN
    assert [itos[token] for token in answer] == [pointer_chasing.node_token(final_node)]

    batch = pointer_chasing.build_pointer_chasing_batch(3, 5, 6, stoi, rng=random.Random(2))
    _assert_prompt_mask_and_metric(batch)
    assert batch.idx.size(1) <= pointer_chasing.required_block_size(5, 6)
    decoded = _decoded_answer(batch, 0, itos)
    assert len(decoded) == 2
    assert decoded[0].startswith("n")
    assert decoded[-1] == pointer_chasing.EOS_TOKEN


def test_state_machine_solving_and_batching():
    _, stoi, itos = state_machine.build_state_machine_vocab(num_states=3, alphabet_size=2)
    transition_table = [
        [1, 2],
        [2, 0],
        [0, 1],
    ]
    actions = [0, 1, 1, 0]
    trace, final_state = state_machine.solve_state_machine(
        transition_table=transition_table,
        start_state=0,
        actions=actions,
    )
    assert trace == [1, 0, 2, 0]
    assert final_state == 0
    assert state_machine.required_block_size(3, 2, 4) == 36

    rng = random.Random(5)
    prompt, answer, table, start_state, sampled_actions, sampled_trace, sampled_final = (
        state_machine.sample_state_machine_example(
            num_states=3,
            alphabet_size=2,
            num_steps=4,
            stoi=stoi,
            rng=rng,
        )
    )
    solved_trace, solved_final = state_machine.solve_state_machine(table, start_state, sampled_actions)
    assert solved_trace == sampled_trace
    assert solved_final == sampled_final
    for row in table:
        assert len(set(row)) == len(row)
    for action in range(2):
        assert sorted(row[action] for row in table) == [0, 1, 2]
    decoded_prompt = [itos[token] for token in prompt]
    assert decoded_prompt[0] == state_machine.STATES_TOKEN
    table_start = decoded_prompt.index(state_machine.TABLE_TOKEN) + 1
    table_end = decoded_prompt.index(state_machine.START_TOKEN)
    table_triples = list(zip(*[iter(decoded_prompt[table_start:table_end])] * 3))
    expected_triples = {
        (
            state_machine.state_token(source),
            state_machine.action_token(action),
            state_machine.state_token(table[source][action]),
        )
        for source in range(3)
        for action in range(2)
    }
    assert set(table_triples) == expected_triples
    assert [(source, action) for source, action, _ in table_triples] != [
        (state_machine.state_token(source), state_machine.action_token(action))
        for source in range(3)
        for action in range(2)
    ]
    assert [itos[token] for token in answer] == [state_machine.state_token(sampled_final)]

    batch = state_machine.build_state_machine_batch(
        3,
        3,
        2,
        4,
        stoi,
        rng=random.Random(6),
    )
    _assert_prompt_mask_and_metric(batch)
    assert batch.idx.size(1) <= state_machine.required_block_size(3, 2, 4)
    decoded = _decoded_answer(batch, 0, itos)
    assert len(decoded) == 2
    assert decoded[0].startswith("s")
    assert decoded[-1] == state_machine.EOS_TOKEN


def test_state_machine_level_zero_mixture():
    _, stoi, itos = state_machine.build_state_machine_vocab(num_states=4, alphabet_size=2)
    saw_copy = False
    saw_source_only = False
    saw_action_lookup = False

    for seed in range(1_000):
        prompt, answer, table, start_state, actions, trace, final_state = state_machine.sample_state_machine_example(
            num_states=4,
            alphabet_size=2,
            num_steps=0,
            stoi=stoi,
            rng=random.Random(seed),
        )
        solved_trace, solved_final = state_machine.solve_state_machine(table, start_state, actions)
        decoded_prompt = [itos[token] for token in prompt]
        assert solved_trace == trace
        assert solved_final == final_state
        assert [itos[token] for token in answer] == [state_machine.state_token(final_state)]
        assert decoded_prompt[0] == state_machine.STATES_TOKEN
        assert decoded_prompt[5] == state_machine.ALPHABET_TOKEN
        assert decoded_prompt[8] == state_machine.TABLE_TOKEN
        table_start = decoded_prompt.index(state_machine.TABLE_TOKEN) + 1
        table_end = decoded_prompt.index(state_machine.START_TOKEN)
        table_triples = list(zip(*[iter(decoded_prompt[table_start:table_end])] * 3))
        source_action_pairs = [(source, action) for source, action, _ in table_triples]
        canonical_pairs = [
            (state_machine.state_token(source), state_machine.action_token(action))
            for source in range(4)
            for action in range(2)
        ]
        row_constant = all(len(set(row)) == 1 for row in table)
        column_constant = all(len({row[action] for row in table}) == 1 for action in range(2))
        assert len(table_triples) == len(canonical_pairs)
        assert set(source_action_pairs) == set(canonical_pairs)
        assert source_action_pairs == canonical_pairs
        saw_copy = saw_copy or not actions
        saw_source_only = saw_source_only or (bool(actions) and row_constant and not column_constant)
        saw_action_lookup = saw_action_lookup or (bool(actions) and start_state == 0 and not row_constant)

    assert saw_copy
    assert saw_source_only
    assert saw_action_lookup

    batch = state_machine.build_state_machine_batch(
        3,
        4,
        2,
        0,
        stoi,
        rng=random.Random(22),
    )
    _assert_prompt_mask_and_metric(batch)
    assert batch.idx.size(1) <= state_machine.required_block_size(4, 2, 0)
    decoded = _decoded_answer(batch, 0, itos)
    assert len(decoded) == 2
    assert decoded[0].startswith("s")
    assert decoded[-1] == state_machine.EOS_TOKEN


def test_state_machine_custom_sizes():
    _, stoi, _ = state_machine.build_state_machine_vocab(num_states=5, alphabet_size=3)
    assert state_machine.state_token(4) in stoi
    assert state_machine.action_token(2) in stoi
    assert state_machine.state_token(5) not in stoi
    assert state_machine.action_token(3) not in stoi

    batch = state_machine.build_state_machine_batch(
        2,
        5,
        3,
        7,
        stoi,
        rng=random.Random(7),
    )
    _assert_prompt_mask_and_metric(batch)
    assert batch.idx.size(1) <= state_machine.required_block_size(5, 3, 7)

    with pytest.raises(ValueError, match="alphabet_size"):
        state_machine.build_state_machine_vocab(num_states=2, alphabet_size=3)


def test_random_graph_walk_generation_and_batching():
    _, stoi, itos = random_graph_walk.build_random_graph_walk_vocab(
        num_states=4,
        label_pool_size=4,
    )
    transition_table = [
        [(0, 1), (1, 2)],
        [(1, 3), (2, 0)],
        [(2, 2), (3, 1)],
        [(3, 0), (0, 3)],
    ]
    trace, final_state = random_graph_walk.solve_random_graph_walk(
        transition_table,
        start_state=0,
        actions=[1, 3],
    )
    assert trace == [2, 1]
    assert final_state == 1

    with pytest.raises(ValueError, match="legal"):
        random_graph_walk.solve_random_graph_walk(
            transition_table,
            start_state=0,
            actions=[2],
        )

    sampled_table = random_graph_walk.sample_random_graph_walk_table(
        num_states=4,
        label_pool_size=4,
        rng=random.Random(11),
    )
    assert all(len(row) == random_graph_walk.OUT_DEGREE for row in sampled_table)
    assert all(len({label for label, _ in row}) == random_graph_walk.OUT_DEGREE for row in sampled_table)
    label_counts = {}
    for row in sampled_table:
        for label, _ in row:
            label_counts[label] = label_counts.get(label, 0) + 1
    assert any(count > 1 for count in label_counts.values())

    rng = random.Random(12)
    prompt, answer, table, start_state, actions, sampled_trace, sampled_final = (
        random_graph_walk.sample_random_graph_walk_example(
            num_states=4,
            label_pool_size=4,
            num_steps=5,
            stoi=stoi,
            rng=rng,
        )
    )
    solved_trace, solved_final = random_graph_walk.solve_random_graph_walk(table, start_state, actions)
    assert solved_trace == sampled_trace
    assert solved_final == sampled_final
    assert [itos[token] for token in prompt][0] == random_graph_walk.STATES_TOKEN
    decoded_prompt = [itos[token] for token in prompt]
    assert random_graph_walk.START_TOKEN in decoded_prompt
    assert [itos[token] for token in answer] == [random_graph_walk.label_token(action) for action in actions]

    batch = random_graph_walk.build_random_graph_walk_batch(
        batch_size=3,
        num_states=4,
        label_pool_size=4,
        num_steps=5,
        stoi=stoi,
        rng=random.Random(13),
    )
    _assert_prompt_mask_and_metric(batch)
    assert batch.idx.size(1) <= random_graph_walk.required_block_size(4, 4, 5)
    decoded = _decoded_answer(batch, 0, itos)
    assert decoded[-2].startswith("t")
    assert decoded[-1] == random_graph_walk.EOS_TOKEN


def test_othello_trace_generation_batching_and_legality(tmp_path):
    _, stoi, itos = othello.build_othello_vocab(othello_train_games=16, othello_val_games=8)
    data_dir = str(tmp_path / "othello_data")
    othello.ensure_othello_datasets(
        othello_data_dir=data_dir,
        othello_train_games=16,
        othello_val_games=8,
        othello_dataset_seed=29,
    )

    prompt, answer, sampled_trace = othello.sample_othello_example(
        stoi,
        random.Random(25),
        split="val",
        othello_data_dir=data_dir,
        othello_train_games=16,
        othello_val_games=8,
        othello_dataset_seed=29,
    )
    decoded_prompt = [itos[token] for token in prompt]
    decoded_answer = [itos[token] for token in answer]
    assert decoded_prompt == []
    assert len(answer) == othello.MAX_MOVES
    assert decoded_answer[: len(sampled_trace)] == [othello.move_token(square) for square in sampled_trace]
    assert all(token == "<pad>" for token in decoded_answer[len(sampled_trace) :])

    opening_prompt, _, _ = othello.sample_othello_example(
        stoi,
        random.Random(25),
        split="val",
        othello_data_dir=data_dir,
        othello_train_games=16,
        othello_val_games=8,
        othello_dataset_seed=29,
        othello_prepend_opening=True,
    )
    assert [itos[token] for token in opening_prompt] == [othello.move_token(square) for square in othello.OPENING_PREFIX]

    legal_prefix_len, all_legal = othello.legal_prefix_length(answer)
    assert legal_prefix_len == len(sampled_trace)
    assert all_legal

    illegal_prefix_len, illegal = othello.legal_prefix_length([stoi[othello.move_token(28)]])
    assert illegal_prefix_len == 0
    assert not illegal

    batch = othello.build_othello_batch(
        2,
        stoi,
        rng=random.Random(26),
        split="val",
        othello_data_dir=data_dir,
        othello_train_games=16,
        othello_val_games=8,
        othello_dataset_seed=29,
    )
    _assert_prompt_mask_and_metric(batch)
    assert batch.idx.size(1) <= othello.required_block_size(othello_train_games=16, othello_val_games=8)
    decoded = _decoded_answer(batch, 0, itos)
    assert decoded[-1] == "<eos>"
    assert all(token.startswith("m") or token == "<pad>" for token in decoded[:-1])


def test_othello_dataset_loads_dense_arrays_once(tmp_path, monkeypatch):
    data_dir = str(tmp_path / "othello_data")
    othello.ensure_othello_datasets(
        othello_data_dir=data_dir,
        othello_train_games=1001,
        othello_val_games=8,
        othello_dataset_seed=31,
    )
    othello._DATASET_CACHE.clear()

    load_calls = 0
    original_np_load = othello.np.load

    def counting_np_load(*args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        return original_np_load(*args, **kwargs)

    monkeypatch.setattr(othello.np, "load", counting_np_load)

    dataset = othello.load_othello_dataset(
        split="train",
        othello_data_dir=data_dir,
        othello_train_games=1001,
        othello_val_games=8,
        othello_dataset_seed=31,
    )

    assert dataset.traces.shape == (1001, othello.MAX_MOVES)
    assert dataset.lengths.shape == (1001,)
    assert load_calls == 2

    for seed in range(5):
        dataset.sample_trace(random.Random(seed))
    assert load_calls == 2

    cached_dataset = othello.load_othello_dataset(
        split="train",
        othello_data_dir=data_dir,
        othello_train_games=1001,
        othello_val_games=8,
        othello_dataset_seed=31,
    )
    assert cached_dataset is dataset
    assert load_calls == 2


def test_othello_legacy_chunked_dataset_migrates_to_dense_arrays(tmp_path):
    data_dir = tmp_path / "othello_data"
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    train_traces_a = torch.full((1000, othello.MAX_MOVES), othello.PAD_MOVE, dtype=torch.uint8).numpy()
    train_lengths_a = torch.ones(1000, dtype=torch.uint8).numpy()
    train_traces_b = torch.full((1, othello.MAX_MOVES), othello.PAD_MOVE, dtype=torch.uint8).numpy()
    train_lengths_b = torch.ones(1, dtype=torch.uint8).numpy()
    val_traces = torch.full((8, othello.MAX_MOVES), othello.PAD_MOVE, dtype=torch.uint8).numpy()
    val_lengths = torch.ones(8, dtype=torch.uint8).numpy()

    np.save(train_dir / "traces_000000.npy", train_traces_a)
    np.save(train_dir / "lengths_000000.npy", train_lengths_a)
    np.save(train_dir / "traces_000001.npy", train_traces_b)
    np.save(train_dir / "lengths_000001.npy", train_lengths_b)
    np.save(val_dir / "traces_000000.npy", val_traces)
    np.save(val_dir / "lengths_000000.npy", val_lengths)

    legacy_metadata = {
        "dataset_version": 2,
        "train_games": 1001,
        "val_games": 8,
        "dataset_seed": 31,
        "max_moves": othello.MAX_MOVES,
        "chunk_size": othello.DATASET_CHUNK_SIZE,
    }
    (data_dir / "metadata.json").write_text(json.dumps(legacy_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    othello._DATASET_CACHE.clear()

    dataset = othello.load_othello_dataset(
        split="train",
        othello_data_dir=str(data_dir),
        othello_train_games=1001,
        othello_val_games=8,
        othello_dataset_seed=31,
    )

    assert dataset.traces.shape == (1001, othello.MAX_MOVES)
    assert dataset.lengths.shape == (1001,)
    assert (data_dir / "train_traces.npy").exists()
    assert (data_dir / "train_lengths.npy").exists()
    assert (data_dir / "val_traces.npy").exists()
    assert (data_dir / "val_lengths.npy").exists()
    assert not train_dir.exists()
    assert not val_dir.exists()
    metadata = json.loads((data_dir / "metadata.json").read_text())
    assert metadata["dataset_version"] == othello.DATASET_VERSION
    assert metadata["storage_format"] == "dense_split_arrays"


def test_othello_initial_legal_moves_and_flip():
    board = othello.initial_board()
    active_player, legal_mask = othello.active_player_and_legal_moves(board, othello.BLACK)
    legal_squares = {idx for idx, is_legal in enumerate(legal_mask.reshape(-1).tolist()) if is_legal}

    assert active_player == othello.BLACK
    assert legal_squares == {19, 26, 37, 44}

    next_board = othello.apply_move(board, (2, 3), othello.BLACK)
    assert next_board[2, 3] == othello.BLACK
    assert next_board[3, 3] == othello.BLACK


def test_tracking_generation_and_batching():
    _, stoi, itos = tracking.build_tracking_vocab(num_objects=5)
    rng = random.Random(13)
    _, answer, ops, query_pos, final_object = tracking.sample_tracking_example(
        5,
        6,
        stoi,
        rng,
    )
    _, final_state = tracking.solve_tracking(5, ops)
    assert final_state[query_pos] == final_object
    assert [itos[token] for token in answer] == [tracking.obj_token(final_object)]

    batch = tracking.build_tracking_batch(2, 5, 6, stoi, rng=random.Random(14))
    _assert_prompt_mask_and_metric(batch)
    assert batch.idx.size(1) <= tracking.required_block_size(5, 6)
    decoded = _decoded_answer(batch, 0, itos)
    assert len(decoded) == 2
    assert decoded[0].startswith("o")


def test_permutation_generation_and_batching():
    _, stoi, itos = permutation.build_permutation_vocab(num_objects=4)
    batch = permutation.build_permutation_batch(
        batch_size=1,
        num_objects=4,
        num_swaps=3,
        stoi=stoi,
        rng=random.Random(15),
    )
    _assert_prompt_mask_and_metric(batch)
    assert batch.idx.shape == (1, permutation.required_block_size(4, 3))
    decoded = _decoded_answer(batch, 0, itos)
    assert decoded[-1] == permutation.EOS_TOKEN
