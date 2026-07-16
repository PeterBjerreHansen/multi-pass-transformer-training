from __future__ import annotations

import random

import pytest
import torch

from models import CausalTransformer, TransformerConfig
from tasks.bbh import permutation, pointer_chasing, state_machine, tracking
from tasks.trace import othello, random_graph_walk, shortest_path


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


def test_shortest_path_generation_is_unique_deterministic_and_parseable():
    config = (8, 3, 2, 5)
    for seed in range(250):
        first = shortest_path.sample_unique_shortest_path_graph(*config, random.Random(seed))
        second = shortest_path.sample_unique_shortest_path_graph(*config, random.Random(seed))
        assert first == second
        edges, start, goal, target_path = first
        solved_path, path_count = shortest_path.solve_shortest_path(
            config[0],
            edges,
            start,
            goal,
        )
        assert path_count == 1
        assert solved_path == target_path
        assert len(target_path) == config[1] + 1
        assert len(edges) == config[1] + config[3]
        assert max(
            sum(source == node for source, _target in edges)
            for node in range(config[0])
        ) <= config[2]

    _vocab, stoi, _itos = shortest_path.build_shortest_path_vocab(*config)
    prompt, answer, edges, start, goal, target_path = shortest_path.sample_shortest_path_example(
        *config,
        stoi,
        random.Random(8128),
    )
    parsed_edges, parsed_start, parsed_goal = shortest_path.parse_prompt_metadata(
        prompt,
        num_nodes=config[0],
        edge_count=config[1] + config[3],
    )
    assert set(parsed_edges) == set(edges)
    assert parsed_start == start
    assert parsed_goal == goal
    assert answer == [stoi[shortest_path.node_token(node)] for node in target_path]
    assert shortest_path.legal_prefix_length(
        prompt,
        answer,
        num_nodes=config[0],
        edge_count=config[1] + config[3],
    ) == (len(answer), True)

    corrupted = list(answer)
    corrupted[0] = stoi[shortest_path.node_token((start + 1) % config[0])]
    assert shortest_path.legal_prefix_length(
        prompt,
        corrupted,
        num_nodes=config[0],
        edge_count=config[1] + config[3],
    )[1] is False


def test_shortest_path_fixed_example_can_be_overfit_and_generated():
    torch.manual_seed(123)
    config = (8, 3, 2, 5)
    vocab, stoi, _itos = shortest_path.build_shortest_path_vocab(*config)
    batch = shortest_path.build_shortest_path_batch(
        1,
        *config,
        stoi,
        device="cpu",
        rng=random.Random(17),
    )
    model = CausalTransformer(
        TransformerConfig(
            block_size=batch.idx.shape[1],
            vocab_size=len(vocab),
            n_layer=2,
            n_head=2,
            n_embd=32,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    for _step in range(100):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.idx)
        loss = model.calc_loss(output.logits, batch.targets)
        loss.backward()
        optimizer.step()

    prompt_len = int(batch.prompt_lengths[0])
    output_len = int(batch.output_lengths[0])
    generated = model.generate(
        batch.idx[:, :prompt_len],
        output_len,
        do_sample=False,
        inference_mode="recompute",
    )
    expected = batch.targets[:, prompt_len - 1 : prompt_len - 1 + output_len]
    assert loss.item() < 0.01
    assert torch.equal(generated[:, prompt_len : prompt_len + output_len], expected)
