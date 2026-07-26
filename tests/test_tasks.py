from __future__ import annotations

import random

import pytest
import torch

from models import CausalTransformer, TransformerConfig
from tasks.bbh import permutation, pointer_chasing, state_machine, tracking
from tasks.trace import othello, random_graph_walk, shortest_path


class _ForcedChoiceRandom(random.Random):
    forced_choice: str

    def choices(self, population, weights=None, *, cum_weights=None, k=1):
        assert self.forced_choice in population
        return [self.forced_choice] * k


def test_bbh_task_solvers_match_sampled_answers():
    rng = random.Random(3)

    _vocab, stoi, _ = pointer_chasing.build_pointer_chasing_vocab(9)
    _prompt, _answer, pointers, start, final = pointer_chasing.sample_pointer_chasing_example(9, 4, stoi, rng)
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


def test_state_machine_level_zero_uses_shuffled_full_table_factorizations():
    num_states = 4
    alphabet_size = 2
    _vocab, stoi, _ = state_machine.build_state_machine_vocab(
        num_states,
        alphabet_size,
    )
    prefix_len = 1 + num_states + 1 + alphabet_size + 1
    table_token_len = 3 * num_states * alphabet_size
    canonical_pairs = [
        (source, action)
        for source in range(num_states)
        for action in range(alphabet_size)
    ]

    starts_by_part = {}
    table_orders_by_part = {}
    prompt_lengths = set()
    for part, expected_weight in state_machine.LEVEL_ZERO_PART_WEIGHTS:
        assert expected_weight in {20, 40}
        starts = set()
        orders = set()
        for seed in range(32):
            rng = _ForcedChoiceRandom(seed)
            rng.forced_choice = part
            sample = state_machine.sample_state_machine_example(
                num_states,
                alphabet_size,
                0,
                stoi,
                rng,
            )
            prompt, answer, table, start, actions, trace, final = sample
            prompt_lengths.add(len(prompt))
            starts.add(start)

            table_tokens = prompt[prefix_len : prefix_len + table_token_len]
            pairs = []
            for offset in range(0, table_token_len, 3):
                source_id, action_id, target_id = table_tokens[offset : offset + 3]
                source = source_id - stoi[state_machine.state_token(0)]
                action = action_id - stoi[state_machine.action_token(0)]
                target = target_id - stoi[state_machine.state_token(0)]
                assert table[source][action] == target
                pairs.append((source, action))
            assert sorted(pairs) == canonical_pairs
            orders.add(tuple(pairs))

            assert len(actions) == 1
            assert trace == [final]
            assert answer == [stoi[state_machine.state_token(final)]]
            assert state_machine.solve_state_machine(table, start, actions)[1] == final

            if part == "source_only_full_table":
                assert all(len(set(row)) == 1 for row in table)
            elif part == "action_only_full_table":
                assert all(
                    len({row[action] for row in table}) == 1
                    for action in range(alphabet_size)
                )
            else:
                assert part == "full_lookup"
                assert all(len(set(row)) == alphabet_size for row in table)

        starts_by_part[part] = starts
        table_orders_by_part[part] = orders

    assert dict(state_machine.LEVEL_ZERO_PART_WEIGHTS) == {
        "source_only_full_table": 40,
        "action_only_full_table": 40,
        "full_lookup": 20,
    }
    assert all(starts == set(range(num_states)) for starts in starts_by_part.values())
    assert all(len(orders) > 1 for orders in table_orders_by_part.values())
    assert all(tuple(canonical_pairs) not in orders for orders in table_orders_by_part.values())

    level_one_prompt = state_machine.sample_state_machine_example(
        num_states,
        alphabet_size,
        1,
        stoi,
        random.Random(100),
    )[0]
    assert prompt_lengths == {len(level_one_prompt)}


def test_pointer_chasing_level_scales_odd_cycle_without_shortcuts():
    max_level = 8
    label_pool_size = 2 * max_level + 3
    _vocab, stoi, _ = pointer_chasing.build_pointer_chasing_vocab(label_pool_size)

    for level in range(1, max_level + 1):
        prompt, answer, pointers, start, final = pointer_chasing.sample_pointer_chasing_example(
            label_pool_size,
            level,
            stoi,
            random.Random(100 + level),
        )
        active_nodes = {
            source for source, target in enumerate(pointers)
            if source != target
        }
        trace, solved_final = pointer_chasing.solve_pointer_chasing(
            pointers,
            start,
            level,
        )

        assert len(active_nodes) == 2 * level + 1
        assert active_nodes == set(range(2 * level + 1))
        assert start in active_nodes
        assert solved_final == final
        assert answer == [stoi[pointer_chasing.node_token(final)]]
        assert len({start, *trace}) == level + 1
        assert final not in [start, *trace[:-1]]
        assert pointer_chasing.solve_pointer_chasing(
            pointers,
            final,
            level + 1,
        )[1] == start
        assert prompt.index(stoi["<query>"]) == 3 * len(active_nodes)
        assert pointer_chasing.required_block_size(
            label_pool_size,
            level,
        ) == len(prompt) + 3


def test_pointer_chasing_rejects_too_small_label_pool():
    with pytest.raises(ValueError, match="2 \\* num_hops \\+ 1"):
        pointer_chasing.required_block_size(num_nodes=8, num_hops=4)
    with pytest.raises(ValueError, match="at least 1"):
        pointer_chasing.active_num_nodes(0)
    with pytest.raises(ValueError, match="at least 3"):
        pointer_chasing.build_pointer_chasing_vocab(2)


def test_pointer_chasing_level_one_is_learnable_with_full_vocabulary():
    torch.manual_seed(1337)
    label_pool_size = pointer_chasing.DEFAULT_NUM_NODES
    vocab, stoi, _ = pointer_chasing.build_pointer_chasing_vocab(label_pool_size)
    model = CausalTransformer(
        TransformerConfig(
            block_size=pointer_chasing.required_block_size(
                label_pool_size,
                pointer_chasing.DEFAULT_MAX_LEVEL,
            ),
            vocab_size=len(vocab),
            n_layer=4,
            n_head=4,
            n_embd=128,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    train_rng = random.Random(1337)

    for _step in range(400):
        batch = pointer_chasing.build_pointer_chasing_batch(
            batch_size=64,
            num_nodes=label_pool_size,
            num_hops=1,
            stoi=stoi,
            device="cpu",
            rng=train_rng,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.idx)
        loss = model.calc_loss(output.logits, batch.targets)
        loss.backward()
        optimizer.step()

    eval_batch = pointer_chasing.build_pointer_chasing_batch(
        batch_size=256,
        num_nodes=label_pool_size,
        num_hops=1,
        stoi=stoi,
        device="cpu",
        rng=random.Random(2027),
    )
    with torch.no_grad():
        logits = model(eval_batch.idx).logits
    rows = torch.arange(eval_batch.idx.size(0))
    answer_positions = eval_batch.prompt_lengths - 1
    predictions = logits[rows, answer_positions].argmax(dim=-1)
    targets = eval_batch.targets[rows, answer_positions]
    assert (predictions == targets).float().mean().item() >= 0.99


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
