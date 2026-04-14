import random

import torch

from tasks import arithmetic
from tasks import boolean_rpn
from tasks import dyck
from tasks import order_deduction
from tasks import permutation
from tasks import tracking
from tasks import truth_graph
from tasks import walk


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


def test_walk_generation_final_and_trace():
    _, stoi, itos = walk.build_walk_vocab(max_steps=8)
    rng = random.Random(1)
    prompt, answer, moves, final_state = walk.sample_walk_example(6, stoi, rng, supervision="final")
    states, solved_final = walk.solve_walk(moves)
    assert solved_final == final_state
    assert [itos[token] for token in answer] == [walk.FINAL_TOKEN, str(final_state[0]), str(final_state[1])]

    batch = walk.build_walk_batch(3, 6, stoi, supervision="trace", rng=random.Random(2))
    _assert_prompt_mask_and_metric(batch)
    assert batch.idx.size(1) <= walk.required_block_size(6, supervision="trace")
    decoded = _decoded_answer(batch, 0, itos)
    assert decoded.count(walk.STATE_TOKEN) == 6
    assert decoded[-4] == walk.FINAL_TOKEN


def test_dyck_generation_final_and_trace():
    _, stoi, itos = dyck.build_dyck_vocab(max_bracket_types=4)
    rng = random.Random(3)
    _, answer, prefix, completion = dyck.sample_dyck_example(
        prefix_length=8,
        max_depth=4,
        bracket_types=3,
        stoi=stoi,
        rng=rng,
        supervision="final",
    )
    _, solved_completion = dyck.solve_dyck_prefix(prefix)
    assert solved_completion == completion
    assert [itos[token] for token in answer] == [dyck.FINAL_TOKEN, *completion]

    batch = dyck.build_dyck_batch(2, 8, 4, 3, stoi, supervision="trace", rng=random.Random(4))
    _assert_prompt_mask_and_metric(batch)
    assert batch.idx.size(1) <= dyck.required_block_size(8, 4, supervision="trace")
    decoded = _decoded_answer(batch, 0, itos)
    assert dyck.STACK_TOKEN in decoded
    assert dyck.FINAL_TOKEN in decoded


def test_boolean_rpn_generation_final_and_trace():
    _, stoi, itos = boolean_rpn.build_boolean_rpn_vocab()
    rng = random.Random(5)
    _, answer, expr, result = boolean_rpn.sample_boolean_rpn_example(5, stoi, rng, supervision="final")
    _, solved_result = boolean_rpn.eval_bool_rpn(expr)
    assert solved_result == result
    assert [itos[token] for token in answer] == [
        boolean_rpn.FINAL_TOKEN,
        boolean_rpn.TRUE_TOKEN if result else boolean_rpn.FALSE_TOKEN,
    ]

    batch = boolean_rpn.build_boolean_rpn_batch(2, 5, stoi, supervision="trace", rng=random.Random(6))
    _assert_prompt_mask_and_metric(batch)
    assert batch.idx.size(1) <= boolean_rpn.required_block_size(5, supervision="trace")
    decoded = _decoded_answer(batch, 0, itos)
    assert boolean_rpn.STACK_TOKEN in decoded
    assert boolean_rpn.FINAL_TOKEN in decoded


def test_arithmetic_generation_final_and_trace():
    _, stoi, itos = arithmetic.build_arithmetic_vocab(max_modulus=10)
    rng = random.Random(7)
    _, answer, initial, operations, final = arithmetic.sample_arithmetic_example(
        6,
        10,
        stoi,
        rng,
        supervision="final",
    )
    _, solved_final = arithmetic.eval_modular_arithmetic(initial, operations, 10)
    assert solved_final == final
    assert [itos[token] for token in answer] == [arithmetic.FINAL_TOKEN, str(final)]

    batch = arithmetic.build_arithmetic_batch(2, 6, 10, stoi, supervision="trace", rng=random.Random(8))
    _assert_prompt_mask_and_metric(batch)
    assert batch.idx.size(1) <= arithmetic.required_block_size(6, supervision="trace")
    decoded = _decoded_answer(batch, 0, itos)
    assert decoded.count(arithmetic.STATE_TOKEN) == 6
    assert arithmetic.FINAL_TOKEN in decoded


def test_truth_graph_generation_final_and_trace():
    _, stoi, itos = truth_graph.build_truth_vocab(max_num_vars=8)
    rng = random.Random(9)
    _, answer, definitions, query_index, final = truth_graph.sample_truth_example(
        6,
        stoi,
        rng,
        supervision="final",
    )
    values = truth_graph.eval_truth_program(definitions)
    assert values[query_index] == final
    assert [itos[token] for token in answer] == [
        truth_graph.FINAL_TOKEN,
        truth_graph.TRUE_TOKEN if final else truth_graph.FALSE_TOKEN,
    ]

    batch = truth_graph.build_truth_batch(2, 6, stoi, supervision="trace", rng=random.Random(10))
    _assert_prompt_mask_and_metric(batch)
    assert batch.idx.size(1) <= truth_graph.required_block_size(6, supervision="trace")
    decoded = _decoded_answer(batch, 0, itos)
    assert decoded.count(truth_graph.STATE_TOKEN) == 6
    assert truth_graph.FINAL_TOKEN in decoded


def test_order_deduction_generation_final_and_trace():
    _, stoi, itos = order_deduction.build_order_vocab(max_num_objects=6)
    rng = random.Random(11)
    _, answer, order, (left, right), final = order_deduction.sample_order_example(
        5,
        stoi,
        rng,
        supervision="final",
    )
    rank = {obj: index for index, obj in enumerate(order)}
    assert (rank[left] < rank[right]) == final
    assert [itos[token] for token in answer] == [
        order_deduction.FINAL_TOKEN,
        order_deduction.TRUE_TOKEN if final else order_deduction.FALSE_TOKEN,
    ]

    batch = order_deduction.build_order_batch(2, 5, stoi, supervision="trace", rng=random.Random(12))
    _assert_prompt_mask_and_metric(batch)
    assert batch.idx.size(1) <= order_deduction.required_block_size(5, supervision="trace")
    decoded = _decoded_answer(batch, 0, itos)
    assert order_deduction.TRACE_TOKEN in decoded
    assert order_deduction.FINAL_TOKEN in decoded


def test_tracking_generation_final_and_trace():
    _, stoi, itos = tracking.build_tracking_vocab(num_objects=5)
    rng = random.Random(13)
    _, answer, ops, query_pos, final_object = tracking.sample_tracking_example(
        5,
        6,
        stoi,
        rng,
        supervision="final",
    )
    _, final_state = tracking.solve_tracking(5, ops)
    assert final_state[query_pos] == final_object
    assert [itos[token] for token in answer] == [tracking.FINAL_TOKEN, tracking.obj_token(final_object)]

    batch = tracking.build_tracking_batch(2, 5, 6, stoi, supervision="trace", rng=random.Random(14))
    _assert_prompt_mask_and_metric(batch)
    assert batch.idx.size(1) <= tracking.required_block_size(5, 6, supervision="trace")
    decoded = _decoded_answer(batch, 0, itos)
    assert decoded.count(tracking.STATE_TOKEN) == 6
    assert tracking.FINAL_TOKEN in decoded


def test_permutation_trace_supervision_keeps_final_mode_compatible():
    _, stoi, itos = permutation.build_permutation_vocab(num_objects=4)
    final_batch = permutation.build_permutation_batch(
        batch_size=1,
        num_objects=4,
        num_swaps=3,
        stoi=stoi,
        rng=random.Random(15),
        supervision="final",
    )
    assert final_batch.idx.shape == (1, permutation.required_block_size(4, 3))

    trace_batch = permutation.build_permutation_batch(
        batch_size=1,
        num_objects=4,
        num_swaps=3,
        stoi=stoi,
        rng=random.Random(15),
        supervision="trace",
    )
    _assert_prompt_mask_and_metric(trace_batch)
    assert trace_batch.idx.shape == (1, permutation.required_block_size(4, 3, supervision="trace"))
    decoded = _decoded_answer(trace_batch, 0, itos)
    assert decoded.count(permutation.TRACE_TOKEN) == 3
    assert permutation.FINAL_TOKEN in decoded
