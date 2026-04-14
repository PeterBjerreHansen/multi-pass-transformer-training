import random

from tasks import arithmetic
from tasks import boolean_rpn
from tasks import dyck
from tasks import order_deduction
from tasks import permutation
from tasks import tracking
from tasks import truth_graph
from tasks import walk
from tasks.clrs_text import ClrsTextExample, encode_example, decode_ids


def _tokens(ids, itos):
    return [itos[int(token)] for token in ids]


def _print_example(name, prompt_tokens, answer_tokens):
    print(f"\n{name}")
    print("  prompt:", " ".join(prompt_tokens))
    print("  answer:", " ".join(answer_tokens))


def test_print_symbolic_task_examples():
    walk_tokens, walk_stoi, walk_itos = walk.build_walk_vocab(max_steps=8)
    prompt, answer, moves, final_state = walk.sample_walk_example(
        num_steps=6,
        stoi=walk_stoi,
        rng=random.Random(101),
        supervision="trace",
    )
    states, solved_final = walk.solve_walk(moves)
    assert solved_final == final_state
    _print_example("walk trace", _tokens(prompt, walk_itos), _tokens(answer, walk_itos))

    _, dyck_stoi, dyck_itos = dyck.build_dyck_vocab(max_bracket_types=4)
    prompt, answer, prefix, completion = dyck.sample_dyck_example(
        prefix_length=8,
        max_depth=4,
        bracket_types=3,
        stoi=dyck_stoi,
        rng=random.Random(102),
        supervision="trace",
    )
    _, solved_completion = dyck.solve_dyck_prefix(prefix)
    assert solved_completion == completion
    _print_example("dyck trace", _tokens(prompt, dyck_itos), _tokens(answer, dyck_itos))

    _, bool_stoi, bool_itos = boolean_rpn.build_boolean_rpn_vocab()
    prompt, answer, expr, result = boolean_rpn.sample_boolean_rpn_example(
        num_binary_ops=4,
        stoi=bool_stoi,
        rng=random.Random(103),
        supervision="trace",
    )
    _, solved_result = boolean_rpn.eval_bool_rpn(expr)
    assert solved_result == result
    _print_example("boolean_rpn trace", _tokens(prompt, bool_itos), _tokens(answer, bool_itos))

    _, arith_stoi, arith_itos = arithmetic.build_arithmetic_vocab(max_modulus=10)
    prompt, answer, initial, operations, final = arithmetic.sample_arithmetic_example(
        num_steps=5,
        modulus=10,
        stoi=arith_stoi,
        rng=random.Random(104),
        supervision="trace",
    )
    _, solved_final = arithmetic.eval_modular_arithmetic(initial, operations, 10)
    assert solved_final == final
    _print_example("arithmetic trace", _tokens(prompt, arith_itos), _tokens(answer, arith_itos))

    _, truth_stoi, truth_itos = truth_graph.build_truth_vocab(max_num_vars=6)
    prompt, answer, definitions, query_index, final = truth_graph.sample_truth_example(
        num_vars=5,
        stoi=truth_stoi,
        rng=random.Random(105),
        supervision="trace",
    )
    values = truth_graph.eval_truth_program(definitions)
    assert values[query_index] == final
    _print_example("truth_graph trace", _tokens(prompt, truth_itos), _tokens(answer, truth_itos))

    _, order_stoi, order_itos = order_deduction.build_order_vocab(max_num_objects=6)
    prompt, answer, order, (left, right), final = order_deduction.sample_order_example(
        num_objects=5,
        stoi=order_stoi,
        rng=random.Random(106),
        supervision="trace",
    )
    rank = {obj: index for index, obj in enumerate(order)}
    assert (rank[left] < rank[right]) == final
    _print_example("order_deduction trace", _tokens(prompt, order_itos), _tokens(answer, order_itos))

    _, tracking_stoi, tracking_itos = tracking.build_tracking_vocab(num_objects=5)
    prompt, answer, ops, query_pos, final_object = tracking.sample_tracking_example(
        num_objects=5,
        num_ops=5,
        stoi=tracking_stoi,
        rng=random.Random(107),
        supervision="trace",
    )
    _, final_state = tracking.solve_tracking(5, ops)
    assert final_state[query_pos] == final_object
    _print_example("tracking trace", _tokens(prompt, tracking_itos), _tokens(answer, tracking_itos))


def test_print_permutation_and_clrs_examples():
    _, perm_stoi, perm_itos = permutation.build_permutation_vocab(num_objects=4)
    batch = permutation.build_permutation_batch(
        batch_size=1,
        num_objects=4,
        num_swaps=3,
        stoi=perm_stoi,
        rng=random.Random(108),
        supervision="trace",
    )
    prompt_len = int(batch.prompt_lengths[0].item())
    output_len = int(batch.output_lengths[0].item())
    prompt = batch.idx[0, :prompt_len].tolist()
    answer = batch.targets[0, prompt_len - 1 : prompt_len - 1 + output_len].tolist()
    _print_example("permutation trace", _tokens(prompt, perm_itos), _tokens(answer, perm_itos))
    assert _tokens(answer, perm_itos).count(permutation.TRACE_TOKEN) == 3

    clrs_example = ClrsTextExample(
        question="minimum: key: [0.3 0.1], initial_trace: 0 trace | min:",
        answer="1 | 1",
        algo_name="minimum",
        length=2,
    )
    encoded = encode_example(clrs_example)
    print("\nclrs_text byte example")
    print("  prompt:", decode_ids(encoded.idx[: encoded.prompt_length]))
    print("  answer:", decode_ids(encoded.targets[encoded.prompt_length - 1 :]))
    assert "minimum" in decode_ids(encoded.idx[: encoded.prompt_length])
