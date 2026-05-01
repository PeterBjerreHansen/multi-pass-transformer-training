import random

from tasks.bbh import permutation
from tasks.bbh import pointer_chasing
from tasks.bbh import state_machine
from tasks.bbh import tracking
from tasks.trace import random_graph_walk
from tasks.trace import othello


def _tokens(ids, itos):
    return [itos[int(token)] for token in ids]


def _print_example(name, prompt_tokens, answer_tokens):
    print(f"\n{name}")
    print("  prompt:", " ".join(prompt_tokens))
    print("  answer:", " ".join(answer_tokens))


def test_print_symbolic_task_examples(tmp_path):
    _, pointer_stoi, pointer_itos = pointer_chasing.build_pointer_chasing_vocab(num_nodes=8)
    prompt, answer, pointers, start_node, final_node = pointer_chasing.sample_pointer_chasing_example(
        num_nodes=8,
        num_hops=6,
        stoi=pointer_stoi,
        rng=random.Random(101),
    )
    _, solved_final = pointer_chasing.solve_pointer_chasing(pointers, start_node, 6)
    assert solved_final == final_node
    _print_example("pointer_chasing final", _tokens(prompt, pointer_itos), _tokens(answer, pointer_itos))

    _, sm_stoi, sm_itos = state_machine.build_state_machine_vocab(num_states=5, alphabet_size=3)
    prompt, answer, table, start_state, actions, trace, final_state = state_machine.sample_state_machine_example(
        num_states=5,
        alphabet_size=3,
        num_steps=5,
        stoi=sm_stoi,
        rng=random.Random(109),
    )
    solved_trace, solved_final = state_machine.solve_state_machine(table, start_state, actions)
    assert solved_trace == trace
    assert solved_final == final_state
    _print_example("state_machine final", _tokens(prompt, sm_itos), _tokens(answer, sm_itos))

    _, lsw_stoi, lsw_itos = random_graph_walk.build_random_graph_walk_vocab(
        num_states=5,
        label_pool_size=4,
    )
    prompt, answer, transition_table, start_state, actions, trace, final_state = (
        random_graph_walk.sample_random_graph_walk_example(
            num_states=5,
            label_pool_size=4,
            num_steps=5,
            stoi=lsw_stoi,
            rng=random.Random(110),
        )
    )
    solved_trace, solved_final = random_graph_walk.solve_random_graph_walk(
        transition_table,
        start_state,
        actions,
    )
    assert solved_trace == trace
    assert solved_final == final_state
    assert _tokens(answer, lsw_itos) == [random_graph_walk.label_token(action) for action in actions]
    _print_example("random_graph_walk trace", _tokens(prompt, lsw_itos), _tokens(answer, lsw_itos))

    _, othello_stoi, othello_itos = othello.build_othello_vocab(othello_train_games=16, othello_val_games=8)
    prompt, answer, sampled_trace = othello.sample_othello_example(
        stoi=othello_stoi,
        rng=random.Random(104),
        split="val",
        othello_data_dir=str(tmp_path / "othello_data"),
        othello_train_games=16,
        othello_val_games=8,
        othello_dataset_seed=17,
    )
    legal_prefix_len, all_legal = othello.legal_prefix_length(answer)
    assert legal_prefix_len == len(sampled_trace)
    assert all_legal
    _print_example("othello trace", _tokens(prompt, othello_itos), _tokens(answer, othello_itos))

    _, tracking_stoi, tracking_itos = tracking.build_tracking_vocab(num_objects=5)
    prompt, answer, ops, query_pos, final_object = tracking.sample_tracking_example(
        num_objects=5,
        num_ops=5,
        stoi=tracking_stoi,
        rng=random.Random(107),
    )
    _, final_state = tracking.solve_tracking(5, ops)
    assert final_state[query_pos] == final_object
    _print_example("tracking final", _tokens(prompt, tracking_itos), _tokens(answer, tracking_itos))


def test_print_permutation_example():
    _, perm_stoi, perm_itos = permutation.build_permutation_vocab(num_objects=4)
    prompt, answer, swaps, final_state = permutation.sample_permutation_example(
        num_objects=4,
        num_swaps=3,
        stoi=perm_stoi,
        rng=random.Random(108),
    )
    assert permutation.solve_permutation(4, swaps) == final_state
    _print_example("permutation final", _tokens(prompt, perm_itos), _tokens(answer, perm_itos))
