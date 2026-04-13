import random

import torch

from tasks import permutation
from tasks import repl


def test_permutation_batch_targets_match_sampled_swaps():
    num_objects = 4
    num_swaps = 5
    _, stoi, itos = permutation.build_permutation_vocab(num_objects)
    batch = permutation.build_permutation_batch(
        batch_size=3,
        num_objects=num_objects,
        num_swaps=num_swaps,
        stoi=stoi,
        rng=random.Random(7),
    )

    assert batch.idx.shape == (3, permutation.required_block_size(num_objects, num_swaps))
    assert batch.targets.shape == batch.idx.shape

    for row in range(batch.idx.size(0)):
        prompt_len = int(batch.prompt_lengths[row].item())
        output_len = int(batch.output_lengths[row].item())
        ids = batch.idx[row].tolist()
        targets = batch.targets[row]
        metric_mask = batch.metric_mask[row]

        assert torch.equal(targets[: prompt_len - 1], torch.full((prompt_len - 1,), -1))
        assert not metric_mask[: prompt_len - 1].any()
        assert metric_mask[prompt_len - 1 : prompt_len - 1 + output_len].all()

        state = list(range(num_objects))
        pos = num_objects + 1
        while pos < prompt_len - 1:
            assert ids[pos] == stoi[permutation.SWAP_TOKEN]
            i = int(itos[ids[pos + 1]][1:])
            j = int(itos[ids[pos + 2]][1:])
            state[i], state[j] = state[j], state[i]
            pos += 3

        expected_suffix = [stoi[f"o{obj}"] for obj in state] + [stoi[permutation.EOS_TOKEN]]
        actual_suffix = targets[prompt_len - 1 : prompt_len - 1 + output_len].tolist()
        assert actual_suffix == expected_suffix


def test_repl_sampled_traces_verify_against_python_execution():
    _, stoi, itos = repl.build_repl_vocab(value_mod=7, max_num_vars=4)
    rng = random.Random(13)

    for _ in range(10):
        ids = repl.sample_repl_trace_ids(
            max_num_vars=4,
            program_length=5,
            stoi=stoi,
            operations=["add", "sub", "mul"],
            operand_patterns=["num_num", "var_num", "num_var", "var_var"],
            value_mod=7,
            randomize_num_vars=True,
            print_prob=0.5,
            rng=rng,
        )
        result = repl.verify_repl_trace_ids(ids, itos, value_mod=7)
        assert result.ok, result.python_source


def test_repl_batch_masks_suffix_loss_and_reconstructs_valid_traces():
    _, stoi, itos = repl.build_repl_vocab(value_mod=5, max_num_vars=3)
    batch = repl.build_repl_batch(
        batch_size=4,
        max_num_vars=3,
        program_length=3,
        stoi=stoi,
        operations=["add", "sub"],
        operand_patterns=["num_num", "var_num", "num_var", "var_var"],
        value_mod=5,
        randomize_num_vars=True,
        print_prob=0.5,
        loss_on="suffix",
        rng=random.Random(19),
    )

    for row in range(batch.idx.size(0)):
        prompt_len = int(batch.prompt_lengths[row].item())
        output_len = int(batch.output_lengths[row].item())
        target_suffix = batch.targets[row, prompt_len - 1 : prompt_len - 1 + output_len]

        assert torch.equal(batch.targets[row, : prompt_len - 1], torch.full((prompt_len - 1,), -1))
        assert not batch.metric_mask[row, : prompt_len - 1].any()
        assert batch.metric_mask[row, prompt_len - 1 : prompt_len - 1 + output_len].all()
        assert not batch.metric_mask[row, prompt_len - 1 + output_len :].any()

        full_ids = batch.idx[row, :prompt_len].tolist() + target_suffix.tolist()
        result = repl.verify_repl_trace_ids(full_ids, itos, value_mod=5)
        assert result.ok, result.python_source
