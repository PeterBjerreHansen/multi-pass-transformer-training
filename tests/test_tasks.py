import random

import torch

from tasks import permutation


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
