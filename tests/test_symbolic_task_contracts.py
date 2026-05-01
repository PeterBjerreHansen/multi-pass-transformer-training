import random

import torch

from tasks.bbh import permutation, pointer_chasing, state_machine, tracking
from tasks.common import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, SEP_TOKEN
from tasks.trace import othello, random_graph_walk


def _assert_symbolic_batch_contract(batch, *, stoi, vocab_size: int, required_block_size: int):
    assert batch.idx.shape == batch.targets.shape == batch.metric_mask.shape
    assert batch.idx.size(0) == 3
    assert batch.idx.size(1) <= required_block_size
    assert int(batch.idx.min()) >= 0
    assert int(batch.idx.max()) < vocab_size

    valid_targets = batch.targets[batch.targets != -1]
    assert valid_targets.numel() > 0
    assert int(valid_targets.min()) >= 0
    assert int(valid_targets.max()) < vocab_size

    for row in range(batch.idx.size(0)):
        prompt_len = int(batch.prompt_lengths[row])
        output_len = int(batch.output_lengths[row])
        active_idx_len = prompt_len + output_len - 1
        suffix_start = prompt_len - 1
        suffix_end = suffix_start + output_len

        idx_row = batch.idx[row]
        target_row = batch.targets[row]
        metric_mask = batch.metric_mask[row]
        target_suffix = target_row[suffix_start:suffix_end]

        assert int(idx_row[0]) == stoi[BOS_TOKEN]
        assert int(idx_row[prompt_len - 1]) == stoi[SEP_TOKEN]
        assert int(target_suffix[-1]) == stoi[EOS_TOKEN]
        assert not (idx_row[:active_idx_len] == stoi[PAD_TOKEN]).any()

        expected_mask = torch.zeros_like(metric_mask)
        expected_mask[suffix_start:suffix_end] = True
        assert torch.equal(metric_mask, expected_mask)
        assert torch.equal(target_row[:suffix_start], torch.full((suffix_start,), -1))
        assert torch.equal(target_row[suffix_end:], torch.full_like(target_row[suffix_end:], -1))
        assert torch.equal(idx_row[active_idx_len:], torch.full_like(idx_row[active_idx_len:], stoi[PAD_TOKEN]))


def test_symbolic_task_batches_follow_contract(tmp_path):
    cases = [
        (
            "permutation",
            permutation.build_permutation_vocab(num_objects=4),
            permutation.build_permutation_batch(
                batch_size=3,
                num_objects=4,
                num_swaps=5,
                stoi=permutation.build_permutation_vocab(num_objects=4)[1],
                device="cpu",
                rng=random.Random(2001),
            ),
            permutation.required_block_size(num_objects=4, num_swaps=5),
        ),
        (
            "tracking",
            tracking.build_tracking_vocab(num_objects=4),
            tracking.build_tracking_batch(
                batch_size=3,
                num_objects=4,
                num_ops=5,
                stoi=tracking.build_tracking_vocab(num_objects=4)[1],
                device="cpu",
                rng=random.Random(2002),
            ),
            tracking.required_block_size(num_objects=4, num_ops=5),
        ),
        (
            "pointer_chasing",
            pointer_chasing.build_pointer_chasing_vocab(num_nodes=12),
            pointer_chasing.build_pointer_chasing_batch(
                batch_size=3,
                num_nodes=12,
                num_hops=5,
                stoi=pointer_chasing.build_pointer_chasing_vocab(num_nodes=12)[1],
                device="cpu",
                rng=random.Random(2003),
            ),
            pointer_chasing.required_block_size(num_nodes=12, num_hops=5),
        ),
        (
            "state_machine",
            state_machine.build_state_machine_vocab(num_states=4, alphabet_size=2),
            state_machine.build_state_machine_batch(
                batch_size=3,
                num_states=4,
                alphabet_size=2,
                num_steps=5,
                stoi=state_machine.build_state_machine_vocab(num_states=4, alphabet_size=2)[1],
                device="cpu",
                rng=random.Random(2004),
            ),
            state_machine.required_block_size(num_states=4, alphabet_size=2, num_steps=5),
        ),
        (
            "random_graph_walk",
            random_graph_walk.build_random_graph_walk_vocab(num_states=4, label_pool_size=4),
            random_graph_walk.build_random_graph_walk_batch(
                batch_size=3,
                num_states=4,
                label_pool_size=4,
                num_steps=5,
                stoi=random_graph_walk.build_random_graph_walk_vocab(num_states=4, label_pool_size=4)[1],
                device="cpu",
                rng=random.Random(2005),
            ),
            random_graph_walk.required_block_size(num_states=4, label_pool_size=4, num_steps=5),
        ),
    ]

    for task_name, vocab_triplet, batch, required_block_size in cases:
        vocab, stoi, itos = vocab_triplet
        assert len(vocab) == len(stoi) == len(itos)
        assert set(stoi.values()) == set(range(len(vocab)))
        assert all(itos[index] == token for token, index in stoi.items())
        _assert_symbolic_batch_contract(
            batch,
            stoi=stoi,
            vocab_size=len(vocab),
            required_block_size=required_block_size,
        )

    vocab, stoi, itos = othello.build_othello_vocab(othello_train_games=16, othello_val_games=8)
    batch = othello.build_othello_batch(
        batch_size=3,
        stoi=stoi,
        device="cpu",
        rng=random.Random(2006),
        split="val",
        othello_data_dir=str(tmp_path / "othello_data"),
        othello_train_games=16,
        othello_val_games=8,
        othello_dataset_seed=11,
    )
    _assert_symbolic_batch_contract(
        batch,
        stoi=stoi,
        vocab_size=len(vocab),
        required_block_size=othello.required_block_size(othello_train_games=16, othello_val_games=8),
    )
