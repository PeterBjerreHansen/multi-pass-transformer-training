import random
from types import SimpleNamespace

import torch
import torch.nn as nn

from train_permutation import evaluate_batches


class TeacherForcingPerfectBadGenerator(nn.Module):
    def forward(self, idx):
        logits = torch.zeros(idx.size(0), idx.size(1), 3)
        logits[..., 1] = 10.0
        return logits

    def calc_loss(self, logits, targets):
        return torch.tensor(0.0)

    def generate(self, prompt, max_new_tokens, do_sample, generation_mode):
        wrong_suffix = torch.zeros(
            prompt.size(0),
            max_new_tokens,
            dtype=prompt.dtype,
            device=prompt.device,
        )
        return torch.cat((prompt, wrong_suffix), dim=1)


def _teacher_forced_metrics(logits: torch.Tensor, batch) -> tuple[float, float]:
    preds = logits.argmax(dim=-1)
    mask = batch.metric_mask
    correct = preds == batch.targets
    token_accuracy = correct[mask].float().mean().item()
    exact_match = (correct | ~mask).all(dim=1).float().mean().item()
    return exact_match, token_accuracy


def test_evaluate_batches_uses_generation_metrics_not_teacher_forcing():
    batch = SimpleNamespace(
        idx=torch.tensor([[2, 2, 2]]),
        targets=torch.tensor([[-1, 1, 1]]),
        metric_mask=torch.tensor([[False, True, True]]),
        prompt_lengths=torch.tensor([2]),
        output_lengths=torch.tensor([2]),
    )
    args = SimpleNamespace(
        architecture="transformer",
        generation_mode="greedy",
        eval_batches=1,
    )
    model = TeacherForcingPerfectBadGenerator()

    logits = model(batch.idx)
    teacher_forced_exact_match, teacher_forced_token_accuracy = _teacher_forced_metrics(logits, batch)
    assert teacher_forced_exact_match == 1.0
    assert teacher_forced_token_accuracy == 1.0

    metrics = evaluate_batches(
        model,
        args,
        batch_builder=lambda _: batch,
        rng=random.Random(0),
    )
    assert metrics["exact_match"] == 0.0
    assert metrics["token_accuracy"] == 0.0
