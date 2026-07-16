from __future__ import annotations

from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch

from experiments.common import (
    RunArtifacts,
    build_stale_memory_source_plan,
    evaluate_prebuilt_batches,
    forward_and_loss,
    gradient_norms,
    load_checkpoint_payload,
    new_stale_memory_stats,
    restore_checkpoint_state,
    runtime_resource_stats,
    save_latest_checkpoint,
    update_stale_memory_stats,
    validate_model_args,
)
from experiments.summarize_ablation import recommend
from experiments.eval_diagnostics import (
    memory_interventions,
    pass_dynamics,
    refinement_robustness,
    teacher_forced_schedule_gap,
)
from experiments.train_bbh import BBH_TASKS, build_fixed_eval_batches, parse_args as parse_bbh_args
from models import JointMemoryTapeTransformer, MemoryTapeConfig, MemoryTapeTransformer, MultiPassConfig
from tasks.bbh import pointer_chasing


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        architecture="memory_tape",
        inference_mode="recompute",
        token_selection="sample",
        pass_loss_weights=[0, 0, 1],
        seed=17,
        device="cpu",
        batch_size=2,
        eval_batches=2,
        task="pointer_chasing",
        num_nodes=4,
        curriculum_start_level=0,
        max_level=2,
    )


def test_fixed_eval_batches_are_identical_every_time():
    args = _args()
    _vocab, stoi, _ = pointer_chasing.build_pointer_chasing_vocab(4)
    task = BBH_TASKS["pointer_chasing"]
    a = build_fixed_eval_batches(args, task, stoi, 2)
    b = build_fixed_eval_batches(args, task, stoi, 2)
    for batch_a, batch_b in zip(a, b):
        assert torch.equal(batch_a.idx, batch_b.idx)
        assert torch.equal(batch_a.targets, batch_b.targets)


def test_evaluation_sampling_is_repeatable_and_does_not_change_global_rng():
    args = _args()
    model = MemoryTapeTransformer(MemoryTapeConfig(24, 11, 1, 1, 8, 3))
    _vocab, stoi, _ = pointer_chasing.build_pointer_chasing_vocab(4)
    batch = pointer_chasing.build_pointer_chasing_batch(2, 4, 2, stoi, device="cpu", rng=random.Random(2))
    before = torch.get_rng_state().clone()
    first = evaluate_prebuilt_batches(model, args, [batch], generation_seed=123)
    middle = torch.get_rng_state().clone()
    second = evaluate_prebuilt_batches(model, args, [batch], generation_seed=123)
    after = torch.get_rng_state().clone()
    assert first["exact_match"] == second["exact_match"]
    assert first["token_accuracy"] == second["token_accuracy"]
    assert torch.equal(before, middle)
    assert torch.equal(before, after)


def test_memory_interventions_pass_dynamics_and_schedule_gap_return_finite_values():
    model = MemoryTapeTransformer(MemoryTapeConfig(24, 11, 1, 1, 8, 3))
    _vocab, stoi, _ = pointer_chasing.build_pointer_chasing_vocab(4)
    batch = pointer_chasing.build_pointer_chasing_batch(2, 4, 2, stoi, device="cpu", rng=random.Random(2))
    full_output = model(batch.idx)
    interventions = memory_interventions(model, batch, seed=3)
    assert interventions["losses"]["correct"] == model.calc_loss(full_output.logits, batch.targets).item()
    assert set(interventions["losses"]) == {
        "correct", "zero_memory_bank", "masked_memory_source", "cross_example",
        "causal_position_resample", "causal_prefix_mean", "extra_lag"
    }
    assert interventions["losses"]["zero_memory_bank"] == interventions["losses"]["masked_memory_source"]
    assert interventions["loss_deltas"]["correct"] == 0.0
    dynamics = pass_dynamics(model, batch, extra_passes=2)
    assert len(dynamics["trained_passes"]) == 3
    assert len(dynamics["extra_passes"]) == 2
    assert all(torch.isfinite(torch.tensor(item["loss"])) for item in dynamics["extra_passes"])
    robustness = refinement_robustness(model, batch)
    assert robustness["final_pass"] == 3
    assert [item["source_pass"] for item in robustness["sources"]] == [1, 2]
    assert robustness["sources"][-1]["loss_delta_from_latest"] == 0.0
    assert robustness["sources"][-1]["loss"] == model.calc_loss(
        full_output.logits,
        batch.targets,
    ).item()
    assert all(torch.isfinite(torch.tensor(item["loss"])) for item in robustness["sources"])

    model.eval()
    schedule_gap = teacher_forced_schedule_gap(model, batch, horizon=2)
    assert schedule_gap["horizon"] == 2
    assert [item["count"] for item in schedule_gap["positions"]] == [2, 2]
    assert schedule_gap["overall"]["count"] == 4
    first = schedule_gap["positions"][0]
    assert first["logit_kl"] < 1e-6
    assert abs(first["nll_delta"]) < 1e-6
    assert first["top1_agreement"] == 1.0
    assert first["memory_rms_delta"] < 1e-6
    for position in schedule_gap["positions"]:
        for name, value in position.items():
            if name not in {"generated_position", "count"}:
                assert torch.isfinite(torch.tensor(value)), name


def test_joint_memory_tape_diagnostics_return_finite_values():
    model = JointMemoryTapeTransformer(MultiPassConfig(24, 11, 1, 1, 8, 3))
    _vocab, stoi, _ = pointer_chasing.build_pointer_chasing_vocab(4)
    batch = pointer_chasing.build_pointer_chasing_batch(2, 4, 2, stoi, device="cpu", rng=random.Random(2))
    interventions = memory_interventions(model, batch, seed=3)
    assert all(torch.isfinite(torch.tensor(value)) for value in interventions["losses"].values())
    assert {"zero_memory_bank", "masked_memory_source"} <= set(interventions["losses"])
    dynamics = pass_dynamics(model, batch, extra_passes=2)
    assert len(dynamics["extra_passes"]) == 2
    schedule_gap = teacher_forced_schedule_gap(model, batch, horizon=2)
    assert schedule_gap["overall"]["count"] == 4
    assert all(torch.isfinite(torch.tensor(value)) for value in schedule_gap["overall"].values())


def test_stale_memory_plan_only_corrupts_teacher_forced_suffix_memory():
    _vocab, stoi, _ = pointer_chasing.build_pointer_chasing_vocab(4)
    batch = pointer_chasing.build_pointer_chasing_batch(
        2,
        4,
        2,
        stoi,
        device="cpu",
        rng=random.Random(2),
    )
    routing, stats = build_stale_memory_source_plan(
        batch,
        n_pass=4,
        probability=1.0,
        generator=torch.Generator(device="cpu").manual_seed(41),
    )
    assert routing is not None
    eligible_per_pass = sum(int(length.item()) - 1 for length in batch.output_lengths)
    assert stats["eligible_routes"] == eligible_per_pass * 2
    assert stats["stale_routes"] == stats["eligible_routes"]
    assert stats["realized_stale_fraction"] == 1.0
    assert sum(stats["stale_source_pass_counts"].values()) == stats["stale_routes"]

    for row in range(batch.idx.shape[0]):
        prompt_len = int(batch.prompt_lengths[row].item())
        suffix_memory_len = int(batch.output_lengths[row].item()) - 1
        suffix = slice(prompt_len, prompt_len + suffix_memory_len)
        assert torch.equal(routing[0, row], torch.zeros_like(routing[0, row]))
        assert torch.equal(routing[1, row], torch.ones_like(routing[1, row]))
        for pass_index in range(2, 4):
            expected_latest = torch.full_like(routing[pass_index, row], pass_index)
            assert torch.equal(routing[pass_index, row, :prompt_len], expected_latest[:prompt_len])
            assert torch.equal(
                routing[pass_index, row, prompt_len + suffix_memory_len :],
                expected_latest[prompt_len + suffix_memory_len :],
            )
            assert (routing[pass_index, row, suffix] >= 1).all()
            assert (routing[pass_index, row, suffix] < pass_index).all()


def test_stale_memory_rng_is_separate_reproducible_and_zero_prob_is_noop():
    _vocab, stoi, _ = pointer_chasing.build_pointer_chasing_vocab(4)
    batch = pointer_chasing.build_pointer_chasing_batch(
        2,
        4,
        2,
        stoi,
        device="cpu",
        rng=random.Random(2),
    )
    generator = torch.Generator(device="cpu").manual_seed(73)
    state = generator.get_state()
    first, first_stats = build_stale_memory_source_plan(
        batch,
        n_pass=4,
        probability=0.5,
        generator=generator,
    )
    restored = torch.Generator(device="cpu")
    restored.set_state(state)
    second, second_stats = build_stale_memory_source_plan(
        batch,
        n_pass=4,
        probability=0.5,
        generator=restored,
    )
    assert torch.equal(first, second)
    assert first_stats == second_stats

    before = generator.get_state()
    routing, zero_stats = build_stale_memory_source_plan(
        batch,
        n_pass=4,
        probability=0.0,
        generator=generator,
    )
    assert routing is None
    assert zero_stats["stale_routes"] == 0
    assert torch.equal(before, generator.get_state())

    cumulative = new_stale_memory_stats(0.5)
    update_stale_memory_stats(cumulative, first_stats)
    update_stale_memory_stats(cumulative, second_stats)
    assert cumulative["eligible_routes"] == 2 * first_stats["eligible_routes"]
    assert cumulative["stale_routes"] == 2 * first_stats["stale_routes"]
    assert cumulative["realized_stale_fraction"] == first_stats["realized_stale_fraction"]


def test_stale_memory_probability_validation():
    args = parse_bbh_args([
        "--preset", "pointer_chasing_smoke",
        "--architecture", "memory_tape",
        "--stale-memory-prob", "0.25",
    ])
    validate_model_args(args)
    assert args.stale_memory_prob == 0.25

    args.stale_memory_prob = 1.01
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_model_args(args)
    args.stale_memory_prob = 0.25
    args.n_pass = 2
    args.pass_loss_weights = [1, 1]
    with pytest.raises(ValueError, match="n-pass >= 3"):
        validate_model_args(args)


def test_gradient_norms_cover_memory_subsystems_after_backward():
    model = MemoryTapeTransformer(MemoryTapeConfig(24, 11, 1, 1, 8, 3))
    _vocab, stoi, _ = pointer_chasing.build_pointer_chasing_vocab(4)
    batch = pointer_chasing.build_pointer_chasing_batch(2, 4, 2, stoi, device="cpu", rng=random.Random(2))
    output = model(batch.idx)
    loss = model.calc_total_loss(output, batch.targets, [0, 0, 1]).loss
    loss.backward()
    norms = gradient_norms(model)
    assert {"global", "backbone", "memory_writer", "memory_attention", "memory_gate"} <= set(norms)
    assert all(torch.isfinite(torch.tensor(value)) and value > 0 for value in norms.values())


def test_joint_memory_tape_gradient_norms_cover_memory_attention_after_backward():
    model = JointMemoryTapeTransformer(MultiPassConfig(24, 11, 1, 1, 8, 3))
    _vocab, stoi, _ = pointer_chasing.build_pointer_chasing_vocab(4)
    batch = pointer_chasing.build_pointer_chasing_batch(2, 4, 2, stoi, device="cpu", rng=random.Random(2))
    output = model(batch.idx)
    loss = model.calc_total_loss(output, batch.targets, [0, 0, 1]).loss
    loss.backward()
    norms = gradient_norms(model)
    assert {"global", "backbone", "memory_writer", "memory_attention"} <= set(norms)
    assert "memory_gate" not in norms
    assert all(torch.isfinite(torch.tensor(value)) and value > 0 for value in norms.values())
    memory_reader_grads = (
        model.transformer.h[0].joint_attn.c_mem_kv.weight.grad,
        model.transformer.h[0].ln_mem_kv.weight.grad,
    )
    assert all(gradient is not None for gradient in memory_reader_grads)
    expected_memory_norm = sum(
        gradient.detach().float().square().sum()
        for gradient in memory_reader_grads
        if gradient is not None
    ).sqrt().item()
    assert norms["memory_attention"] == pytest.approx(expected_memory_norm)


def _one_step(model, optimizer, tokens, targets):
    optimizer.zero_grad(set_to_none=True)
    output = model(tokens)
    loss = model.calc_total_loss(output, targets, [0, 0, 1]).loss
    loss.backward()
    optimizer.step()


def test_checkpoint_resume_reproduces_next_optimizer_step(tmp_path):
    torch.manual_seed(101)
    config = MemoryTapeConfig(8, 13, 1, 1, 8, 3)
    model = MemoryTapeTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(55)
    batch1 = torch.randint(0, 13, (2, 6), generator=gen)
    target1 = torch.randint(0, 13, (2, 6), generator=gen)
    batch2 = torch.randint(0, 13, (2, 6), generator=gen)
    target2 = torch.randint(0, 13, (2, 6), generator=gen)

    _one_step(model, optimizer, batch1, target1)
    artifacts = RunArtifacts(tmp_path, tmp_path / "config.json", tmp_path / "metrics.jsonl", tmp_path / "latest.pt")
    args = SimpleNamespace(example=True)
    local_rng = random.Random(9)
    save_latest_checkpoint(
        artifacts,
        model=model,
        optimizer=optimizer,
        args=args,
        step=1,
        extra_state={"local_rng": local_rng.getstate()},
    )
    _one_step(model, optimizer, batch2, target2)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}

    torch.manual_seed(999)
    restored_model = MemoryTapeTransformer(config)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    checkpoint = load_checkpoint_payload(tmp_path / "latest.pt", device="cpu")
    restore_checkpoint_state(checkpoint, model=restored_model, optimizer=restored_optimizer, device="cpu")
    _one_step(restored_model, restored_optimizer, batch2, target2)
    for name, value in restored_model.state_dict().items():
        assert torch.equal(value, expected[name]), name


def test_stale_memory_rng_checkpoint_resume_reproduces_next_step(tmp_path):
    torch.manual_seed(101)
    config = MemoryTapeConfig(24, 11, 1, 1, 8, 4)
    model = MemoryTapeTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = SimpleNamespace(
        architecture="memory_tape",
        pass_loss_weights=[0, 0, 1, 1],
    )
    _vocab, stoi, _ = pointer_chasing.build_pointer_chasing_vocab(4)
    task_rng = random.Random(31)
    batches = [
        pointer_chasing.build_pointer_chasing_batch(
            2,
            4,
            2,
            stoi,
            device="cpu",
            rng=task_rng,
        )
        for _ in range(2)
    ]
    stale_rng = torch.Generator(device="cpu").manual_seed(97)

    def step(local_model, local_optimizer, batch, generator):
        routing, _stats = build_stale_memory_source_plan(
            batch,
            n_pass=4,
            probability=0.5,
            generator=generator,
        )
        local_optimizer.zero_grad(set_to_none=True)
        loss, _output, _pass_losses = forward_and_loss(
            local_model,
            batch,
            args,
            memory_source_passes=routing,
        )
        loss.backward()
        local_optimizer.step()

    step(model, optimizer, batches[0], stale_rng)
    artifacts = RunArtifacts(
        tmp_path,
        tmp_path / "config.json",
        tmp_path / "metrics.jsonl",
        tmp_path / "latest.pt",
    )
    save_latest_checkpoint(
        artifacts,
        model=model,
        optimizer=optimizer,
        args=args,
        step=1,
        extra_state={"stale_memory_rng_state": stale_rng.get_state()},
    )
    step(model, optimizer, batches[1], stale_rng)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}

    restored_model = MemoryTapeTransformer(config)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    checkpoint = load_checkpoint_payload(tmp_path / "latest.pt", device="cpu")
    restore_checkpoint_state(
        checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
        device="cpu",
    )
    restored_rng = torch.Generator(device="cpu")
    restored_rng.set_state(checkpoint["extra_state"]["stale_memory_rng_state"])
    step(restored_model, restored_optimizer, batches[1], restored_rng)
    for name, value in restored_model.state_dict().items():
        assert torch.equal(value, expected[name]), name


def test_cli_has_only_two_inference_modes_and_no_cache_source():
    args = parse_bbh_args([
        "--preset", "pointer_chasing_smoke",
        "--architecture", "memory_tape",
        "--inference-mode", "append_recurrent",
    ])
    assert args.inference_mode == "append_recurrent"
    assert not hasattr(args, "cache_source")
    assert not hasattr(args, "memory_tape_gate")


def test_main_presets_preserve_established_experiment_scales():
    from experiments.presets import BBH_PRESETS, TRACE_PRESETS

    pointer = BBH_PRESETS["pointer_chasing_main"].values
    assert pointer["num_nodes"] == 8
    assert pointer["lr"] == 1e-4
    assert pointer["eval_interval"] == 5_000
    assert pointer["batch_size"] == 64

    graph = TRACE_PRESETS["random_graph_walk_main"].values
    assert graph["lr"] == 3e-4
    assert graph["eval_interval"] == 1_000
    assert graph["inference_mode"] == "append_recurrent"

    othello_main = TRACE_PRESETS["othello_main"].values
    assert othello_main["othello_train_games"] == 5_000_000
    assert othello_main["othello_val_games"] == 1_024
    assert othello_main["batch_size"] == 128
    assert othello_main["eval_interval"] == 5_000


def test_memory_update_direct_default_matches_experiment_default():
    from models import MemoryUpdateConfig

    config = MemoryUpdateConfig(
        block_size=8,
        vocab_size=13,
        n_layer=1,
        n_head=1,
        n_embd=8,
        n_pass=3,
    )
    assert config.use_memory_gate is False


def test_runtime_resource_stats_reports_peak_rss():
    stats = runtime_resource_stats("cpu")
    assert stats["process_peak_rss_bytes"] > 0


def test_ablation_recommendation_accepts_noninferior_efficiency_win():
    control = {
        str(seed): {
            "drift.append_recurrent.token_legality": 0.80,
            "train.train_tok_per_s": 100.0,
            "model.non_embedding_parameters": 1000.0,
        }
        for seed in range(3)
    }
    treatment = {
        str(seed): {
            "drift.append_recurrent.token_legality": 0.795,
            "train.train_tok_per_s": 115.0,
            "model.non_embedding_parameters": 1000.0,
        }
        for seed in range(3)
    }
    result = recommend(control, treatment, mode="pareto")
    assert result["quality_noninferior"]
    assert result["efficiency_win"]
    assert result["recommend_merge"]
