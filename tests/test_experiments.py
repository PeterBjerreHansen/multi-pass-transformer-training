from __future__ import annotations

from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch

from experiments.common import (
    RunArtifacts,
    evaluate_prebuilt_batches,
    gradient_norms,
    load_checkpoint_payload,
    restore_checkpoint_state,
    runtime_resource_stats,
    sample_train_position_offset,
    save_latest_checkpoint,
)
from experiments.summarize_ablation import recommend
from experiments.eval_diagnostics import memory_interventions, pass_dynamics, teacher_forced_schedule_gap
from experiments.eval_othello import build_eval_examples, legal_set_step_metrics
from experiments.presets import BBH_PRESETS, TRACE_PRESETS
from experiments.train_bbh import BBH_TASKS, build_fixed_eval_batches, parse_args as parse_bbh_args
from models import JointMemoryTapeTransformer, MemoryTapeConfig, MemoryTapeTransformer, MultiPassConfig
from tasks.bbh import pointer_chasing
from tasks.trace import othello, random_graph_walk
from tasks.trace.registry import TRACE_TASKS


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

    path = TRACE_PRESETS["shortest_path_main"].values
    assert path["num_nodes"] == 24
    assert path["shortest_path_length"] == 6
    assert path["branching_factor"] == 3
    assert path["distractor_edges"] == 40


def test_othello_prefix_examples_and_legal_set_metrics_are_deterministic():
    _vocab, stoi, _itos = othello.build_othello_vocab(
        othello_train_games=1,
        othello_val_games=1,
    )
    traces = [othello.random_game_trace64(seed=7), othello.random_game_trace64(seed=8)]
    first = build_eval_examples(
        traces,
        stoi=stoi,
        evaluation_mode="all",
        prefix_fractions=(0.25, 0.5, 0.75),
        rng=random.Random(99),
    )
    second = build_eval_examples(
        traces,
        stoi=stoi,
        evaluation_mode="all",
        prefix_fractions=(0.25, 0.5, 0.75),
        rng=random.Random(99),
    )
    assert first == second
    assert {example.protocol for example in first} == {
        "full-game",
        "random-prefix",
        "prefix-grid-0.25",
        "prefix-grid-0.5",
        "prefix-grid-0.75",
    }
    assert all(0 <= example.cut < len(example.trace_move_ids) for example in first)

    legal_ids = othello.legal_move_token_ids_after_prefix(())
    logits = torch.zeros(len(stoi))
    metrics = legal_set_step_metrics(logits, legal_ids, legal_ids[0])
    assert metrics["legal_set_size"] == len(legal_ids)
    assert metrics["legal_probability_mass"] == pytest.approx(len(legal_ids) / len(stoi))
    assert metrics["legal_set_nll"] == pytest.approx(
        -torch.log(torch.tensor(len(legal_ids) / len(stoi))).item()
    )


def test_trace_registry_preserves_seeded_task_behavior(tmp_path):
    graph_args = SimpleNamespace(
        num_states=4,
        label_pool_size=4,
        max_level=5,
        batch_size=3,
        device="cpu",
    )
    graph_task = TRACE_TASKS["random_graph_walk"]
    direct_vocab = random_graph_walk.build_random_graph_walk_vocab(4, 4)
    assert graph_task.build_vocab(graph_args) == direct_vocab
    assert graph_task.required_block_size(graph_args) == random_graph_walk.required_block_size(4, 4, 5)
    direct_graph_batch = random_graph_walk.build_random_graph_walk_batch(
        batch_size=3,
        num_states=4,
        label_pool_size=4,
        num_steps=5,
        stoi=direct_vocab[1],
        device="cpu",
        rng=random.Random(2026),
    )
    registered_graph_batch = graph_task.build_batch(
        graph_args,
        direct_vocab[1],
        random.Random(2026),
        split="train",
    )
    assert torch.equal(registered_graph_batch.idx, direct_graph_batch.idx)
    assert torch.equal(registered_graph_batch.targets, direct_graph_batch.targets)

    othello_args = SimpleNamespace(
        batch_size=3,
        device="cpu",
        othello_data_dir=str(tmp_path / "othello"),
        othello_train_games=8,
        othello_val_games=4,
        othello_dataset_seed=31,
        othello_prepend_opening=False,
    )
    othello_task = TRACE_TASKS["othello"]
    direct_othello_vocab = othello.build_othello_vocab(
        othello_train_games=8,
        othello_val_games=4,
    )
    assert othello_task.build_vocab(othello_args) == direct_othello_vocab
    assert othello_task.required_block_size(othello_args) == othello.required_block_size(
        othello_prepend_opening=False,
        othello_train_games=8,
        othello_val_games=4,
    )
    direct_othello_batch = othello.build_othello_batch(
        batch_size=3,
        stoi=direct_othello_vocab[1],
        device="cpu",
        rng=random.Random(2027),
        split="val",
        othello_data_dir=othello_args.othello_data_dir,
        othello_train_games=8,
        othello_val_games=4,
        othello_dataset_seed=31,
        othello_prepend_opening=False,
    )
    registered_othello_batch = othello_task.build_batch(
        othello_args,
        direct_othello_vocab[1],
        random.Random(2027),
        split="val",
    )
    assert torch.equal(registered_othello_batch.idx, direct_othello_batch.idx)
    assert torch.equal(registered_othello_batch.targets, direct_othello_batch.targets)


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


def test_gate_init_ablation_presets_differ_only_in_gate_initialization():
    control = TRACE_PRESETS["shortest_path_gate_init_control"].values
    treatment = TRACE_PRESETS["shortest_path_gate_init_unit"].values
    assert control["memory_gate_init"] == 0.1
    assert treatment["memory_gate_init"] == 1.0
    assert {
        key: value for key, value in control.items() if key != "memory_gate_init"
    } == {
        key: value for key, value in treatment.items() if key != "memory_gate_init"
    }


def test_main_bbh_preset_contract_is_frozen():
    common = {
        "model_size": "small",
        "n_pass": 4,
        "pass_loss_weights": [0.0, 0.0, 1.0, 1.0],
        "batch_size": 64,
        "train_steps": 50_000,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "eval_interval": 5_000,
        "eval_batches": 4,
        "seed": 1337,
        "max_level": 64,
        "curriculum_threshold": 0.95,
        "review_easier_every": 2,
        "token_selection": "argmax",
        "inference_mode": "recompute",
        "memory_gate_init": 0.1,
    }
    task_contracts = {
        "pointer_chasing_main": {
            "task": "pointer_chasing",
            "num_nodes": 8,
            "curriculum_start_level": 0,
        },
        "tracking_main": {
            "task": "tracking",
            "num_objects": 4,
            "curriculum_start_level": 1,
        },
        "permutation_main": {
            "task": "permutation",
            "num_objects": 4,
            "curriculum_start_level": 1,
        },
        "state_machine_main": {
            "task": "state_machine",
            "num_states": 4,
            "alphabet_size": 2,
            "curriculum_start_level": 0,
        },
    }
    for name, task_values in task_contracts.items():
        values = BBH_PRESETS[name].values
        for key, expected in {**common, **task_values}.items():
            assert values[key] == expected, f"{name}.{key} changed"


def test_main_trace_preset_contract_is_frozen():
    common = {
        "model_size": "small",
        "n_pass": 4,
        "pass_loss_weights": [0.0, 0.0, 1.0, 1.0],
        "weight_decay": 0.0,
        "seed": 1337,
        "inference_mode": "append_recurrent",
        "memory_gate_init": 0.1,
    }
    contracts = {
        "random_graph_walk_main": {
            "task": "random_graph_walk",
            "batch_size": 64,
            "train_steps": 50_000,
            "lr": 3e-4,
            "eval_interval": 1_000,
            "eval_batches": 4,
            "num_states": 6,
            "label_pool_size": 4,
            "max_level": 32,
        },
        "othello_main": {
            "task": "othello",
            "batch_size": 128,
            "train_steps": 500_000,
            "lr": 3e-4,
            "eval_interval": 5_000,
            "eval_batches": 1,
            "othello_train_games": 5_000_000,
            "othello_val_games": 1_024,
        },
        "shortest_path_main": {
            "task": "shortest_path",
            "batch_size": 64,
            "train_steps": 50_000,
            "lr": 3e-4,
            "eval_interval": 1_000,
            "eval_batches": 4,
            "num_nodes": 24,
            "shortest_path_length": 6,
            "branching_factor": 3,
            "distractor_edges": 40,
        },
    }
    for name, contract in contracts.items():
        values = TRACE_PRESETS[name].values
        for key, expected in {**common, **contract}.items():
            assert values[key] == expected, f"{name}.{key} changed"


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


def test_position_offset_sampling_is_deterministic():
    args = SimpleNamespace(train_position_offset_max=7)
    first = random.Random(12)
    second = random.Random(12)
    assert [sample_train_position_offset(args, first) for _ in range(20)] == [
        sample_train_position_offset(args, second) for _ in range(20)
    ]


def test_ablation_recommendation_accepts_task_specific_quality_metric():
    control = {
        str(seed): {"drift.append_recurrent.optimal_path": 0.30}
        for seed in range(3)
    }
    treatment = {
        str(seed): {"drift.append_recurrent.optimal_path": value}
        for seed, value in enumerate((0.32, 0.31, 0.30))
    }
    result = recommend(
        control,
        treatment,
        mode="quality-only",
        quality_metric="drift.append_recurrent.optimal_path",
    )
    assert result["quality_metric"] == "drift.append_recurrent.optimal_path"
    assert result["quality_win"]
    assert result["recommend_merge"]
