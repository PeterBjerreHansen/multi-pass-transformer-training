from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from experiments.common import validate_model_args
from model_factory import build_model, supports_append_recurrent
from models import LoopedTransformer, LoopedTransformerConfig


def looped_model(
    *,
    layout: str = "sandwich",
    persistent_input: bool = False,
    n_pass: int = 3,
) -> LoopedTransformer:
    torch.manual_seed(7)
    return LoopedTransformer(
        LoopedTransformerConfig(
            block_size=10,
            vocab_size=17,
            n_layer=4,
            n_head=2,
            n_embd=8,
            n_pass=n_pass,
            loop_layout=layout,
            persistent_input=persistent_input,
        )
    )


def test_sandwich_config_requires_a_prelude_core_and_coda():
    with pytest.raises(ValueError, match="at least three"):
        LoopedTransformerConfig(10, 17, 2, 1, 8, 3, loop_layout="sandwich")
    with pytest.raises(ValueError, match="loop_layout"):
        LoopedTransformerConfig(10, 17, 4, 1, 8, 3, loop_layout="unknown")


def test_training_reads_out_only_final_state_while_diagnostics_capture_iterations():
    model = looped_model(n_pass=5)
    tokens = torch.randint(0, 17, (2, 6))
    training_output = model(tokens)
    diagnostic_output = model.forward_iterations(tokens)
    assert len(training_output.passes) == 1
    assert len(diagnostic_output.passes) == 5
    assert training_output.logits.shape == (2, 6, 17)
    assert torch.allclose(training_output.logits, diagnostic_output.logits)
    states = tuple(item.memory_states for item in diagnostic_output.passes)
    assert all(state is not None and state.shape == (2, 6, 8) for state in states)


def test_full_and_sandwich_layouts_are_exactly_parameter_matched():
    full = looped_model(layout="full", n_pass=4)
    sandwich = looped_model(layout="sandwich", n_pass=7)
    assert full.get_num_params() == sandwich.get_num_params()
    assert tuple(full.state_dict()) == tuple(sandwich.state_dict())


def test_four_layer_matched_compute_control_has_sixteen_block_applications():
    tokens = torch.randint(0, 17, (1, 5))
    counts = {}
    for name, model in {
        "full": looped_model(layout="full", n_pass=4),
        "sandwich": looped_model(layout="sandwich", n_pass=7),
    }.items():
        calls = [0]
        hooks = [
            block.register_forward_hook(lambda *_args: calls.__setitem__(0, calls[0] + 1))
            for block in model.transformer.h
        ]
        model(tokens)
        for hook in hooks:
            hook.remove()
        counts[name] = calls[0]
    assert counts == {"full": 16, "sandwich": 16}


def test_prelude_and_coda_run_once_while_core_repeats():
    model = looped_model(layout="sandwich", n_pass=5)
    calls = [0, 0, 0, 0]
    hooks = [
        block.register_forward_hook(
            lambda _module, _inputs, _output, index=index: calls.__setitem__(index, calls[index] + 1)
        )
        for index, block in enumerate(model.transformer.h)
    ]
    model(torch.randint(0, 17, (1, 5)))
    for hook in hooks:
        hook.remove()
    assert calls == [1, 5, 5, 1]


def test_persistent_input_toggle_is_parameter_matched_and_controls_gradients():
    tokens = torch.randint(0, 17, (2, 6))
    targets = torch.randint(0, 17, (2, 6))
    for enabled in (False, True):
        model = looped_model(persistent_input=enabled)
        output = model(tokens)
        model.calc_loss(output.logits, targets).backward()
        injection_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith(("input_projection", "input_step_bias", "retention_log_scale"))
        ]
        if enabled:
            assert all(parameter.grad is not None for parameter in injection_parameters)
            assert sum(parameter.grad.abs().sum().item() for parameter in injection_parameters) > 0
        else:
            assert all(parameter.grad is None for parameter in injection_parameters)


def test_initial_persistent_write_is_stable_and_channelwise():
    model = looped_model(persistent_input=True)
    alpha, delta = model.input_write_coefficients()
    assert alpha.shape == delta.shape == (8,)
    assert torch.allclose(delta, torch.full_like(delta, 0.1), atol=1e-7)
    assert torch.allclose(alpha, torch.full_like(alpha, torch.exp(torch.tensor(-0.1))), atol=1e-7)


def test_looped_transformer_is_causal():
    model = looped_model(persistent_input=True)
    model.eval()
    prefix = torch.tensor([[1, 2, 3, 4]])
    a = torch.cat((prefix, torch.tensor([[5, 6]])), dim=1)
    b = torch.cat((prefix, torch.tensor([[8, 9]])), dim=1)
    logits_a = model(a).logits[:, : prefix.shape[1]]
    logits_b = model(b).logits[:, : prefix.shape[1]]
    assert torch.allclose(logits_a, logits_b, atol=1e-6, rtol=1e-5)


def test_looped_generation_is_recompute_only():
    model = looped_model()
    prompt = torch.tensor([[1, 2, 3]])
    generated = model.generate(prompt, 2, do_sample=False, inference_mode="recompute")
    assert generated.shape == (1, 5)
    with pytest.raises(ValueError, match="only supports"):
        model.generate(prompt, 1, do_sample=False, inference_mode="append_recurrent")
    assert not supports_append_recurrent("looped_transformer")


def test_factory_builds_both_loop_ablation_settings():
    base = dict(
        architecture="looped_transformer",
        n_layer=4,
        n_head=2,
        n_embd=8,
        n_pass=3,
    )
    off = build_model(
        SimpleNamespace(**base, loop_layout="full", loop_persistent_input="off"),
        17,
        10,
        "cpu",
    )
    on = build_model(
        SimpleNamespace(**base, loop_layout="sandwich", loop_persistent_input="on"),
        17,
        10,
        "cpu",
    )
    assert isinstance(off, LoopedTransformer)
    assert off.config.loop_layout == "full"
    assert not off.config.persistent_input
    assert on.config.loop_layout == "sandwich"
    assert on.config.persistent_input


def test_cli_validation_rejects_append_mode_for_depth_recurrence():
    args = SimpleNamespace(
        architecture="looped_transformer",
        model_size="small",
        n_layer=4,
        n_head=2,
        n_embd=8,
        n_pass=4,
        pass_loss_weights=[0, 0, 0, 1],
        inference_mode="append_recurrent",
        loop_layout="full",
        loop_persistent_input="off",
    )
    with pytest.raises(ValueError, match="recompute"):
        validate_model_args(args)
