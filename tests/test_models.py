from types import SimpleNamespace

import pytest
import torch

from model_factory import build_model
from models import (
    CausalTransformer,
    MemoryBlock,
    MemoryConcatTransformer,
    MemoryTapeConfig,
    MemoryTapeTransformer,
    MemoryUpdateConfig,
    MemoryUpdateTransformer,
    MultiPassConfig,
    TransformerConfig,
)


def _finite_nonzero_gradient_exists(model) -> bool:
    found = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        assert torch.isfinite(parameter.grad).all()
        found = found or parameter.grad.abs().sum().item() > 0
    return found


def test_causal_transformer_forward_loss_backward_and_generate():
    torch.manual_seed(11)
    config = TransformerConfig(
        block_size=8,
        vocab_size=17,
        n_layer=1,
        n_head=2,
        n_embd=8,
    )
    model = CausalTransformer(config)
    idx = torch.randint(0, config.vocab_size, (2, 6))
    targets = torch.randint(0, config.vocab_size, (2, 6))

    logits = model(idx)
    assert logits.shape == (2, 6, config.vocab_size)

    loss = model.calc_loss(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert _finite_nonzero_gradient_exists(model)

    generated = model.generate(idx[:, :4], max_new_tokens=2, do_sample=False)
    assert generated.shape == (2, 6)


def test_multipass_transformers_forward_loss_and_backward():
    cases = (
        (MemoryTapeTransformer, MemoryTapeConfig, {}),
        (MemoryTapeTransformer, MemoryTapeConfig, {"memory_tape_gate": "none"}),
        (MemoryTapeTransformer, MemoryTapeConfig, {"memory_tape_gate": "scalar"}),
        (MemoryConcatTransformer, MultiPassConfig, {}),
        (MemoryUpdateTransformer, MemoryUpdateConfig, {}),
        (MemoryUpdateTransformer, MemoryUpdateConfig, {"use_memory_gate": False}),
    )

    for model_cls, config_cls, config_kwargs in cases:
        torch.manual_seed(23)
        config = config_cls(
            block_size=8,
            vocab_size=17,
            n_layer=1,
            n_head=2,
            n_embd=8,
            n_pass=3,
            **config_kwargs,
        )
        model = model_cls(config)
        idx = torch.randint(0, config.vocab_size, (2, 6))
        targets = torch.randint(0, config.vocab_size, (2, 6))

        logits_per_pass = model(idx, return_all_logits=True)
        assert len(logits_per_pass) == config.n_pass
        for logits in logits_per_pass:
            assert logits.shape == (2, 6, config.vocab_size)

        pass_logits, pass_memories = model.forward_passes(idx)
        assert len(pass_logits) == config.n_pass
        assert len(pass_memories) == config.n_pass
        for logits, memory in zip(pass_logits, pass_memories):
            assert logits.shape == (2, 6, config.vocab_size)
            assert memory.shape == (2, 6, config.n_embd)

        loss, pass_losses = model.calc_total_loss(
            logits_per_pass,
            targets,
            loss_weights=[0.1, 0.5, 1.0],
        )
        assert len(pass_losses) == config.n_pass
        assert torch.isfinite(loss)
        loss.backward()
        assert _finite_nonzero_gradient_exists(model)


def test_memory_tape_gate_modes():
    torch.manual_seed(29)
    n_embd = 8
    x = torch.randn(2, 6, n_embd)
    memory_a = torch.randn(2, 6, n_embd)
    memory_b = torch.randn(2, 6, n_embd)

    config = MemoryTapeConfig(
        block_size=8,
        vocab_size=17,
        n_layer=1,
        n_head=2,
        n_embd=n_embd,
        n_pass=2,
        memory_tape_gate="tanh",
    )
    block = MemoryBlock(config)
    assert block.memory_tape_gate == "tanh"
    assert block.memory_gate.item() == 0.0
    out_a = block(x, memory_a)
    out_b = block(x, memory_b)
    assert torch.allclose(out_a, out_b)

    config = MemoryTapeConfig(
        block_size=8,
        vocab_size=17,
        n_layer=1,
        n_head=2,
        n_embd=n_embd,
        n_pass=2,
        memory_tape_gate="scalar",
    )
    block = MemoryBlock(config)
    assert block.memory_tape_gate == "scalar"
    assert block.memory_gate.item() == pytest.approx(0.2)
    out_a = block(x, memory_a)
    out_b = block(x, memory_b)
    assert not torch.allclose(out_a, out_b)

    config = MemoryTapeConfig(
        block_size=8,
        vocab_size=17,
        n_layer=1,
        n_head=2,
        n_embd=n_embd,
        n_pass=2,
        memory_tape_gate="none",
    )
    block = MemoryBlock(config)
    assert block.memory_tape_gate == "none"
    assert block.memory_gate is None
    out_a = block(x, memory_a)
    out_b = block(x, memory_b)
    assert not torch.allclose(out_a, out_b)


def test_closed_memory_tape_first_pass_matches_causal_transformer_initialization():
    seed = 31
    transformer_config = TransformerConfig(block_size=8, vocab_size=17, n_layer=2, n_head=2, n_embd=8)
    idx = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ],
        dtype=torch.long,
    )

    for gate_mode in ("none", "tanh", "scalar"):
        multipass_config = MemoryTapeConfig(
            block_size=8,
            vocab_size=17,
            n_layer=2,
            n_head=2,
            n_embd=8,
            n_pass=3,
            memory_tape_gate=gate_mode,
        )

        torch.manual_seed(seed)
        baseline = CausalTransformer(transformer_config)
        torch.manual_seed(seed)
        memory_tape = MemoryTapeTransformer(multipass_config)

        with torch.no_grad():
            baseline_logits = baseline(idx)
            memory_tape_logits = memory_tape(idx, return_all_logits=True)[0]

        assert torch.equal(baseline.transformer.wte.weight, memory_tape.transformer.wte.weight)
        assert torch.equal(baseline.transformer.wpe.weight, memory_tape.transformer.wpe.weight)
        assert torch.equal(baseline.lm_head.weight, memory_tape.lm_head.weight)
        for baseline_block, memory_block in zip(baseline.transformer.h, memory_tape.transformer.h):
            assert torch.equal(baseline_block.attn.c_attn.weight, memory_block.attn.c_attn.weight)
            assert torch.equal(baseline_block.attn.c_proj.weight, memory_block.attn.c_proj.weight)
            assert torch.equal(baseline_block.mlp.c_fc.weight, memory_block.mlp.c_fc.weight)
            assert torch.equal(baseline_block.mlp.c_proj.weight, memory_block.mlp.c_proj.weight)
        assert torch.equal(baseline_logits, memory_tape_logits)


def test_config_round_trips_for_future_checkpointing():
    config = TransformerConfig(block_size=8, vocab_size=17, n_layer=1, n_head=2, n_embd=8)
    restored = TransformerConfig.from_dict(config.to_dict())
    assert restored.to_dict() == config.to_dict()

    multipass_config = MultiPassConfig(block_size=8, vocab_size=17, n_layer=1, n_head=2, n_embd=8, n_pass=3)
    restored_multipass = MultiPassConfig.from_dict(multipass_config.to_dict())
    assert restored_multipass.to_dict() == multipass_config.to_dict()

    memory_tape_config = MemoryTapeConfig(
        block_size=8,
        vocab_size=17,
        n_layer=1,
        n_head=2,
        n_embd=8,
        n_pass=3,
        memory_tape_gate="scalar",
    )
    restored_memory_tape = MemoryTapeConfig.from_dict(memory_tape_config.to_dict())
    assert restored_memory_tape.to_dict() == memory_tape_config.to_dict()
    legacy_memory_tape = MemoryTapeConfig.from_dict(
        {
            "block_size": 8,
            "vocab_size": 17,
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 8,
            "n_pass": 3,
        }
    )
    assert legacy_memory_tape.memory_tape_gate == "tanh"
    with pytest.raises(ValueError, match="memory_tape_gate must be one of"):
        MemoryTapeConfig(
            block_size=8,
            vocab_size=17,
            n_layer=1,
            n_head=2,
            n_embd=8,
            n_pass=3,
            memory_tape_gate="bad_gate",
        )
    none_memory_tape = MemoryTapeConfig(
        block_size=8,
        vocab_size=17,
        n_layer=1,
        n_head=2,
        n_embd=8,
        n_pass=3,
        memory_tape_gate="None",
    )
    assert none_memory_tape.memory_tape_gate == "none"


@pytest.mark.parametrize(
    "architecture,extra_args,model_cls,config_cls",
    (
        ("transformer", {}, CausalTransformer, TransformerConfig),
        ("memory_concat", {}, MemoryConcatTransformer, MultiPassConfig),
        ("memory_tape", {}, MemoryTapeTransformer, MemoryTapeConfig),
        ("memory_tape", {"memory_tape_gate": "none"}, MemoryTapeTransformer, MemoryTapeConfig),
        ("memory_tape", {"memory_tape_gate": "scalar"}, MemoryTapeTransformer, MemoryTapeConfig),
        ("memory_update", {}, MemoryUpdateTransformer, MemoryUpdateConfig),
        ("memory_update", {"memory_update_gate": "on"}, MemoryUpdateTransformer, MemoryUpdateConfig),
    ),
)
def test_build_model_constructs_expected_model_and_config_classes(
    architecture,
    extra_args,
    model_cls,
    config_cls,
):
    arg_values = dict(
        architecture=architecture,
        n_layer=1,
        n_head=2,
        n_embd=8,
        n_pass=3,
        memory_tape_gate="tanh",
        memory_update_gate="off",
        memory_gate_bias=-1.0,
    )
    arg_values.update(extra_args)
    args = SimpleNamespace(**arg_values)

    model = build_model(args, vocab_size=17, block_size=8, device="cpu")
    assert isinstance(model, model_cls)
    assert isinstance(model.config, config_cls)

    idx = torch.randint(0, model.config.vocab_size, (2, 6))
    logits = model(idx)
    assert logits.shape == (2, 6, model.config.vocab_size)

    if architecture == "memory_tape":
        assert model.transformer.h[0].memory_tape_gate == model.config.memory_tape_gate

    memory_update_config = MemoryUpdateConfig(
        block_size=8,
        vocab_size=17,
        n_layer=1,
        n_head=2,
        n_embd=8,
        n_pass=3,
        memory_gate_bias=-2.0,
        use_memory_gate=False,
    )
    restored_memory_update = MemoryUpdateConfig.from_dict(memory_update_config.to_dict())
    assert restored_memory_update.to_dict() == memory_update_config.to_dict()

    legacy_memory_update = MemoryUpdateConfig.from_dict(
        {
            "block_size": 8,
            "vocab_size": 17,
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 8,
            "n_pass": 3,
        }
    )
    assert legacy_memory_update.memory_gate_bias == -1.0
    assert legacy_memory_update.use_memory_gate is True
