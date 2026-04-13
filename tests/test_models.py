import torch

from models import (
    CausalTransformer,
    MemoryConcatTransformer,
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
        (MemoryTapeTransformer, MultiPassConfig, {}),
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

        loss, pass_losses = model.calc_total_loss(
            logits_per_pass,
            targets,
            loss_weights=[0.1, 0.5, 1.0],
        )
        assert len(pass_losses) == config.n_pass
        assert torch.isfinite(loss)
        loss.backward()
        assert _finite_nonzero_gradient_exists(model)


def test_config_round_trips_for_future_checkpointing():
    config = TransformerConfig(block_size=8, vocab_size=17, n_layer=1, n_head=2, n_embd=8)
    restored = TransformerConfig.from_dict(config.to_dict())
    assert restored.to_dict() == config.to_dict()

    multipass_config = MultiPassConfig(block_size=8, vocab_size=17, n_layer=1, n_head=2, n_embd=8, n_pass=3)
    restored_multipass = MultiPassConfig.from_dict(multipass_config.to_dict())
    assert restored_multipass.to_dict() == multipass_config.to_dict()

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
