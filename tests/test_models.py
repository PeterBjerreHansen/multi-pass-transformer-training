from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from model_factory import build_model
from models import (
    CausalCrossAttention,
    CausalTokenMemoryAttention,
    CausalTransformer,
    JointMemoryTapeTransformer,
    LayerNorm,
    MemoryBlock,
    MemoryConcatTransformer,
    MemoryTapeConfig,
    MemoryTapeTransformer,
    MemoryUpdateConfig,
    MemoryUpdateTransformer,
    MultiPassConfig,
    TransformerConfig,
    normalize_pass_weights,
    sample_next_token,
    shift_right,
)


def tiny_memory_model(*, block_size: int = 12, n_pass: int = 3) -> MemoryTapeTransformer:
    torch.manual_seed(7)
    return MemoryTapeTransformer(
        MemoryTapeConfig(
            block_size=block_size,
            vocab_size=19,
            n_layer=2,
            n_head=2,
            n_embd=8,
            n_pass=n_pass,
        )
    )


def tiny_joint_memory_model(*, block_size: int = 12, n_pass: int = 3) -> JointMemoryTapeTransformer:
    torch.manual_seed(7)
    return JointMemoryTapeTransformer(
        MultiPassConfig(
            block_size=block_size,
            vocab_size=19,
            n_layer=2,
            n_head=2,
            n_embd=8,
            n_pass=n_pass,
        )
    )


def test_shift_right_is_exact():
    memory = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    shifted = shift_right(memory)
    assert torch.equal(shifted[:, 0], torch.zeros_like(shifted[:, 0]))
    assert torch.equal(shifted[:, 1:], memory[:, :-1])


def test_pass_weights_are_always_normalized():
    a = normalize_pass_weights([0, 0, 1, 1], 4, device=torch.device("cpu"), dtype=torch.float32)
    b = normalize_pass_weights([0, 0, 0.5, 0.5], 4, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(a, b)
    assert a.sum().item() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        normalize_pass_weights([0, 0, 0], 3, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(ValueError):
        normalize_pass_weights([1, -1], 2, device=torch.device("cpu"), dtype=torch.float32)


def test_equivalent_relative_pass_weights_give_identical_loss():
    model = tiny_memory_model(n_pass=4)
    tokens = torch.randint(0, 19, (2, 7))
    targets = torch.randint(0, 19, (2, 7))
    output = model(tokens)
    loss_a = model.calc_total_loss(output, targets, [0, 0, 1, 1]).loss
    loss_b = model.calc_total_loss(output, targets, [0, 0, 0.5, 0.5]).loss
    assert torch.equal(loss_a, loss_b)


def test_zero_memory_produces_exact_zero_cross_attention_output():
    config = MemoryTapeConfig(8, 17, 1, 2, 8, 2)
    attention = CausalCrossAttention(config)
    query = torch.randn(2, 6, 8)
    output = attention(query, torch.zeros_like(query))
    assert torch.equal(output, torch.zeros_like(output))


def test_cross_attention_manual_and_sdpa_paths_agree():
    config = MemoryTapeConfig(8, 17, 1, 2, 8, 2)
    attention = CausalCrossAttention(config)
    if not attention.flash:
        pytest.skip("scaled_dot_product_attention is unavailable")
    query = torch.randn(2, 6, 8)
    memory = torch.randn(2, 6, 8)
    expected = attention(query, memory)
    attention.flash = False
    actual = attention(query, memory)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_cross_attention_supports_independent_memory_width():
    config = TransformerConfig(8, 17, 1, 2, 8)
    attention = CausalCrossAttention(config, memory_dim=4)
    query = torch.randn(2, 6, 8)
    memory = torch.randn(2, 6, 4)
    expected = attention(query, memory)
    attention.flash = False
    actual = attention(query, memory)
    assert actual.shape == query.shape
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_token_memory_attention_manual_and_sdpa_paths_agree():
    config = MultiPassConfig(8, 17, 1, 2, 8, 2)
    attention = CausalTokenMemoryAttention(config)
    if not attention.flash:
        pytest.skip("scaled_dot_product_attention is unavailable")
    query = torch.randn(2, 6, 8)
    token_sources = torch.randn(2, 6, 8)
    memory = torch.randn(2, 6, 8)
    expected = attention(query, token_sources, memory)
    expected_token_only = attention(
        query,
        token_sources,
        memory,
        include_memory_source=False,
    )
    attention.flash = False
    actual = attention(query, token_sources, memory)
    actual_token_only = attention(
        query,
        token_sources,
        memory,
        include_memory_source=False,
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.allclose(actual_token_only, expected_token_only, atol=1e-6, rtol=1e-5)


def test_zero_memory_bank_dilutes_token_attention_until_the_bank_is_masked():
    config = MultiPassConfig(8, 17, 1, 1, 4, 2)
    attention = CausalTokenMemoryAttention(config)
    with torch.no_grad():
        attention.c_q.weight.zero_()
        attention.c_tok_kv.weight.zero_()
        attention.c_tok_kv.weight[config.n_embd :, :].copy_(torch.eye(config.n_embd))
        attention.c_mem_kv.weight.zero_()
        attention.c_proj.weight.copy_(torch.eye(config.n_embd))

    query = torch.randn(1, 4, config.n_embd)
    token_sources = torch.randn_like(query)
    zero_memory = torch.zeros_like(query)
    with_null_bank = attention(query, token_sources, zero_memory)
    token_only = attention(
        query,
        token_sources,
        zero_memory,
        include_memory_source=False,
    )

    assert token_only.abs().sum().item() > 0
    assert torch.allclose(with_null_bank, 0.5 * token_only, atol=1e-6, rtol=0)


def test_masked_memory_source_is_independent_of_memory_values():
    config = MultiPassConfig(8, 17, 1, 2, 8, 2)
    attention = CausalTokenMemoryAttention(config)
    query = torch.randn(2, 6, 8)
    token_sources = torch.randn_like(query)
    memory_a = torch.randn_like(query)
    memory_b = torch.randn_like(query)
    output_a = attention(query, token_sources, memory_a, include_memory_source=False)
    output_b = attention(query, token_sources, memory_b, include_memory_source=False)
    assert torch.equal(output_a, output_b)


def test_joint_memory_tape_masked_pass_excludes_memory_at_every_layer():
    model = tiny_joint_memory_model(n_pass=2)
    model.eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    token_stream = model.embed_tokens(tokens)
    memory_a = torch.randn_like(token_stream)
    memory_b = torch.randn_like(token_stream)
    output_a = model.forward_pass_without_memory_source(token_stream, memory_a)
    output_b = model.forward_pass_without_memory_source(token_stream, memory_b)
    assert torch.equal(output_a.logits, output_b.logits)
    assert torch.equal(output_a.memory_states, output_b.memory_states)


def test_memory_block_has_no_first_pass_intercept():
    config = MemoryTapeConfig(8, 17, 1, 2, 8, 2)
    block = MemoryBlock(config)
    block.eval()
    hidden = torch.randn(2, 6, 8)
    after_self = hidden + block.attn(block.ln_self(hidden))
    expected = after_self + block.mlp(block.ln_mlp(after_self))
    actual = block(hidden, torch.zeros_like(hidden))
    assert torch.equal(actual, expected)


def test_memory_tape_uses_standard_memory_normalization_and_shared_writer():
    model = tiny_memory_model()
    assert isinstance(model.transformer.h[0].ln_mem_kv, LayerNorm)
    hidden = torch.randn(2, 6, 8)
    assert torch.equal(model.write_memory(hidden), model.mem_head(model.ln_mem(hidden)))


@pytest.mark.parametrize("memory_dim", [2, 4, 8])
def test_memory_tape_supports_independent_memory_width(memory_dim):
    model = MemoryTapeTransformer(
        MemoryTapeConfig(12, 19, 2, 2, 8, 3, n_memory_embd=memory_dim)
    )
    tokens = torch.randint(0, 19, (2, 8))
    output = model(tokens)
    assert model.memory_dim == memory_dim
    assert all(item.memory_states.shape == (2, 8, memory_dim) for item in output.passes)
    hidden = torch.randn(2, 8, 8)
    assert torch.equal(model.write_memory(hidden), model.ln_mem(model.mem_head(hidden)))
    state = model.prefill_recurrent(tokens[:, :5])
    assert state.memory_states.shape == (2, 5, memory_dim)
    next_token = state.next_token_logits.argmax(dim=-1, keepdim=True)
    assert model.recurrent_step(state, next_token).memory_states.shape == (2, 6, memory_dim)


def test_narrower_memory_reduces_parameters_and_receives_gradients():
    models = [
        MemoryTapeTransformer(MemoryTapeConfig(12, 19, 2, 2, 8, 3, n_memory_embd=width))
        for width in (2, 4, 8)
    ]
    assert [model.get_num_params() for model in models] == sorted(model.get_num_params() for model in models)
    model = models[0]
    tokens = torch.randint(0, 19, (2, 8))
    targets = torch.randint(0, 19, (2, 8))
    loss = model.calc_total_loss(model(tokens), targets, [0, 0, 1]).loss
    loss.backward()
    assert model.mem_head.weight.grad is not None
    assert model.transformer.h[0].cross_attn.c_kv.weight.grad is not None
    with pytest.raises(ValueError, match="positive"):
        MemoryTapeConfig(12, 19, 2, 2, 8, 3, n_memory_embd=0)


def test_causal_transformer_structured_output_and_generation():
    model = CausalTransformer(TransformerConfig(8, 17, 1, 1, 8))
    tokens = torch.randint(0, 17, (2, 6))
    output = model(tokens)
    assert len(output.passes) == 1
    assert output.logits.shape == (2, 6, 17)
    generated = model.generate(tokens[:, :4], 2, do_sample=False)
    assert generated.shape == (2, 6)
    with pytest.raises(ValueError):
        model.generate(tokens[:, :4], 1, inference_mode="append_recurrent")


def test_multipass_variants_return_all_passes_and_finite_losses():
    cases = [
        (MemoryTapeTransformer, MemoryTapeConfig(8, 17, 1, 1, 8, 3)),
        (JointMemoryTapeTransformer, MultiPassConfig(8, 17, 1, 1, 8, 3)),
        (MemoryConcatTransformer, MultiPassConfig(8, 17, 1, 1, 8, 3)),
        (MemoryUpdateTransformer, MemoryUpdateConfig(8, 17, 1, 1, 8, 3)),
    ]
    tokens = torch.randint(0, 17, (2, 6))
    targets = torch.randint(0, 17, (2, 6))
    for cls, config in cases:
        model = cls(config)
        output = model(tokens)
        assert len(output.passes) == 3
        assert all(item.memory_states is not None for item in output.passes)
        loss_output = model.calc_total_loss(output, targets, [0, 0, 1])
        assert torch.isfinite(loss_output.loss)
        assert loss_output.normalized_pass_weights.tolist() == [0.0, 0.0, 1.0]


def test_memory_tape_is_causal_in_tokens_and_emitted_memory():
    model = tiny_memory_model()
    model.eval()
    prefix = torch.tensor([[1, 2, 3, 4]])
    a = torch.cat((prefix, torch.tensor([[5, 6, 7, 8]])), dim=1)
    b = torch.cat((prefix, torch.tensor([[9, 10, 11, 12]])), dim=1)
    out_a = model(a)
    out_b = model(b)
    for pass_a, pass_b in zip(out_a.passes, out_b.passes):
        assert torch.allclose(pass_a.logits[:, :4], pass_b.logits[:, :4], atol=1e-6, rtol=0)
        assert pass_a.memory_states is not None and pass_b.memory_states is not None
        assert torch.allclose(pass_a.memory_states[:, :4], pass_b.memory_states[:, :4], atol=1e-6, rtol=0)


def test_joint_memory_tape_is_causal_in_tokens_and_emitted_memory():
    model = tiny_joint_memory_model()
    model.eval()
    prefix = torch.tensor([[1, 2, 3, 4]])
    a = torch.cat((prefix, torch.tensor([[5, 6, 7, 8]])), dim=1)
    b = torch.cat((prefix, torch.tensor([[9, 10, 11, 12]])), dim=1)
    out_a = model(a)
    out_b = model(b)
    for pass_a, pass_b in zip(out_a.passes, out_b.passes):
        assert torch.allclose(pass_a.logits[:, :4], pass_b.logits[:, :4], atol=1e-6, rtol=0)
        assert pass_a.memory_states is not None and pass_b.memory_states is not None
        assert torch.allclose(pass_a.memory_states[:, :4], pass_b.memory_states[:, :4], atol=1e-6, rtol=0)


@pytest.mark.parametrize("model_factory", [tiny_memory_model, tiny_joint_memory_model])
def test_previous_memory_at_t_cannot_affect_position_t(model_factory):
    model = model_factory(n_pass=2)
    model.eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    token_stream = model.embed_tokens(tokens)
    base = torch.zeros_like(token_stream)
    changed = base.clone()
    changed[:, 2, :] = torch.randn_like(changed[:, 2, :])
    out_base = model.forward_pass(token_stream, base)
    out_changed = model.forward_pass(token_stream, changed)
    assert torch.allclose(out_base.logits[:, :3], out_changed.logits[:, :3], atol=1e-6, rtol=0)
    assert not torch.allclose(out_base.logits[:, 3:], out_changed.logits[:, 3:])


def test_final_pass_loss_reaches_memory_writer_and_reader():
    model = tiny_memory_model(n_pass=3)
    tokens = torch.randint(0, 19, (2, 8))
    targets = torch.randint(0, 19, (2, 8))
    output = model(tokens)
    loss = model.calc_total_loss(output, targets, [0, 0, 1]).loss
    loss.backward()
    assert model.mem_head.weight.grad is not None
    assert model.mem_head.weight.grad.abs().sum().item() > 0
    reader = model.transformer.h[0].cross_attn.c_kv.weight
    assert reader.grad is not None
    assert reader.grad.abs().sum().item() > 0


def test_joint_memory_tape_final_pass_loss_reaches_memory_writer_and_reader():
    model = tiny_joint_memory_model(n_pass=3)
    tokens = torch.randint(0, 19, (2, 8))
    targets = torch.randint(0, 19, (2, 8))
    output = model(tokens)
    loss = model.calc_total_loss(output, targets, [0, 0, 1]).loss
    loss.backward()
    assert model.mem_head.weight.grad is not None
    assert model.mem_head.weight.grad.abs().sum().item() > 0
    for reader in (
        model.transformer.h[0].joint_attn.c_tok_kv.weight,
        model.transformer.h[0].joint_attn.c_mem_kv.weight,
    ):
        assert reader.grad is not None
        assert reader.grad.abs().sum().item() > 0


def test_final_pass_can_be_reproduced_from_previous_pass_memory_input():
    model = tiny_memory_model()
    tokens = torch.randint(0, 19, (2, 8))
    output = model(tokens)
    previous = output.passes[-2].memory_states
    assert previous is not None
    reproduced = model.forward_pass(model.embed_tokens(tokens), previous)
    assert torch.equal(reproduced.logits, output.logits)
    assert reproduced.memory_states is not None
    assert torch.equal(reproduced.memory_states, output.final_memory)


def test_recurrent_prefill_uses_last_pass_memory_and_append_is_immutable():
    model = tiny_memory_model(block_size=10)
    tokens = torch.randint(0, 19, (2, 5))
    output = model(tokens)
    state = model.prefill_recurrent(tokens)
    assert torch.equal(state.memory_states, output.final_memory)
    old_memory = state.memory_states.clone()
    next_token = state.next_token_logits.argmax(dim=-1, keepdim=True)
    next_state = model.recurrent_step(state, next_token)
    assert next_state.memory_states.shape[1] == old_memory.shape[1] + 1
    assert torch.equal(next_state.memory_states[:, :-1], old_memory)


def test_append_recurrent_matches_manual_two_token_rollout():
    model = tiny_memory_model(block_size=10)
    prompt = torch.tensor([[1, 2, 3, 4]])
    state = model.prefill_recurrent(prompt)
    first = state.next_token_logits.argmax(dim=-1, keepdim=True)
    state = model.recurrent_step(state, first)
    second = state.next_token_logits.argmax(dim=-1, keepdim=True)
    expected = torch.cat((prompt, first, second), dim=1)
    actual = model.generate(prompt, 2, do_sample=False, inference_mode="append_recurrent")
    assert torch.equal(actual, expected)


def test_append_recurrent_context_guard_allows_final_unprocessed_token():
    model = tiny_memory_model(block_size=8)
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6]])
    allowed = model.generate(prompt, 3, do_sample=False, inference_mode="append_recurrent")
    assert allowed.shape[1] == 9
    with pytest.raises(ValueError, match="prompt_length"):
        model.generate(prompt, 4, do_sample=False, inference_mode="append_recurrent")


def test_generation_restores_mode_and_validates_sampling_even_for_zero_tokens():
    model = tiny_memory_model()
    model.train()
    tokens = torch.tensor([[1, 2]])
    returned = model.generate(tokens, 0, inference_mode="append_recurrent")
    assert returned is tokens
    assert model.training
    with pytest.raises(ValueError, match="temperature"):
        model.generate(tokens, 0, temperature=-1)
    with pytest.raises(ValueError, match="top_k"):
        sample_next_token(torch.randn(1, 3), top_k=0)


@pytest.mark.parametrize(
    "model",
    [
        MemoryTapeTransformer(MemoryTapeConfig(8, 17, 1, 1, 8, 3)),
        JointMemoryTapeTransformer(MultiPassConfig(8, 17, 1, 1, 8, 3)),
        MemoryConcatTransformer(MultiPassConfig(8, 17, 1, 1, 8, 3)),
        MemoryUpdateTransformer(MemoryUpdateConfig(8, 17, 1, 1, 8, 3)),
    ],
)
def test_all_multipass_variants_support_append_recurrent(model):
    prompt = torch.tensor([[1, 2, 3]])
    generated = model.generate(prompt, 2, do_sample=False, inference_mode="append_recurrent")
    assert generated.shape == (1, 5)


def test_model_factory_constructs_all_variants():
    base = dict(
        n_layer=1,
        n_head=1,
        n_embd=8,
        n_pass=3,
        pass_loss_weights=[0, 0, 1],
        memory_update_gate="off",
        memory_gate_bias=-1.0,
    )
    expected = {
        "transformer": CausalTransformer,
        "memory_tape": MemoryTapeTransformer,
        "joint_memory_tape": JointMemoryTapeTransformer,
        "memory_concat": MemoryConcatTransformer,
        "memory_update": MemoryUpdateTransformer,
    }
    for architecture, cls in expected.items():
        model = build_model(SimpleNamespace(architecture=architecture, **base), 17, 8, "cpu")
        assert isinstance(model, cls)


def test_model_factory_applies_memory_gate_init():
    args = SimpleNamespace(
        architecture="memory_tape",
        n_layer=2,
        n_head=1,
        n_embd=8,
        n_pass=3,
        memory_gate_init=1.0,
    )
    model = build_model(args, 17, 8, "cpu")
    assert model.config.memory_gate_init == 1.0
    assert model.memory_gate_stats()["effective"] == [1.0, 1.0]
