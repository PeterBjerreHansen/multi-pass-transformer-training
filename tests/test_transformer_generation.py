import torch

from models import (
    MemoryConcatTransformer,
    MemoryTapeTransformer,
    MemoryUpdateConfig,
    MemoryUpdateTransformer,
    MultiPassConfig,
)


def make_config(config_cls, block_size: int, **kwargs):
    return config_cls(
        block_size=block_size,
        vocab_size=19,
        n_layer=2,
        n_head=2,
        n_embd=16,
        n_pass=3,
        **kwargs,
    )


def test_generation_modes_run():
    cases = (
        (MemoryTapeTransformer, MultiPassConfig, {}),
        (MemoryConcatTransformer, MultiPassConfig, {}),
        (MemoryUpdateTransformer, MemoryUpdateConfig, {}),
        (MemoryUpdateTransformer, MemoryUpdateConfig, {"use_memory_gate": False}),
    )

    for model_cls, config_cls, config_kwargs in cases:
        torch.manual_seed(1234)
        config = make_config(config_cls, block_size=8, **config_kwargs)
        model = model_cls(config).eval()
        prompt = torch.randint(0, config.vocab_size, (2, 7))

        recompute = model.generate(prompt.clone(), max_new_tokens=1, do_sample=False, generation_mode="recompute")
        greedy = model.generate(prompt.clone(), max_new_tokens=1, do_sample=False, generation_mode="greedy")
        assert torch.equal(recompute, greedy)

        generated = model.generate(prompt.clone(), max_new_tokens=4, do_sample=False, generation_mode="greedy")
        assert generated.shape == (2, 11)
