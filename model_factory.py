from models import (
    CausalTransformer,
    MemoryConcatTransformer,
    MemoryTapeTransformer,
    MemoryUpdateConfig,
    MemoryUpdateTransformer,
    MultiPassConfig,
    TransformerConfig,
)


ARCHITECTURES = ("transformer", "memory_tape", "memory_concat", "memory_update")


def is_multi_pass_architecture(architecture: str) -> bool:
    return architecture != "transformer"


def build_model(args, vocab_size: int, block_size: int, device: str):
    if args.architecture == "transformer":
        config = TransformerConfig(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
        )
        model = CausalTransformer(config)
    elif args.architecture == "memory_concat":
        config = MultiPassConfig(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            n_pass=args.n_pass,
        )
        model = MemoryConcatTransformer(config)
    elif args.architecture == "memory_update":
        config = MemoryUpdateConfig(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            n_pass=args.n_pass,
            memory_gate_bias=args.memory_gate_bias,
            use_memory_gate=args.memory_update_gate == "on",
        )
        model = MemoryUpdateTransformer(config)
    elif args.architecture == "memory_tape":
        config = MultiPassConfig(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            n_pass=args.n_pass,
        )
        model = MemoryTapeTransformer(config)
    else:
        raise ValueError(f"Unsupported architecture: {args.architecture}")

    return model.to(device)
