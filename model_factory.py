from __future__ import annotations

from models import (
    CausalTransformer,
    MemoryAddTransformer,
    MemoryTapeConfig,
    MemoryTapeTransformer,
    MultiPassConfig,
    TransformerConfig,
)


ARCHITECTURES = (
    "transformer",
    "memory_tape",
    "memory_add",
)


def is_multi_pass_architecture(architecture: str) -> bool:
    return architecture != "transformer"


def build_model(args, vocab_size: int, block_size: int, device: str):
    max_position_embeddings = getattr(args, "max_position_embeddings", None) or block_size
    required_positions = block_size + max(
        getattr(args, "train_position_offset_max", 0),
        getattr(args, "eval_position_offset", 0),
    )
    if max_position_embeddings < required_positions:
        raise ValueError(
            f"--max-position-embeddings must be at least {required_positions} for the configured offsets"
        )
    common = dict(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        max_position_embeddings=max_position_embeddings,
    )

    if args.architecture == "transformer":
        model = CausalTransformer(TransformerConfig(**common))
    elif args.architecture == "memory_tape":
        model = MemoryTapeTransformer(
            MemoryTapeConfig(
                **common,
                n_pass=args.n_pass,
            )
        )
    elif args.architecture == "memory_add":
        model = MemoryAddTransformer(
            MultiPassConfig(
                **common,
                n_pass=args.n_pass,
            )
        )
    else:
        raise ValueError(f"Unsupported architecture: {args.architecture}")

    return model.to(device)
