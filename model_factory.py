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


def resolve_memory_read_layers(pattern: str, n_layer: int) -> tuple[int, ...] | None:
    if pattern == "all":
        return None
    if pattern == "early":
        return (0,)
    if pattern == "middle":
        return (n_layer // 2,)
    if pattern == "late":
        return (n_layer - 1,)
    raise ValueError(f"unsupported memory read pattern: {pattern}")


def is_multi_pass_architecture(architecture: str) -> bool:
    return architecture != "transformer"


def build_model(args, vocab_size: int, block_size: int, device: str):
    common = dict(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )

    if args.architecture == "transformer":
        model = CausalTransformer(TransformerConfig(**common))
    elif args.architecture == "memory_tape":
        model = MemoryTapeTransformer(
            MemoryTapeConfig(
                **common,
                n_pass=args.n_pass,
                memory_read_layers=resolve_memory_read_layers(
                    getattr(args, "memory_read_pattern", "all"),
                    args.n_layer,
                ),
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
