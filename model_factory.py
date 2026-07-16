from __future__ import annotations

from models import (
    CausalTransformer,
    JointMemoryTapeTransformer,
    MemoryConcatTransformer,
    MemoryTapeConfig,
    MemoryTapeTransformer,
    MemoryUpdateConfig,
    MemoryUpdateTransformer,
    MultiPassConfig,
    TransformerConfig,
)


ARCHITECTURES = ("transformer", "memory_tape", "joint_memory_tape", "memory_concat", "memory_update")


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
    elif args.architecture == "joint_memory_tape":
        model = JointMemoryTapeTransformer(
            MultiPassConfig(
                **common,
                n_pass=args.n_pass,
            )
        )
    elif args.architecture == "memory_concat":
        model = MemoryConcatTransformer(
            MultiPassConfig(
                **common,
                n_pass=args.n_pass,
            )
        )
    elif args.architecture == "memory_update":
        model = MemoryUpdateTransformer(
            MemoryUpdateConfig(
                **common,
                n_pass=args.n_pass,
                memory_gate_bias=args.memory_gate_bias,
                use_memory_gate=args.memory_update_gate == "on",
            )
        )
    else:
        raise ValueError(f"Unsupported architecture: {args.architecture}")

    return model.to(device)
