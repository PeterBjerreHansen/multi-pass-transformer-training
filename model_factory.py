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
                # Old run configs predate this CLI option. Their checkpoints
                # retain the original 0.1 initialization contract.
                memory_gate_init=getattr(args, "memory_gate_init", 0.1),
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
