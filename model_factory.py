from __future__ import annotations

from models import (
    CausalTransformer,
    LoopedTransformer,
    LoopedTransformerConfig,
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
    "looped_transformer",
)


RECURRENT_TAPE_ARCHITECTURES = frozenset({"memory_tape", "memory_add"})


def uses_pass_loss_weights(architecture: str) -> bool:
    """Whether training exposes and weights a language-model loss per pass."""
    return architecture in RECURRENT_TAPE_ARCHITECTURES


def supports_append_recurrent(architecture: str) -> bool:
    """Whether generation can reuse an aligned recurrent tape across tokens."""
    return architecture in RECURRENT_TAPE_ARCHITECTURES


def supports_memory_diagnostics(architecture: str) -> bool:
    return architecture in RECURRENT_TAPE_ARCHITECTURES


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
            )
        )
    elif args.architecture == "memory_add":
        model = MemoryAddTransformer(
            MultiPassConfig(
                **common,
                n_pass=args.n_pass,
            )
        )
    elif args.architecture == "looped_transformer":
        model = LoopedTransformer(
            LoopedTransformerConfig(
                **common,
                n_pass=args.n_pass,
                loop_layout=getattr(args, "loop_layout", "sandwich"),
                persistent_input=getattr(args, "loop_persistent_input", "off") == "on",
            )
        )
    else:
        raise ValueError(f"Unsupported architecture: {args.architecture}")

    return model.to(device)
