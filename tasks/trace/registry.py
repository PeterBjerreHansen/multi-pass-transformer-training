from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable

from tasks.trace import othello, random_graph_walk


@dataclass(frozen=True)
class TraceTask:
    """Adapter for the behavior shared by fixed-suffix trace tasks."""

    name: str
    build_vocab_fn: Callable
    required_block_size_fn: Callable
    build_batch_fn: Callable
    generation_metrics_fn: Callable
    format_metrics_fn: Callable
    legality_prefix_fn: Callable
    valid_target_mask_fn: Callable

    def build_vocab(self, args):
        return self.build_vocab_fn(args)

    def required_block_size(self, args) -> int:
        return int(self.required_block_size_fn(args))

    def build_batch(self, args, stoi, rng: random.Random, *, split: str):
        return self.build_batch_fn(args, stoi, rng, split)

    def generation_metrics(self, model, batch, args, *, inference_mode: str | None = None):
        return self.generation_metrics_fn(model, batch, args, inference_mode)

    def format_metrics(self, metrics: dict[str, float]) -> str:
        return self.format_metrics_fn(metrics)

    def legality_prefix(
        self,
        args,
        prompt_tokens: list[int],
        generated_tokens: list[int],
    ) -> tuple[int, bool]:
        return self.legality_prefix_fn(args, prompt_tokens, generated_tokens)

    def valid_target_mask(self, args, target_tokens: list[int]) -> list[bool]:
        return self.valid_target_mask_fn(args, target_tokens)


def _all_target_positions(_args, target_tokens: list[int]) -> list[bool]:
    return [True] * len(target_tokens)


def _random_graph_walk_vocab(args):
    return random_graph_walk.build_random_graph_walk_vocab(
        args.num_states,
        args.label_pool_size,
    )


def _random_graph_walk_block_size(args) -> int:
    return random_graph_walk.required_block_size(
        args.num_states,
        args.label_pool_size,
        args.max_level,
    )


def _random_graph_walk_batch(args, stoi, rng: random.Random, _split: str):
    return random_graph_walk.build_random_graph_walk_batch(
        batch_size=args.batch_size,
        num_states=args.num_states,
        label_pool_size=args.label_pool_size,
        num_steps=args.max_level,
        stoi=stoi,
        device=args.device,
        rng=rng,
    )


def _random_graph_walk_metrics(model, batch, args, inference_mode: str | None):
    return random_graph_walk.random_graph_walk_generation_metrics(
        model,
        batch,
        args,
        inference_mode=inference_mode,
        num_states=args.num_states,
        label_pool_size=args.label_pool_size,
    )


def _random_graph_walk_legality(args, prompt_tokens, generated_tokens):
    return random_graph_walk.legal_prefix_length(
        prompt_tokens,
        generated_tokens,
        num_states=args.num_states,
        label_pool_size=args.label_pool_size,
    )


def _othello_vocab(args):
    return othello.build_othello_vocab(
        othello_train_games=args.othello_train_games,
        othello_val_games=args.othello_val_games,
    )


def _othello_block_size(args) -> int:
    return othello.required_block_size(
        othello_prepend_opening=args.othello_prepend_opening,
        othello_train_games=args.othello_train_games,
        othello_val_games=args.othello_val_games,
    )


def _othello_batch(args, stoi, rng: random.Random, split: str):
    return othello.build_othello_batch(
        batch_size=args.batch_size,
        stoi=stoi,
        device=args.device,
        rng=rng,
        split=split,
        othello_data_dir=args.othello_data_dir,
        othello_train_games=args.othello_train_games,
        othello_val_games=args.othello_val_games,
        othello_dataset_seed=args.othello_dataset_seed,
        othello_prepend_opening=args.othello_prepend_opening,
    )


def _othello_metrics(model, batch, args, inference_mode: str | None):
    return othello.othello_generation_metrics(
        model,
        batch,
        args,
        inference_mode=inference_mode,
    )


def _othello_legality(_args, _prompt_tokens, generated_tokens):
    return othello.legal_prefix_length(generated_tokens)


TRACE_TASKS: dict[str, TraceTask] = {
    "random_graph_walk": TraceTask(
        name="random_graph_walk",
        build_vocab_fn=_random_graph_walk_vocab,
        required_block_size_fn=_random_graph_walk_block_size,
        build_batch_fn=_random_graph_walk_batch,
        generation_metrics_fn=_random_graph_walk_metrics,
        format_metrics_fn=random_graph_walk.format_random_graph_walk_eval_metrics,
        legality_prefix_fn=_random_graph_walk_legality,
        valid_target_mask_fn=_all_target_positions,
    ),
    "othello": TraceTask(
        name="othello",
        build_vocab_fn=_othello_vocab,
        required_block_size_fn=_othello_block_size,
        build_batch_fn=_othello_batch,
        generation_metrics_fn=_othello_metrics,
        format_metrics_fn=othello.format_othello_eval_metrics,
        legality_prefix_fn=_othello_legality,
        valid_target_mask_fn=_all_target_positions,
    ),
}


def get_trace_task(name: str) -> TraceTask:
    try:
        return TRACE_TASKS[name]
    except KeyError as error:
        raise ValueError(f"unsupported trace task: {name}") from error


__all__ = ["TRACE_TASKS", "TraceTask", "get_trace_task"]
