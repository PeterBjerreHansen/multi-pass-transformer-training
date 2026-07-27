from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable

from tasks.trace import othello, shortest_path


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


def _shortest_path_vocab(args):
    distribution = getattr(args, "shortest_path_distribution", "legacy")
    if distribution != "legacy":
        return shortest_path.build_distribution_shortest_path_vocab(distribution)
    return shortest_path.build_shortest_path_vocab(
        args.num_nodes,
        args.shortest_path_length,
        args.branching_factor,
        args.distractor_edges,
    )


def _shortest_path_block_size(args) -> int:
    distribution = getattr(args, "shortest_path_distribution", "legacy")
    if distribution != "legacy":
        return shortest_path.required_distribution_block_size(distribution)
    return shortest_path.required_block_size(
        args.num_nodes,
        args.shortest_path_length,
        args.branching_factor,
        args.distractor_edges,
    )


def _shortest_path_batch(args, stoi, rng: random.Random, _split: str):
    distribution = getattr(args, "shortest_path_distribution", "legacy")
    if distribution != "legacy":
        return shortest_path.build_distribution_shortest_path_batch(
            batch_size=args.batch_size,
            distribution_name=distribution,
            stoi=stoi,
            device=args.device,
            rng=rng,
        )
    return shortest_path.build_shortest_path_batch(
        batch_size=args.batch_size,
        num_nodes=args.num_nodes,
        path_length=args.shortest_path_length,
        branching_factor=args.branching_factor,
        distractor_edges=args.distractor_edges,
        stoi=stoi,
        device=args.device,
        rng=rng,
    )


def _shortest_path_metrics(model, batch, args, inference_mode: str | None):
    distribution = getattr(args, "shortest_path_distribution", "legacy")
    if distribution != "legacy":
        return shortest_path.shortest_path_generation_metrics(
            model,
            batch,
            args,
            inference_mode=inference_mode,
        )
    return shortest_path.shortest_path_generation_metrics(
        model,
        batch,
        args,
        inference_mode=inference_mode,
        num_nodes=args.num_nodes,
        edge_count=args.shortest_path_length + args.distractor_edges,
        legacy_metric_semantics=True,
    )


def _shortest_path_legality(args, prompt_tokens, generated_tokens):
    distribution = getattr(args, "shortest_path_distribution", "legacy")
    if distribution != "legacy":
        return shortest_path.legal_prefix_length(
            prompt_tokens,
            generated_tokens,
        )
    return shortest_path.legal_prefix_length(
        prompt_tokens,
        generated_tokens,
        num_nodes=args.num_nodes,
        edge_count=args.shortest_path_length + args.distractor_edges,
    )


TRACE_TASKS: dict[str, TraceTask] = {
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
    "shortest_path": TraceTask(
        name="shortest_path",
        build_vocab_fn=_shortest_path_vocab,
        required_block_size_fn=_shortest_path_block_size,
        build_batch_fn=_shortest_path_batch,
        generation_metrics_fn=_shortest_path_metrics,
        format_metrics_fn=shortest_path.format_shortest_path_eval_metrics,
        legality_prefix_fn=_shortest_path_legality,
        valid_target_mask_fn=_all_target_positions,
    ),
}


def get_trace_task(name: str) -> TraceTask:
    try:
        return TRACE_TASKS[name]
    except KeyError as error:
        raise ValueError(f"unsupported trace task: {name}") from error


__all__ = ["TRACE_TASKS", "TraceTask", "get_trace_task"]
