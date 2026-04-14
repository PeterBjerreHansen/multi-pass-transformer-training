from dataclasses import dataclass
from typing import Callable

import random

from tasks import arithmetic
from tasks import boolean_rpn
from tasks import dyck
from tasks import order_deduction
from tasks import permutation
from tasks import tracking
from tasks import truth_graph
from tasks import walk


@dataclass(frozen=True)
class BBHSymbolicTaskSpec:
    name: str
    min_level: int
    default_start_level: int
    default_max_level: int
    build_vocab: Callable[..., tuple[list[str], dict[str, int], dict[int, str]]]
    required_block_size: Callable[..., int]
    build_batch: Callable[..., object]


def _build_walk_vocab(max_level: int):
    return walk.build_walk_vocab(max_steps=max_level)


def _walk_block_size(max_level: int, supervision: str) -> int:
    return walk.required_block_size(num_steps=max_level, supervision=supervision)


def _walk_batch(batch_size: int, level: int, stoi, supervision: str, device: str, rng: random.Random):
    return walk.build_walk_batch(batch_size, level, stoi, supervision=supervision, device=device, rng=rng)


def _build_dyck_vocab(max_level: int):
    return dyck.build_dyck_vocab(max_bracket_types=4)


def _dyck_depth(level: int) -> int:
    return max(1, min(level, 8))


def _dyck_block_size(max_level: int, supervision: str) -> int:
    return dyck.required_block_size(
        prefix_length=max_level,
        max_depth=_dyck_depth(max_level),
        supervision=supervision,
    )


def _dyck_batch(batch_size: int, level: int, stoi, supervision: str, device: str, rng: random.Random):
    return dyck.build_dyck_batch(
        batch_size=batch_size,
        prefix_length=level,
        max_depth=_dyck_depth(level),
        bracket_types=4,
        stoi=stoi,
        supervision=supervision,
        device=device,
        rng=rng,
    )


def _build_boolean_rpn_vocab(max_level: int):
    return boolean_rpn.build_boolean_rpn_vocab()


def _boolean_rpn_block_size(max_level: int, supervision: str) -> int:
    return boolean_rpn.required_block_size(num_binary_ops=max_level, supervision=supervision)


def _boolean_rpn_batch(batch_size: int, level: int, stoi, supervision: str, device: str, rng: random.Random):
    return boolean_rpn.build_boolean_rpn_batch(batch_size, level, stoi, supervision=supervision, device=device, rng=rng)


def _build_arithmetic_vocab(max_level: int):
    return arithmetic.build_arithmetic_vocab(max_modulus=10)


def _arithmetic_block_size(max_level: int, supervision: str) -> int:
    return arithmetic.required_block_size(num_steps=max_level, supervision=supervision)


def _arithmetic_batch(batch_size: int, level: int, stoi, supervision: str, device: str, rng: random.Random):
    return arithmetic.build_arithmetic_batch(
        batch_size=batch_size,
        num_steps=level,
        modulus=10,
        stoi=stoi,
        supervision=supervision,
        device=device,
        rng=rng,
    )


def _build_truth_graph_vocab(max_level: int):
    return truth_graph.build_truth_vocab(max_num_vars=max_level)


def _truth_graph_block_size(max_level: int, supervision: str) -> int:
    return truth_graph.required_block_size(num_vars=max_level, supervision=supervision)


def _truth_graph_batch(batch_size: int, level: int, stoi, supervision: str, device: str, rng: random.Random):
    return truth_graph.build_truth_batch(batch_size, level, stoi, supervision=supervision, device=device, rng=rng)


def _build_order_deduction_vocab(max_level: int):
    return order_deduction.build_order_vocab(max_num_objects=max_level)


def _order_deduction_block_size(max_level: int, supervision: str) -> int:
    return order_deduction.required_block_size(num_objects=max_level, supervision=supervision)


def _order_deduction_batch(batch_size: int, level: int, stoi, supervision: str, device: str, rng: random.Random):
    return order_deduction.build_order_batch(batch_size, level, stoi, supervision=supervision, device=device, rng=rng)


TRACKING_NUM_OBJECTS = 4


def _build_tracking_vocab(max_level: int):
    return tracking.build_tracking_vocab(num_objects=TRACKING_NUM_OBJECTS)


def _tracking_block_size(max_level: int, supervision: str) -> int:
    return tracking.required_block_size(
        num_objects=TRACKING_NUM_OBJECTS,
        num_ops=max_level,
        supervision=supervision,
    )


def _tracking_batch(batch_size: int, level: int, stoi, supervision: str, device: str, rng: random.Random):
    return tracking.build_tracking_batch(
        batch_size=batch_size,
        num_objects=TRACKING_NUM_OBJECTS,
        num_ops=level,
        stoi=stoi,
        supervision=supervision,
        device=device,
        rng=rng,
    )


def _build_permutation_vocab(max_level: int, *, num_objects: int = 4):
    return permutation.build_permutation_vocab(num_objects=num_objects)


def _permutation_block_size(max_level: int, supervision: str, *, num_objects: int = 4) -> int:
    return permutation.required_block_size(
        num_objects=num_objects,
        num_swaps=max_level,
        supervision=supervision,
    )


def _permutation_batch(
    batch_size: int,
    level: int,
    stoi,
    supervision: str,
    device: str,
    rng: random.Random,
    *,
    num_objects: int = 4,
):
    return permutation.build_permutation_batch(
        batch_size=batch_size,
        num_objects=num_objects,
        num_swaps=level,
        stoi=stoi,
        supervision=supervision,
        device=device,
        rng=rng,
    )


TASKS = {
    "walk": BBHSymbolicTaskSpec(
        name="walk",
        min_level=1,
        default_start_level=1,
        default_max_level=64,
        build_vocab=_build_walk_vocab,
        required_block_size=_walk_block_size,
        build_batch=_walk_batch,
    ),
    "dyck": BBHSymbolicTaskSpec(
        name="dyck",
        min_level=1,
        default_start_level=1,
        default_max_level=64,
        build_vocab=_build_dyck_vocab,
        required_block_size=_dyck_block_size,
        build_batch=_dyck_batch,
    ),
    "boolean_rpn": BBHSymbolicTaskSpec(
        name="boolean_rpn",
        min_level=1,
        default_start_level=1,
        default_max_level=32,
        build_vocab=_build_boolean_rpn_vocab,
        required_block_size=_boolean_rpn_block_size,
        build_batch=_boolean_rpn_batch,
    ),
    "arithmetic": BBHSymbolicTaskSpec(
        name="arithmetic",
        min_level=1,
        default_start_level=1,
        default_max_level=64,
        build_vocab=_build_arithmetic_vocab,
        required_block_size=_arithmetic_block_size,
        build_batch=_arithmetic_batch,
    ),
    "truth_graph": BBHSymbolicTaskSpec(
        name="truth_graph",
        min_level=2,
        default_start_level=2,
        default_max_level=32,
        build_vocab=_build_truth_graph_vocab,
        required_block_size=_truth_graph_block_size,
        build_batch=_truth_graph_batch,
    ),
    "order_deduction": BBHSymbolicTaskSpec(
        name="order_deduction",
        min_level=2,
        default_start_level=2,
        default_max_level=16,
        build_vocab=_build_order_deduction_vocab,
        required_block_size=_order_deduction_block_size,
        build_batch=_order_deduction_batch,
    ),
    "tracking": BBHSymbolicTaskSpec(
        name="tracking",
        min_level=1,
        default_start_level=1,
        default_max_level=64,
        build_vocab=_build_tracking_vocab,
        required_block_size=_tracking_block_size,
        build_batch=_tracking_batch,
    ),
    "permutation": BBHSymbolicTaskSpec(
        name="permutation",
        min_level=0,
        default_start_level=1,
        default_max_level=64,
        build_vocab=_build_permutation_vocab,
        required_block_size=_permutation_block_size,
        build_batch=_permutation_batch,
    ),
}
