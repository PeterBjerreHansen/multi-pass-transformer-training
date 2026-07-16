"""Unique shortest-path generation as a fixed-suffix trace task.

Each example serializes a shuffled directed acyclic graph, a start node, and a
goal node. The graph is constructed to have exactly one shortest path, and the
target is the complete node sequence from start through goal.
"""
from __future__ import annotations

from collections import deque
import random
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from tasks.common import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SEP_TOKEN,
    SymbolicBatch,
    build_batch_from_sequences,
    build_vocab,
    make_sequence,
)


NODES_TOKEN = "<nodes>"
EDGES_TOKEN = "<edges>"
START_TOKEN = "<start>"
GOAL_TOKEN = "<goal>"
NODE_TOKEN_OFFSET = 8
DEFAULT_NUM_NODES = 24
DEFAULT_PATH_LENGTH = 6
DEFAULT_BRANCHING_FACTOR = 3
DEFAULT_DISTRACTOR_EDGES = 40


def node_token(index: int) -> str:
    if index < 0:
        raise ValueError("node index must be non-negative")
    return f"n{index}"


def required_block_size(
    num_nodes: int,
    path_length: int,
    branching_factor: int,
    distractor_edges: int,
) -> int:
    _validate_sizes(num_nodes, path_length, branching_factor, distractor_edges)
    edge_count = path_length + distractor_edges
    prompt_tokens = num_nodes + 2 * edge_count + 6
    answer_tokens = path_length + 1
    return 2 + prompt_tokens + answer_tokens


def build_shortest_path_vocab(
    num_nodes: int,
    path_length: int,
    branching_factor: int,
    distractor_edges: int,
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    _validate_sizes(num_nodes, path_length, branching_factor, distractor_edges)
    tokens = [
        PAD_TOKEN,
        BOS_TOKEN,
        SEP_TOKEN,
        EOS_TOKEN,
        NODES_TOKEN,
        EDGES_TOKEN,
        START_TOKEN,
        GOAL_TOKEN,
    ]
    tokens.extend(node_token(index) for index in range(num_nodes))
    return build_vocab(tokens)


def solve_shortest_path(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    start: int,
    goal: int,
) -> tuple[list[int], int]:
    """Return one shortest path and the number of shortest paths, capped at two."""
    if num_nodes < 2:
        raise ValueError("num_nodes must be at least 2")
    if not 0 <= start < num_nodes or not 0 <= goal < num_nodes:
        raise ValueError("start and goal must be valid node indices")
    if start == goal:
        raise ValueError("start and goal must differ")

    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    seen = set()
    for source, target in edges:
        if not 0 <= source < num_nodes or not 0 <= target < num_nodes:
            raise ValueError("edge endpoint must be a valid node index")
        if source == target:
            raise ValueError("self edges are not allowed")
        if (source, target) in seen:
            raise ValueError("duplicate edges are not allowed")
        seen.add((source, target))
        adjacency[source].append(target)
    for targets in adjacency:
        targets.sort()

    distances = [-1] * num_nodes
    path_counts = [0] * num_nodes
    parents: list[int | None] = [None] * num_nodes
    distances[start] = 0
    path_counts[start] = 1
    queue = deque([start])
    while queue:
        source = queue.popleft()
        for target in adjacency[source]:
            candidate_distance = distances[source] + 1
            if distances[target] == -1:
                distances[target] = candidate_distance
                path_counts[target] = path_counts[source]
                parents[target] = source
                queue.append(target)
            elif distances[target] == candidate_distance:
                path_counts[target] = min(2, path_counts[target] + path_counts[source])

    if distances[goal] < 0:
        raise ValueError("goal is unreachable from start")
    path = [goal]
    current = goal
    while current != start:
        parent = parents[current]
        if parent is None:
            raise RuntimeError("shortest-path reconstruction failed")
        path.append(parent)
        current = parent
    path.reverse()
    return path, path_counts[goal]


def sample_unique_shortest_path_graph(
    num_nodes: int,
    path_length: int,
    branching_factor: int,
    distractor_edges: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], int, int, list[int]]:
    _validate_sizes(num_nodes, path_length, branching_factor, distractor_edges)
    path = rng.sample(range(num_nodes), path_length + 1)
    start, goal = path[0], path[-1]

    topological_order = list(path)
    remaining = [node for node in range(num_nodes) if node not in path]
    rng.shuffle(remaining)
    for node in remaining:
        topological_order.insert(rng.randrange(len(topological_order) + 1), node)
    rank = {node: index for index, node in enumerate(topological_order)}

    edges = {(path[index], path[index + 1]) for index in range(path_length)}
    out_degrees = [0] * num_nodes
    for source, _target in edges:
        out_degrees[source] += 1

    candidates = [
        (source, target)
        for source in range(num_nodes)
        for target in range(num_nodes)
        if rank[source] < rank[target] and (source, target) not in edges
    ]
    rng.shuffle(candidates)
    accepted = 0
    for source, target in candidates:
        if accepted == distractor_edges:
            break
        if out_degrees[source] >= branching_factor:
            continue
        candidate_edges = [*edges, (source, target)]
        try:
            candidate_path, path_count = solve_shortest_path(
                num_nodes,
                candidate_edges,
                start,
                goal,
            )
        except ValueError:
            continue
        if path_count != 1 or candidate_path != path:
            continue
        edges.add((source, target))
        out_degrees[source] += 1
        accepted += 1

    if accepted != distractor_edges:
        raise ValueError(
            "could not construct the requested number of distractor edges; "
            "increase num_nodes or branching_factor, or reduce distractor_edges"
        )
    result = sorted(edges)
    rng.shuffle(result)
    return result, start, goal, path


def sample_shortest_path_example(
    num_nodes: int,
    path_length: int,
    branching_factor: int,
    distractor_edges: int,
    stoi: Dict[str, int],
    rng: random.Random,
) -> tuple[list[int], list[int], list[tuple[int, int]], int, int, list[int]]:
    edges, start, goal, path = sample_unique_shortest_path_graph(
        num_nodes,
        path_length,
        branching_factor,
        distractor_edges,
        rng,
    )
    prompt = [stoi[NODES_TOKEN]]
    prompt.extend(stoi[node_token(index)] for index in range(num_nodes))
    prompt.append(stoi[EDGES_TOKEN])
    serialized_edges = list(edges)
    rng.shuffle(serialized_edges)
    for source, target in serialized_edges:
        prompt.extend((stoi[node_token(source)], stoi[node_token(target)]))
    prompt.extend(
        (
            stoi[START_TOKEN],
            stoi[node_token(start)],
            stoi[GOAL_TOKEN],
            stoi[node_token(goal)],
        )
    )
    answer = [stoi[node_token(node)] for node in path]
    return prompt, answer, edges, start, goal, path


def build_shortest_path_batch(
    batch_size: int,
    num_nodes: int,
    path_length: int,
    branching_factor: int,
    distractor_edges: int,
    stoi: Dict[str, int],
    device=None,
    rng: random.Random | None = None,
) -> SymbolicBatch:
    _validate_sizes(num_nodes, path_length, branching_factor, distractor_edges)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rng = rng or random.Random()
    rows = []
    for _ in range(batch_size):
        prompt, answer, *_metadata = sample_shortest_path_example(
            num_nodes,
            path_length,
            branching_factor,
            distractor_edges,
            stoi,
            rng,
        )
        rows.append(make_sequence(prompt, answer, stoi))
    return build_batch_from_sequences(rows, pad_id=stoi[PAD_TOKEN], device=device)


def parse_prompt_metadata(
    prompt_tokens: Sequence[int],
    *,
    num_nodes: int,
    edge_count: int,
) -> tuple[list[tuple[int, int]], int, int]:
    expected_length = num_nodes + 2 * edge_count + 6
    if len(prompt_tokens) != expected_length:
        raise ValueError("prompt_tokens has unexpected length")
    if prompt_tokens[0] != 4:
        raise ValueError("prompt must begin with <nodes>")
    listed_nodes = [
        token_id_to_node(token_id, num_nodes=num_nodes)
        for token_id in prompt_tokens[1 : 1 + num_nodes]
    ]
    if any(node is None for node in listed_nodes) or set(listed_nodes) != set(range(num_nodes)):
        raise ValueError("prompt node list must contain every node exactly once")
    edges_marker = 1 + num_nodes
    if prompt_tokens[edges_marker] != 5:
        raise ValueError("prompt must contain <edges> after the node list")

    edge_start = edges_marker + 1
    edge_end = edge_start + 2 * edge_count
    edges = []
    for offset in range(edge_start, edge_end, 2):
        source = token_id_to_node(prompt_tokens[offset], num_nodes=num_nodes)
        target = token_id_to_node(prompt_tokens[offset + 1], num_nodes=num_nodes)
        if source is None or target is None:
            raise ValueError("prompt edge contains an invalid node token")
        edges.append((source, target))
    if len(set(edges)) != len(edges):
        raise ValueError("prompt contains duplicate edges")
    if prompt_tokens[edge_end] != 6 or prompt_tokens[edge_end + 2] != 7:
        raise ValueError("prompt must end with <start> node <goal> node")
    start = token_id_to_node(prompt_tokens[edge_end + 1], num_nodes=num_nodes)
    goal = token_id_to_node(prompt_tokens[edge_end + 3], num_nodes=num_nodes)
    if start is None or goal is None:
        raise ValueError("prompt start or goal token is invalid")
    return edges, start, goal


def token_id_to_node(token_id: int, *, num_nodes: int) -> int | None:
    node = int(token_id) - NODE_TOKEN_OFFSET
    return node if 0 <= node < num_nodes else None


def legal_prefix_length(
    prompt_tokens: Sequence[int],
    generated_node_token_ids: Sequence[int],
    *,
    num_nodes: int,
    edge_count: int,
) -> tuple[int, bool]:
    edges, start, goal = parse_prompt_metadata(
        prompt_tokens,
        num_nodes=num_nodes,
        edge_count=edge_count,
    )
    target_path, path_count = solve_shortest_path(num_nodes, edges, start, goal)
    if path_count != 1:
        raise ValueError("shortest-path prompt does not have a unique shortest path")
    edge_set = set(edges)
    decoded = [
        token_id_to_node(token_id, num_nodes=num_nodes)
        for token_id in generated_node_token_ids
    ]
    if not decoded or decoded[0] != start:
        return 0, False
    legal_tokens = 1
    for previous, current in zip(decoded, decoded[1:]):
        if current is None or previous is None or (previous, current) not in edge_set:
            return legal_tokens, False
        legal_tokens += 1
    return legal_tokens, decoded == target_path and decoded[-1] == goal


@torch.no_grad()
def shortest_path_generation_metrics(
    model,
    batch,
    args,
    *,
    inference_mode: str | None = None,
    num_nodes: int,
    edge_count: int,
    **_unused,
) -> dict[str, float]:
    mode = "recompute" if args.architecture == "transformer" else (inference_mode or args.inference_mode)
    do_sample = getattr(args, "token_selection", "argmax") == "sample"
    totals = {
        "token_legality": 0.0,
        "sequence_legality": 0.0,
        "valid_edge_rate": 0.0,
        "goal_reached": 0.0,
        "optimal_path": 0.0,
        "exact_path": 0.0,
        "mean_generated_path_length": 0.0,
    }

    for row in range(batch.idx.shape[0]):
        prompt_len = int(batch.prompt_lengths[row].item())
        output_len = int(batch.output_lengths[row].item())
        prompt = batch.idx[row : row + 1, :prompt_len]
        target_suffix = batch.targets[
            row,
            prompt_len - 1 : prompt_len - 1 + output_len,
        ].tolist()
        generated = model.generate(
            prompt,
            max_new_tokens=output_len,
            do_sample=do_sample,
            inference_mode=mode,
        )
        generated_suffix = generated[0, prompt_len : prompt_len + output_len].tolist()
        eos_position = next(
            (position for position, token_id in enumerate(generated_suffix) if token_id == 3),
            None,
        )
        generated_path_ids = (
            generated_suffix if eos_position is None else generated_suffix[:eos_position]
        )
        prompt_tokens = batch.idx[row, 1 : prompt_len - 1].tolist()
        edges, _start, goal = parse_prompt_metadata(
            prompt_tokens,
            num_nodes=num_nodes,
            edge_count=edge_count,
        )
        target_path_ids = target_suffix[:-1]
        legal_length, _all_legal = legal_prefix_length(
            prompt_tokens,
            generated_path_ids,
            num_nodes=num_nodes,
            edge_count=edge_count,
        )
        decoded_path = [
            token_id_to_node(token_id, num_nodes=num_nodes)
            for token_id in generated_path_ids
        ]
        path_is_edge_valid = legal_length == len(generated_path_ids)
        goal_reached = bool(decoded_path and decoded_path[-1] == goal and path_is_edge_valid)
        exact_path = generated_path_ids == target_path_ids
        complete = eos_position is not None and exact_path
        totals["token_legality"] += min(1.0, legal_length / max(len(target_path_ids), 1))
        totals["sequence_legality"] += float(complete)
        totals["valid_edge_rate"] += legal_length / max(len(generated_path_ids), 1)
        totals["goal_reached"] += float(goal_reached)
        totals["optimal_path"] += float(exact_path)
        totals["exact_path"] += float(complete)
        totals["mean_generated_path_length"] += float(len(generated_path_ids))

    count = int(batch.idx.shape[0])
    return {key: value / count for key, value in totals.items()}


def format_shortest_path_eval_metrics(metrics: dict[str, float]) -> str:
    return (
        f"optimal {metrics['optimal_path']:.3f} | "
        f"goal {metrics['goal_reached']:.3f} | "
        f"edge_valid {metrics['valid_edge_rate']:.3f}"
    )


def _validate_sizes(
    num_nodes: int,
    path_length: int,
    branching_factor: int,
    distractor_edges: int,
) -> None:
    if num_nodes < 2:
        raise ValueError("num_nodes must be at least 2")
    if path_length < 1:
        raise ValueError("path_length must be positive")
    if path_length + 1 > num_nodes:
        raise ValueError("path_length requires at least path_length + 1 nodes")
    if not 1 <= branching_factor < num_nodes:
        raise ValueError("branching_factor must be in [1, num_nodes)")
    if distractor_edges < 0:
        raise ValueError("distractor_edges must be non-negative")
    maximum_edges = num_nodes * branching_factor
    if path_length + distractor_edges > maximum_edges:
        raise ValueError("requested edges exceed the branching-factor capacity")


__all__ = [
    "DEFAULT_BRANCHING_FACTOR",
    "DEFAULT_DISTRACTOR_EDGES",
    "DEFAULT_NUM_NODES",
    "DEFAULT_PATH_LENGTH",
    "build_shortest_path_batch",
    "build_shortest_path_vocab",
    "format_shortest_path_eval_metrics",
    "legal_prefix_length",
    "node_token",
    "parse_prompt_metadata",
    "required_block_size",
    "sample_shortest_path_example",
    "sample_unique_shortest_path_graph",
    "shortest_path_generation_metrics",
    "solve_shortest_path",
]
