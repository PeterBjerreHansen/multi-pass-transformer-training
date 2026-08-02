"""Unique shortest-path generation as a fixed-suffix trace task.

Each example serializes a shuffled directed acyclic graph, a start node, and a
goal node. The graph is constructed to have exactly one shortest path, and the
target is the complete node sequence from start through goal.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Sequence, Tuple

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
PATH_LENGTH_BUCKETS = ("short", "medium", "long")


@dataclass(frozen=True)
class ShortestPathDistribution:
    """A compact distribution over solver-verified shortest-path examples."""

    name: str
    min_nodes: int
    max_nodes: int
    min_path_length: int
    max_path_length: int
    max_out_degree: int
    min_detours: int
    max_detours: int
    max_detour_penalty: int
    min_edge_probability: float
    max_edge_probability: float


SHORTEST_PATH_DISTRIBUTIONS = {
    "easy": ShortestPathDistribution(
        name="easy",
        min_nodes=8,
        max_nodes=12,
        min_path_length=3,
        max_path_length=4,
        max_out_degree=2,
        min_detours=1,
        max_detours=2,
        max_detour_penalty=3,
        min_edge_probability=0.05,
        max_edge_probability=0.25,
    ),
    "main": ShortestPathDistribution(
        name="main",
        min_nodes=16,
        max_nodes=26,
        min_path_length=5,
        max_path_length=10,
        max_out_degree=2,
        min_detours=4,
        max_detours=6,
        max_detour_penalty=2,
        min_edge_probability=0.05,
        max_edge_probability=0.20,
    ),
}


def get_shortest_path_distribution(name: str) -> ShortestPathDistribution:
    try:
        return SHORTEST_PATH_DISTRIBUTIONS[name]
    except KeyError as error:
        raise ValueError(f"unsupported shortest-path distribution: {name}") from error


def path_length_bucket(path_length: int) -> str:
    if path_length < 1:
        raise ValueError("path length must be positive")
    if path_length <= 6:
        return "short"
    if path_length <= 8:
        return "medium"
    return "long"


def node_token(index: int) -> str:
    if index < 0:
        raise ValueError("node index must be non-negative")
    return f"n{index}"


def required_block_size(distribution_name: str) -> int:
    """Return a safe maximum block size for every example in a distribution."""
    distribution = get_shortest_path_distribution(distribution_name)
    max_edges = distribution.max_out_degree * (distribution.max_nodes - 1)
    prompt_tokens = distribution.max_nodes + 2 * max_edges + 6
    answer_tokens = distribution.max_path_length + 1
    return 2 + prompt_tokens + answer_tokens


def build_shortest_path_vocab(
    distribution_name: str,
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    distribution = get_shortest_path_distribution(distribution_name)
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
    tokens.extend(node_token(index) for index in range(distribution.max_nodes))
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


def permute_graph_labels(
    edges: Sequence[tuple[int, int]],
    path: Sequence[int],
    permutation: Sequence[int],
) -> tuple[list[tuple[int, int]], list[int]]:
    """Apply an arbitrary bijective node relabeling to a graph and its path."""
    num_nodes = len(permutation)
    if sorted(int(node) for node in permutation) != list(range(num_nodes)):
        raise ValueError("permutation must contain every node label exactly once")
    if not path:
        raise ValueError("path must not be empty")
    for source, target in edges:
        if not 0 <= source < num_nodes or not 0 <= target < num_nodes:
            raise ValueError("edge endpoint is outside the permutation")
    if any(not 0 <= node < num_nodes for node in path):
        raise ValueError("path node is outside the permutation")
    mapped_edges = [
        (int(permutation[source]), int(permutation[target]))
        for source, target in edges
    ]
    mapped_path = [int(permutation[node]) for node in path]
    return mapped_edges, mapped_path


def sample_shortest_path_graph(
    distribution_name: str,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], int, int, list[int], int]:
    """Sample a varied DAG with a solver-verified unique shortest path.

    A short route and several longer, randomly shaped alternatives establish
    the task signal. Additional topologically valid edges are sampled with a
    random density, following the distributional spirit of CLRS graph samplers.
    Labels are independently permuted only after graph construction.
    """
    distribution = get_shortest_path_distribution(distribution_name)
    path_length = rng.randint(
        distribution.min_path_length,
        distribution.max_path_length,
    )
    # Each minimally sized detour requires one internal node plus one feeder.
    # Sampling path length first keeps every length equally represented while
    # preventing impossible long-path/small-graph combinations.
    minimum_nodes = max(
        distribution.min_nodes,
        path_length + 1 + 2 * distribution.min_detours,
    )
    if minimum_nodes > distribution.max_nodes:
        raise ValueError(
            f"{distribution.name} cannot fit path length {path_length} "
            f"and {distribution.min_detours} detours"
        )
    num_nodes = rng.randint(minimum_nodes, distribution.max_nodes)
    path = list(range(path_length + 1))
    next_node = len(path)
    remaining_nodes = num_nodes - next_node
    detour_count = rng.randint(
        distribution.min_detours,
        min(
            distribution.max_detours,
            path_length,
            remaining_nodes // 2,
        ),
    )
    unused_branches = set(range(path_length))
    detours: dict[int, tuple[list[int], int, int]] = {}
    for detour_index in range(detour_count):
        reserve = 2 * (detour_count - detour_index - 1)
        available_for_detour = num_nodes - next_node - reserve - 1
        feasible = []
        for branch in unused_branches:
            for rejoin in range(branch + 1, path_length + 1):
                direct_span = rejoin - branch
                for penalty in range(1, distribution.max_detour_penalty + 1):
                    internal_nodes = direct_span + penalty - 1
                    if internal_nodes <= available_for_detour:
                        feasible.append(
                            (branch, rejoin, internal_nodes)
                        )
        if not feasible:
            break
        branch, rejoin, internal_count = rng.choice(feasible)
        unused_branches.remove(branch)
        detour_nodes = list(range(next_node, next_node + internal_count))
        next_node += internal_count
        feeder = next_node
        next_node += 1
        detours[branch] = (detour_nodes, rejoin, feeder)

    if len(detours) < distribution.min_detours:
        raise RuntimeError("distribution could not allocate its minimum detours")

    edges = {(path[index], path[index + 1]) for index in range(path_length)}
    for branch, (detour_nodes, rejoin, feeder) in detours.items():
        detour_path = [path[branch], *detour_nodes, path[rejoin]]
        edges.update(zip(detour_path, detour_path[1:]))
        edges.add((feeder, detour_nodes[0]))

    topological_order = []
    for index in range(path_length):
        topological_order.append(path[index])
        if index in detours:
            detour_nodes, _rejoin, feeder = detours[index]
            topological_order.append(feeder)
            topological_order.extend(detour_nodes)
    topological_order.append(path[-1])
    while next_node < num_nodes:
        topological_order.insert(
            rng.randrange(len(topological_order) + 1),
            next_node,
        )
        next_node += 1
    rank = {node: index for index, node in enumerate(topological_order)}

    out_degrees = [0] * num_nodes
    for source, _target in edges:
        out_degrees[source] += 1
    edge_probability = rng.uniform(
        distribution.min_edge_probability,
        distribution.max_edge_probability,
    )
    background_candidates = [
        (source, target)
        for source in range(num_nodes)
        for target in range(num_nodes)
        if rank[source] < rank[target] and (source, target) not in edges
    ]
    rng.shuffle(background_candidates)
    for source, target in background_candidates:
        if (
            out_degrees[source] >= distribution.max_out_degree
            or rng.random() >= edge_probability
        ):
            continue
        candidate_edges = [*edges, (source, target)]
        candidate_path, path_count = solve_shortest_path(
            num_nodes,
            candidate_edges,
            path[0],
            path[-1],
        )
        if path_count != 1 or candidate_path != path:
            continue
        edges.add((source, target))
        out_degrees[source] += 1

    permutation = list(range(num_nodes))
    rng.shuffle(permutation)
    mapped_edges, mapped_path = permute_graph_labels(edges, path, permutation)
    rng.shuffle(mapped_edges)
    solved_path, path_count = solve_shortest_path(
        num_nodes,
        mapped_edges,
        mapped_path[0],
        mapped_path[-1],
    )
    if path_count != 1 or solved_path != mapped_path:
        raise RuntimeError(
            "generated graph failed its final shortest-path verification"
        )
    return (
        mapped_edges,
        mapped_path[0],
        mapped_path[-1],
        mapped_path,
        num_nodes,
    )


def sample_shortest_path_example(
    distribution_name: str,
    stoi: Dict[str, int],
    rng: random.Random,
) -> tuple[list[int], list[int], list[tuple[int, int]], int, int, list[int]]:
    edges, start, goal, path, num_nodes = sample_shortest_path_graph(
        distribution_name,
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
    distribution_name: str,
    stoi: Dict[str, int],
    device=None,
    rng: random.Random | None = None,
) -> SymbolicBatch:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    get_shortest_path_distribution(distribution_name)
    rng = rng or random.Random()
    rows = []
    for _ in range(batch_size):
        prompt, answer, *_metadata = sample_shortest_path_example(
            distribution_name,
            stoi,
            rng,
        )
        rows.append(make_sequence(prompt, answer, stoi))
    return build_batch_from_sequences(rows, pad_id=stoi[PAD_TOKEN], device=device)


def parse_prompt_metadata(
    prompt_tokens: Sequence[int],
) -> tuple[list[tuple[int, int]], int, int]:
    tokens = [int(token_id) for token_id in prompt_tokens]
    if not tokens or tokens[0] != 4:
        raise ValueError("prompt must begin with <nodes>")
    try:
        edges_marker = tokens.index(5, 1)
        start_marker = tokens.index(6, edges_marker + 1)
    except ValueError as error:
        raise ValueError("prompt is missing a required marker") from error

    num_nodes = edges_marker - 1
    if num_nodes < 2:
        raise ValueError("prompt must list at least two nodes")
    listed_nodes = [
        token_id_to_node(token_id, num_nodes=num_nodes)
        for token_id in tokens[1:edges_marker]
    ]
    if any(node is None for node in listed_nodes) or set(listed_nodes) != set(range(num_nodes)):
        raise ValueError("prompt node list must contain every node exactly once")

    edge_start = edges_marker + 1
    edge_end = start_marker
    if (edge_end - edge_start) % 2 != 0:
        raise ValueError("prompt edge list must contain source-target pairs")
    edges = []
    for offset in range(edge_start, edge_end, 2):
        source = token_id_to_node(tokens[offset], num_nodes=num_nodes)
        target = token_id_to_node(tokens[offset + 1], num_nodes=num_nodes)
        if source is None or target is None:
            raise ValueError("prompt edge contains an invalid node token")
        edges.append((source, target))
    if len(set(edges)) != len(edges):
        raise ValueError("prompt contains duplicate edges")
    if (
        len(tokens) != start_marker + 4
        or tokens[start_marker] != 6
        or tokens[start_marker + 2] != 7
    ):
        raise ValueError("prompt must end with <start> node <goal> node")
    start = token_id_to_node(tokens[start_marker + 1], num_nodes=num_nodes)
    goal = token_id_to_node(tokens[start_marker + 3], num_nodes=num_nodes)
    if start is None or goal is None:
        raise ValueError("prompt start or goal token is invalid")
    return edges, start, goal


def token_id_to_node(token_id: int, *, num_nodes: int) -> int | None:
    node = int(token_id) - NODE_TOKEN_OFFSET
    return node if 0 <= node < num_nodes else None


__all__ = [
    "SHORTEST_PATH_DISTRIBUTIONS",
    "PATH_LENGTH_BUCKETS",
    "ShortestPathDistribution",
    "build_shortest_path_batch",
    "build_shortest_path_vocab",
    "get_shortest_path_distribution",
    "node_token",
    "path_length_bucket",
    "parse_prompt_metadata",
    "permute_graph_labels",
    "required_block_size",
    "sample_shortest_path_example",
    "sample_shortest_path_graph",
    "solve_shortest_path",
]
