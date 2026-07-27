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
    "smoke": ShortestPathDistribution(
        name="smoke",
        min_nodes=7,
        max_nodes=7,
        min_path_length=2,
        max_path_length=3,
        max_out_degree=2,
        min_detours=1,
        max_detours=1,
        max_detour_penalty=2,
        min_edge_probability=0.05,
        max_edge_probability=0.25,
    ),
    "main": ShortestPathDistribution(
        name="main",
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
}


def get_shortest_path_distribution(name: str) -> ShortestPathDistribution:
    try:
        return SHORTEST_PATH_DISTRIBUTIONS[name]
    except KeyError as error:
        raise ValueError(f"unsupported shortest-path distribution: {name}") from error


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


def required_distribution_block_size(distribution_name: str) -> int:
    """Return a safe maximum block size for every example in a distribution."""
    distribution = get_shortest_path_distribution(distribution_name)
    max_edges = distribution.max_out_degree * (distribution.max_nodes - 1)
    prompt_tokens = distribution.max_nodes + 2 * max_edges + 6
    answer_tokens = distribution.max_path_length + 1
    return 2 + prompt_tokens + answer_tokens


def build_distribution_shortest_path_vocab(
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


def sample_distribution_shortest_path_graph(
    distribution_name: str,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], int, int, list[int], int]:
    """Sample a varied DAG with a solver-verified unique shortest path.

    A short route and one or two longer, randomly shaped alternatives establish
    the task signal. Additional topologically valid edges are sampled with a
    random density, following the distributional spirit of CLRS graph samplers.
    Labels are independently permuted only after graph construction.
    """
    distribution = get_shortest_path_distribution(distribution_name)
    num_nodes = rng.randint(distribution.min_nodes, distribution.max_nodes)
    path_length = rng.randint(
        distribution.min_path_length,
        distribution.max_path_length,
    )
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


def sample_distribution_shortest_path_example(
    distribution_name: str,
    stoi: Dict[str, int],
    rng: random.Random,
) -> tuple[list[int], list[int], list[tuple[int, int]], int, int, list[int]]:
    edges, start, goal, path, num_nodes = sample_distribution_shortest_path_graph(
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


def build_distribution_shortest_path_batch(
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
        prompt, answer, *_metadata = sample_distribution_shortest_path_example(
            distribution_name,
            stoi,
            rng,
        )
        rows.append(make_sequence(prompt, answer, stoi))
    return build_batch_from_sequences(rows, pad_id=stoi[PAD_TOKEN], device=device)


def parse_prompt_metadata(
    prompt_tokens: Sequence[int],
    *,
    num_nodes: int | None = None,
    edge_count: int | None = None,
) -> tuple[list[tuple[int, int]], int, int]:
    tokens = [int(token_id) for token_id in prompt_tokens]
    if not tokens or tokens[0] != 4:
        raise ValueError("prompt must begin with <nodes>")
    try:
        edges_marker = tokens.index(5, 1)
        start_marker = tokens.index(6, edges_marker + 1)
    except ValueError as error:
        raise ValueError("prompt is missing a required marker") from error

    inferred_num_nodes = edges_marker - 1
    if inferred_num_nodes < 2:
        raise ValueError("prompt must list at least two nodes")
    if num_nodes is not None and inferred_num_nodes != num_nodes:
        raise ValueError("prompt node count does not match num_nodes")
    num_nodes = inferred_num_nodes
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
    inferred_edge_count = (edge_end - edge_start) // 2
    if edge_count is not None and inferred_edge_count != edge_count:
        raise ValueError("prompt edge count does not match edge_count")
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


def legal_prefix_length(
    prompt_tokens: Sequence[int],
    generated_node_token_ids: Sequence[int],
    *,
    num_nodes: int | None = None,
    edge_count: int | None = None,
) -> tuple[int, bool]:
    edges, start, goal = parse_prompt_metadata(
        prompt_tokens,
        num_nodes=num_nodes,
        edge_count=edge_count,
    )
    if num_nodes is None:
        num_nodes = [int(token_id) for token_id in prompt_tokens].index(5) - 1
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


def graph_structure_metrics(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    start: int,
    goal: int,
    shortest_path: Sequence[int],
) -> dict[str, float]:
    """Measure realized connectivity without treating edge count as a task class."""
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    reverse: list[list[int]] = [[] for _ in range(num_nodes)]
    for source, target in edges:
        adjacency[source].append(target)
        reverse[target].append(source)

    def reachable_from(seed: int, graph: Sequence[Sequence[int]]) -> set[int]:
        reached = {seed}
        stack = [seed]
        while stack:
            node = stack.pop()
            for neighbor in graph[node]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    stack.append(neighbor)
        return reached

    reachable = reachable_from(start, adjacency)
    can_reach_goal = reachable_from(goal, reverse)
    relevant_edges = sum(
        source in reachable and target in can_reach_goal
        for source, target in edges
    )

    memo: dict[int, int] = {}

    def route_count(node: int, visiting: set[int]) -> int:
        if node == goal:
            return 1
        if node in memo:
            return memo[node]
        if node in visiting:
            raise ValueError("shortest-path structure metrics require a DAG")
        total = 0
        for target in adjacency[node]:
            total = min(2, total + route_count(target, visiting | {node}))
        memo[node] = total
        return total

    capped_route_count = route_count(start, set())
    decision_points = sum(
        len(adjacency[node]) > 1
        for node in shortest_path[:-1]
    )
    random_legal_probability = 1.0
    for node in shortest_path[:-1]:
        random_legal_probability /= len(adjacency[node])
    return {
        "node_count": float(num_nodes),
        "edge_count": float(len(edges)),
        "mean_out_degree": len(edges) / num_nodes,
        "max_out_degree": float(
            max((len(targets) for targets in adjacency), default=0)
        ),
        "reachable_node_fraction": len(reachable) / num_nodes,
        "goal_reachable_node_fraction": len(can_reach_goal) / num_nodes,
        "multi_route": float(capped_route_count > 1),
        "decision_points": float(decision_points),
        "relevant_edge_fraction": relevant_edges / max(len(edges), 1),
        "random_legal_path_probability": random_legal_probability,
    }


@torch.no_grad()
def shortest_path_generation_metrics(
    model,
    batch,
    args,
    *,
    inference_mode: str | None = None,
    num_nodes: int | None = None,
    edge_count: int | None = None,
    legacy_metric_semantics: bool = False,
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
        "mean_node_count": 0.0,
        "mean_edge_count": 0.0,
        "mean_out_degree": 0.0,
        "mean_max_out_degree": 0.0,
        "mean_reachable_node_fraction": 0.0,
        "mean_goal_reachable_node_fraction": 0.0,
        "multi_route_fraction": 0.0,
        "mean_decision_points": 0.0,
        "mean_relevant_edge_fraction": 0.0,
        "mean_random_legal_path_probability": 0.0,
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
        edges, start, goal = parse_prompt_metadata(
            prompt_tokens,
            num_nodes=num_nodes,
            edge_count=edge_count,
        )
        row_num_nodes = (
            num_nodes
            if num_nodes is not None
            else prompt_tokens.index(5) - 1
        )
        target_path_ids = target_suffix[:-1]
        legal_length, _all_legal = legal_prefix_length(
            prompt_tokens,
            generated_path_ids,
            num_nodes=row_num_nodes,
            edge_count=edge_count,
        )
        decoded_path = [
            token_id_to_node(token_id, num_nodes=row_num_nodes)
            for token_id in generated_path_ids
        ]
        edge_set = set(edges)
        valid_edges = sum(
            previous is not None
            and current is not None
            and (previous, current) in edge_set
            for previous, current in zip(decoded_path, decoded_path[1:])
        )
        edge_total = max(len(decoded_path) - 1, 0)
        path_is_edge_valid = bool(
            decoded_path
            and decoded_path[0] == start
            and all(node is not None for node in decoded_path)
            and valid_edges == edge_total
        )
        goal_reached = bool(
            path_is_edge_valid
            and decoded_path[-1] == goal
        )
        exact_path = generated_path_ids == target_path_ids
        complete = eos_position is not None and exact_path
        target_path = [
            token_id_to_node(token_id, num_nodes=row_num_nodes)
            for token_id in target_path_ids
        ]
        if any(node is None for node in target_path):
            raise ValueError("target path contains an invalid node token")
        structure = graph_structure_metrics(
            row_num_nodes,
            edges,
            start,
            goal,
            target_path,
        )
        totals["token_legality"] += min(1.0, legal_length / max(len(target_path_ids), 1))
        if legacy_metric_semantics:
            totals["sequence_legality"] += float(complete)
            totals["valid_edge_rate"] += legal_length / max(
                len(generated_path_ids),
                1,
            )
        else:
            totals["sequence_legality"] += float(path_is_edge_valid)
            totals["valid_edge_rate"] += valid_edges / max(edge_total, 1)
        totals["goal_reached"] += float(goal_reached)
        totals["optimal_path"] += float(exact_path)
        totals["exact_path"] += float(complete)
        totals["mean_generated_path_length"] += float(len(generated_path_ids))
        totals["mean_node_count"] += structure["node_count"]
        totals["mean_edge_count"] += structure["edge_count"]
        totals["mean_out_degree"] += structure["mean_out_degree"]
        totals["mean_max_out_degree"] += structure["max_out_degree"]
        totals["mean_reachable_node_fraction"] += structure[
            "reachable_node_fraction"
        ]
        totals["mean_goal_reachable_node_fraction"] += structure[
            "goal_reachable_node_fraction"
        ]
        totals["multi_route_fraction"] += structure["multi_route"]
        totals["mean_decision_points"] += structure["decision_points"]
        totals["mean_relevant_edge_fraction"] += structure["relevant_edge_fraction"]
        totals["mean_random_legal_path_probability"] += structure[
            "random_legal_path_probability"
        ]

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
    "SHORTEST_PATH_DISTRIBUTIONS",
    "ShortestPathDistribution",
    "build_distribution_shortest_path_batch",
    "build_distribution_shortest_path_vocab",
    "build_shortest_path_batch",
    "build_shortest_path_vocab",
    "format_shortest_path_eval_metrics",
    "get_shortest_path_distribution",
    "graph_structure_metrics",
    "legal_prefix_length",
    "node_token",
    "parse_prompt_metadata",
    "permute_graph_labels",
    "required_distribution_block_size",
    "required_block_size",
    "sample_distribution_shortest_path_example",
    "sample_distribution_shortest_path_graph",
    "sample_shortest_path_example",
    "sample_unique_shortest_path_graph",
    "shortest_path_generation_metrics",
    "solve_shortest_path",
]
