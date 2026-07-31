"""Task-specific evaluation for shortest-path trace generation."""
from __future__ import annotations

from typing import Sequence

import torch

from tasks.trace import shortest_path


def graph_structure_metrics(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    start: int,
    goal: int,
    target_path: Sequence[int],
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
        for node in target_path[:-1]
    )
    random_legal_probability = 1.0
    for node in target_path[:-1]:
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
def generation_metrics(
    model,
    batch,
    args,
    *,
    inference_mode: str | None = None,
    **_unused,
) -> dict[str, float]:
    """Evaluate generated paths against graph legality and the optimal target."""
    mode = (
        "recompute"
        if args.architecture == "transformer"
        else (inference_mode or args.inference_mode)
    )
    do_sample = getattr(args, "token_selection", "argmax") == "sample"
    totals = {
        "token_legality": 0.0,
        "sequence_legality": 0.0,
        "valid_edge_rate": 0.0,
        "goal_reached": 0.0,
        "optimal_path": 0.0,
        "exact_path": 0.0,
        "mean_generated_path_length": 0.0,
        "mean_target_path_length": 0.0,
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
    bucket_counts = {
        bucket: 0
        for bucket in shortest_path.PATH_LENGTH_BUCKETS
    }
    bucket_optimal = {
        bucket: 0.0
        for bucket in shortest_path.PATH_LENGTH_BUCKETS
    }
    step_counts = {
        step: 0
        for step in range(
            1,
            shortest_path.get_shortest_path_distribution(
                args.shortest_path_distribution
            ).max_path_length
            + 1,
        )
    }
    step_correct = {step: 0.0 for step in step_counts}

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
        generated_suffix = generated[
            0,
            prompt_len : prompt_len + output_len,
        ].tolist()
        eos_position = next(
            (
                position
                for position, token_id in enumerate(generated_suffix)
                if token_id == 3
            ),
            None,
        )
        generated_path_ids = (
            generated_suffix
            if eos_position is None
            else generated_suffix[:eos_position]
        )
        prompt_tokens = batch.idx[row, 1 : prompt_len - 1].tolist()
        edges, start, goal = shortest_path.parse_prompt_metadata(prompt_tokens)
        row_num_nodes = prompt_tokens.index(5) - 1
        target_path_ids = target_suffix[:-1]
        target_path_length = len(target_path_ids) - 1
        bucket = shortest_path.path_length_bucket(target_path_length)
        legal_length, _all_legal = shortest_path.legal_prefix_length(
            prompt_tokens,
            generated_path_ids,
        )
        decoded_path = [
            shortest_path.token_id_to_node(
                token_id,
                num_nodes=row_num_nodes,
            )
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
            shortest_path.token_id_to_node(
                token_id,
                num_nodes=row_num_nodes,
            )
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
        totals["token_legality"] += min(
            1.0,
            legal_length / max(len(target_path_ids), 1),
        )
        totals["sequence_legality"] += float(path_is_edge_valid)
        totals["valid_edge_rate"] += valid_edges / max(edge_total, 1)
        totals["goal_reached"] += float(goal_reached)
        totals["optimal_path"] += float(exact_path)
        totals["exact_path"] += float(complete)
        totals["mean_generated_path_length"] += float(len(generated_path_ids))
        totals["mean_target_path_length"] += float(target_path_length)
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
        totals["mean_relevant_edge_fraction"] += structure[
            "relevant_edge_fraction"
        ]
        totals["mean_random_legal_path_probability"] += structure[
            "random_legal_path_probability"
        ]
        bucket_counts[bucket] += 1
        bucket_optimal[bucket] += float(exact_path)
        for step in range(1, len(target_path_ids)):
            step_counts[step] += 1
            step_correct[step] += float(
                step < len(generated_path_ids)
                and generated_path_ids[step] == target_path_ids[step]
            )

    count = int(batch.idx.shape[0])
    result = {key: value / count for key, value in totals.items()}
    for bucket in shortest_path.PATH_LENGTH_BUCKETS:
        bucket_count = bucket_counts[bucket]
        if bucket_count:
            result[f"optimal_path_{bucket}"] = (
                bucket_optimal[bucket] / bucket_count
            )
            result[f"optimal_path_{bucket}__weight"] = float(bucket_count)
            result[f"examples_{bucket}__sum"] = float(bucket_count)
    for step, step_count in step_counts.items():
        if step_count:
            metric = f"path_step_{step}_accuracy"
            result[metric] = step_correct[step] / step_count
            result[f"{metric}__weight"] = float(step_count)
            result[f"path_step_{step}_examples__sum"] = float(step_count)
    return result


def format_metrics(metrics: dict[str, float]) -> str:
    fields = [
        f"optimal {metrics['optimal_path']:.3f}",
        f"goal {metrics['goal_reached']:.3f}",
        f"edge_valid {metrics['valid_edge_rate']:.3f}",
    ]
    fields.extend(
        f"{bucket} {metrics[f'optimal_path_{bucket}']:.3f}"
        for bucket in shortest_path.PATH_LENGTH_BUCKETS
        if f"optimal_path_{bucket}" in metrics
    )
    step_numbers = sorted(
        int(key.removeprefix("path_step_").removesuffix("_accuracy"))
        for key in metrics
        if key.startswith("path_step_") and key.endswith("_accuracy")
    )
    step_fields = [
        f"{step}:{metrics[f'path_step_{step}_accuracy']:.3f}"
        for step in step_numbers
    ]
    if step_fields:
        fields.append(f"step_acc [{', '.join(step_fields)}]")
    return " | ".join(fields)


__all__ = [
    "format_metrics",
    "generation_metrics",
    "graph_structure_metrics",
]
