#!/usr/bin/env python3
"""Select and report an adaptive shortest-path calibration sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median


METRIC_NAMES = (
    "loss",
    "exact_path",
    "optimal_path",
    "goal_reached",
    "valid_edge_rate",
    "token_legality",
    "sequence_legality",
)
STAGE_ONE_TARGETS = (0.15, 0.45)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select stage-two variants or summarize final calibration results."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--root", type=Path, required=True)
    select.add_argument("--stage-one-seed", type=int, default=1337)
    select.add_argument("--final-seeds", type=int, nargs="+", default=[1337, 2027, 4099])

    report = subparsers.add_parser("report")
    report.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_dir(root: Path, variant: str, architecture: str, seed: int) -> Path:
    return root / "runs" / variant / architecture / f"seed_{seed}"


def _summary(root: Path, variant: str, architecture: str, seed: int, mode: str) -> dict:
    path = _run_dir(root, variant, architecture, seed) / "drift" / mode / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing evaluation summary: {path}")
    payload = _read_json(path)
    if int(payload.get("evaluation_examples") or 0) < 1:
        raise ValueError(f"evaluation contains no examples: {path}")
    return payload


def _variant_spec(run_dir: Path) -> dict[str, int]:
    args = _read_json(run_dir / "config.json")["args"]
    return {
        "num_nodes": int(args["num_nodes"]),
        "path_length": int(args["shortest_path_length"]),
        "branching_factor": int(args["branching_factor"]),
        "distractor_edges": int(args["distractor_edges"]),
    }


def select_variants(root: Path, stage_one_seed: int, final_seeds: list[int]) -> dict:
    run_root = root / "runs"
    candidates = []
    for run_dir in sorted(run_root.glob(f"*/transformer/seed_{stage_one_seed}")):
        variant = run_dir.parents[1].name
        summary = _summary(root, variant, "transformer", stage_one_seed, "recompute")
        metrics = summary["metrics"]
        candidates.append(
            {
                "variant": variant,
                **_variant_spec(run_dir),
                "evaluation_examples": int(summary["evaluation_examples"]),
                "exact_path": float(metrics["exact_path"]),
                "valid_edge_rate": float(metrics["valid_edge_rate"]),
                "loss": float(metrics["loss"]),
            }
        )
    if len(candidates) < 2:
        raise ValueError("stage one must contain at least two evaluated variants")

    selected = []
    remaining = list(candidates)
    for target in STAGE_ONE_TARGETS:
        choice = min(
            remaining,
            key=lambda row: (
                abs(row["exact_path"] - target),
                abs(row["valid_edge_rate"] - 0.5),
                row["variant"],
            ),
        )
        selected.append({**choice, "selection_target": target})
        remaining.remove(choice)

    payload = {
        "stage_one_seed": stage_one_seed,
        "stage_one_targets": list(STAGE_ONE_TARGETS),
        "final_seeds": final_seeds,
        "selection_rule": (
            "Select distinct variants closest to 15% and 45% exact-path accuracy; "
            "break ties by valid-edge rate closest to 50%."
        ),
        "candidates": sorted(candidates, key=lambda row: row["exact_path"]),
        "selected": selected,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "calibration_selection.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    with (root / "calibration_selection.tsv").open("w", encoding="utf-8") as handle:
        for row in selected:
            print(
                row["variant"],
                row["num_nodes"],
                row["path_length"],
                row["branching_factor"],
                row["distractor_edges"],
                sep="\t",
                file=handle,
            )
    return payload


def _aggregate_metric(rows: list[dict], name: str) -> dict:
    values = [float(row[name]) for row in rows]
    return {"median": median(values), "values": values}


def _diagnostic_row(run_dir: Path) -> dict:
    payload = _read_json(run_dir / "diagnostics.json")
    zero_delta = payload["memory_interventions"]["loss_deltas"]["zero_memory_bank"]
    gate = payload["pass_dynamics"]["memory_gate_stats"]["mean_abs_effective"]
    schedule = payload["teacher_forced_schedule_gap"]["overall"]
    return {
        "zero_memory_loss_delta": float(zero_delta),
        "mean_abs_memory_gate": float(gate),
        "schedule_nll_delta": float(schedule["nll_delta"]),
        "schedule_top1_agreement": float(schedule["top1_agreement"]),
    }


def _architecture_summary(
    root: Path,
    variant: str,
    architecture: str,
    seeds: list[int],
) -> dict:
    modes = ["recompute"] if architecture == "transformer" else ["recompute", "append_recurrent"]
    result: dict[str, object] = {"modes": {}}
    for mode in modes:
        seed_rows = []
        for seed in seeds:
            payload = _summary(root, variant, architecture, seed, mode)
            seed_rows.append(
                {
                    "seed": seed,
                    "evaluation_examples": int(payload["evaluation_examples"]),
                    **{
                        name: float(payload["metrics"][name])
                        for name in METRIC_NAMES
                        if name in payload["metrics"]
                    },
                }
            )
        result["modes"][mode] = {
            "seeds": seed_rows,
            "aggregate": {
                name: _aggregate_metric(seed_rows, name)
                for name in METRIC_NAMES
                if all(name in row for row in seed_rows)
            },
        }
    if architecture == "memory_tape":
        diagnostics = [
            {"seed": seed, **_diagnostic_row(_run_dir(root, variant, architecture, seed))}
            for seed in seeds
        ]
        result["diagnostics"] = {
            "seeds": diagnostics,
            "aggregate": {
                name: _aggregate_metric(diagnostics, name)
                for name in (
                    "zero_memory_loss_delta",
                    "mean_abs_memory_gate",
                    "schedule_nll_delta",
                    "schedule_top1_agreement",
                )
            },
        }
    return result


def _count_in_band(values: list[float], low: float, high: float) -> int:
    return sum(low <= value <= high for value in values)


def _judge_variant(result: dict, seed_count: int) -> dict:
    transformer = result["architectures"]["transformer"]["modes"]["recompute"]
    memory = result["architectures"]["memory_tape"]["modes"]["append_recurrent"]
    memory_recompute = result["architectures"]["memory_tape"]["modes"]["recompute"]
    diagnostics = result["architectures"]["memory_tape"]["diagnostics"]["aggregate"]

    transformer_exact = transformer["aggregate"]["exact_path"]
    memory_exact = memory["aggregate"]["exact_path"]
    required_seed_count = min(2, seed_count)
    median_band = all(
        0.15 <= metric["median"] <= 0.85
        for metric in (transformer_exact, memory_exact)
    )
    stable_seeds = all(
        _count_in_band(metric["values"], 0.10, 0.90) >= required_seed_count
        for metric in (transformer_exact, memory_exact)
    )
    difficulty_calibrated = median_band and stable_seeds

    zero_delta = abs(diagnostics["zero_memory_loss_delta"]["median"])
    gate = diagnostics["mean_abs_memory_gate"]["median"]
    memory_sensitive = zero_delta >= 0.01 and gate >= 0.01

    append_exact = memory_exact["median"]
    recompute_exact = memory_recompute["aggregate"]["exact_path"]["median"]
    inference_gap = abs(append_exact - recompute_exact)
    inference_stable = inference_gap <= 0.05

    return {
        "difficulty_calibrated": difficulty_calibrated,
        "memory_sensitive": memory_sensitive,
        "inference_stable": inference_stable,
        "recommended_benchmark": (
            difficulty_calibrated and memory_sensitive and inference_stable
        ),
        "transformer_median_exact_path": transformer_exact["median"],
        "memory_tape_append_median_exact_path": append_exact,
        "memory_tape_recompute_median_exact_path": recompute_exact,
        "append_recompute_exact_path_gap": inference_gap,
        "median_zero_memory_loss_delta": diagnostics["zero_memory_loss_delta"]["median"],
        "median_mean_abs_memory_gate": gate,
    }


def _write_csv(root: Path, results: list[dict]) -> None:
    fieldnames = [
        "variant",
        "architecture",
        "mode",
        "seed",
        "evaluation_examples",
        *METRIC_NAMES,
        "zero_memory_loss_delta",
        "mean_abs_memory_gate",
        "schedule_nll_delta",
        "schedule_top1_agreement",
    ]
    with (root / "calibration_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            variant = result["variant"]
            memory_diagnostics = {
                row["seed"]: row
                for row in result["architectures"]["memory_tape"]["diagnostics"]["seeds"]
            }
            for architecture, architecture_result in result["architectures"].items():
                for mode, mode_result in architecture_result["modes"].items():
                    for row in mode_result["seeds"]:
                        output = {
                            "variant": variant,
                            "architecture": architecture,
                            "mode": mode,
                            **row,
                        }
                        if architecture == "memory_tape":
                            output.update(memory_diagnostics[row["seed"]])
                        writer.writerow(output)


def _write_markdown(root: Path, results: list[dict], recommendation: dict) -> None:
    lines = [
        "# Shortest-path calibration report",
        "",
        "| Variant | Transformer exact | MemoryTape recompute | MemoryTape append | "
        "Zero-memory Δloss | Gate | Decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        judgment = result["judgment"]
        decision = "recommended" if judgment["recommended_benchmark"] else "not qualified"
        lines.append(
            f"| {result['variant']} "
            f"| {judgment['transformer_median_exact_path']:.3f} "
            f"| {judgment['memory_tape_recompute_median_exact_path']:.3f} "
            f"| {judgment['memory_tape_append_median_exact_path']:.3f} "
            f"| {judgment['median_zero_memory_loss_delta']:.4f} "
            f"| {judgment['median_mean_abs_memory_gate']:.4f} "
            f"| {decision} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            recommendation["message"],
            "",
            "A static configuration qualifies only when both architectures have median "
            "exact-path accuracy between 15% and 85%, at least two seeds lie between "
            "10% and 90%, MemoryTape has a meaningful gate and zero-memory intervention, "
            "and append/recompute exact accuracy differs by no more than five points.",
        ]
    )
    (root / "calibration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def report(root: Path) -> dict:
    selection = _read_json(root / "calibration_selection.json")
    seeds = [int(seed) for seed in selection["final_seeds"]]
    results = []
    for selected in selection["selected"]:
        variant = selected["variant"]
        result = {
            "variant": variant,
            "spec": {
                key: selected[key]
                for key in (
                    "num_nodes",
                    "path_length",
                    "branching_factor",
                    "distractor_edges",
                )
            },
            "architectures": {
                architecture: _architecture_summary(root, variant, architecture, seeds)
                for architecture in ("transformer", "memory_tape")
            },
        }
        result["judgment"] = _judge_variant(result, len(seeds))
        results.append(result)

    qualified = [
        result for result in results if result["judgment"]["recommended_benchmark"]
    ]
    if qualified:
        chosen = min(
            qualified,
            key=lambda result: (
                abs(result["judgment"]["transformer_median_exact_path"] - 0.5)
                + abs(result["judgment"]["memory_tape_append_median_exact_path"] - 0.5),
                result["variant"],
            ),
        )
        recommendation = {
            "status": "qualified",
            "variant": chosen["variant"],
            "spec": chosen["spec"],
            "message": (
                f"Use `{chosen['variant']}` as the fixed shortest-path benchmark. "
                "It is learnable without saturation, memory-sensitive, and stable "
                "across inference schedules."
            ),
        }
    elif any(result["judgment"]["difficulty_calibrated"] for result in results):
        recommendation = {
            "status": "planning_only",
            "variant": None,
            "message": (
                "At least one difficulty is calibrated for planning quality, but none "
                "shows sufficiently causal MemoryTape use. Keep it as a planning "
                "benchmark, not as the primary memory-ablation benchmark."
            ),
        }
    else:
        recommendation = {
            "status": "needs_curriculum",
            "variant": None,
            "message": (
                "Neither confirmed difficulty lands in the learnable, non-saturated "
                "band. Static shortest-path difficulty remains cliff-like; implement "
                "a staged curriculum rather than spending more compute on a fixed grid."
            ),
        }

    payload = {
        "selection": selection,
        "results": results,
        "recommendation": recommendation,
    }
    (root / "calibration_report.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(root, results)
    _write_markdown(root, results, recommendation)
    return payload


def main() -> None:
    args = parse_args()
    if args.command == "select":
        payload = select_variants(args.root, args.stage_one_seed, args.final_seeds)
        for row in payload["selected"]:
            print(
                f"selected {row['variant']}: exact={row['exact_path']:.3f}, "
                f"valid_edge={row['valid_edge_rate']:.3f}, "
                f"target={row['selection_target']:.2f}"
            )
    else:
        payload = report(args.root)
        print(payload["recommendation"]["message"])
        print(f"report: {args.root / 'calibration_report.md'}")


if __name__ == "__main__":
    main()

