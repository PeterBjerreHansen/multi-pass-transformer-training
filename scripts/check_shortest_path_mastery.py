#!/usr/bin/env python3
"""Require every selected architecture to master a shortest-path distribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--distribution", choices=["smoke", "main"], required=True)
    parser.add_argument("--architectures", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--minimum-examples", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    failures = []
    for architecture in args.architectures:
        modes = ["recompute"] if architecture == "transformer" else [
            "recompute",
            "append_recurrent",
        ]
        for mode in modes:
            path = (
                args.root
                / args.distribution
                / architecture
                / f"seed_{args.seed}"
                / "drift"
                / mode
                / "summary.json"
            )
            if not path.is_file():
                raise FileNotFoundError(f"missing mastery evaluation: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            examples = int(payload["evaluation_examples"])
            metrics = payload["metrics"]
            row = {
                "architecture": architecture,
                "mode": mode,
                "evaluation_examples": examples,
                "exact_path": float(metrics["exact_path"]),
                "sequence_legality": float(metrics["sequence_legality"]),
                "goal_reached": float(metrics["goal_reached"]),
            }
            rows.append(row)
            if (
                examples < args.minimum_examples
                or row["exact_path"] != 1.0
                or row["sequence_legality"] != 1.0
                or row["goal_reached"] != 1.0
            ):
                failures.append(row)

    output = {
        "distribution": args.distribution,
        "seed": args.seed,
        "minimum_examples": args.minimum_examples,
        "mastered": not failures,
        "rows": rows,
    }
    output_path = args.root / f"{args.distribution}_mastery.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(
            f"{row['architecture']:20s} {row['mode']:18s} "
            f"exact={row['exact_path']:.4f} "
            f"legal={row['sequence_legality']:.4f} "
            f"goal={row['goal_reached']:.4f} "
            f"n={row['evaluation_examples']}"
        )
    print(f"mastery report: {output_path}")
    if failures:
        raise SystemExit(
            f"{args.distribution} is not mastered by every architecture and mode"
        )


if __name__ == "__main__":
    main()
