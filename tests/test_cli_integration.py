from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONPATH": str(ROOT)})
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_bbh_training_cli_writes_restorable_checkpoint(tmp_path):
    run_dir = tmp_path / "bbh"
    result = _run(
        "-m", "experiments.train_bbh",
        "--preset", "pointer_chasing_smoke",
        "--architecture", "joint_memory_tape",
        "--device", "cpu",
        "--run-dir", str(run_dir),
    )
    assert "architecture: joint_memory_tape" in result.stdout
    assert (run_dir / "latest.pt").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
    events = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    evaluation = next(event for event in events if event["event"] == "eval")
    assert "gradient_norms" in evaluation
    assert evaluation["gradient_norms"]["global"]["max"] > 0


def test_variable_depth_training_logs_sampled_depth(tmp_path):
    run_dir = tmp_path / "variable_depth"
    _run(
        "-m", "experiments.train_trace",
        "--preset", "random_graph_walk_smoke",
        "--architecture", "memory_tape",
        "--train-pass-range", "2", "6",
        "--sampled-tail-loss-weights", "0.3", "0.7",
        "--device", "cpu",
        "--run-dir", str(run_dir),
    )
    events = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    evaluation = next(event for event in events if event["event"] == "eval")
    assert 2 <= evaluation["sampled_n_pass"] <= 6
    assert sum(evaluation["sampled_pass_histogram"].values()) == 1
    assert evaluation["effective_pass_loss_weights"][-2:] == [0.3, 0.7]


def test_trace_training_drift_and_diagnostics_cli(tmp_path):
    run_dir = tmp_path / "trace"
    _run(
        "-m", "experiments.train_trace",
        "--preset", "random_graph_walk_smoke",
        "--architecture", "joint_memory_tape",
        "--device", "cpu",
        "--run-dir", str(run_dir),
    )

    diagnostics = tmp_path / "diagnostics.json"
    _run(
        "-m", "experiments.eval_diagnostics",
        "--input-run-dir", str(run_dir),
        "--device", "cpu",
        "--batch-size", "2",
        "--eval-batches", "1",
        "--extra-passes", "2",
        "--output", str(diagnostics),
    )
    payload = json.loads(diagnostics.read_text(encoding="utf-8"))
    assert "memory_interventions" in payload
    assert len(payload["pass_dynamics"]["extra_passes"]) == 2
    assert payload["teacher_forced_schedule_gap"]["horizon"] == 16
    assert payload["teacher_forced_schedule_gap"]["overall"]["count"] > 0

    trace_events = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    trace_evaluation = next(event for event in trace_events if event["event"] == "eval")
    assert trace_evaluation["gradient_norms"]["global"]["mean"] > 0

    drift_dir = tmp_path / "drift"
    _run(
        "-m", "experiments.eval_trace_drift",
        "--input-run-dir", str(run_dir),
        "--inference-mode", "append_recurrent",
        "--token-selection", "argmax",
        "--device", "cpu",
        "--eval-batches", "1",
        "--run-dir", str(drift_dir),
    )
    summary = json.loads((drift_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["effective_inference_mode"] == "append_recurrent"
    assert (drift_dir / "per_position.jsonl").exists()


def test_othello_random_prefix_evaluation_cli(tmp_path):
    run_dir = tmp_path / "othello"
    data_dir = tmp_path / "othello_data"
    _run(
        "-m", "experiments.train_trace",
        "--preset", "othello_smoke",
        "--architecture", "memory_tape",
        "--othello-data-dir", str(data_dir),
        "--othello-train-games", "8",
        "--othello-val-games", "4",
        "--device", "cpu",
        "--run-dir", str(run_dir),
    )

    output_dir = tmp_path / "othello_eval"
    _run(
        "-m", "experiments.eval_othello",
        "--input-run-dir", str(run_dir),
        "--output-dir", str(output_dir),
        "--evaluation-mode", "random-prefix",
        "--inference-modes", "recompute", "append_recurrent",
        "--token-selection", "argmax",
        "--examples", "1",
        "--device", "cpu",
    )
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["evaluated_inference_modes"] == ["recompute", "append_recurrent"]
    for mode in summary["evaluated_inference_modes"]:
        overall = summary["modes"][mode]["overall"]
        assert overall["count"] == 1
        assert overall["teacher_move_count"] > 0
        assert 0.0 <= overall["teacher_forced"]["legal_probability_mass"] <= 1.0
    rows = [
        json.loads(line)
        for line in (output_dir / "per_example.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2


def test_shortest_path_training_resume_drift_and_diagnostics_cli(tmp_path):
    run_dir = tmp_path / "shortest_path"
    result = _run(
        "-m", "experiments.train_trace",
        "--preset", "shortest_path_smoke",
        "--architecture", "memory_tape",
        "--device", "cpu",
        "--run-dir", str(run_dir),
    )
    assert "task: shortest_path" in result.stdout
    events = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    evaluation = next(event for event in events if event["event"] == "eval")
    for metric in ("valid_edge_rate", "goal_reached", "optimal_path", "exact_path"):
        assert metric in evaluation["metrics"]

    _run(
        "-m", "experiments.train_trace",
        "--preset", "shortest_path_smoke",
        "--resume-from", str(run_dir),
        "--train-steps", "1",
        "--device", "cpu",
        "--run-dir", str(run_dir),
    )
    resumed_events = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event.get("event") == "eval" and event.get("step") == 2 for event in resumed_events)

    for inference_mode in ("recompute", "append_recurrent"):
        drift_dir = tmp_path / f"drift_{inference_mode}"
        _run(
            "-m", "experiments.eval_trace_drift",
            "--input-run-dir", str(run_dir),
            "--inference-mode", inference_mode,
            "--token-selection", "argmax",
            "--device", "cpu",
            "--eval-batches", "1",
            "--run-dir", str(drift_dir),
        )
        summary = json.loads((drift_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["task"] == "shortest_path"
        assert "optimal_path" in summary["metrics"]
        assert (drift_dir / "per_position.jsonl").exists()

    diagnostics = tmp_path / "shortest_path_diagnostics.json"
    _run(
        "-m", "experiments.eval_diagnostics",
        "--input-run-dir", str(run_dir),
        "--device", "cpu",
        "--batch-size", "2",
        "--eval-batches", "1",
        "--extra-passes", "1",
        "--schedule-gap-horizon", "4",
        "--output", str(diagnostics),
    )
    payload = json.loads(diagnostics.read_text(encoding="utf-8"))
    assert payload["task"] == "shortest_path"
    assert payload["teacher_forced_schedule_gap"]["overall"]["count"] > 0
