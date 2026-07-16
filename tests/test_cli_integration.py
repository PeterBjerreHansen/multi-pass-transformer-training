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


def test_independent_memory_width_is_saved_in_run_config(tmp_path):
    run_dir = tmp_path / "memory_width"
    _run(
        "-m", "experiments.train_trace",
        "--preset", "random_graph_walk_smoke",
        "--architecture", "memory_tape",
        "--n-memory-embd", "8",
        "--device", "cpu",
        "--run-dir", str(run_dir),
    )
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["model_config"]["n_memory_embd"] == 8
    assert config["model_stats"]["memory_bytes_per_token"] == 32


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
