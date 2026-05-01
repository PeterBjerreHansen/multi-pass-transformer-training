import json
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def _train_tiny_trace_run(tmp_path: Path, *, architecture: str) -> Path:
    run_dir = tmp_path / f"{architecture}_trace_run"
    cmd = [
        sys.executable,
        "-m",
        "experiments.train_trace",
        "--preset",
        "random_graph_walk_smoke",
        "--architecture",
        architecture,
        "--device",
        "cpu",
        "--batch-size",
        "1",
        "--train-steps",
        "1",
        "--eval-interval",
        "1",
        "--eval-batches",
        "1",
        "--run-dir",
        str(run_dir),
    ]
    subprocess.run(cmd, cwd=ROOT_DIR, check=True, capture_output=True, text=True)
    return run_dir


def test_eval_trace_drift_runs_on_saved_trace_checkpoint(tmp_path):
    train_run_dir = _train_tiny_trace_run(tmp_path, architecture="memory_tape")
    eval_run_dir = tmp_path / "drift_eval"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.eval_trace_drift",
            "--input-run-dir",
            str(train_run_dir),
            "--run-dir",
            str(eval_run_dir),
            "--device",
            "cpu",
            "--eval-batches",
            "1",
            "--inference-mode",
            "final_pass",
            "--cache-source",
            "penultimate",
            "--token-selection",
            "argmax",
        ],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote" in result.stdout
    assert (eval_run_dir / "config.json").exists()
    assert (eval_run_dir / "summary.jsonl").exists()
    assert (eval_run_dir / "per_position.jsonl").exists()

    summary = [json.loads(line) for line in (eval_run_dir / "summary.jsonl").read_text().splitlines()]
    assert len(summary) == 1
    assert summary[0]["event"] == "trace_drift_summary"
    assert summary[0]["inference_mode"] == "final_pass"
    assert summary[0]["effective_inference_mode"] == "final_pass"
    assert summary[0]["cache_source"] == "penultimate"
    assert "token_legality" in summary[0]["metrics"]
    assert "sequence_legality" in summary[0]["metrics"]

    per_position = [json.loads(line) for line in (eval_run_dir / "per_position.jsonl").read_text().splitlines()]
    assert per_position
    assert per_position[0]["event"] == "per_position"
    assert per_position[0]["inference_mode"] == "final_pass"
    assert per_position[0]["effective_inference_mode"] == "final_pass"
    assert per_position[0]["cache_source"] == "penultimate"
    assert "token_legality" in per_position[0]


def test_eval_trace_drift_handles_transformer_runs(tmp_path):
    train_run_dir = _train_tiny_trace_run(tmp_path, architecture="transformer")
    eval_run_dir = tmp_path / "transformer_drift_eval"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.eval_trace_drift",
            "--input-run-dir",
            str(train_run_dir),
            "--run-dir",
            str(eval_run_dir),
            "--device",
            "cpu",
            "--eval-batches",
            "1",
            "--inference-mode",
            "final_pass",
            "--token-selection",
            "argmax",
        ],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = [json.loads(line) for line in (eval_run_dir / "summary.jsonl").read_text().splitlines()]
    assert summary[0]["architecture"] == "transformer"
    assert summary[0]["inference_mode"] == "final_pass"
    assert summary[0]["effective_inference_mode"] == "recompute"
