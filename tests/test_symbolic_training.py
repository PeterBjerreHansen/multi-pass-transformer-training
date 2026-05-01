import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.train_bbh import BBH_TASKS, build_training_objects as build_bbh_training_objects
from experiments.train_bbh import parse_args as parse_bbh_args
from experiments.train_trace import build_training_objects as build_trace_training_objects
from experiments.train_trace import parse_args as parse_trace_args


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_custom_block_size_must_cover_required_task_size():
    args = parse_bbh_args(
        [
            "--preset",
            "pointer_chasing_main",
            "--device",
            "cpu",
        ]
    )
    args.block_size = 8

    with pytest.raises(ValueError, match="required block size"):
        from experiments.train_bbh import validate_task_args

        validate_task_args(args)


def test_bbh_training_objects_respect_task_defaults():
    args = parse_bbh_args(
        [
            "--preset",
            "pointer_chasing_smoke",
            "--device",
            "cpu",
        ]
    )
    task, block_size, vocab, stoi, model, optimizer = build_bbh_training_objects(args)

    assert task.name == "pointer_chasing"
    assert task is BBH_TASKS["pointer_chasing"]
    assert "n3" in stoi
    assert "n4" not in stoi
    assert block_size == task.required_block_size(args, args.max_level)


def test_trace_random_graph_walk_cli_accepts_custom_sizes():
    args = parse_trace_args(
        [
            "--preset",
            "random_graph_walk_smoke",
            "--device",
            "cpu",
        ]
    )
    block_size, vocab, stoi, model, optimizer = build_trace_training_objects(args)

    assert "s3" in stoi
    assert "t3" in stoi
    assert block_size >= 1


def test_trace_othello_preset_builds_training_objects():
    args = parse_trace_args(
        [
            "--preset",
            "othello_smoke",
            "--device",
            "cpu",
        ]
    )
    block_size, vocab, stoi, model, optimizer = build_trace_training_objects(args)

    assert "m28" in stoi
    assert block_size == 62


def test_trace_run_dir_writes_run_artifacts(tmp_path):
    run_dir = tmp_path / "trace_run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.train_trace",
            "--preset",
            "random_graph_walk_smoke",
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
        ],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "task: random_graph_walk" in result.stdout
    assert (run_dir / "config.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "latest.pt").exists()

    config = json.loads((run_dir / "config.json").read_text())
    assert config["args"]["run_dir"] == str(run_dir)
    assert config["args"]["inference_mode"] == "recompute"


def test_trace_resume_from_latest_checkpoint_continues_steps(tmp_path):
    run_dir = tmp_path / "resume_trace"
    base_cmd = [
        sys.executable,
        "-m",
        "experiments.train_trace",
        "--preset",
        "random_graph_walk_smoke",
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

    subprocess.run(base_cmd, cwd=ROOT_DIR, check=True, capture_output=True, text=True)
    checkpoint = run_dir / "latest.pt"
    assert checkpoint.exists()

    resumed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.train_trace",
            "--resume-from",
            str(run_dir),
            "--train-steps",
            "1",
        ],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "resume_from:" in resumed.stdout
    assert "device: cpu" in resumed.stdout
    assert "step     2" in resumed.stdout

    events = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    train_steps = [event["step"] for event in events if event["event"] == "train_step"]
    assert train_steps[:2] == [0, 1]
    assert train_steps[-1] == 2


def test_bbh_resume_from_latest_checkpoint_continues_steps(tmp_path):
    run_dir = tmp_path / "resume_bbh"
    base_cmd = [
        sys.executable,
        "-m",
        "experiments.train_bbh",
        "--preset",
        "pointer_chasing_smoke",
        "--architecture",
        "memory_concat",
        "--device",
        "cpu",
        "--train-steps",
        "1",
        "--run-dir",
        str(run_dir),
    ]

    subprocess.run(base_cmd, cwd=ROOT_DIR, check=True, capture_output=True, text=True)

    resumed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.train_bbh",
            "--resume-from",
            str(run_dir),
            "--train-steps",
            "1",
        ],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "resume_from:" in resumed.stdout
    assert "device: cpu" in resumed.stdout
    assert "step     2" in resumed.stdout

    events = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    train_steps = [event["step"] for event in events if event["event"] == "train_step"]
    assert train_steps[:2] == [1, 2]
    assert train_steps[-1] == 2


def test_bbh_entrypoint_runs_multipass_step():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.train_bbh",
            "--preset",
            "pointer_chasing_smoke",
            "--architecture",
            "memory_concat",
            "--device",
            "cpu",
            "--train-steps",
            "1",
        ],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "architecture: memory_concat" in result.stdout
    assert "cache_source: last" in result.stdout
    assert "token_selection: argmax" in result.stdout


def test_trace_random_graph_walk_entrypoint_runs_level_trace_step():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.train_trace",
            "--preset",
            "random_graph_walk_smoke",
            "--device",
            "cpu",
            "--architecture",
            "transformer",
            "--batch-size",
            "1",
            "--train-steps",
            "1",
            "--eval-interval",
            "1",
            "--eval-batches",
            "1",
            "--token-selection",
            "argmax",
        ],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "training_mode: trace_fixed" in result.stdout
    assert "level: 2" in result.stdout
    assert "token_legality" in result.stdout
    assert "sequence_legality" in result.stdout


def test_trace_othello_entrypoint_runs_legality_eval(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.train_trace",
            "--preset",
            "othello_smoke",
            "--device",
            "cpu",
            "--architecture",
            "transformer",
            "--batch-size",
            "1",
            "--train-steps",
            "1",
            "--eval-interval",
            "1",
            "--eval-batches",
            "1",
            "--othello-data-dir",
            str(tmp_path / "othello_data"),
            "--othello-train-games",
            "16",
            "--othello-val-games",
            "8",
            "--othello-dataset-seed",
            "9",
            "--token-selection",
            "argmax",
        ],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "task: othello" in result.stdout
    assert "token_legality" in result.stdout
    assert "sequence_legality" in result.stdout
    assert "mean_legal_len" in result.stdout
