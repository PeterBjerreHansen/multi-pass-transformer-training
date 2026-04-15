import subprocess
import sys
from pathlib import Path

from tasks.bbh_symbolic_registry import TASKS


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_bbh_symbolic_registry_builds_all_task_specs():
    for task_name, spec in TASKS.items():
        vocab, stoi, _ = spec.build_vocab(spec.default_max_level)
        batch = spec.build_batch(
            2,
            spec.default_start_level,
            stoi,
            "final",
            "cpu",
            __import__("random").Random(100),
        )
        assert task_name == spec.name
        assert spec.min_level <= spec.default_start_level <= spec.default_max_level
        assert len(vocab) > 0
        assert batch.idx.size(0) == 2
        assert batch.targets.shape == batch.idx.shape
        assert batch.idx.size(1) <= spec.required_block_size(spec.default_max_level, "final")


def test_bbh_curriculum_entrypoint_runs_one_cpu_step(tmp_path):
    log_path = tmp_path / "bbh_symbolic.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "train_bbh_curriculum.py",
            "--task",
            "walk",
            "--architecture",
            "transformer",
            "--device",
            "cpu",
            "--model-size",
            "tiny",
            "--n-layer",
            "1",
            "--n-head",
            "1",
            "--n-embd",
            "8",
            "--curriculum-start-level",
            "1",
            "--max-level",
            "2",
            "--batch-size",
            "1",
            "--train-steps",
            "1",
            "--eval-interval",
            "1",
            "--eval-batches",
            "1",
            "--log-jsonl",
            str(log_path),
        ],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "task_family: bbh_symbolic" in result.stdout
    assert "training_mode: answer_curriculum" in result.stdout
    assert "task: walk" in result.stdout
    assert "log_policy: promotions" in result.stdout
    assert "promotion_history:" in result.stdout

    events = [line.split('"event": "')[1].split('"', 1)[0] for line in log_path.read_text().splitlines()]
    assert "run_start" in events
    assert "run_end" in events
    assert "train_step" not in events
    assert "eval" not in events
    assert "eval_easier" not in events


def test_bbh_curriculum_entrypoint_runs_permutation_task():
    result = subprocess.run(
        [
            sys.executable,
            "train_bbh_curriculum.py",
            "--task",
            "permutation",
            "--architecture",
            "transformer",
            "--device",
            "cpu",
            "--model-size",
            "tiny",
            "--n-layer",
            "1",
            "--n-head",
            "1",
            "--n-embd",
            "8",
            "--num-objects",
            "3",
            "--curriculum-start-level",
            "1",
            "--max-level",
            "2",
            "--batch-size",
            "1",
            "--train-steps",
            "1",
            "--eval-interval",
            "1",
            "--eval-batches",
            "1",
        ],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "task: permutation" in result.stdout
    assert "num_objects: 3" in result.stdout
    assert "promotion_history:" in result.stdout


def test_bbh_trace_entrypoint_runs_fixed_level_trace_step():
    result = subprocess.run(
        [
            sys.executable,
            "train_bbh_trace.py",
            "--task",
            "permutation",
            "--architecture",
            "transformer",
            "--device",
            "cpu",
            "--model-size",
            "tiny",
            "--n-layer",
            "1",
            "--n-head",
            "1",
            "--n-embd",
            "8",
            "--num-objects",
            "3",
            "--max-level",
            "2",
            "--eval-levels",
            "1,2",
            "--batch-size",
            "1",
            "--train-steps",
            "1",
            "--eval-interval",
            "1",
            "--eval-batches",
            "1",
        ],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "training_mode: trace_fixed" in result.stdout
    assert "supervision: trace" in result.stdout
    assert "fixed_level: 2" in result.stdout
    assert "eval_levels: 1,2" in result.stdout
    assert "eval_trace level" in result.stdout
