import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_training_entrypoint_help_commands():
    for module_name in (
        "experiments.train_bbh",
        "experiments.train_trace",
        "experiments.eval_trace_drift",
    ):
        result = subprocess.run(
            [sys.executable, "-m", module_name, "--help"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "usage:" in result.stdout


def test_symbolic_entrypoints_have_updated_clis():
    curriculum = subprocess.run(
        [sys.executable, "-m", "experiments.train_bbh", "--help"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    trace = subprocess.run(
        [sys.executable, "-m", "experiments.train_trace", "--help"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--preset" in curriculum.stdout
    assert "--token-selection" in curriculum.stdout
    assert "--run-dir" in curriculum.stdout
    assert "--architecture" in curriculum.stdout
    assert "--memory-tape-gate" in curriculum.stdout
    assert "pointer_chasing_main" in curriculum.stdout
    assert "tracking_smoke" in curriculum.stdout
    assert "--cache-source" not in curriculum.stdout
    assert "--batch-size" not in curriculum.stdout
    assert "--eval-interval" not in curriculum.stdout
    assert "--lr" not in curriculum.stdout
    assert "--num-nodes" not in curriculum.stdout
    assert "--curriculum-threshold" not in curriculum.stdout

    assert "--preset" in trace.stdout
    assert "--architecture" in trace.stdout
    assert "random_graph_walk_main" in trace.stdout
    assert "othello_main" in trace.stdout

    for help_text in (
        curriculum.stdout,
        trace.stdout,
    ):
        assert "--generation-mode" not in help_text
        assert "--greedy-cache" not in help_text
        assert "--results-dir" not in help_text
        assert "--log-jsonl" not in help_text
        assert "--compare-generation-modes" not in help_text
        assert "--task" not in help_text
