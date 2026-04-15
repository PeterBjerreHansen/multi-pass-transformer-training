import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_training_entrypoint_help_commands():
    for script_name in (
        "train_bbh_curriculum.py",
        "train_bbh_trace.py",
    ):
        result = subprocess.run(
            [sys.executable, script_name, "--help"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "usage:" in result.stdout


def test_bbh_entrypoints_have_separate_regime_clis():
    curriculum = subprocess.run(
        [sys.executable, "train_bbh_curriculum.py", "--help"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    trace = subprocess.run(
        [sys.executable, "train_bbh_trace.py", "--help"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--supervision" not in curriculum.stdout
    assert "--curriculum-start-level" in curriculum.stdout
    assert "--eval-levels" not in curriculum.stdout
    assert "--supervision" not in trace.stdout
    assert "--curriculum-start-level" not in trace.stdout
    assert "--eval-levels" in trace.stdout
