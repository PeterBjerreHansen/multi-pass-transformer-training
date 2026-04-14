import re
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_training_entrypoint_help_commands():
    for script_name in ("train_permutation.py", "train_repl.py"):
        result = subprocess.run(
            [sys.executable, script_name, "--help"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "usage:" in result.stdout


def test_training_entrypoint_help_omits_fixed_mode_flags():
    obsolete_flags = {
        "train_permutation.py": ("--eval-num-swaps", "--train-num-swaps"),
        "train_repl.py": ("--eval-program-lengths",),
    }

    for script_name, script_flags in obsolete_flags.items():
        result = subprocess.run(
            [sys.executable, script_name, "--help"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        assert not re.search(r"(^|[\s\[])--curriculum(?=[\s,\]\n])", result.stdout)
        for flag in script_flags:
            assert flag not in result.stdout


def test_training_entrypoints_reject_fixed_mode_flags():
    cases = (
        ("train_permutation.py", "--curriculum"),
        ("train_permutation.py", "--eval-num-swaps"),
        ("train_permutation.py", "--train-num-swaps"),
        ("train_repl.py", "--curriculum"),
        ("train_repl.py", "--eval-program-lengths"),
    )

    for script_name, flag in cases:
        result = subprocess.run(
            [sys.executable, script_name, flag],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "unrecognized arguments" in result.stderr
