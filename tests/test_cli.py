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
