import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_all_project_workflows_live_under_scripts():
    assert not (ROOT / "runs").exists()
    assert (ROOT / "scripts" / "bbh" / "run.sh").is_file()
    assert (ROOT / "scripts" / "trace" / "run.sh").is_file()
    assert (ROOT / "scripts" / "trace" / "eval_othello.sh").is_file()
    assert (
        ROOT / "scripts" / "trace" / "compare_shortest_path_difficulty.sh"
    ).is_file()
    assert (ROOT / "scripts" / "trace" / "test_shortest_path.sh").is_file()
    assert (
        ROOT / "scripts" / "trace" / "ablate_shortest_path_gate_init.sh"
    ).is_file()
    assert (ROOT / "scripts" / "test_smoke.sh").is_file()
    assert not (ROOT / "scripts" / "local").exists()
    assert not (ROOT / "scripts" / "drift").exists()
    assert not (ROOT / "scripts" / "ablations").exists()


def test_launcher_architecture_registry_matches_model_factory():
    from model_factory import ARCHITECTURES

    env = os.environ.copy()
    env["MATRIX_LIB"] = str(ROOT / "scripts" / "lib" / "model_matrix.sh")
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "${MATRIX_LIB}"; printf "%s\\n" "${MPT_ARCHITECTURES[@]}"',
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert tuple(result.stdout.splitlines()) == ARCHITECTURES


def test_canonical_training_launchers_do_not_accept_scientific_overrides():
    launchers = [
        ROOT / "scripts" / "bbh" / "run.sh",
        ROOT / "scripts" / "trace" / "run.sh",
    ]
    prohibited = (
        "TRAIN_STEPS",
        "EVAL_INTERVAL",
        "EVAL_BATCHES",
        "BATCH_SIZE",
        "MEMORY_GATE_INIT",
        "TOKEN_SELECTION",
    )
    for launcher in launchers:
        text = launcher.read_text(encoding="utf-8")
        assert "memory_add" in text, f"{launcher} omits memory_add from its default matrix"
        assert "memory_state" in text, f"{launcher} omits memory_state from its default matrix"
        for variable in prohibited:
            assert variable not in text, f"{launcher} accepts scientific override {variable}"


def test_shortest_path_workflows_use_only_easy_and_main_distributions():
    difficulty = (
        ROOT / "scripts" / "trace" / "compare_shortest_path_difficulty.sh"
    ).read_text(encoding="utf-8")
    assert 'SHORTEST_PATH_DISTRIBUTIONS="${SHORTEST_PATH_DISTRIBUTIONS:-easy}"' in difficulty
    assert 'ARCHITECTURES="${ARCHITECTURES:-transformer}"' in difficulty
    assert 'TRAIN_STEPS="${TRAIN_STEPS:-10000}"' in difficulty
    assert 'MIN_QUAL_EXAMPLES="${MIN_QUAL_EXAMPLES:-4096}"' in difficulty
    assert "--shortest-path-distribution" in difficulty
    assert "SHORTEST_PATH_VARIANTS" not in difficulty


def test_trace_launcher_runs_task_matrix_into_task_first_results(tmp_path):
    bin_dir, log_path = _fake_python(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "DEVICE": "cpu",
            "TASKS": "shortest_path othello",
            "ARCHITECTURES": "memory_add",
            "SEEDS": "1337",
            "RESULT_ROOT": str(tmp_path / "results"),
        }
    )
    subprocess.run(
        ["bash", str(ROOT / "scripts" / "trace" / "run.sh")],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    calls = log_path.read_text(encoding="utf-8").splitlines()
    training_calls = [
        call for call in calls if "-m experiments.train_trace" in call
    ]
    assert len(training_calls) == 2
    assert "--preset shortest_path_main" in training_calls[0]
    assert "--preset othello_main" in training_calls[1]
    assert all("--architecture memory_add" in call for call in training_calls)
    assert f"--run-dir {tmp_path}/results/shortest_path/memory_add/seed_1337" in training_calls[0]
    assert f"--run-dir {tmp_path}/results/othello/memory_add/seed_1337" in training_calls[1]


def _fake_python(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "python_args.txt"
    executable = bin_dir / "python"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {log_path!s}\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return bin_dir, log_path


def test_bbh_launcher_passes_new_architecture_names_without_suffixes(tmp_path):
    bin_dir, log_path = _fake_python(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "TASKS": "permutation",
            "ARCHITECTURES": "memory_add memory_state",
            "SEEDS": "1337",
            "RESULT_ROOT": str(tmp_path / "results"),
        }
    )
    subprocess.run(
        ["bash", str(ROOT / "scripts" / "bbh" / "run.sh")],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    calls = log_path.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert "--architecture memory_add" in calls[0]
    assert "--architecture memory_state" in calls[1]
    assert all("memory_state#" not in call for call in calls)


def test_bbh_launcher_rejects_entire_bad_matrix_before_starting(tmp_path):
    bin_dir, log_path = _fake_python(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "TASKS": "permutation",
            "ARCHITECTURES": "memory_add memory_state#",
            "SEEDS": "1337",
        }
    )
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "bbh" / "run.sh")],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "invalid architecture in matrix: memory_state#" in result.stderr
    assert not log_path.exists()
