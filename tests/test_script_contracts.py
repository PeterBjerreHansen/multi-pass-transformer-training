import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_all_project_workflows_live_under_scripts():
    assert not (ROOT / "runs").exists()
    assert (ROOT / "scripts" / "bbh" / "10_bbh_curriculum.sh").is_file()
    assert (ROOT / "scripts" / "trace" / "10_shortest_path_trace.sh").is_file()
    assert (ROOT / "scripts" / "local" / "10_main_matrix_pilot.sh").is_file()
    assert (
        ROOT / "scripts" / "local" / "40_shortest_path_mastery.sh"
    ).is_file()
    assert not (
        ROOT / "scripts" / "local" / "40_shortest_path_overnight_calibration.sh"
    ).exists()
    assert (ROOT / "scripts" / "ablations" / "10_memory_gate_init.sh").is_file()


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
        ROOT / "scripts" / "bbh" / "10_bbh_curriculum.sh",
        ROOT / "scripts" / "trace" / "10_shortest_path_trace.sh",
        ROOT / "scripts" / "trace" / "10_othello_trace.sh",
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


def test_local_workflows_are_explicitly_parameterized():
    local = (ROOT / "scripts" / "local" / "10_main_matrix_pilot.sh").read_text(
        encoding="utf-8"
    )
    assert "TRAIN_STEPS" in local
    assert "results/local_pilots" in local


def test_shortest_path_workflows_use_only_smoke_and_main_distributions():
    difficulty = (
        ROOT / "scripts" / "local" / "30_trace_difficulty_sweep.sh"
    ).read_text(encoding="utf-8")
    assert 'SHORTEST_PATH_DISTRIBUTIONS="${SHORTEST_PATH_DISTRIBUTIONS:-smoke main}"' in difficulty
    assert "--shortest-path-distribution" in difficulty
    assert "SHORTEST_PATH_VARIANTS" not in difficulty

    mastery = (
        ROOT / "scripts" / "local" / "40_shortest_path_mastery.sh"
    ).read_text(encoding="utf-8")
    assert "run_distribution smoke" in mastery
    assert "run_distribution main" in mastery
    assert mastery.index("run_distribution smoke") < mastery.index(
        "run_distribution main"
    )
    assert "check_shortest_path_mastery.py" in mastery


def test_shortest_path_mastery_gate_requires_perfect_held_out_metrics(tmp_path):
    summary_dir = (
        tmp_path
        / "smoke"
        / "transformer"
        / "seed_1337"
        / "drift"
        / "recompute"
    )
    summary_dir.mkdir(parents=True)
    summary_path = summary_dir / "summary.json"
    payload = {
        "evaluation_examples": 1024,
        "metrics": {
            "exact_path": 1.0,
            "sequence_legality": 1.0,
            "goal_reached": 1.0,
        },
    }
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    command = [
        "python",
        str(ROOT / "scripts" / "check_shortest_path_mastery.py"),
        "--root",
        str(tmp_path),
        "--distribution",
        "smoke",
        "--architectures",
        "transformer",
        "--minimum-examples",
        "1024",
    ]
    mastered = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert mastered.returncode == 0

    payload["metrics"]["exact_path"] = 1023 / 1024
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    imperfect = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert imperfect.returncode != 0
    assert "not mastered" in imperfect.stderr


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
        ["bash", str(ROOT / "scripts" / "bbh" / "10_bbh_curriculum.sh")],
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
        ["bash", str(ROOT / "scripts" / "bbh" / "10_bbh_curriculum.sh")],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "invalid architecture in matrix: memory_state#" in result.stderr
    assert not log_path.exists()
