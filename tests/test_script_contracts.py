from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_all_project_workflows_live_under_scripts():
    assert not (ROOT / "runs").exists()
    assert (ROOT / "scripts" / "bbh" / "10_bbh_curriculum.sh").is_file()
    assert (ROOT / "scripts" / "trace" / "10_shortest_path_trace.sh").is_file()
    assert (ROOT / "scripts" / "local" / "10_main_matrix_pilot.sh").is_file()
    assert (ROOT / "scripts" / "ablations" / "10_memory_gate_init.sh").is_file()


def test_canonical_training_launchers_do_not_accept_scientific_overrides():
    launchers = [
        ROOT / "scripts" / "bbh" / "10_bbh_curriculum.sh",
        ROOT / "scripts" / "trace" / "10_random_graph_walk_trace.sh",
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
        for variable in prohibited:
            assert variable not in text, f"{launcher} accepts scientific override {variable}"


def test_local_workflows_are_explicitly_parameterized():
    local = (ROOT / "scripts" / "local" / "10_main_matrix_pilot.sh").read_text(
        encoding="utf-8"
    )
    assert "TRAIN_STEPS" in local
    assert "results/local_pilots" in local
