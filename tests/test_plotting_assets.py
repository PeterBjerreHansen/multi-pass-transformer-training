from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_readme_uses_local_figure_paths_and_no_fetch_helper_exists():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    expected = {
        "bbh_curriculum_fig.png",
        "inference_pattern_fig.png",
        "generation_fig.png",
        "multipass_training_fig.png",
        "mismatch_fig.png",
        "trace_plot_figs.png",
    }
    for filename in expected:
        assert f"figures/{filename}" in readme
        figure_path = ROOT / "figures" / filename
        assert figure_path.is_file()
        assert figure_path.stat().st_size > 0
    assert "drift_plots_othello.png" not in readme
    assert not (ROOT / "figures" / "drift_plots_othello.png").exists()


def test_plotting_notebooks_are_valid_output_free_python():
    expected = {
        "01_learning_and_compute.ipynb",
        "02_deployment_and_othello.ipynb",
        "03_ablation_diagnostics.ipynb",
    }
    paths = {path.name: path for path in (ROOT / "figures").glob("*.ipynb")}
    assert expected <= paths.keys()

    for name in expected:
        payload = json.loads(paths[name].read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        for index, cell in enumerate(payload["cells"]):
            if cell["cell_type"] != "code":
                continue
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None
            compile("".join(cell["source"]), f"{name}:cell-{index}", "exec")


def test_plotting_loaders_follow_current_artifact_schemas(tmp_path):
    pytest.importorskip("matplotlib")
    from figures.plotting_utils import (
        load_ablation_rows,
        load_diagnostic_records,
        load_drift_records,
        load_othello_examples,
        load_training_records,
    )

    run_dir = tmp_path / "control" / "seed_1337"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "args": {
                    "task": "shortest_path",
                    "architecture": "memory_tape",
                    "preset": "shortest_path_main",
                    "seed": 1337,
                    "device": "cpu",
                    "n_pass": 4,
                },
                "model_stats": {
                    "total_parameters": 1200,
                    "non_embedding_parameters": 1000,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.jsonl").write_text(
        json.dumps(
            {
                "event": "eval",
                "step": 10,
                "train_loss": 1.5,
                "pass_losses": [1.8, 1.6, 1.5, 1.4],
                "train_tok_per_s": 100.0,
                "metrics": {"loss": 1.4, "optimal_path": 0.5},
                "gradient_norms": {"memory_writer": {"mean": 0.2, "max": 0.3}},
                "memory_gate_stats": {"mean_abs_effective": 0.4, "max_abs_effective": 0.5},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    drift_dir = run_dir / "drift" / "append_recurrent"
    drift_dir.mkdir(parents=True)
    (drift_dir / "summary.json").write_text(
        json.dumps(
            {
                "input_run_dir": str(run_dir),
                "task": "shortest_path",
                "architecture": "memory_tape",
                "inference_mode": "append_recurrent",
                "effective_inference_mode": "append_recurrent",
                "evaluation_examples": 16,
                "metrics": {
                    "optimal_path": 0.5,
                    "eval_output_tok_per_s": 25.0,
                },
            }
        ),
        encoding="utf-8",
    )
    (drift_dir / "per_position.jsonl").write_text(
        json.dumps({"position": 0, "count": 16, "token_legality": 0.75}) + "\n",
        encoding="utf-8",
    )

    diagnostics = {
        "input_run_dir": str(run_dir),
        "task": "shortest_path",
        "architecture": "memory_tape",
        "memory_interventions": {
            "loss_deltas": {"correct": 0.0, "masked_memory_source": 0.2}
        },
        "pass_dynamics": {
            "trained_passes": [{"pass": 1, "loss": 1.8}],
            "extra_passes": [],
        },
        "teacher_forced_schedule_gap": {
            "positions": [
                {
                    "generated_position": 1,
                    "count": 16,
                    "nll_delta": 0.1,
                    "memory_rms_delta": 0.2,
                }
            ],
            "overall": {"count": 16, "nll_delta": 0.1},
        },
    }
    (run_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics),
        encoding="utf-8",
    )

    othello_dir = run_dir / "othello_eval"
    othello_dir.mkdir()
    (othello_dir / "summary.json").write_text(
        json.dumps({"task": "othello", "input_run_dir": str(run_dir)}),
        encoding="utf-8",
    )
    (othello_dir / "per_example.jsonl").write_text(
        json.dumps(
            {
                "example_index": 0,
                "protocol": "random-prefix",
                "inference_mode": "append_recurrent",
                "prompt_moves": 12,
                "prompt_bucket": "1-15",
                "reference_suffix_moves": 40,
                "suffix_bucket": "31-45",
                "free_generation": {"legal_move_fraction": 0.8},
                "teacher_forced": {
                    "legal_probability_mass": 0.9,
                    "move_count": 40.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with (tmp_path / "per_seed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variant", "seed", "drift.append_recurrent.optimal_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "variant": "control",
                "seed": "1337",
                "drift.append_recurrent.optimal_path": "0.5",
            }
        )

    training = load_training_records(tmp_path)
    assert training[0]["pass_4_loss"] == 1.4
    assert training[0]["gradient_memory_writer_mean"] == 0.2
    assert training[0]["memory_gate_mean_abs_effective"] == 0.4

    drift = load_drift_records(tmp_path)
    assert drift[0]["optimal_path"] == 0.5
    assert drift[0]["per_position"][0]["token_legality"] == 0.75

    loaded_diagnostics = load_diagnostic_records(tmp_path)
    assert loaded_diagnostics[0][
        "memory_interventions.loss_deltas.masked_memory_source"
    ] == 0.2
    assert loaded_diagnostics[0]["payload"]["pass_dynamics"]["trained_passes"][0]["pass"] == 1

    othello = load_othello_examples(tmp_path)
    assert othello[0]["free_generation.legal_move_fraction"] == 0.8
    assert othello[0]["teacher_forced.legal_probability_mass"] == 0.9

    ablation = load_ablation_rows(tmp_path)
    assert ablation[0]["drift.append_recurrent.optimal_path"] == 0.5
