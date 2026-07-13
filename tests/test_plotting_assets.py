from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_plotting_notebooks_are_valid_and_use_current_inference_modes():
    for name in ("plot_trace.ipynb", "plot_drift.ipynb"):
        path = ROOT / "figures" / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        assert payload["cells"]
        text = path.read_text(encoding="utf-8")
        assert "cache_source" not in text
        assert "penultimate" not in text
        assert "final_pass" not in text

    drift_text = (ROOT / "figures" / "plot_drift.ipynb").read_text(encoding="utf-8")
    assert "append_recurrent" in drift_text
    assert "recompute" in drift_text


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
