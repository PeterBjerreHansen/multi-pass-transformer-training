from __future__ import annotations

from pathlib import Path


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
