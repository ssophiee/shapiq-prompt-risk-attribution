"""Smoke tests for attribution plotting."""

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
from shapiq_attribution.visualize import plot_results

matplotlib.use("Agg")


def test_plot_results_saves_figure(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that plot_results writes a PNG named after the title."""
    monkeypatch.chdir(tmp_path)
    steps = ["Step 1: a", "Step 2: b", "Step 3: c"]
    sv = SimpleNamespace(values=np.array([0.3, 0.1, -0.2, 0.05]))
    ksii = SimpleNamespace(
        interactions=[(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)],
        values=np.array([0.1, -0.2, 0.05, 0.02, -0.01, 0.0]),
    )

    plot_results(steps, sv, ksii, title="Unit Test Plot")

    assert (tmp_path / "Unit_Test_Plot.png").exists()
