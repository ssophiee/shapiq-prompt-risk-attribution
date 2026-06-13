"""Tests for the drift-monitoring feature pipeline.

Feature extraction and CSV logging are pure and fast; ``build_baseline`` is
exercised with a deterministic predictor stub. The Evidently report is covered
by a single integration test that asserts an HTML dashboard is produced.
"""

from __future__ import annotations

import csv
from pathlib import Path

from shapiq_attribution import monitoring


class DummyPredictor:
    """Predictor stub returning a fixed probability for any prompt."""

    def __init__(self, probability: float) -> None:
        """Store the probability the stub should always return."""
        self.probability = probability

    def predict_proba(self, prompt: str) -> float:
        """Return the configured probability regardless of input."""
        del prompt
        return self.probability


def test_extract_features_computes_expected_values() -> None:
    """Features are character length, word count, and the risk probability."""
    features = monitoring.extract_features("two words", 0.42)

    assert features == {"prompt_len": 9.0, "token_count": 2.0, "p_risky": 0.42}
    assert set(features) == set(monitoring.FEATURE_COLUMNS)


def test_log_prediction_writes_header_then_appends(tmp_path: Path) -> None:
    """The first write adds a header; later writes append rows only."""
    path = tmp_path / "nested" / "predictions.csv"

    monitoring.log_prediction(monitoring.extract_features("hello there", 0.7), path)
    monitoring.log_prediction(monitoring.extract_features("bye", 0.1), path)

    with path.open() as handle:
        rows = list(csv.DictReader(handle))

    assert [r["p_risky"] for r in rows] == ["0.7", "0.1"]
    assert list(rows[0]) == list(monitoring.FEATURE_COLUMNS)


def test_build_baseline_rows_match_dataset(tmp_path: Path) -> None:
    """One baseline row is written per dataset example, with model probabilities."""
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        '{"prompt": "a harmful request", "label": 1, "source": "advbench"}\n'
        '{"prompt": "hi", "label": 0, "source": "wildguard"}\n',
        encoding="utf-8",
    )
    out_path = tmp_path / "baseline.csv"

    monitoring.build_baseline(DummyPredictor(0.6), dataset, out_path)

    with out_path.open() as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0] == {"prompt_len": "17.0", "token_count": "3.0", "p_risky": "0.6"}
    assert all(row["p_risky"] == "0.6" for row in rows)


def test_generate_report_writes_html(tmp_path: Path) -> None:
    """The Evidently report renders an HTML dashboard from baseline + live CSVs."""
    reference = tmp_path / "baseline.csv"
    current = tmp_path / "predictions.csv"
    for prompt, prob in [("short prompt", 0.3), ("another prompt here", 0.5), ("third one", 0.4)]:
        monitoring.log_prediction(monitoring.extract_features(prompt, prob), reference)
    for prompt, prob in [("live request now", 0.9), ("another live one entirely", 0.95)]:
        monitoring.log_prediction(monitoring.extract_features(prompt, prob), current)

    out_path = tmp_path / "report.html"
    result = monitoring.generate_report(reference, current, out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
