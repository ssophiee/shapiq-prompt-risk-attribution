"""Tests for W&B model registry validation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from shapiq_attribution.data import write_prompt_risk_jsonl
from shapiq_attribution.model_registry import (
    ValidationThresholds,
    registry_target_from_artifact,
    select_evaluation_examples,
    validate_local_model,
    write_validation_report,
)


class StubPredictor:
    """Deterministic predictor for registry validation tests."""

    def predict_proba_batch(self, prompts: list[str]) -> list[float]:
        """Return high risk for prompts containing the word risky."""
        return [0.9 if "risky" in prompt else 0.1 for prompt in prompts]


def _write_model_files(model_dir: Path) -> None:
    """Create the minimum files checked before loading a model."""
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")


def _write_test_data(path: Path) -> None:
    """Create balanced registry validation data."""
    write_prompt_risk_jsonl(
        [
            {"prompt": "safe question one", "label": 0, "source": "fixture"},
            {"prompt": "risky request one", "label": 1, "source": "fixture"},
            {"prompt": "safe question two", "label": 0, "source": "fixture"},
            {"prompt": "risky request two", "label": 1, "source": "fixture"},
        ],
        path,
    )


@pytest.mark.parametrize(
    ("artifact_path", "expected"),
    [
        (
            "wandb-registry-Models/prompt-risk-distilbert:staging",
            "wandb-registry-Models/prompt-risk-distilbert",
        ),
        (
            "team/model-registry/prompt-risk-distilbert:v3",
            "team/model-registry/prompt-risk-distilbert",
        ),
    ],
)
def test_registry_target_supports_current_and_course_paths(artifact_path: str, expected: str) -> None:
    """Test Registry target extraction for both W&B path conventions."""
    assert registry_target_from_artifact(artifact_path) == expected


def test_registry_target_requires_alias_or_version() -> None:
    """Test that an ambiguous artifact path is rejected."""
    with pytest.raises(ValueError, match="version or alias"):
        registry_target_from_artifact("wandb-registry-Models/prompt-risk-distilbert")


def test_select_evaluation_examples_is_balanced_and_deterministic(tmp_path: Path) -> None:
    """Test deterministic selection with both labels."""
    data_path = tmp_path / "test.jsonl"
    _write_test_data(data_path)

    from shapiq_attribution.data import load_prompt_risk_jsonl

    examples = load_prompt_risk_jsonl(data_path)
    selected = select_evaluation_examples(examples, max_samples=4)

    assert len(selected) == 4
    assert [example["label"] for example in selected] == [0, 0, 1, 1]
    assert selected == select_evaluation_examples(examples, max_samples=4)


def test_validate_local_model_passes_meaningful_thresholds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test successful staged-model validation."""
    model_dir = tmp_path / "model"
    test_data = tmp_path / "test.jsonl"
    _write_model_files(model_dir)
    _write_test_data(test_data)
    monkeypatch.setattr(
        "shapiq_attribution.model_registry.PromptRiskPredictor.from_pretrained",
        lambda *args, **kwargs: StubPredictor(),
    )

    result = validate_local_model(
        model_dir=model_dir,
        test_data=test_data,
        artifact_path="wandb-registry-Models/prompt-risk-distilbert:v1",
        thresholds=ValidationThresholds(),
        max_samples=4,
        batch_size=2,
    )

    assert result.passed
    assert result.sample_count == 4
    assert result.metrics["f1"] == 1.0
    assert result.metrics["roc_auc"] == 1.0


def test_validate_local_model_rejects_missing_weights(tmp_path: Path) -> None:
    """Test that incomplete artifacts fail before model loading."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}\n", encoding="utf-8")

    result = validate_local_model(
        model_dir=model_dir,
        test_data=tmp_path / "unused.jsonl",
        artifact_path="team/model-registry/prompt-risk-distilbert:v1",
        thresholds=ValidationThresholds(),
    )

    assert not result.passed
    assert result.checks == {"required_model_files": False}


def test_validation_report_records_failure(tmp_path: Path) -> None:
    """Test that reports expose failed model gates."""
    from shapiq_attribution.model_registry import ValidationResult

    result = ValidationResult(
        artifact_path="team/model-registry/prompt-risk-distilbert:v2",
        sample_count=4,
        metrics={"f1": 0.5, "roc_auc": 0.6},
        checks={"f1_threshold": False, "roc_auc_threshold": False},
        errors=["f1_threshold", "roc_auc_threshold"],
    )
    markdown_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"

    write_validation_report(result, markdown_path, json_path, ValidationThresholds())

    assert "Status: **FAIL**" in markdown_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["thresholds"]["min_roc_auc"] == 0.85
