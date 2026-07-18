"""Tests for evaluation metrics."""

from types import SimpleNamespace

import torch
from shapiq_attribution.evaluate import compute_metrics, evaluate_model
from torch.utils.data import DataLoader


def test_compute_metrics_perfect_predictions() -> None:
    """Test that perfect predictions score 1.0 on every metric."""
    labels = [0, 1, 0, 1]
    predictions = [0, 1, 0, 1]
    probabilities = [0.1, 0.9, 0.2, 0.8]

    metrics = compute_metrics(labels=labels, predictions=predictions, probabilities=probabilities)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_compute_metrics_all_wrong_predictions() -> None:
    """Test that inverted predictions score 0.0 on accuracy and f1."""
    labels = [0, 1, 0, 1]
    predictions = [1, 0, 1, 0]
    probabilities = [0.9, 0.1, 0.8, 0.2]

    metrics = compute_metrics(labels=labels, predictions=predictions, probabilities=probabilities)

    assert metrics["accuracy"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["roc_auc"] == 0.0


def test_compute_metrics_single_class_skips_roc_auc() -> None:
    """Test that roc_auc is omitted when only one class is present."""
    labels = [1, 1, 1]
    predictions = [1, 1, 0]
    probabilities = [0.9, 0.8, 0.4]

    metrics = compute_metrics(labels=labels, predictions=predictions, probabilities=probabilities)

    assert "roc_auc" not in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_compute_metrics_zero_division_is_safe() -> None:
    """Test that precision/recall fall back to 0.0 instead of raising."""
    labels = [0, 0, 1]
    predictions = [0, 0, 0]
    probabilities = [0.1, 0.2, 0.3]

    metrics = compute_metrics(labels=labels, predictions=predictions, probabilities=probabilities)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_compute_metrics_values_are_floats() -> None:
    """Test that every metric is a plain Python float."""
    metrics = compute_metrics(labels=[0, 1], predictions=[0, 1], probabilities=[0.2, 0.8])

    assert all(isinstance(value, float) for value in metrics.values())


class DummyClassifier(torch.nn.Module):
    """Sequence-classification stub that always predicts the risky class."""

    def __init__(self) -> None:
        """Initialize the stub."""
        super().__init__()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> SimpleNamespace:
        """Return a fixed loss and risky-leaning logits per example."""
        del input_ids, attention_mask
        batch_size = labels.shape[0]
        return SimpleNamespace(
            loss=torch.tensor(0.7),
            logits=torch.tensor([[0.0, 2.0]]).repeat(batch_size, 1),
        )


def test_evaluate_model_aggregates_metrics_and_loss() -> None:
    """Test that evaluate_model reports loss and metrics over the dataloader."""
    dataset = [
        {
            "input_ids": torch.ones(4, dtype=torch.long),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "labels": torch.tensor(label),
        }
        for label in [1, 0, 1, 1]
    ]
    dataloader = DataLoader(dataset, batch_size=2)

    metrics = evaluate_model(model=DummyClassifier(), dataloader=dataloader, device=torch.device("cpu"))

    # The stub predicts risky for everything, so accuracy is the risky fraction.
    assert metrics["accuracy"] == 0.75
    assert metrics["recall"] == 1.0
    assert abs(metrics["loss"] - 0.7) < 1e-6
