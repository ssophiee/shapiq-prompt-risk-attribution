"""Evaluation utilities for prompt-risk classification."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader


def compute_metrics(labels: list[int], predictions: list[int], probabilities: list[float]) -> dict[str, float]:
    """Compute binary classification metrics.

    Args:
        labels: True binary labels.
        predictions: Predicted binary labels.
        probabilities: Predicted probability for the risky class.

    Returns:
        Binary classification metrics.
    """
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }
    if len(set(labels)) > 1:
        metrics["roc_auc"] = roc_auc_score(labels, probabilities)
    return {key: float(value) for key, value in metrics.items()}


def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
    """Evaluate a prompt-risk classifier.

    Args:
        model: Sequence classification model.
        dataloader: Evaluation dataloader.
        device: Evaluation device.

    Returns:
        Loss and classification metrics.
    """
    model.eval()
    losses = []
    labels = []
    predictions = []
    probabilities = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[:, 1]

            losses.append(float(outputs.loss.detach().cpu()))
            labels.extend(batch["labels"].detach().cpu().tolist())
            predictions.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
            probabilities.extend(probs.detach().cpu().tolist())

    metrics = compute_metrics(labels=labels, predictions=predictions, probabilities=probabilities)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics
