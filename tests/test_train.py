"""Tests for prompt-risk training helpers."""

import torch
from shapiq_attribution.data import PromptRiskTextDataset, make_prompt_risk_example
from shapiq_attribution.evaluate import compute_metrics


class DummyTokenizer:
    """Small tokenizer stub for dataset tests."""

    def __call__(
        self,
        text: str,
        truncation: bool,
        padding: str,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        """Return deterministic fake token IDs."""
        del text, truncation, padding, return_tensors
        return {
            "input_ids": torch.ones((1, max_length), dtype=torch.long),
            "attention_mask": torch.ones((1, max_length), dtype=torch.long),
        }


def test_prompt_risk_text_dataset_returns_labels() -> None:
    """Test that text examples are tokenized with labels."""
    examples = [make_prompt_risk_example(prompt="risky prompt", label=1, source="fixture")]
    dataset = PromptRiskTextDataset(examples=examples, tokenizer=DummyTokenizer(), max_length=8)

    item = dataset[0]

    assert item["input_ids"].shape == torch.Size([8])
    assert item["attention_mask"].shape == torch.Size([8])
    assert item["labels"].item() == 1


def test_compute_metrics_returns_binary_scores() -> None:
    """Test binary classification metric computation."""
    metrics = compute_metrics(labels=[0, 1, 1, 0], predictions=[0, 1, 0, 0], probabilities=[0.1, 0.9, 0.4, 0.2])

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["f1"] > 0.0
    assert metrics["roc_auc"] == 1.0
