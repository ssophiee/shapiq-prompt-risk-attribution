"""Tests for prompt-risk model utilities."""

from types import SimpleNamespace

import torch
from shapiq_attribution.model import PromptRiskPredictor


class DummyTokenizer:
    """Tokenizer stub for predictor tests."""

    def __call__(
        self,
        prompt: str,
        truncation: bool,
        padding: str,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        """Return deterministic fake token IDs."""
        del prompt, truncation, padding, return_tensors
        return {
            "input_ids": torch.ones((1, max_length), dtype=torch.long),
            "attention_mask": torch.ones((1, max_length), dtype=torch.long),
        }


class DummyModel(torch.nn.Module):
    """Model stub for predictor tests."""

    def __init__(self) -> None:
        """Initialize the model stub."""
        super().__init__()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        """Return fixed logits."""
        del input_ids, attention_mask
        return SimpleNamespace(logits=torch.tensor([[0.0, 2.0]]))


def test_prompt_risk_predictor_returns_risky_probability() -> None:
    """Test that the predictor returns the risky-class probability."""
    predictor = PromptRiskPredictor(
        model=DummyModel(),
        tokenizer=DummyTokenizer(),
        max_length=8,
        device=torch.device("cpu"),
    )

    probability = predictor.predict_proba("any prompt")

    assert probability > 0.8
    assert probability < 0.9


def test_prompt_risk_predictor_is_callable() -> None:
    """Test that the predictor can be called directly."""
    predictor = PromptRiskPredictor(
        model=DummyModel(),
        tokenizer=DummyTokenizer(),
        max_length=8,
        device=torch.device("cpu"),
    )

    assert predictor("masked prompt") == predictor.predict_proba("masked prompt")
