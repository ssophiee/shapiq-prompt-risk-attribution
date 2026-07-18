"""Tests for prompt-risk model utilities."""

from types import SimpleNamespace

import pytest
import torch
from shapiq_attribution.model import PromptRiskPredictor, save_prompt_risk_model


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


class BatchDummyTokenizer:
    """Tokenizer stub that also handles batched inputs."""

    def __call__(self, prompt, truncation, padding, max_length, return_tensors):
        """Return fake token IDs for one prompt or a batch of prompts."""
        del truncation, padding, max_length, return_tensors
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        return {
            "input_ids": torch.ones((batch_size, 4), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 4), dtype=torch.long),
        }


class BatchDummyModel(torch.nn.Module):
    """Model stub returning the same risky-leaning logits per example."""

    def __init__(self) -> None:
        """Initialize the model stub."""
        super().__init__()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        """Return fixed logits for every example in the batch."""
        del attention_mask
        return SimpleNamespace(logits=torch.tensor([[0.0, 2.0]]).repeat(input_ids.shape[0], 1))


def test_predict_proba_batch_matches_single_predictions() -> None:
    """Test that the batched path agrees with per-prompt predictions."""
    predictor = PromptRiskPredictor(
        model=BatchDummyModel(),
        tokenizer=BatchDummyTokenizer(),
        max_length=8,
        device=torch.device("cpu"),
    )

    batch = predictor.predict_proba_batch(["prompt one", "prompt two"])

    assert len(batch) == 2
    assert all(abs(probability - predictor.predict_proba("prompt one")) < 1e-6 for probability in batch)


def test_predict_proba_batch_returns_empty_for_no_prompts() -> None:
    """Test that an empty batch short-circuits without tokenizing."""
    predictor = PromptRiskPredictor(
        model=BatchDummyModel(),
        tokenizer=BatchDummyTokenizer(),
        max_length=8,
        device=torch.device("cpu"),
    )

    assert predictor.predict_proba_batch([]) == []


def test_save_prompt_risk_model_writes_model_and_tokenizer(tmp_path) -> None:
    """Test that saving delegates to both save_pretrained implementations."""

    class SavableModel(BatchDummyModel):
        def save_pretrained(self, output_dir) -> None:
            (output_dir / "model.safetensors").write_text("weights")

    class SavableTokenizer(BatchDummyTokenizer):
        def save_pretrained(self, output_dir) -> None:
            (output_dir / "tokenizer.json").write_text("vocab")

    output_dir = tmp_path / "checkpoint"
    save_prompt_risk_model(SavableModel(), SavableTokenizer(), output_dir)

    assert (output_dir / "model.safetensors").exists()
    assert (output_dir / "tokenizer.json").exists()


def test_from_pretrained_builds_predictor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that from_pretrained wires the loaded model and tokenizer."""
    monkeypatch.setattr(
        "shapiq_attribution.model.AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda path: DummyTokenizer()),
    )
    monkeypatch.setattr(
        "shapiq_attribution.model.AutoModelForSequenceClassification",
        SimpleNamespace(from_pretrained=lambda path, num_labels: DummyModel()),
    )

    predictor = PromptRiskPredictor.from_pretrained("models/fake", max_length=8, device="cpu")

    assert 0.8 < predictor.predict_proba("any prompt") < 0.9
