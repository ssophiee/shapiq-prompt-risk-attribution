"""Model utilities for prompt-risk classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_prompt_risk_tokenizer(model_name_or_path: str) -> Any:
    """Load a tokenizer for prompt-risk classification.

    Args:
        model_name_or_path: Hugging Face model name or local model directory.

    Returns:
        A Hugging Face tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_name_or_path)


def load_prompt_risk_model(model_name_or_path: str, num_labels: int = 2) -> torch.nn.Module:
    """Load a sequence classification model for prompt-risk classification.

    Args:
        model_name_or_path: Hugging Face model name or local model directory.
        num_labels: Number of output labels.

    Returns:
        A Hugging Face sequence classification model.
    """
    return AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)


def save_prompt_risk_model(model: torch.nn.Module, tokenizer: Any, output_dir: str | Path) -> None:
    """Save a prompt-risk model and tokenizer.

    Args:
        model: Hugging Face sequence classification model.
        tokenizer: Hugging Face tokenizer.
        output_dir: Output directory.
    """
    model_output_dir = Path(output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)


class PromptRiskPredictor:
    """Convenience wrapper for prompt-risk probability prediction."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        max_length: int,
        device: torch.device,
    ) -> None:
        """Initialize the predictor.

        Args:
            model: Prompt-risk classifier.
            tokenizer: Prompt-risk tokenizer.
            max_length: Maximum tokenized sequence length.
            device: Torch device.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        max_length: int = 128,
        device: str | torch.device = "cpu",
    ) -> PromptRiskPredictor:
        """Load a predictor from a Hugging Face model name or local model directory.

        Args:
            model_name_or_path: Hugging Face model name or local model directory.
            max_length: Maximum tokenized sequence length.
            device: Torch device name or device object.

        Returns:
            Prompt-risk predictor.
        """
        torch_device = torch.device(device)
        model_path = str(model_name_or_path)
        tokenizer = load_prompt_risk_tokenizer(model_path)
        model = load_prompt_risk_model(model_path)
        return cls(model=model, tokenizer=tokenizer, max_length=max_length, device=torch_device)

    def tokenize(self, prompt: str) -> dict[str, torch.Tensor]:
        """Tokenize a prompt for the prompt-risk classifier.

        Args:
            prompt: Input prompt text.

        Returns:
            Tokenized model inputs on the predictor device.
        """
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in inputs.items()}

    def predict_proba(self, prompt: str) -> float:
        """Predict the risky-class probability for a prompt.

        Args:
            prompt: Input prompt text.

        Returns:
            Probability assigned to the risky class.
        """
        inputs = self.tokenize(prompt)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)

        return float(probabilities[0, 1].detach().cpu())

    def predict_proba_batch(self, prompts: list[str]) -> list[float]:
        """Predict risky-class probabilities for many prompts in one forward pass.

        Equivalent to calling :meth:`predict_proba` on each prompt — padding is
        ignored via the attention mask, so per-prompt probabilities are
        independent of batch composition. Padding is dynamic (to the longest
        sequence in the batch, capped at ``max_length``) rather than a fixed
        ``max_length``, so short coalition texts are not padded out to 128
        tokens — this is what makes the batched pass cheaper than the loop.

        Args:
            prompts: Input prompt texts.

        Returns:
            Probability assigned to the risky class for each prompt, in order.
        """
        if not prompts:
            return []

        inputs = self.tokenizer(
            prompts,
            truncation=True,
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)

        return probabilities[:, 1].detach().cpu().tolist()

    def __call__(self, prompt: str) -> float:
        """Predict the risky-class probability for a prompt.

        Args:
            prompt: Input prompt text.

        Returns:
            Probability assigned to the risky class.
        """
        return self.predict_proba(prompt)
