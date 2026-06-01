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
