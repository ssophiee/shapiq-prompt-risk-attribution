"""Tests for the attribution value-function core."""

from types import SimpleNamespace

import numpy as np
import torch
from shapiq_attribution.attribution import build_prompt, compute_answer_logprob, make_value_function

VOCAB_SIZE = 4


class DummyChatTokenizer:
    """Tokenizer stub covering the two interfaces attribution.py uses."""

    def apply_chat_template(self, messages: list[dict], tokenize: bool, add_generation_prompt: bool) -> str:
        """Concatenate message contents into one prompt string."""
        del tokenize, add_generation_prompt
        return "\n".join(message["content"] for message in messages)

    def __call__(self, text: str, return_tensors: str) -> SimpleNamespace:
        """Encode one token per whitespace-separated word."""
        del return_tensors
        n_tokens = max(len(text.split()), 1)
        return SimpleNamespace(input_ids=torch.zeros((1, n_tokens), dtype=torch.long))


class DummyCausalLM(torch.nn.Module):
    """Causal LM stub with uniform next-token logits."""

    def __init__(self) -> None:
        """Initialize the stub."""
        super().__init__()
        self.device = torch.device("cpu")

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        """Return uniform logits over the vocabulary."""
        return SimpleNamespace(logits=torch.zeros((1, input_ids.shape[1], VOCAB_SIZE)))


def test_build_prompt_includes_present_steps() -> None:
    """Test that only the coalition's steps end up in the prompt."""
    prompt = build_prompt(
        question="is this risky?",
        few_shot="Q: example",
        system="you are careful",
        present_steps=["Step 1: think", "Step 3: conclude"],
        tokenizer=DummyChatTokenizer(),
    )

    assert "Step 1: think" in prompt
    assert "Step 3: conclude" in prompt
    assert "is this risky?" in prompt


def test_build_prompt_empty_coalition_uses_placeholder() -> None:
    """Test that an empty coalition yields the no-reasoning placeholder."""
    prompt = build_prompt(
        question="q",
        few_shot="",
        system="s",
        present_steps=[],
        tokenizer=DummyChatTokenizer(),
    )

    assert "(no reasoning provided)" in prompt


def test_compute_answer_logprob_uniform_model() -> None:
    """Test that uniform logits give a mean log-probability of -log(vocab)."""
    logprob = compute_answer_logprob(
        prompt="one two three",
        target=" four five",
        model=DummyCausalLM(),
        tokenizer=DummyChatTokenizer(),
    )

    assert np.isclose(logprob, -np.log(VOCAB_SIZE))


def test_make_value_function_scores_each_coalition() -> None:
    """Test that the value function returns one score per coalition row."""
    value_fn = make_value_function(
        question="q",
        few_shot="fs",
        system="s",
        cot_steps=["Step 1: a", "Step 2: b"],
        target_answer=" risky",
        model=DummyCausalLM(),
        tokenizer=DummyChatTokenizer(),
    )

    coalitions = np.array([[False, False], [True, False], [True, True]])
    scores = value_fn(coalitions)

    assert scores.shape == (3,)
    assert np.allclose(scores, -np.log(VOCAB_SIZE))
