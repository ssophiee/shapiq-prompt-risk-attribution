"""Tests for deterministic prompt-risk dataset splitting."""

from shapiq_attribution.data import make_prompt_risk_example
from shapiq_attribution.split_data import split_examples, validate_split_sizes


def test_split_examples_is_deterministic() -> None:
    """Test that the same seed creates the same splits."""
    examples = [make_prompt_risk_example(prompt=f"safe {index}", label=0, source="fixture") for index in range(10)] + [
        make_prompt_risk_example(prompt=f"risky {index}", label=1, source="fixture") for index in range(10)
    ]

    first_split = split_examples(examples, train_size=0.8, val_size=0.1, test_size=0.1, seed=42, stratify=True)
    second_split = split_examples(examples, train_size=0.8, val_size=0.1, test_size=0.1, seed=42, stratify=True)

    assert first_split == second_split


def test_split_examples_preserves_all_examples() -> None:
    """Test that every example appears in exactly one split."""
    examples = [
        make_prompt_risk_example(prompt=f"example {index}", label=index % 2, source="fixture") for index in range(20)
    ]

    train, val, test = split_examples(
        examples,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        seed=12,
        stratify=True,
    )

    split_prompts = {example["prompt"] for example in train + val + test}

    assert len(train) + len(val) + len(test) == len(examples)
    assert split_prompts == {example["prompt"] for example in examples}


def test_split_examples_keeps_both_labels_when_stratified() -> None:
    """Test that stratified splitting keeps both labels in the training split."""
    examples = [make_prompt_risk_example(prompt=f"safe {index}", label=0, source="fixture") for index in range(10)] + [
        make_prompt_risk_example(prompt=f"risky {index}", label=1, source="fixture") for index in range(10)
    ]

    train, val, test = split_examples(
        examples,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        seed=12,
        stratify=True,
    )

    assert {example["label"] for example in train} == {0, 1}
    assert {example["label"] for example in val} == {0, 1}
    assert {example["label"] for example in test} == {0, 1}


def test_validate_split_sizes_rejects_invalid_sum() -> None:
    """Test that split fractions must sum to one."""
    try:
        validate_split_sizes(train_size=0.8, val_size=0.2, test_size=0.2)
    except ValueError as error:
        assert "sum to 1.0" in str(error)
    else:
        raise AssertionError("Expected ValueError")
