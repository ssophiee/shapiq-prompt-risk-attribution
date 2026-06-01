"""Hydra entrypoint for deterministic prompt-risk train/validation/test splits."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from shapiq_attribution.data import PromptRiskExample, load_prompt_risk_jsonl, write_prompt_risk_jsonl


def validate_split_sizes(train_size: float, val_size: float, test_size: float) -> None:
    """Validate split fractions.

    Args:
        train_size: Fraction of examples assigned to the training split.
        val_size: Fraction of examples assigned to the validation split.
        test_size: Fraction of examples assigned to the test split.

    Raises:
        ValueError: If split fractions are invalid.
    """
    total = train_size + val_size + test_size
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("split sizes must be positive")
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"split sizes must sum to 1.0, got {total}")


def split_examples(
    examples: list[PromptRiskExample],
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
    stratify: bool,
) -> tuple[list[PromptRiskExample], list[PromptRiskExample], list[PromptRiskExample]]:
    """Split prompt-risk examples into train, validation, and test sets.

    Args:
        examples: Normalized prompt-risk examples.
        train_size: Fraction of examples assigned to the training split.
        val_size: Fraction of examples assigned to the validation split.
        test_size: Fraction of examples assigned to the test split.
        seed: Random seed used for deterministic shuffling.
        stratify: Whether to split separately per label.

    Returns:
        Train, validation, and test examples.
    """
    validate_split_sizes(train_size=train_size, val_size=val_size, test_size=test_size)

    rng = random.Random(seed)
    groups: dict[int, list[PromptRiskExample]] = defaultdict(list)
    if stratify:
        for example in examples:
            groups[example["label"]].append(example)
    else:
        groups[0] = list(examples)

    train: list[PromptRiskExample] = []
    val: list[PromptRiskExample] = []
    test: list[PromptRiskExample] = []

    for group in groups.values():
        shuffled = list(group)
        rng.shuffle(shuffled)

        n_examples = len(shuffled)
        n_train = int(n_examples * train_size)
        n_val = int(n_examples * val_size)

        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train : n_train + n_val])
        test.extend(shuffled[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def split_dataset(cfg: DictConfig) -> None:
    """Create train, validation, and test JSONL files from the configured dataset.

    Args:
        cfg: Hydra configuration.
    """
    examples = load_prompt_risk_jsonl(to_absolute_path(cfg.data.input_path))
    train, val, test = split_examples(
        examples=examples,
        train_size=cfg.data.train_size,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
        seed=cfg.data.seed,
        stratify=cfg.data.stratify,
    )

    train_path = Path(to_absolute_path(cfg.data.train_path))
    val_path = Path(to_absolute_path(cfg.data.val_path))
    test_path = Path(to_absolute_path(cfg.data.test_path))

    write_prompt_risk_jsonl(train, train_path)
    write_prompt_risk_jsonl(val, val_path)
    write_prompt_risk_jsonl(test, test_path)

    print(f"train: {len(train)} examples -> {train_path}")
    print(f"val: {len(val)} examples -> {val_path}")
    print(f"test: {len(test)} examples -> {test_path}")


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Run the split-data pipeline step.

    Args:
        cfg: Hydra configuration.
    """
    split_dataset(cfg)


if __name__ == "__main__":
    main()
