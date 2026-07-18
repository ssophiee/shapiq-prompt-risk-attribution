"""PyTorch Lightning data module for prompt-risk classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
from torch.utils.data import DataLoader

from shapiq_attribution.data import (
    PromptRiskTextDataset,
    load_prompt_risk_jsonl,
)
from shapiq_attribution.model import load_prompt_risk_tokenizer


class PromptRiskDataModule(L.LightningDataModule):
    """Provide prompt-risk dataloaders for Lightning training."""

    def __init__(
        self,
        train_path: str | Path,
        val_path: str | Path,
        test_path: str | Path,
        model_name: str,
        max_length: int,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        tokenizer: Any | None = None,
    ) -> None:
        """Initialize the prompt-risk data module.

        Args:
            train_path: Path to the training JSONL file.
            val_path: Path to the validation JSONL file.
            test_path: Path to the test JSONL file.
            model_name: Hugging Face tokenizer name or path.
            max_length: Maximum tokenized sequence length.
            batch_size: Batch size per device.
            num_workers: Number of workers per dataloader.
            pin_memory: Whether to use pinned memory.
            persistent_workers: Whether workers persist between epochs.
            tokenizer: Optional tokenizer override for tests.
        """
        super().__init__()
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.test_path = Path(test_path)
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.tokenizer = tokenizer

        self.train_dataset: PromptRiskTextDataset | None = None
        self.val_dataset: PromptRiskTextDataset | None = None
        self.test_dataset: PromptRiskTextDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Load datasets required for the current Lightning stage.

        Args:
            stage: Current Lightning stage.
        """
        if self.tokenizer is None:
            self.tokenizer = load_prompt_risk_tokenizer(self.model_name)

        if stage in {None, "fit"}:
            if self.train_dataset is None:
                self.train_dataset = self._load_dataset(self.train_path)
            if self.val_dataset is None:
                self.val_dataset = self._load_dataset(self.val_path)

        if stage == "validate" and self.val_dataset is None:
            self.val_dataset = self._load_dataset(self.val_path)

        if stage in {None, "test"} and self.test_dataset is None:
            self.test_dataset = self._load_dataset(self.test_path)

    def _load_dataset(self, path: Path) -> PromptRiskTextDataset:
        """Load and tokenize one prompt-risk dataset.

        Args:
            path: JSONL dataset path.

        Returns:
            Tokenized prompt-risk dataset.
        """
        examples = load_prompt_risk_jsonl(path)
        return PromptRiskTextDataset(
            examples=examples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

    def _make_dataloader(
        self,
        dataset: PromptRiskTextDataset,
        shuffle: bool,
    ) -> DataLoader:
        """Create a configured dataloader.

        Args:
            dataset: Dataset to load.
            shuffle: Whether to shuffle examples.

        Returns:
            Configured dataloader.
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("setup('fit') must run before train_dataloader()")
        return self._make_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("setup('fit') or setup('validate') must run before val_dataloader()")
        return self._make_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError("setup('test') must run before test_dataloader()")
        return self._make_dataloader(self.test_dataset, shuffle=False)
