"""PyTorch Lightning data module for prompt-risk classification."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from shapiq_attribution.data import (
    PromptRiskDataset,
    PromptRiskExample,
    PromptRiskTextDataset,
    load_prompt_risk_jsonl,
)
from shapiq_attribution.model import load_prompt_risk_tokenizer


class PromptRiskBatchCollator:
    """Tokenize and dynamically pad a batch of prompt-risk examples."""

    def __init__(self, tokenizer: Any, max_length: int) -> None:
        """Initialize the batch collator.

        Args:
            tokenizer: Hugging Face tokenizer.
            max_length: Maximum tokenized sequence length.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: list[PromptRiskExample]) -> dict[str, torch.Tensor]:
        """Tokenize examples and pad only to the longest sequence in the batch."""
        batch = self.tokenizer(
            [example["prompt"] for example in examples],
            truncation=True,
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor([example["label"] for example in examples], dtype=torch.long)
        return dict(batch)


class LengthGroupedSampler(Sampler[int]):
    """Shuffle examples in large groups and batch examples of similar lengths."""

    def __init__(self, lengths: list[int], batch_size: int, seed: int) -> None:
        """Initialize the sampler.

        Args:
            lengths: Approximate sequence length for every dataset example.
            batch_size: Number of examples per training batch.
            seed: Base seed used for deterministic epoch-level shuffling.
        """
        self.lengths = lengths
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        """Return the number of sampled examples."""
        return len(self.lengths)

    def __iter__(self) -> Iterator[int]:
        """Return shuffled indices grouped into approximately equal-length batches."""
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.lengths), generator=generator).tolist()

        multiplier = min(max(len(indices) // (self.batch_size * 4), 1), 50)
        group_size = multiplier * self.batch_size
        groups = [indices[start : start + group_size] for start in range(0, len(indices), group_size)]
        grouped_indices = [
            index for group in groups for index in sorted(group, key=self.lengths.__getitem__, reverse=True)
        ]
        return iter(grouped_indices)

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic shuffle for an epoch."""
        self.epoch = epoch


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
        dynamic_padding: bool = False,
        length_grouped_batching: bool = False,
        seed: int = 0,
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
            dynamic_padding: Whether to tokenize batches and pad to their longest sequence.
            length_grouped_batching: Whether training batches group examples of similar lengths.
            seed: Base seed for deterministic length-grouped sampling.
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
        self.dynamic_padding = dynamic_padding
        self.length_grouped_batching = length_grouped_batching
        self.seed = seed
        self.tokenizer = tokenizer

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self._train_lengths: list[int] | None = None

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

    def _load_dataset(self, path: Path) -> Dataset:
        """Load and tokenize one prompt-risk dataset.

        Args:
            path: JSONL dataset path.

        Returns:
            Tokenized prompt-risk dataset.
        """
        if self.dynamic_padding:
            return PromptRiskDataset(path)

        examples = load_prompt_risk_jsonl(path)
        return PromptRiskTextDataset(
            examples=examples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

    def _make_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool,
    ) -> DataLoader:
        """Create a configured dataloader.

        Args:
            dataset: Dataset to load.
            shuffle: Whether to shuffle examples.

        Returns:
            Configured dataloader.
        """
        collate_fn = None
        if self.dynamic_padding:
            if self.tokenizer is None:
                raise RuntimeError("setup() must initialize the tokenizer before creating a dataloader")
            collate_fn = PromptRiskBatchCollator(self.tokenizer, self.max_length)

        sampler = None
        if shuffle and self.length_grouped_batching:
            if not isinstance(dataset, PromptRiskDataset):
                raise RuntimeError("length-grouped batching requires the dynamic prompt-risk dataset")
            if self.tokenizer is None:
                raise RuntimeError("setup() must initialize the tokenizer before length grouping")
            if self._train_lengths is None:
                tokenized = self.tokenizer(
                    [example["prompt"] for example in dataset.examples],
                    truncation=True,
                    padding=False,
                    max_length=self.max_length,
                )
                self._train_lengths = [len(input_ids) for input_ids in tokenized["input_ids"]]
            sampler = LengthGroupedSampler(self._train_lengths, self.batch_size, self.seed)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn,
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
