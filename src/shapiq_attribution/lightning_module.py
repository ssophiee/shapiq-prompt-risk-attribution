"""PyTorch Lightning module for prompt-risk classification."""

from __future__ import annotations

from typing import Any, Literal

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)
from transformers import get_linear_schedule_with_warmup

from shapiq_attribution.model import load_prompt_risk_model


class PromptRiskLightningModule(L.LightningModule):
    """Train a Hugging Face prompt-risk classifier."""

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        learning_rate: float,
        weight_decay: float,
        model: nn.Module | None = None,
    ) -> None:
        """Initialize the Lightning training module.

        Args:
            model_name: Hugging Face model name or local model path.
            num_labels: Number of classification labels.
            learning_rate: AdamW learning rate.
            weight_decay: AdamW weight decay.
            model: Optional model override, primarily for testing.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = model or load_prompt_risk_model(
            model_name,
            num_labels=num_labels,
        )
        metrics = MetricCollection(
            {
                "accuracy": BinaryAccuracy(),
                "precision": BinaryPrecision(),
                "recall": BinaryRecall(),
                "f1": BinaryF1Score(),
                "roc_auc": BinaryAUROC(),
            }
        )
        self.validation_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, **batch: torch.Tensor) -> Any:
        """Run a model forward pass.

        Args:
            **batch: Tokenized model inputs.

        Returns:
            Hugging Face model output.
        """
        return self.model(**batch)

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Run one training step.

        Args:
            batch: Tokenized model inputs and labels.
            batch_idx: Batch index.

        Returns:
            Training loss.
        """
        del batch_idx
        outputs = self(**batch)

        self.log(
            "train/loss",
            outputs.loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["labels"].size(0),
        )
        return outputs.loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Run one validation step.

        Args:
            batch: Tokenized model inputs and labels.
            batch_idx: Batch index.
        """
        del batch_idx
        self._evaluation_step(batch, stage="val")

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Run one test step.

        Args:
            batch: Tokenized model inputs and labels.
            batch_idx: Batch index.
        """
        del batch_idx
        self._evaluation_step(batch, stage="test")

    def _evaluation_step(
        self,
        batch: dict[str, torch.Tensor],
        stage: Literal["val", "test"],
    ) -> None:
        """Update evaluation loss and metric states.

        Args:
            batch: Tokenized model inputs and labels.
            stage: Evaluation stage.
        """
        outputs = self(**batch)
        probabilities = torch.softmax(outputs.logits, dim=-1)[:, 1]
        batch_size = batch["labels"].size(0)

        if stage == "val":
            self.validation_metrics.update(probabilities, batch["labels"])
        else:
            self.test_metrics.update(probabilities, batch["labels"])

        self.log(
            f"{stage}/loss",
            outputs.loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )

    def on_validation_epoch_end(self) -> None:
        """Compute and log globally synchronized validation metrics."""
        metrics = self.validation_metrics.compute()
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        self.validation_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Compute and log globally synchronized test metrics."""
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, sync_dist=True)
        self.test_metrics.reset()

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Create the optimizer and step-wise learning-rate scheduler.

        Returns:
            Lightning optimizer and scheduler configuration.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
