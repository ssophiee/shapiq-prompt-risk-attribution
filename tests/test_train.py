"""Tests for prompt-risk training helpers."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import lightning as L
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from shapiq_attribution.data import PromptRiskTextDataset, make_prompt_risk_example
from shapiq_attribution.evaluate import compute_metrics
from shapiq_attribution.lightning_data_module import PromptRiskDataModule
from shapiq_attribution.lightning_module import PromptRiskLightningModule
from shapiq_attribution.train import (
    _create_wandb_logger,
    _load_wandb_api_key_from_secret,
)
from torch import nn
from torch.nn import functional as F


class DummyTokenizer:
    """Small tokenizer stub for dataset tests."""

    def __call__(
        self,
        text: str,
        truncation: bool,
        padding: str,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        """Return deterministic fake token IDs."""
        del text, truncation, padding, return_tensors
        return {
            "input_ids": torch.ones((1, max_length), dtype=torch.long),
            "attention_mask": torch.ones((1, max_length), dtype=torch.long),
        }


class DummySequenceClassifier(nn.Module):
    """Small sequence classifier used for Lightning unit tests."""

    def __init__(self) -> None:
        """Initialize a linear classifier."""
        super().__init__()
        self.classifier = nn.Linear(4, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> SimpleNamespace:
        """Return logits and loss in Hugging Face format."""
        del attention_mask
        logits = self.classifier(input_ids.float())
        loss = F.cross_entropy(logits, labels)
        return SimpleNamespace(loss=loss, logits=logits)


def test_prompt_risk_text_dataset_returns_labels() -> None:
    """Test that text examples are tokenized with labels."""
    examples = [make_prompt_risk_example(prompt="risky prompt", label=1, source="fixture")]
    dataset = PromptRiskTextDataset(examples=examples, tokenizer=DummyTokenizer(), max_length=8)

    item = dataset[0]

    assert item["input_ids"].shape == torch.Size([8])
    assert item["attention_mask"].shape == torch.Size([8])
    assert item["labels"].item() == 1


def test_compute_metrics_returns_binary_scores() -> None:
    """Test binary classification metric computation."""
    metrics = compute_metrics(labels=[0, 1, 1, 0], predictions=[0, 1, 0, 0], probabilities=[0.1, 0.9, 0.4, 0.2])

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["f1"] > 0.0
    assert metrics["roc_auc"] == 1.0


def test_lightning_module_training_step_returns_loss() -> None:
    """Test that the Lightning module returns a loss and configures AdamW."""
    module = PromptRiskLightningModule(
        model_name="dummy-model",
        num_labels=2,
        learning_rate=1e-3,
        weight_decay=0.01,
        model=DummySequenceClassifier(),
    )
    batch = {
        "input_ids": torch.tensor(
            [
                [1, 2, 3, 4],
                [4, 3, 2, 1],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.ones((2, 4), dtype=torch.long),
        "labels": torch.tensor([0, 1], dtype=torch.long),
    }

    with patch.object(module, "log") as log_mock:
        loss = module.training_step(batch, batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    log_mock.assert_called_once()


def test_lightning_module_computes_validation_and_test_metrics() -> None:
    """Test that validation and test metrics use separate states."""
    module = PromptRiskLightningModule(
        model_name="dummy-model",
        num_labels=2,
        learning_rate=1e-3,
        weight_decay=0.01,
        model=DummySequenceClassifier(),
    )
    batch = {
        "input_ids": torch.tensor(
            [
                [1, 2, 3, 4],
                [4, 3, 2, 1],
                [1, 1, 1, 1],
                [4, 4, 4, 4],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.ones((4, 4), dtype=torch.long),
        "labels": torch.tensor([0, 1, 0, 1], dtype=torch.long),
    }

    with (
        patch.object(module, "log"),
        patch.object(module, "log_dict") as log_dict_mock,
    ):
        module.validation_step(batch, batch_idx=0)
        module.on_validation_epoch_end()

        validation_metrics = log_dict_mock.call_args.args[0]

        module.test_step(batch, batch_idx=0)
        module.on_test_epoch_end()

        test_metrics = log_dict_mock.call_args.args[0]

    assert set(validation_metrics) == {
        "val/accuracy",
        "val/precision",
        "val/recall",
        "val/f1",
        "val/roc_auc",
    }
    assert set(test_metrics) == {
        "test/accuracy",
        "test/precision",
        "test/recall",
        "test/f1",
        "test/roc_auc",
    }
    assert all(torch.isfinite(value) for value in validation_metrics.values())
    assert all(torch.isfinite(value) for value in test_metrics.values())


def test_prompt_risk_data_module_provides_all_dataloaders(
    tmp_path: Path,
) -> None:
    """Test that the Lightning data module loads all dataset splits."""
    records = [
        {"prompt": "safe prompt", "label": 0, "source": "fixture"},
        {"prompt": "risky prompt", "label": 1, "source": "fixture"},
    ]
    serialized_records = "\n".join(json.dumps(record) for record in records)

    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    test_path = tmp_path / "test.jsonl"

    for path in (train_path, val_path, test_path):
        path.write_text(f"{serialized_records}\n", encoding="utf-8")

    data_module = PromptRiskDataModule(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        model_name="dummy-model",
        max_length=4,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=True,
        tokenizer=DummyTokenizer(),
    )

    data_module.setup(stage="fit")
    train_batch = next(iter(data_module.train_dataloader()))
    val_batch = next(iter(data_module.val_dataloader()))

    data_module.setup(stage="test")
    test_batch = next(iter(data_module.test_dataloader()))

    assert train_batch["input_ids"].shape == torch.Size([2, 4])
    assert val_batch["input_ids"].shape == torch.Size([2, 4])
    assert test_batch["input_ids"].shape == torch.Size([2, 4])

    assert train_batch["labels"].shape == torch.Size([2])
    assert val_batch["labels"].shape == torch.Size([2])
    assert test_batch["labels"].shape == torch.Size([2])

    assert data_module.persistent_workers is False


def test_lightning_trainer_runs_end_to_end(tmp_path: Path) -> None:
    """Test one complete Lightning train, validation, and test cycle."""
    records = [
        {"prompt": "safe prompt", "label": 0, "source": "fixture"},
        {"prompt": "risky prompt", "label": 1, "source": "fixture"},
    ]
    serialized_records = "\n".join(json.dumps(record) for record in records)

    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    test_path = tmp_path / "test.jsonl"

    for path in (train_path, val_path, test_path):
        path.write_text(f"{serialized_records}\n", encoding="utf-8")

    data_module = PromptRiskDataModule(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        model_name="dummy-model",
        max_length=4,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        tokenizer=DummyTokenizer(),
    )
    model = PromptRiskLightningModule(
        model_name="dummy-model",
        num_labels=2,
        learning_rate=1e-3,
        weight_decay=0.01,
        model=DummySequenceClassifier(),
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=tmp_path / "checkpoints",
        filename="prompt-risk-{epoch:02d}",
        monitor="val/roc_auc",
        mode="max",
        save_top_k=1,
        save_last="link",
        auto_insert_metric_name=False,
    )

    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        callbacks=[checkpoint_callback],
        enable_model_summary=False,
    )

    trainer.fit(model=model, datamodule=data_module)
    optimizer = trainer.optimizers[0]

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] <= 1e-3
    assert optimizer.param_groups[0]["weight_decay"] == 0.01
    assert len(trainer.lr_scheduler_configs) == 1
    assert trainer.lr_scheduler_configs[0].interval == "step"
    assert Path(checkpoint_callback.best_model_path).is_file()
    assert (tmp_path / "checkpoints" / "last.ckpt").is_file()

    test_results = trainer.test(
        model=model,
        datamodule=data_module,
        ckpt_path="best",
    )

    assert trainer.global_step == 1
    assert len(test_results) == 1
    assert "test/loss" in test_results[0]
    assert "test/accuracy" in test_results[0]
    assert "test/f1" in test_results[0]
    assert "test/roc_auc" in test_results[0]


def test_wandb_logger_can_be_disabled() -> None:
    """Test that W&B can be disabled for tests and local runs."""
    cfg = OmegaConf.create(
        {
            "wandb": {
                "mode": "disabled",
            }
        }
    )

    logger = _create_wandb_logger(cfg)

    assert logger is False


def test_wandb_api_key_is_loaded_from_secret_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test loading the W&B API key from Secret Manager."""
    secret_resource = "projects/test-project/secrets/wandb-api-key/versions/latest"
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_API_KEY_SECRET", secret_resource)

    response = SimpleNamespace(
        payload=SimpleNamespace(data=b"test-api-key\n"),
    )

    with patch(
        "shapiq_attribution.train.secretmanager.SecretManagerServiceClient",
    ) as client_class:
        client_class.return_value.access_secret_version.return_value = response

        _load_wandb_api_key_from_secret()

        client_class.return_value.access_secret_version.assert_called_once_with(
            request={"name": secret_resource},
        )

    assert os.environ["WANDB_API_KEY"] == "test-api-key"


def test_existing_wandb_api_key_skips_secret_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that an existing W&B API key takes precedence."""
    monkeypatch.setenv("WANDB_API_KEY", "existing-api-key")
    monkeypatch.setenv(
        "WANDB_API_KEY_SECRET",
        "projects/test-project/secrets/wandb-api-key/versions/latest",
    )

    with patch(
        "shapiq_attribution.train.secretmanager.SecretManagerServiceClient",
    ) as client_class:
        _load_wandb_api_key_from_secret()

    client_class.assert_not_called()
    assert os.environ["WANDB_API_KEY"] == "existing-api-key"
