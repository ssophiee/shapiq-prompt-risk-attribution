"""Hydra entrypoint for Lightning prompt-risk classifier training."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import hydra
import lightning as L
from google.cloud import secretmanager
from hydra.utils import to_absolute_path
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from shapiq_attribution.lightning_data_module import PromptRiskDataModule
from shapiq_attribution.lightning_module import PromptRiskLightningModule
from shapiq_attribution.model import (
    load_prompt_risk_tokenizer,
    save_prompt_risk_model,
)


def _extract_metrics(
    results: Sequence[Mapping[str, float]],
    stage: str,
) -> dict[str, float]:
    """Extract one stage's metrics from Lightning results.

    Args:
        results: Results returned by Trainer.validate or Trainer.test.
        stage: Metric prefix to remove.

    Returns:
        Metrics without their stage prefix.
    """
    if not results:
        return {}

    prefix = f"{stage}/"
    return {key.removeprefix(prefix): float(value) for key, value in results[0].items() if key.startswith(prefix)}


def _parse_save_last(
    value: object,
) -> bool | Literal["link"]:
    """Parse the Lightning save-last checkpoint option.

    Args:
        value: Hydra checkpoint option.

    Returns:
        Boolean checkpoint option or the link strategy.

    Raises:
        ValueError: If the option is unsupported.
    """
    if value == "link":
        return "link"
    if isinstance(value, bool):
        return value
    raise ValueError("checkpoint.save_last must be true, false, or link")


def _load_wandb_api_key_from_secret() -> None:
    """Load the W&B API key from GCP Secret Manager when configured."""
    if os.environ.get("WANDB_API_KEY"):
        return

    secret_resource = os.environ.get("WANDB_API_KEY_SECRET")
    if not secret_resource:
        return

    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(
        request={"name": secret_resource},
    )
    api_key = response.payload.data.decode("utf-8").strip()
    if not api_key:
        raise RuntimeError("The configured W&B API key secret is empty")

    os.environ["WANDB_API_KEY"] = api_key


def _create_wandb_logger(
    cfg: DictConfig,
) -> WandbLogger | bool:
    """Create the configured Lightning W&B logger.

    Args:
        cfg: Hydra configuration.

    Returns:
        W&B logger or False when logging is disabled.

    Raises:
        ValueError: If the configured W&B mode is unsupported.
    """
    mode = str(cfg.wandb.mode)
    if mode == "disabled":
        return False
    if mode not in {"online", "offline"}:
        raise ValueError("wandb.mode must be one of: online, offline, disabled")
    if mode == "online":
        _load_wandb_api_key_from_secret()

    save_dir = Path(to_absolute_path(cfg.wandb.save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = WandbLogger(
        project=str(cfg.wandb.project),
        entity=cfg.wandb.entity,
        name=str(cfg.wandb.run_name),
        save_dir=save_dir,
        offline=mode == "offline",
        log_model=bool(cfg.wandb.log_model),
    )

    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(resolved_config, dict):
        logger.log_hyperparams({str(key): value for key, value in resolved_config.items()})

    global_batch_size = (
        int(cfg.training.batch_size)
        * int(cfg.training.devices)
        * int(cfg.training.num_nodes)
        * int(cfg.training.accumulate_grad_batches)
    )
    logger.log_hyperparams({"training.global_batch_size": global_batch_size})
    return logger


def train_model(cfg: DictConfig) -> dict[str, dict[str, float]]:
    """Train and evaluate the configured prompt-risk classifier.

    Args:
        cfg: Hydra configuration.

    Returns:
        Nested validation and test metrics.
    """
    L.seed_everything(
        int(cfg.training.seed),
        workers=True,
    )

    tokenizer = load_prompt_risk_tokenizer(str(cfg.model.name))
    data_module = PromptRiskDataModule(
        train_path=to_absolute_path(cfg.data.train_path),
        val_path=to_absolute_path(cfg.data.val_path),
        test_path=to_absolute_path(cfg.data.test_path),
        model_name=str(cfg.model.name),
        max_length=int(cfg.model.max_length),
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.num_workers),
        pin_memory=bool(cfg.training.pin_memory),
        persistent_workers=bool(cfg.training.persistent_workers),
        tokenizer=tokenizer,
    )
    model = PromptRiskLightningModule(
        model_name=str(cfg.model.name),
        num_labels=int(cfg.model.num_labels),
        learning_rate=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=to_absolute_path(cfg.checkpoint.dirpath),
        filename=str(cfg.checkpoint.filename),
        monitor=str(cfg.checkpoint.monitor),
        mode=str(cfg.checkpoint.mode),
        save_top_k=int(cfg.checkpoint.save_top_k),
        save_last=_parse_save_last(cfg.checkpoint.save_last),
        auto_insert_metric_name=False,
    )

    logger = _create_wandb_logger(cfg)

    callbacks: list[Callback] = [checkpoint_callback]
    if logger is not False:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    trainer = L.Trainer(
        max_epochs=int(cfg.training.epochs),
        accelerator=str(cfg.training.accelerator),
        devices=int(cfg.training.devices),
        strategy=str(cfg.training.strategy),
        num_nodes=int(cfg.training.num_nodes),
        precision=cfg.training.precision,
        accumulate_grad_batches=int(cfg.training.accumulate_grad_batches),
        deterministic=bool(cfg.training.deterministic),
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
        limit_test_batches=cfg.training.limit_test_batches,
        log_every_n_steps=int(cfg.training.log_every_n_steps),
        logger=logger,
        callbacks=callbacks,
    )

    resume_from = cfg.checkpoint.resume_from
    checkpoint_path = to_absolute_path(str(resume_from)) if resume_from else None

    trainer.fit(model=model, datamodule=data_module, ckpt_path=checkpoint_path)

    validation_results = trainer.validate(
        model=model,
        datamodule=data_module,
        ckpt_path="best",
    )

    test_results = trainer.test(
        model=model,
        datamodule=data_module,
        ckpt_path="best",
    )

    metrics = {
        "val": _extract_metrics(validation_results, stage="val"),
        "test": _extract_metrics(test_results, stage="test"),
    }

    if trainer.is_global_zero:
        model_dir = Path(to_absolute_path(cfg.output.model_dir))
        save_prompt_risk_model(
            model=model.model,
            tokenizer=tokenizer,
            output_dir=model_dir,
        )

        metrics_path = Path(to_absolute_path(cfg.output.metrics_path))
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

    trainer.strategy.barrier()
    return metrics


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train",
)
def main(cfg: DictConfig) -> None:
    """Run the Lightning training pipeline.

    Args:
        cfg: Hydra configuration.
    """
    train_model(cfg)


if __name__ == "__main__":
    main()
