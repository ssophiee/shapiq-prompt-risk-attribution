"""Hydra entrypoint for DistilBERT prompt-risk classifier training."""

from __future__ import annotations

import json
import random
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from shapiq_attribution.data import PromptRiskExample, PromptRiskTextDataset, load_prompt_risk_jsonl
from shapiq_attribution.evaluate import evaluate_model
from shapiq_attribution.model import load_prompt_risk_model, load_prompt_risk_tokenizer, save_prompt_risk_model


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training.

    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    """Select the training device.

    Args:
        device_name: Device name or `auto`.

    Returns:
        Torch device.

    Raises:
        ValueError: If a requested accelerator is unavailable.
    """
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return torch.device(device_name)


def make_dataloader(
    examples: list[PromptRiskExample],
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a DataLoader for prompt-risk examples.

    Args:
        examples: Prompt-risk examples.
        tokenizer: Hugging Face tokenizer.
        max_length: Maximum tokenized sequence length.
        batch_size: Batch size.
        shuffle: Whether to shuffle examples.

    Returns:
        Prompt-risk dataloader.
    """
    dataset = PromptRiskTextDataset(examples=examples, tokenizer=tokenizer, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(cfg: DictConfig) -> dict[str, dict[str, float]]:
    """Train and evaluate the configured prompt-risk classifier.

    Args:
        cfg: Hydra configuration.

    Returns:
        Nested metrics dictionary.
    """
    set_seed(int(cfg.training.seed))
    device = select_device(str(cfg.training.device))

    train_examples = load_prompt_risk_jsonl(to_absolute_path(cfg.data.train_path))
    val_examples = load_prompt_risk_jsonl(to_absolute_path(cfg.data.val_path))
    test_examples = load_prompt_risk_jsonl(to_absolute_path(cfg.data.test_path))

    tokenizer = load_prompt_risk_tokenizer(str(cfg.model.name))
    model = load_prompt_risk_model(str(cfg.model.name), num_labels=int(cfg.model.num_labels))
    model.to(device)

    train_loader = make_dataloader(
        examples=train_examples,
        tokenizer=tokenizer,
        max_length=int(cfg.model.max_length),
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
    )
    val_loader = make_dataloader(
        examples=val_examples,
        tokenizer=tokenizer,
        max_length=int(cfg.model.max_length),
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
    )
    test_loader = make_dataloader(
        examples=test_examples,
        tokenizer=tokenizer,
        max_length=int(cfg.model.max_length),
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )
    total_steps = len(train_loader) * int(cfg.training.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    try:
        val_metrics: dict[str, float] = {}
        for epoch in range(int(cfg.training.epochs)):
            model.train()
            train_losses = []

            for batch in train_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad()
                outputs = model(**batch)
                outputs.loss.backward()
                optimizer.step()
                scheduler.step()
                train_losses.append(float(outputs.loss.detach().cpu()))

            val_metrics = evaluate_model(model=model, dataloader=val_loader, device=device)
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": float(np.mean(train_losses)),
                    **{f"val/{key}": value for key, value in val_metrics.items()},
                }
            )

        test_metrics = evaluate_model(model=model, dataloader=test_loader, device=device)
        wandb.log({f"test/{key}": value for key, value in test_metrics.items()})

        model_dir = Path(to_absolute_path(cfg.output.model_dir))
        save_prompt_risk_model(model=model, tokenizer=tokenizer, output_dir=model_dir)

        metrics = {"val": val_metrics, "test": test_metrics}
        metrics_path = Path(to_absolute_path(cfg.output.metrics_path))
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        return metrics
    finally:
        run.finish()


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Run the training pipeline step.

    Args:
        cfg: Hydra configuration.
    """
    train_model(cfg)


if __name__ == "__main__":
    main()
