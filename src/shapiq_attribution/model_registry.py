"""Publish, validate, and promote prompt-risk models in the W&B Registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import wandb

from shapiq_attribution.data import PromptRiskExample, load_prompt_risk_jsonl
from shapiq_attribution.evaluate import compute_metrics
from shapiq_attribution.model import PromptRiskPredictor


@dataclass(frozen=True)
class ValidationThresholds:
    """Minimum performance required before a model can be promoted."""

    min_f1: float = 0.70
    min_roc_auc: float = 0.85


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating one staged model."""

    artifact_path: str
    sample_count: int
    metrics: dict[str, float]
    checks: dict[str, bool]
    errors: list[str]

    @property
    def passed(self) -> bool:
        """Return whether every model check passed."""
        return not self.errors and all(self.checks.values())


def registry_target_from_artifact(artifact_path: str) -> str:
    """Remove the version or alias from a W&B artifact path.

    Args:
        artifact_path: Fully qualified W&B artifact path.

    Returns:
        Registry collection path without its version or alias.

    Raises:
        ValueError: If the path does not contain a version or alias.
    """
    target, separator, version = artifact_path.rpartition(":")
    if not separator or not target or not version:
        raise ValueError("artifact path must end in a version or alias, for example ':staging' or ':v3'")
    return target


def select_evaluation_examples(
    examples: Sequence[PromptRiskExample],
    max_samples: int,
) -> list[PromptRiskExample]:
    """Select a deterministic, approximately balanced evaluation subset.

    Args:
        examples: Full evaluation dataset.
        max_samples: Maximum number of examples to select.

    Returns:
        Deterministic evaluation subset containing both labels.

    Raises:
        ValueError: If the sample limit or dataset cannot support both labels.
    """
    if max_samples < 2:
        raise ValueError("max_samples must be at least 2")

    by_label = {
        0: [example for example in examples if example["label"] == 0],
        1: [example for example in examples if example["label"] == 1],
    }
    if not by_label[0] or not by_label[1]:
        raise ValueError("evaluation data must contain both labels")

    per_label = max_samples // 2
    selected = by_label[0][:per_label] + by_label[1][:per_label]
    if len(selected) < max_samples:
        selected.extend(by_label[1][per_label : per_label + max_samples - len(selected)])
    if len(selected) < max_samples:
        selected.extend(by_label[0][per_label : per_label + max_samples - len(selected)])
    return selected


def validate_local_model(
    model_dir: str | Path,
    test_data: str | Path,
    artifact_path: str,
    thresholds: ValidationThresholds,
    max_samples: int = 512,
    batch_size: int = 32,
    max_length: int = 128,
) -> ValidationResult:
    """Load and evaluate a downloaded model artifact.

    Args:
        model_dir: Directory containing the Hugging Face model and tokenizer.
        test_data: DVC-versioned JSONL test split.
        artifact_path: W&B artifact path used for the report.
        thresholds: Minimum acceptable model metrics.
        max_samples: Maximum number of deterministic test examples.
        batch_size: Inference batch size.
        max_length: Maximum tokenizer sequence length.

    Returns:
        Model validation result.
    """
    model_path = Path(model_dir)
    required_files_present = (model_path / "config.json").is_file() and any(
        (model_path / filename).is_file() for filename in ("model.safetensors", "pytorch_model.bin")
    )
    if not required_files_present:
        return ValidationResult(
            artifact_path=artifact_path,
            sample_count=0,
            metrics={},
            checks={"required_model_files": False},
            errors=["The artifact does not contain a loadable Hugging Face model under model/."],
        )

    examples = select_evaluation_examples(load_prompt_risk_jsonl(test_data), max_samples=max_samples)
    predictor = PromptRiskPredictor.from_pretrained(model_path, max_length=max_length, device="cpu")

    probabilities: list[float] = []
    for start in range(0, len(examples), batch_size):
        prompts = [example["prompt"] for example in examples[start : start + batch_size]]
        probabilities.extend(predictor.predict_proba_batch(prompts))

    labels = [example["label"] for example in examples]
    predictions = [int(probability >= 0.5) for probability in probabilities]
    metrics = compute_metrics(labels=labels, predictions=predictions, probabilities=probabilities)
    probabilities_valid = len(probabilities) == len(labels) and all(
        math.isfinite(probability) and 0.0 <= probability <= 1.0 for probability in probabilities
    )
    checks = {
        "required_model_files": True,
        "probabilities_valid": probabilities_valid,
        "both_classes_predicted": set(predictions) == {0, 1},
        "f1_threshold": metrics["f1"] >= thresholds.min_f1,
        "roc_auc_threshold": metrics["roc_auc"] >= thresholds.min_roc_auc,
    }
    errors = [name for name, passed in checks.items() if not passed]
    return ValidationResult(
        artifact_path=artifact_path,
        sample_count=len(examples),
        metrics=metrics,
        checks=checks,
        errors=errors,
    )


def write_validation_report(
    result: ValidationResult,
    markdown_path: str | Path,
    json_path: str | Path,
    thresholds: ValidationThresholds,
) -> None:
    """Write human-readable and machine-readable validation reports.

    Args:
        result: Completed model validation.
        markdown_path: Markdown output path.
        json_path: JSON output path.
        thresholds: Thresholds used for validation.
    """
    status = "PASS" if result.passed else "FAIL"
    lines = [
        "# Staged model validation",
        "",
        f"- Status: **{status}**",
        f"- Artifact: `{result.artifact_path}`",
        f"- Evaluation examples: {result.sample_count}",
        f"- Minimum F1: {thresholds.min_f1:.3f}",
        f"- Minimum ROC-AUC: {thresholds.min_roc_auc:.3f}",
        "",
        "## Metrics",
        "",
    ]
    if result.metrics:
        lines.extend(f"- {name}: {value:.4f}" for name, value in sorted(result.metrics.items()))
    else:
        lines.append("- No metrics were produced.")
    lines.extend(["", "## Checks", ""])
    lines.extend(f"- [{'x' if passed else ' '}] {name}" for name, passed in result.checks.items())
    if result.errors:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {error}" for error in result.errors)

    markdown_output = Path(markdown_path)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_output = Path(json_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    payload = {**asdict(result), "passed": result.passed, "thresholds": asdict(thresholds)}
    json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _wandb_settings() -> tuple[str | None, str | None]:
    """Return optional W&B entity and project settings."""
    return os.environ.get("WANDB_ENTITY") or None, os.environ.get("WANDB_PROJECT") or None


def download_model_artifact(artifact_path: str, output_dir: str | Path) -> Path:
    """Download a model artifact and return its model directory.

    Args:
        artifact_path: Fully qualified W&B artifact path.
        output_dir: Download destination.

    Returns:
        Path to the downloaded Hugging Face model directory.
    """
    entity, project = _wandb_settings()
    overrides = {key: value for key, value in {"entity": entity, "project": project}.items() if value}
    artifact = wandb.Api(overrides=overrides).artifact(artifact_path, type="model")
    root = Path(artifact.download(root=str(output_dir)))
    return root / "model"


def _artifact_metadata(metrics_path: Path) -> dict[str, Any]:
    """Build provenance metadata for a published model."""
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = os.environ.get("GITHUB_SHA", "unknown")

    dvc_lock = Path("dvc.lock")
    dvc_lock_sha256 = hashlib.sha256(dvc_lock.read_bytes()).hexdigest() if dvc_lock.is_file() else "unavailable"
    return {
        "git_commit": git_commit,
        "dvc_lock_sha256": dvc_lock_sha256,
        "metrics": metrics,
    }


def log_model_artifact(
    run: Any,
    model_dir: str | Path,
    metrics_path: str | Path,
    artifact_name: str,
    registry_target: str,
    aliases: Sequence[str] = ("candidate",),
) -> str:
    """Log a trained model and link it to a W&B Registry collection.

    Args:
        run: Active W&B run.
        model_dir: Exported Hugging Face model directory.
        metrics_path: JSON metrics produced by training.
        artifact_name: Name of the source project artifact.
        registry_target: Registry collection path.
        aliases: Registry aliases assigned to the new version.

    Returns:
        Qualified registry artifact path.
    """
    model_path = Path(model_dir)
    metrics_file = Path(metrics_path)
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        metadata=_artifact_metadata(metrics_file),
    )
    artifact.add_dir(str(model_path), name="model")
    artifact.add_file(str(metrics_file), name="reports/metrics.json")
    logged_artifact = run.log_artifact(artifact, aliases=["latest"])
    logged_artifact.wait()
    run.link_artifact(logged_artifact, target_path=registry_target, aliases=list(aliases))
    return f"{registry_target}:{aliases[0]}"


def publish_model_artifact(
    model_dir: str | Path,
    metrics_path: str | Path,
    artifact_name: str,
    registry_target: str,
) -> str:
    """Publish an existing local model through a dedicated W&B run.

    Args:
        model_dir: Exported Hugging Face model directory.
        metrics_path: JSON metrics produced by training.
        artifact_name: Name of the source project artifact.
        registry_target: Registry collection path.

    Returns:
        Qualified registry artifact path.
    """
    entity, project = _wandb_settings()
    if not project:
        raise ValueError("WANDB_PROJECT must be set")
    with wandb.init(entity=entity, project=project, job_type="model-publication") as run:
        return log_model_artifact(
            run=run,
            model_dir=model_dir,
            metrics_path=metrics_path,
            artifact_name=artifact_name,
            registry_target=registry_target,
        )


def promote_model_artifact(artifact_path: str, alias: str = "production") -> str:
    """Promote one tested model version in its existing Registry collection.

    Args:
        artifact_path: Fully qualified staged model artifact.
        alias: Alias assigned after validation.

    Returns:
        Qualified promoted artifact path.
    """
    entity, project = _wandb_settings()
    if not project:
        raise ValueError("WANDB_PROJECT must be set")
    target = registry_target_from_artifact(artifact_path)
    overrides = {key: value for key, value in {"entity": entity, "project": project}.items() if value}
    artifact = wandb.Api(overrides=overrides).artifact(artifact_path, type="model")
    with wandb.init(entity=entity, project=project, job_type="model-promotion") as run:
        run.link_artifact(artifact, target_path=target, aliases=[alias])
    return f"{target}:{alias}"


def _validate_command(args: argparse.Namespace) -> int:
    """Run the validation CLI command."""
    thresholds = ValidationThresholds(min_f1=args.min_f1, min_roc_auc=args.min_roc_auc)
    try:
        model_dir = download_model_artifact(args.artifact_path, args.download_dir)
        result = validate_local_model(
            model_dir=model_dir,
            test_data=args.test_data,
            artifact_path=args.artifact_path,
            thresholds=thresholds,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
    except Exception as error:
        result = ValidationResult(
            artifact_path=args.artifact_path,
            sample_count=0,
            metrics={},
            checks={"validation_completed": False},
            errors=[f"{type(error).__name__}: {error}"],
        )
    write_validation_report(result, args.markdown_report, args.json_report, thresholds)
    print(Path(args.markdown_report).read_text(encoding="utf-8"))
    return 0 if result.passed else 1


def _build_parser() -> argparse.ArgumentParser:
    """Build the model registry command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish = subparsers.add_parser("publish", help="Publish an existing local model")
    publish.add_argument("--model-dir", default="models/prompt_risk_distilbert")
    publish.add_argument("--metrics", default="reports/metrics.json")
    publish.add_argument("--artifact-name", default="prompt-risk-distilbert")
    publish.add_argument("--registry-target", default="wandb-registry-Models/prompt-risk-distilbert")

    validate = subparsers.add_parser("validate", help="Download and test a staged model")
    validate.add_argument("--artifact-path", required=True)
    validate.add_argument("--test-data", default="data/processed/test.jsonl")
    validate.add_argument("--download-dir", default="outputs/registry-model")
    validate.add_argument("--markdown-report", default="model-validation.md")
    validate.add_argument("--json-report", default="model-validation.json")
    validate.add_argument("--max-samples", type=int, default=512)
    validate.add_argument("--batch-size", type=int, default=32)
    validate.add_argument("--max-length", type=int, default=128)
    validate.add_argument("--min-f1", type=float, default=0.70)
    validate.add_argument("--min-roc-auc", type=float, default=0.85)

    promote = subparsers.add_parser("promote", help="Assign a production alias to a tested model")
    promote.add_argument("--artifact-path", required=True)
    promote.add_argument("--alias", default="production")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the model registry command-line interface.

    Args:
        argv: Optional arguments for testing.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    if args.command == "publish":
        artifact_path = publish_model_artifact(
            model_dir=args.model_dir,
            metrics_path=args.metrics,
            artifact_name=args.artifact_name,
            registry_target=args.registry_target,
        )
        print(f"Published model: {artifact_path}")
        return 0
    if args.command == "validate":
        return _validate_command(args)
    if args.command == "promote":
        artifact_path = promote_model_artifact(args.artifact_path, alias=args.alias)
        print(f"Promoted model: {artifact_path}")
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
