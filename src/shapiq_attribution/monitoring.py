"""Production monitoring for the prompt-risk service.

Evidently operates on derived *numerical* features, not raw prompt text. Every
API request is reduced to a row of features and appended to a predictions CSV;
the same extraction over the training set produces a baseline CSV. Evidently
then compares the live feature distribution against that baseline to surface
data drift, plus a health-check test asserting the mean P(risky) has not
collapsed to all-safe or all-risky.

Data flow::

    TRAIN TIME:  train.jsonl --build_baseline--> data/monitoring/baseline.csv ----+
                                                                                  |--> Evidently --> drift_report.html
    LIVE TIME:   POST /predict --log_prediction--> data/monitoring/predictions.csv +

CLI::

    uv run python -m shapiq_attribution.monitoring baseline   # snapshot training distribution
    uv run python -m shapiq_attribution.monitoring report     # build the Evidently HTML report
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import PromptRiskPredictor

# Feature columns Evidently monitors. The order is stable: it is the CSV header.
FEATURE_COLUMNS: tuple[str, ...] = ("prompt_len", "token_count", "p_risky")

# Default on-disk locations (overridable by every callable below).
MONITORING_DIR = Path("data/monitoring")
PREDICTIONS_CSV = MONITORING_DIR / "predictions.csv"
BASELINE_CSV = MONITORING_DIR / "baseline.csv"
REPORT_HTML = Path("reports/monitoring/drift_report.html")

# Health-check bounds on the mean P(risky). A live mean outside this band means
# the model has collapsed (all-safe or all-risky) even if no input drift fired.
P_RISKY_MIN = 0.2
P_RISKY_MAX = 0.8


def extract_features(prompt: str, p_risky: float) -> dict[str, float]:
    """Reduce a prompt and its prediction to the numerical features Evidently tracks.

    Args:
        prompt: Raw prompt text.
        p_risky: Model probability that the prompt is unsafe, in ``[0, 1]``.

    Returns:
        A mapping with keys :data:`FEATURE_COLUMNS`.
    """
    return {
        "prompt_len": float(len(prompt)),
        "token_count": float(len(prompt.split())),
        "p_risky": float(p_risky),
    }


def log_prediction(row: dict[str, float], path: str | Path = PREDICTIONS_CSV) -> None:
    """Append one feature row to the predictions CSV, writing a header if it is new.

    Args:
        row: Feature mapping as returned by :func:`extract_features`.
        path: Destination CSV (parent directories are created on demand).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FEATURE_COLUMNS))
        if write_header:
            writer.writeheader()
        writer.writerow({column: row[column] for column in FEATURE_COLUMNS})


def build_baseline(
    predictor: PromptRiskPredictor,
    dataset_path: str | Path,
    out_path: str | Path = BASELINE_CSV,
) -> Path:
    """Snapshot the training-set feature distribution as the drift reference.

    Args:
        predictor: Loaded prompt-risk predictor.
        dataset_path: Training JSONL (``{"prompt", "label", "source"}`` per line).
        out_path: Destination CSV.

    Returns:
        The path the baseline CSV was written to.
    """
    from .data import load_prompt_risk_jsonl

    examples = load_prompt_risk_jsonl(dataset_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FEATURE_COLUMNS))
        writer.writeheader()
        for example in examples:
            prompt = example["prompt"]
            features = extract_features(prompt, predictor.predict_proba(prompt))
            writer.writerow(features)
    return out_path


def generate_report(
    reference_path: str | Path = BASELINE_CSV,
    current_path: str | Path = PREDICTIONS_CSV,
    out_path: str | Path = REPORT_HTML,
) -> Path:
    """Build the Evidently drift + health report comparing live data to the baseline.

    Runs a per-column drift test (``DataDriftPreset``) over
    :data:`FEATURE_COLUMNS` and a health check asserting the mean ``p_risky``
    stays within ``[P_RISKY_MIN, P_RISKY_MAX]``. The rendered HTML dashboard is
    written to ``out_path``.

    Args:
        reference_path: Baseline CSV from :func:`build_baseline`.
        current_path: Live predictions CSV from :func:`log_prediction`.
        out_path: Destination HTML report.

    Returns:
        The path the HTML report was written to.
    """
    import pandas as pd
    from evidently import DataDefinition, Dataset, Report
    from evidently.metrics import MeanValue
    from evidently.presets import DataDriftPreset
    from evidently.tests import gte, lte

    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)

    definition = DataDefinition(numerical_columns=list(FEATURE_COLUMNS))
    reference = Dataset.from_pandas(reference_df, data_definition=definition)
    current = Dataset.from_pandas(current_df, data_definition=definition)

    report = Report(
        [
            DataDriftPreset(),
            MeanValue(column="p_risky", tests=[gte(P_RISKY_MIN), lte(P_RISKY_MAX)]),
        ],
        include_tests=True,
    )
    snapshot = report.run(current_data=current, reference_data=reference)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out_path))
    return out_path


def cli() -> None:
    """Command-line entry point: ``baseline`` and ``report`` subcommands."""
    import argparse

    parser = argparse.ArgumentParser(description="Prompt-risk monitoring utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline_parser = subparsers.add_parser("baseline", help="Build the training-set feature baseline.")
    baseline_parser.add_argument("--model-dir", default="models/prompt_risk_distilbert")
    baseline_parser.add_argument("--dataset-path", default="data/processed/train.jsonl")
    baseline_parser.add_argument("--out-path", default=str(BASELINE_CSV))

    report_parser = subparsers.add_parser("report", help="Generate the Evidently drift report.")
    report_parser.add_argument("--reference-path", default=str(BASELINE_CSV))
    report_parser.add_argument("--current-path", default=str(PREDICTIONS_CSV))
    report_parser.add_argument("--out-path", default=str(REPORT_HTML))

    args = parser.parse_args()

    if args.command == "baseline":
        from .model import PromptRiskPredictor

        predictor = PromptRiskPredictor.from_pretrained(args.model_dir)
        path = build_baseline(predictor, args.dataset_path, args.out_path)
        print(f"Wrote baseline ({path})")
    elif args.command == "report":
        path = generate_report(args.reference_path, args.current_path, args.out_path)
        print(f"Wrote report ({path})")


if __name__ == "__main__":
    cli()
