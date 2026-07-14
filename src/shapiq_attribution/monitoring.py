"""Production monitoring for the prompt-risk service.

Every API request is logged as one row: the raw prompt text (input–output
collection) plus derived *numerical* features; the same extraction over the
training set produces a baseline CSV. Evidently operates on the numerical
features only — it compares the live feature distribution against that baseline
to surface data drift, plus a health-check test asserting the mean P(risky) has
not collapsed to all-safe or all-risky.

Data flow::

    TRAIN TIME:  train.jsonl --build_baseline--> data/monitoring/baseline.csv ----+
                                                                                  |--> Evidently --> drift_report.html
    LIVE TIME:   POST /predict --log_prediction--> data/monitoring/predictions.csv +

On Cloud Run the local CSV lives on ephemeral disk and dies with the container,
so the deployed API *also* uploads every row to a GCS bucket when
``MONITORING_BUCKET`` is set (one small JSON blob per prediction — GCS objects
are immutable, so append-to-one-CSV is not an option). ``fetch`` downloads those
blobs back into the local predictions CSV, after which ``report`` works as usual::

    DEPLOYED:  POST /predict --log_prediction_gcs--> gs://$MONITORING_BUCKET/predictions/*.json
    LOCAL:     fetch --> data/monitoring/predictions.csv --report--> drift_report.html

CLI::

    uv run python -m shapiq_attribution.monitoring baseline   # snapshot training distribution
    uv run python -m shapiq_attribution.monitoring fetch      # pull deployed predictions from GCS
    uv run python -m shapiq_attribution.monitoring report     # build the Evidently HTML report
"""

from __future__ import annotations

import csv
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import PromptRiskPredictor

# Numerical feature columns Evidently monitors for drift.
FEATURE_COLUMNS: tuple[str, ...] = ("prompt_len", "token_count", "p_risky")

# Full layout of a logged row: raw prompt text first, then the features. The
# order is stable: it is the CSV header. Evidently only sees FEATURE_COLUMNS.
LOGGED_COLUMNS: tuple[str, ...] = ("prompt", *FEATURE_COLUMNS)

# Default on-disk locations (overridable by every callable below).
MONITORING_DIR = Path("data/monitoring")
PREDICTIONS_CSV = MONITORING_DIR / "predictions.csv"
BASELINE_CSV = MONITORING_DIR / "baseline.csv"
REPORT_HTML = Path("reports/monitoring/drift_report.html")

# GCS prefix under which the deployed API drops one JSON blob per prediction.
GCS_PREDICTIONS_PREFIX = "predictions/"

# GCS blob holding the training-set baseline (uploaded by deploy/cloudrun.sh).
GCS_BASELINE_BLOB = "baseline.csv"

# Health-check bounds on the mean P(risky). A live mean outside this band means
# the model has collapsed (all-safe or all-risky) even if no input drift fired.
P_RISKY_MIN = 0.2
P_RISKY_MAX = 0.8


def extract_features(prompt: str, p_risky: float) -> dict[str, float | str]:
    """Reduce a prompt and its prediction to one loggable monitoring row.

    Args:
        prompt: Raw prompt text (kept in the row for input–output collection).
        p_risky: Model probability that the prompt is unsafe, in ``[0, 1]``.

    Returns:
        A mapping with keys :data:`LOGGED_COLUMNS`.
    """
    return {
        "prompt": prompt,
        "prompt_len": float(len(prompt)),
        "token_count": float(len(prompt.split())),
        "p_risky": float(p_risky),
    }


def log_prediction(row: dict[str, float | str], path: str | Path = PREDICTIONS_CSV) -> None:
    """Append one row to the predictions CSV, writing a header if it is new.

    A pre-existing CSV whose header does not match :data:`LOGGED_COLUMNS`
    (e.g. written before the raw prompt was logged) is rotated aside to
    ``*.legacy.csv`` instead of being corrupted by misaligned appends.

    Args:
        row: Row mapping as returned by :func:`extract_features`.
        path: Destination CSV (parent directories are created on demand).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            header = handle.readline().strip()
        if header != ",".join(LOGGED_COLUMNS):
            path.rename(path.with_suffix(".legacy.csv"))
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LOGGED_COLUMNS))
        if write_header:
            writer.writeheader()
        writer.writerow({column: row[column] for column in LOGGED_COLUMNS})


def _gcs_bucket(bucket_name: str, client: object = None):
    """Return a GCS bucket handle, creating a default client if none is given."""
    if client is None:
        from google.cloud import storage

        client = storage.Client()
    return client.bucket(bucket_name)


def log_prediction_gcs(row: dict[str, float | str], bucket_name: str, client: object = None) -> str:
    """Upload one monitoring row as a timestamped JSON blob to a GCS bucket.

    Used by the deployed API, where the local CSV is on ephemeral disk. GCS
    objects are immutable, so each prediction becomes its own small blob under
    :data:`GCS_PREDICTIONS_PREFIX`; :func:`fetch_predictions_gcs` reassembles
    them into a CSV.

    Args:
        row: Row mapping as returned by :func:`extract_features`.
        bucket_name: Name of the target GCS bucket (no ``gs://`` prefix).
        client: Optional ``google.cloud.storage.Client`` (injected in tests).

    Returns:
        The name of the blob that was written.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    blob_name = f"{GCS_PREDICTIONS_PREFIX}{stamp}-{uuid.uuid4().hex[:8]}.json"
    payload = json.dumps({column: row[column] for column in LOGGED_COLUMNS})
    bucket = _gcs_bucket(bucket_name, client)
    bucket.blob(blob_name).upload_from_string(payload, content_type="application/json")
    return blob_name


def fetch_predictions_gcs(
    bucket_name: str,
    out_path: str | Path = PREDICTIONS_CSV,
    client: object = None,
) -> tuple[Path, int]:
    """Download all prediction blobs from GCS into a local predictions CSV.

    Blob names start with a UTC timestamp, so sorting them by name restores
    chronological order. The output CSV is overwritten, not appended to. Blobs
    logged before the raw prompt was collected get an empty ``prompt`` cell.

    Args:
        bucket_name: Bucket the deployed API logs to (no ``gs://`` prefix).
        out_path: Destination CSV (same format as :func:`log_prediction`).
        client: Optional ``google.cloud.storage.Client`` (injected in tests).

    Returns:
        The path written and the number of prediction rows fetched.
    """
    bucket = _gcs_bucket(bucket_name, client)
    blobs = sorted(bucket.client.list_blobs(bucket_name, prefix=GCS_PREDICTIONS_PREFIX), key=lambda b: b.name)
    rows = [json.loads(blob.download_as_text()) for blob in blobs]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LOGGED_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in LOGGED_COLUMNS})
    return out_path, len(rows)


def fetch_baseline_gcs(
    bucket_name: str,
    out_path: str | Path = BASELINE_CSV,
    client: object = None,
) -> Path:
    """Download the training-set baseline CSV from GCS.

    The baseline is uploaded to the bucket by ``deploy/cloudrun.sh``, so the
    deployed drift endpoint can compare against it without the CSV being baked
    into the image (``data/`` is excluded from the build context).

    Args:
        bucket_name: Monitoring bucket name (no ``gs://`` prefix).
        out_path: Where to write the downloaded CSV.
        client: Optional ``google.cloud.storage.Client`` (injected in tests).

    Returns:
        The path the baseline CSV was written to.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bucket = _gcs_bucket(bucket_name, client)
    bucket.blob(GCS_BASELINE_BLOB).download_to_filename(str(out_path))
    return out_path


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
        writer = csv.DictWriter(handle, fieldnames=list(LOGGED_COLUMNS))
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

    # Drift is computed on the numerical features only; the raw prompt column
    # (absent in pre-collection CSVs) is dropped rather than fed to Evidently.
    reference_df = pd.read_csv(reference_path)[list(FEATURE_COLUMNS)]
    current_df = pd.read_csv(current_path)[list(FEATURE_COLUMNS)]

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

    fetch_parser = subparsers.add_parser("fetch", help="Download deployed-API predictions from GCS.")
    fetch_parser.add_argument(
        "--bucket",
        default=os.environ.get("MONITORING_BUCKET"),
        help="GCS bucket the deployed API logs to (default: $MONITORING_BUCKET).",
    )
    fetch_parser.add_argument("--out-path", default=str(PREDICTIONS_CSV))

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
    elif args.command == "fetch":
        if not args.bucket:
            parser.error("no bucket: pass --bucket or set MONITORING_BUCKET")
        path, count = fetch_predictions_gcs(args.bucket, args.out_path)
        print(f"Fetched {count} prediction rows from gs://{args.bucket} ({path})")
    elif args.command == "report":
        path = generate_report(args.reference_path, args.current_path, args.out_path)
        print(f"Wrote report ({path})")


if __name__ == "__main__":
    cli()
