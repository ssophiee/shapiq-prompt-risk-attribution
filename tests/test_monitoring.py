"""Tests for the drift-monitoring feature pipeline.

Feature extraction and CSV logging are pure and fast; ``build_baseline`` is
exercised with a deterministic predictor stub. The Evidently report is covered
by a single integration test that asserts an HTML dashboard is produced.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from shapiq_attribution import monitoring


class FakeBlob:
    """In-memory stand-in for ``google.cloud.storage.Blob``."""

    def __init__(self, name: str, store: dict[str, str]) -> None:
        """Bind the blob name to the shared in-memory object store."""
        self.name = name
        self._store = store

    def upload_from_string(self, data: str, content_type: str | None = None) -> None:
        """Store the payload under this blob's name."""
        del content_type
        self._store[self.name] = data

    def download_as_text(self) -> str:
        """Return the stored payload."""
        return self._store[self.name]

    def download_to_filename(self, filename: str) -> None:
        """Write the stored payload to a local file."""
        Path(filename).write_text(self._store[self.name], encoding="utf-8")


class FakeBucket:
    """In-memory stand-in for ``google.cloud.storage.Bucket``."""

    def __init__(self, client: FakeClient, name: str) -> None:
        """Keep a backref to the client, mirroring the real Bucket API."""
        self.client = client
        self.name = name

    def blob(self, name: str) -> FakeBlob:
        """Return a blob handle inside this bucket."""
        return FakeBlob(name, self.client.store)


class FakeClient:
    """In-memory stand-in for ``google.cloud.storage.Client``."""

    def __init__(self) -> None:
        """Start with an empty object store."""
        self.store: dict[str, str] = {}

    def bucket(self, name: str) -> FakeBucket:
        """Return a bucket handle sharing this client's store."""
        return FakeBucket(self, name)

    def list_blobs(self, bucket_name: str, prefix: str = "") -> list[FakeBlob]:
        """List blobs whose names start with the prefix."""
        del bucket_name
        return [FakeBlob(name, self.store) for name in self.store if name.startswith(prefix)]


class DummyPredictor:
    """Predictor stub returning a fixed probability for any prompt."""

    def __init__(self, probability: float) -> None:
        """Store the probability the stub should always return."""
        self.probability = probability

    def predict_proba(self, prompt: str) -> float:
        """Return the configured probability regardless of input."""
        del prompt
        return self.probability


def test_extract_features_computes_expected_values() -> None:
    """A row keeps the raw prompt plus length, word count, and risk probability."""
    features = monitoring.extract_features("two words", 0.42)

    assert features == {"prompt": "two words", "prompt_len": 9.0, "token_count": 2.0, "p_risky": 0.42}
    assert set(features) == set(monitoring.LOGGED_COLUMNS)


def test_log_prediction_writes_header_then_appends(tmp_path: Path) -> None:
    """The first write adds a header; later writes append rows only."""
    path = tmp_path / "nested" / "predictions.csv"

    monitoring.log_prediction(monitoring.extract_features("hello there", 0.7), path)
    monitoring.log_prediction(monitoring.extract_features("bye", 0.1), path)

    with path.open() as handle:
        rows = list(csv.DictReader(handle))

    assert [r["p_risky"] for r in rows] == ["0.7", "0.1"]
    assert [r["prompt"] for r in rows] == ["hello there", "bye"]
    assert list(rows[0]) == list(monitoring.LOGGED_COLUMNS)


def test_log_prediction_rotates_legacy_csv(tmp_path: Path) -> None:
    """An old-format CSV (no prompt column) is moved aside, not appended to."""
    path = tmp_path / "predictions.csv"
    path.write_text("prompt_len,token_count,p_risky\n5.0,1.0,0.4\n", encoding="utf-8")

    monitoring.log_prediction(monitoring.extract_features("hello", 0.7), path)

    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(monitoring.LOGGED_COLUMNS)
    assert len(rows) == 1
    assert (tmp_path / "predictions.legacy.csv").read_text(encoding="utf-8").startswith("prompt_len")


def test_log_prediction_gcs_writes_one_json_blob() -> None:
    """Each logged prediction becomes one JSON blob under the predictions prefix."""
    client = FakeClient()

    blob_name = monitoring.log_prediction_gcs(monitoring.extract_features("hello there", 0.7), "bucket", client)

    assert blob_name.startswith(monitoring.GCS_PREDICTIONS_PREFIX)
    assert json.loads(client.store[blob_name]) == {
        "prompt": "hello there",
        "prompt_len": 11.0,
        "token_count": 2.0,
        "p_risky": 0.7,
    }


def test_fetch_predictions_gcs_rebuilds_local_csv(tmp_path: Path) -> None:
    """Fetching downloads every logged blob into a well-formed predictions CSV."""
    client = FakeClient()
    monitoring.log_prediction_gcs(monitoring.extract_features("hello there", 0.7), "bucket", client)
    monitoring.log_prediction_gcs(monitoring.extract_features("bye", 0.1), "bucket", client)
    out_path = tmp_path / "nested" / "predictions.csv"

    result, count = monitoring.fetch_predictions_gcs("bucket", out_path, client)

    assert (result, count) == (out_path, 2)
    with out_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(monitoring.LOGGED_COLUMNS)
    assert sorted(row["p_risky"] for row in rows) == ["0.1", "0.7"]
    assert sorted(row["prompt"] for row in rows) == ["bye", "hello there"]


def test_fetch_predictions_gcs_limit_keeps_newest_rows(tmp_path: Path) -> None:
    """A fetch limit drops the oldest blobs so report time stays bounded."""
    client = FakeClient()
    for index in range(3):
        monitoring.log_prediction_gcs(monitoring.extract_features(f"prompt {index}", 0.5), "bucket", client)
    out_path = tmp_path / "predictions.csv"

    _, count = monitoring.fetch_predictions_gcs("bucket", out_path, client, limit=2)

    assert count == 2
    with out_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["prompt"] for row in rows] == ["prompt 1", "prompt 2"]


def test_fetch_baseline_gcs_downloads_csv(tmp_path: Path) -> None:
    """The deploy-time baseline blob is downloaded to the requested path."""
    client = FakeClient()
    client.store[monitoring.GCS_BASELINE_BLOB] = "prompt_len,token_count,p_risky\n5.0,1.0,0.4\n"
    out_path = tmp_path / "nested" / "baseline.csv"

    result = monitoring.fetch_baseline_gcs("bucket", out_path, client)

    assert result == out_path
    assert out_path.read_text(encoding="utf-8").startswith("prompt_len,token_count,p_risky")


def test_build_baseline_rows_match_dataset(tmp_path: Path) -> None:
    """One baseline row is written per dataset example, with model probabilities."""
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        '{"prompt": "a harmful request", "label": 1, "source": "advbench"}\n'
        '{"prompt": "hi", "label": 0, "source": "wildguard"}\n',
        encoding="utf-8",
    )
    out_path = tmp_path / "baseline.csv"

    monitoring.build_baseline(DummyPredictor(0.6), dataset, out_path)

    with out_path.open() as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0] == {"prompt": "a harmful request", "prompt_len": "17.0", "token_count": "3.0", "p_risky": "0.6"}
    assert all(row["p_risky"] == "0.6" for row in rows)


def test_generate_report_writes_html(tmp_path: Path) -> None:
    """The Evidently report renders an HTML dashboard from baseline + live CSVs."""
    reference = tmp_path / "baseline.csv"
    current = tmp_path / "predictions.csv"
    for prompt, prob in [("short prompt", 0.3), ("another prompt here", 0.5), ("third one", 0.4)]:
        monitoring.log_prediction(monitoring.extract_features(prompt, prob), reference)
    for prompt, prob in [("live request now", 0.9), ("another live one entirely", 0.95)]:
        monitoring.log_prediction(monitoring.extract_features(prompt, prob), current)

    out_path = tmp_path / "report.html"
    result = monitoring.generate_report(reference, current, out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
