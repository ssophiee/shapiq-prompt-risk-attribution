"""Tests for the FastAPI prompt-risk service.

The real predictor loads a DistilBERT checkpoint at startup, so these tests
override the ``get_predictor`` dependency with a deterministic stub and avoid the
lifespan model load by constructing a ``TestClient`` without entering it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from shapiq_attribution import api


class DummyPredictor:
    """Deterministic predictor stub returning a fixed probability."""

    def __init__(self, probability: float) -> None:
        """Store the probability the stub should always return."""
        self.probability = probability

    def predict_proba(self, prompt: str) -> float:
        """Return the configured probability regardless of input."""
        del prompt
        return self.probability


@pytest.fixture(autouse=True)
def _no_monitoring_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop /predict from writing monitoring rows to the real data directory."""
    monkeypatch.setattr(api, "log_prediction", lambda *args, **kwargs: None)


@pytest.fixture
def client() -> TestClient:
    """A TestClient with the lifespan disabled (no real model load)."""
    return TestClient(api.app)


def _use_predictor(probability: float) -> DummyPredictor:
    """Override the predictor dependency and return the installed stub."""
    predictor = DummyPredictor(probability)
    api.app.dependency_overrides[api.get_predictor] = lambda: predictor
    return predictor


def teardown_function() -> None:
    """Clear dependency overrides between tests."""
    api.app.dependency_overrides.clear()


def test_index_serves_html(client: TestClient) -> None:
    """The root path returns the web interface."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Prompt-Risk Attribution" in response.text


def test_health_reports_status(client: TestClient) -> None:
    """The health endpoint always responds with a status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_labels_risky_prompt(client: TestClient) -> None:
    """A high probability is labelled risky."""
    _use_predictor(0.92)

    response = client.post("/predict", json={"prompt": "do something harmful"})

    assert response.status_code == 200
    body = response.json()
    assert body["risk_probability"] == pytest.approx(0.92)
    assert body["label"] == "risky"


def test_predict_labels_safe_prompt(client: TestClient) -> None:
    """A low probability is labelled safe."""
    _use_predictor(0.04)

    response = client.post("/predict", json={"prompt": "what is the weather"})

    assert response.status_code == 200
    assert response.json()["label"] == "safe"


def test_predict_rejects_empty_prompt(client: TestClient) -> None:
    """An empty prompt fails request validation."""
    _use_predictor(0.5)

    response = client.post("/predict", json={"prompt": ""})

    assert response.status_code == 422


def test_predict_returns_503_without_model(client: TestClient) -> None:
    """Without a loaded model the predict endpoint reports unavailable."""
    api.state.predictor = None
    response = client.post("/predict", json={"prompt": "hello"})
    assert response.status_code == 503


def test_metrics_exposes_request_and_label_counters(client: TestClient) -> None:
    """/metrics serves Prometheus counters covering requests and predicted labels."""
    _use_predictor(0.92)
    client.post("/predict", json={"prompt": "do something harmful"})

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "api_requests_total" in response.text
    assert 'api_predicted_labels_total{label="risky"}' in response.text
    assert "api_request_duration_seconds" in response.text


def test_monitoring_returns_drift_report_from_local_csvs(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without a bucket, /monitoring renders the Evidently report from local CSVs."""
    from shapiq_attribution import monitoring

    reference = tmp_path / "baseline.csv"
    current = tmp_path / "predictions.csv"
    for prompt, prob in [("short prompt", 0.3), ("another prompt here", 0.5), ("third one", 0.4)]:
        monitoring.log_prediction(monitoring.extract_features(prompt, prob), reference)
    for prompt, prob in [("live request now", 0.9), ("another live one", 0.95)]:
        monitoring.log_prediction(monitoring.extract_features(prompt, prob), current)
    monkeypatch.setattr(api, "MONITORING_BUCKET", None)
    monkeypatch.setattr(monitoring, "BASELINE_CSV", reference)
    monkeypatch.setattr(monitoring, "PREDICTIONS_CSV", current)
    monkeypatch.setattr(monitoring, "REPORT_HTML", tmp_path / "report.html")

    response = client.get("/monitoring")

    assert response.status_code == 200
    assert "<html" in response.text.lower()


def test_monitoring_404s_without_logged_data(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """/monitoring reports 404 when no monitoring CSVs exist yet."""
    from shapiq_attribution import monitoring

    monkeypatch.setattr(api, "MONITORING_BUCKET", None)
    monkeypatch.setattr(monitoring, "BASELINE_CSV", tmp_path / "missing_baseline.csv")
    monkeypatch.setattr(monitoring, "PREDICTIONS_CSV", tmp_path / "missing_predictions.csv")

    response = client.get("/monitoring")

    assert response.status_code == 404


def test_predict_uploads_to_gcs_when_bucket_configured(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """With MONITORING_BUCKET set, each prediction is mirrored to GCS."""
    _use_predictor(0.3)
    uploads: list[tuple[dict[str, float], str]] = []
    monkeypatch.setattr(api, "MONITORING_BUCKET", "some-bucket")
    monkeypatch.setattr(api, "log_prediction_gcs", lambda row, bucket: uploads.append((row, bucket)))

    response = client.post("/predict", json={"prompt": "what is the weather"})

    assert response.status_code == 200
    assert uploads == [
        ({"prompt": "what is the weather", "prompt_len": 19.0, "token_count": 4.0, "p_risky": 0.3}, "some-bucket")
    ]
