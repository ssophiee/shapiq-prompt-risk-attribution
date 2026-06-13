"""Tests for the FastAPI prompt-risk service.

The real predictor loads a DistilBERT checkpoint at startup, so these tests
override the ``get_predictor`` dependency with a deterministic stub and avoid the
lifespan model load by constructing a ``TestClient`` without entering it.
"""

from __future__ import annotations

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
