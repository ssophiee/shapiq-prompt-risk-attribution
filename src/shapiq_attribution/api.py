"""FastAPI service for prompt-risk classification and SHAPIQ attribution.

The service loads a fine-tuned DistilBERT prompt-risk classifier once at startup
and exposes two endpoints:

- ``POST /predict``: P(unsafe) and a risky/safe label for a prompt.
- ``POST /attribute``: word-level Shapley values explaining the prediction.
- ``GET /monitoring``: Evidently drift-detection dashboard over logged predictions.

Each ``/predict`` call also logs one monitoring row — the raw prompt plus
derived numerical features (see ``monitoring.py``) — for input–output
collection and downstream Evidently drift monitoring; this is
best-effort and never blocks a prediction. When ``MONITORING_BUCKET`` is set
(as on Cloud Run, where the local CSV is ephemeral) the same row is uploaded to
that GCS bucket in a background task after the response is sent.

``GET /metrics`` exposes Prometheus system metrics: request counts and latency
histograms per endpoint (via HTTP middleware) plus a counter of predicted
labels, so a collapsed model (all-risky / all-safe) is visible operationally.

Llama Guard is intentionally *not* served here: it is heavy, gated, and slow.
Experiment with it in notebooks / scripts via
``safety_analysis.SafetyAnalysisGame(..., backend="llama_guard")`` or
``experiments/run_attribution.py --backend llama_guard``.

Configuration via environment variables:

- ``MODEL_DIR``: path or HF id of the classifier (default
  ``models/prompt_risk_distilbert``).
- ``RISK_THRESHOLD``: probability above which a prompt is labelled risky
  (default ``0.5``).
- ``MAX_LENGTH``: tokenizer max sequence length (default ``128``).
- ``DEVICE``: force a torch device (``cuda``/``mps``/``cpu``); auto-detected if unset.
- ``MONITORING_BUCKET``: GCS bucket to mirror monitoring rows to (unset = local CSV only).
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from .model import PromptRiskPredictor
from .monitoring import extract_features, log_prediction, log_prediction_gcs
from .web import INDEX_HTML

MODEL_DIR = os.environ.get("MODEL_DIR", "models/prompt_risk_distilbert")
RISK_THRESHOLD = float(os.environ.get("RISK_THRESHOLD", "0.5"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "128"))
MONITORING_BUCKET = os.environ.get("MONITORING_BUCKET")

# ---- System metrics (M28), scraped from GET /metrics ------------------------
REQUEST_COUNT = Counter(
    "api_requests_total",
    "HTTP requests served, by endpoint and status code.",
    ["endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "Wall-clock request latency per endpoint.",
    ["endpoint"],
)
PREDICTED_LABELS = Counter(
    "api_predicted_labels_total",
    "Predictions served, by risk label.",
    ["label"],
)


def _select_device() -> str:
    """Pick a torch device, honouring the ``DEVICE`` env override."""
    override = os.environ.get("DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _State:
    """Holds the loaded predictor for the lifetime of the app."""

    predictor: PromptRiskPredictor | None = None


state = _State()


def _label(probability: float) -> str:
    """Return the risk label for a probability."""
    return "risky" if probability >= RISK_THRESHOLD else "safe"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the predictor at startup and release it at shutdown.

    A load failure (e.g. the model volume is not mounted) is logged rather than
    raised, so the container stays up and ``/health`` reports ``model_loaded:
    false`` instead of crash-looping.
    """
    try:
        state.predictor = PromptRiskPredictor.from_pretrained(
            MODEL_DIR,
            max_length=MAX_LENGTH,
            device=_select_device(),
        )
    except Exception as exc:  # noqa: BLE001 - surfaced via /health, not fatal
        print(f"[api] WARNING: could not load model from {MODEL_DIR!r}: {exc}")
        state.predictor = None
    yield
    state.predictor = None


app = FastAPI(
    title="Prompt-Risk Attribution API",
    description="Classify prompt risk and explain predictions with Shapley interaction values.",
    version="0.0.1",
    lifespan=lifespan,
)

# Prometheus exposition endpoint; the metrics themselves are recorded below.
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def record_request_metrics(request: Request, call_next) -> Response:
    """Record count and latency for every request, labelled by path and status."""
    start = time.perf_counter()
    response = await call_next(request)
    endpoint = request.url.path
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)
    REQUEST_COUNT.labels(endpoint=endpoint, status_code=str(response.status_code)).inc()
    return response


def _log_monitoring(prompt: str, probability: float) -> None:
    """Log one monitoring row locally and, if configured, to GCS.

    Best-effort by design: this runs as a background task after the response is
    sent, and a monitoring failure must never surface to the client.
    """
    try:
        features = extract_features(prompt, probability)
        log_prediction(features)
        if MONITORING_BUCKET:
            log_prediction_gcs(features, MONITORING_BUCKET)
    except Exception as exc:  # noqa: BLE001 - monitoring is non-critical
        print(f"[api] WARNING: could not log prediction features: {exc}")


def get_predictor() -> PromptRiskPredictor:
    """Return the loaded predictor or raise 503 if it is unavailable."""
    if state.predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return state.predictor


PredictorDep = Annotated[PromptRiskPredictor, Depends(get_predictor)]


class PredictRequest(BaseModel):
    """Request body for prompt-risk prediction."""

    prompt: str = Field(..., min_length=1, description="Prompt text to classify.")


class PredictResponse(BaseModel):
    """Prompt-risk prediction result."""

    prompt: str
    risk_probability: float = Field(..., description="P(unsafe) in [0, 1].")
    label: str = Field(..., description="'risky' or 'safe'.")


class AttributeRequest(BaseModel):
    """Request body for SHAPIQ attribution."""

    prompt: str = Field(..., min_length=1, description="Prompt text to explain.")
    budget: int = Field(256, ge=8, le=2048, description="KernelSHAPIQ coalition budget.")
    top_interactions: int = Field(5, ge=0, le=50, description="Number of strongest pairwise interactions to return.")


class WordAttribution(BaseModel):
    """Shapley value for a single word."""

    word: str
    shapley_value: float


class Interaction(BaseModel):
    """Pairwise k-SII interaction between two tokens."""

    tokens: list[str] = Field(..., description="The two interacting token strings.")
    indices: list[int] = Field(..., description="Their token positions.")
    value: float = Field(
        ...,
        description="k-SII interaction value: positive = reinforcing (super-additive), "
        "negative = offsetting (sub-additive).",
    )


class AttributeResponse(BaseModel):
    """Word-level Shapley attribution for a prompt."""

    prompt: str
    risk_probability: float
    label: str
    baseline: float = Field(..., description="P(unsafe) with all tokens masked.")
    words: list[WordAttribution]
    top_interactions: list[Interaction] = Field(
        default_factory=list, description="Strongest pairwise token interactions by magnitude."
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the interactive web interface."""
    return INDEX_HTML


@app.get("/health")
def health() -> dict[str, object]:
    """Report service health and whether the model is loaded."""
    return {"status": "ok", "model_loaded": state.predictor is not None}


@app.get("/monitoring", response_class=HTMLResponse)
def monitoring_report() -> str:
    """Serve the Evidently drift-detection dashboard (M27).

    Compares the feature distribution of logged predictions against the
    training-set baseline and renders the result as HTML. When
    ``MONITORING_BUCKET`` is set (Cloud Run) both sides come from GCS — the
    baseline uploaded at deploy time and the newest logged prediction rows
    (capped so the fetch stays well under the Cloud Run request timeout);
    otherwise the local CSVs under ``data/monitoring`` are used.

    The report is rebuilt on every call, so it can take a few seconds.
    """
    import tempfile

    from .monitoring import (
        BASELINE_CSV,
        PREDICTIONS_CSV,
        REPORT_HTML,
        fetch_baseline_gcs,
        fetch_predictions_gcs,
        generate_report,
    )

    if MONITORING_BUCKET:
        workdir = Path(tempfile.mkdtemp(prefix="monitoring-"))
        try:
            reference_csv = fetch_baseline_gcs(MONITORING_BUCKET, workdir / "baseline.csv")
            current_csv, row_count = fetch_predictions_gcs(MONITORING_BUCKET, workdir / "predictions.csv")
        except Exception as exc:  # noqa: BLE001 - surface GCS trouble as 503, not a stack trace
            raise HTTPException(status_code=503, detail=f"Could not fetch monitoring data: {exc}") from exc
        if row_count == 0:
            raise HTTPException(status_code=404, detail="No predictions logged yet; call /predict first.")
        out_path = workdir / "drift_report.html"
    else:
        reference_csv, current_csv, out_path = BASELINE_CSV, PREDICTIONS_CSV, REPORT_HTML
        if not reference_csv.exists() or not current_csv.exists():
            raise HTTPException(
                status_code=404,
                detail="Local monitoring CSVs missing; run `invoke build-baseline` and call /predict first.",
            )
    return generate_report(reference_csv, current_csv, out_path).read_text(encoding="utf-8")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, predictor: PredictorDep, background_tasks: BackgroundTasks) -> PredictResponse:
    """Classify the risk of a single prompt."""
    probability = predictor.predict_proba(request.prompt)
    label = _label(probability)
    PREDICTED_LABELS.labels(label=label).inc()
    # Monitoring I/O (CSV + optional GCS upload) happens after the response is sent.
    background_tasks.add_task(_log_monitoring, request.prompt, probability)
    return PredictResponse(
        prompt=request.prompt,
        risk_probability=probability,
        label=label,
    )


@app.post("/attribute", response_model=AttributeResponse)
def attribute(
    request: AttributeRequest, predictor: PredictorDep, background_tasks: BackgroundTasks
) -> AttributeResponse:
    """Explain a prompt-risk prediction with word-level Shapley values."""
    # Imported lazily so /predict and /health work even if shapiq is unavailable.
    from .safety_analysis import (
        SafetyAnalysisGame,
        aggregate_to_words,
        run_safety_shapiq,
    )

    game = SafetyAnalysisGame(request.prompt, model=predictor, backend="distilbert")
    if game.n_players == 0:
        raise HTTPException(status_code=422, detail="Prompt has no attributable tokens.")

    sii = run_safety_shapiq(game, budget=request.budget)
    word_sv, word_names = aggregate_to_words(sii, game.token_names)
    words = [
        WordAttribution(word=name, shapley_value=float(word_sv[(index,)])) for index, name in enumerate(word_names)
    ]

    token_names = game.token_names
    n = len(token_names)
    pairs = [
        Interaction(
            tokens=[token_names[i], token_names[j]],
            indices=[i, j],
            value=float(sii[(i, j)]),
        )
        for i in range(n)
        for j in range(i + 1, n)
    ]
    pairs.sort(key=lambda pair: abs(pair.value), reverse=True)

    probability = predictor.predict_proba(request.prompt)
    label = _label(probability)
    PREDICTED_LABELS.labels(label=label).inc()
    # Monitoring I/O (CSV + optional GCS upload) happens after the response is sent.
    background_tasks.add_task(_log_monitoring, request.prompt, probability)
    return AttributeResponse(
        prompt=request.prompt,
        risk_probability=probability,
        label=label,
        baseline=float(game._baseline),
        words=words,
        top_interactions=pairs[: request.top_interactions],
    )
