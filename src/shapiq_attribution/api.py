"""FastAPI service for prompt-risk classification and SHAPIQ attribution.

The service loads a fine-tuned DistilBERT prompt-risk classifier once at startup
and exposes two endpoints:

- ``POST /predict``: P(unsafe) and a risky/safe label for a prompt.
- ``POST /attribute``: word-level Shapley values explaining the prediction.

Each ``/predict`` call also logs a row of derived numerical features (see
``monitoring.py``) for downstream Evidently drift monitoring; this is
best-effort and never blocks a prediction.

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
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated

import torch
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .model import PromptRiskPredictor
from .monitoring import extract_features, log_prediction
from .web import INDEX_HTML

MODEL_DIR = os.environ.get("MODEL_DIR", "models/prompt_risk_distilbert")
RISK_THRESHOLD = float(os.environ.get("RISK_THRESHOLD", "0.5"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "128"))


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


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, predictor: PredictorDep) -> PredictResponse:
    """Classify the risk of a single prompt."""
    probability = predictor.predict_proba(request.prompt)
    # Monitoring is best-effort: a logging failure must never break a prediction.
    try:
        log_prediction(extract_features(request.prompt, probability))
    except Exception as exc:  # noqa: BLE001 - monitoring is non-critical
        print(f"[api] WARNING: could not log prediction features: {exc}")
    return PredictResponse(
        prompt=request.prompt,
        risk_probability=probability,
        label=_label(probability),
    )


@app.post("/attribute", response_model=AttributeResponse)
def attribute(request: AttributeRequest, predictor: PredictorDep) -> AttributeResponse:
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
    # Monitoring is best-effort: a logging failure must never break a request.
    try:
        log_prediction(extract_features(request.prompt, probability))
    except Exception as exc:  # noqa: BLE001 - monitoring is non-critical
        print(f"[api] WARNING: could not log prediction features: {exc}")
    return AttributeResponse(
        prompt=request.prompt,
        risk_probability=probability,
        label=_label(probability),
        baseline=float(game._baseline),
        words=words,
        top_interactions=pairs[: request.top_interactions],
    )
