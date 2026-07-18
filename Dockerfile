# syntax=docker/dockerfile:1
#
# Cloud Run image: same FastAPI service as dockerfiles/api.dockerfile, but with
# the trained model BAKED IN so the container is fully self-contained.
#
# Why a separate Dockerfile at the repo root?
#   - `gcloud run deploy --source .` auto-detects a root `Dockerfile`, so this is
#     the one Cloud Build uses. The local-dev images stay in dockerfiles/.
#   - Cloud Run has no persistent local volumes, so we cannot mount ./models the
#     way docker-compose does. The model is copied into the image instead.
#
# The model lives at models/prompt_risk_distilbert and is DVC-tracked. For the
# COPY below to work via `--source .`, that path must survive BOTH filters:
#   - .gcloudignore  (what gcloud uploads to Cloud Build)
#   - .dockerignore  (what the Docker build sees)
# Both are configured to keep models/prompt_risk_distilbert. Run `dvc pull` first
# if the directory is just a .dvc pointer locally.
#
# Deploy:  ./deploy/cloudrun.sh         (wraps `gcloud run deploy --source .`)

FROM python:3.13-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    MODEL_DIR=/app/models/prompt_risk_distilbert \
    RISK_THRESHOLD=0.5 \
    DEVICE=cpu

# Compiler toolchain: shapiq has no prebuilt wheel for Python 3.13, so it is
# compiled from source during `uv sync`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# 1) Dependency layer — cached unless pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project --extra cpu

# 2) Source layer — changes here don't bust the dependency cache above.
COPY src src/
RUN uv sync --frozen --no-dev --extra cpu

# 3) Model layer — baked in (unlike the local image, which mounts it as a volume).
COPY models/prompt_risk_distilbert /app/models/prompt_risk_distilbert

# Cloud Run injects $PORT (default 8080) and routes traffic to it. Use shell form
# so $PORT expands; `exec` makes uvicorn PID 1 for clean shutdown signals.
# (Cloud Run runs its own startup/liveness probes, so no Docker HEALTHCHECK here.)
# Start the venv binary directly: `uv run` would re-sync the environment on
# every container start — pulling the dev group (mypy, ruff, mkdocs, ...) that
# the image was built without and adding ~85 s to every cold start.
CMD exec /app/.venv/bin/uvicorn shapiq_attribution.api:app --host 0.0.0.0 --port ${PORT:-8080}
