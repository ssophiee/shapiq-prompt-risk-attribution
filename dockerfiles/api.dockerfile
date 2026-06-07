# syntax=docker/dockerfile:1
#
# API image: serves the DistilBERT prompt-risk classifier + SHAPIQ attribution
# (FastAPI/uvicorn) together with the web UI.
#
# The trained model is intentionally NOT baked into the image; it is a
# DVC-tracked artifact mounted at runtime (see docker-compose.yml). This keeps
# code and artifacts on separate lifecycles.
#
# Build:  docker build -t shapiq-api:latest -f dockerfiles/api.dockerfile .
# Run:    docker run -p 8000:8000 -v "$PWD/models:/app/models:ro" shapiq-api:latest

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
RUN uv sync --frozen --no-dev --no-install-project

# 2) Source layer — changes here don't bust the dependency cache above.
COPY src src/
RUN uv sync --frozen --no-dev

EXPOSE 8000

# Liveness/readiness probe (stdlib only — no extra packages needed).
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD uv run python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

ENTRYPOINT ["uv", "run", "uvicorn", "shapiq_attribution.api:app", "--host", "0.0.0.0", "--port", "8000"]
