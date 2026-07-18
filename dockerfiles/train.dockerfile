FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV WANDB_MODE=offline

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY .dvc/config .dvc/config
COPY dockerfiles/train-entrypoint.sh /usr/local/bin/train-entrypoint

RUN chmod +x /usr/local/bin/train-entrypoint

RUN uv sync --frozen --no-dev --no-install-project --extra cpu

COPY configs configs/
COPY src src/
COPY dvc.yaml dvc.lock ./

RUN uv sync --frozen --no-dev --extra cpu

RUN /app/.venv/bin/dvc config core.no_scm true

ENTRYPOINT ["/usr/local/bin/train-entrypoint"]
