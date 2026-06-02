FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV WANDB_MODE=offline

COPY pyproject.toml uv.lock README.md LICENSE ./

RUN uv sync --frozen --no-dev --no-install-project

COPY configs configs/
COPY src src/
COPY dvc.yaml dvc.lock ./

RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "python", "-m", "shapiq_attribution.train"]
