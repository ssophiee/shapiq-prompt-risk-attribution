FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS uv

FROM nvidia/cuda:12.4.1-base-ubuntu22.04

COPY --from=uv /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_NO_CACHE=1


RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*



COPY pyproject.toml uv.lock README.md LICENSE ./
COPY .dvc/config .dvc/config
COPY dockerfiles/train-entrypoint.sh /usr/local/bin/train-entrypoint

RUN chmod +x /usr/local/bin/train-entrypoint

RUN uv sync --python 3.13 --frozen --no-dev --no-install-project --extra cu124

COPY configs configs/
COPY src src/
COPY dvc.yaml dvc.lock ./

RUN uv sync --python 3.13 --frozen --no-dev --extra cu124

RUN /app/.venv/bin/dvc config core.no_scm true

ENTRYPOINT ["/usr/local/bin/train-entrypoint"]
