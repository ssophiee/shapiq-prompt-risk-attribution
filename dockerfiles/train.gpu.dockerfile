FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON=3.13

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md LICENSE ./

RUN uv sync --python 3.13 --frozen --no-dev --no-install-project

COPY configs configs/
COPY src src/
COPY dvc.yaml dvc.lock ./

RUN uv sync --python 3.13 --frozen --no-dev

ENTRYPOINT ["uv", "run", "python", "-m", "shapiq_attribution.train"]
