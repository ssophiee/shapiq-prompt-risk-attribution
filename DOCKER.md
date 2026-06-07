# Containerization

The project is containerized as **one image per MLOps concern**, orchestrated
with Docker Compose profiles. Code lives in images; data and model artifacts are
mounted as volumes, so the two have independent lifecycles.

## Images

| Image                  | Dockerfile                       | Base                      | Purpose                              |
| ---------------------- | -------------------------------- | ------------------------- | ------------------------------------ |
| `shapiq-api:latest`       | `dockerfiles/api.dockerfile`     | `python:3.13-slim`        | Serve classifier + SHAPIQ + web UI   |
| `shapiq-train:latest`     | `dockerfiles/train.dockerfile`   | `python:3.13-slim` (uv)   | CPU training run                     |
| `shapiq-train-gpu:latest` | `dockerfiles/train.gpu.dockerfile` | `pytorch/...cuda12.4`   | GPU training run                     |

> **Why Debian slim, not Alpine?** PyTorch ships only `glibc` (manylinux) wheels.
> Alpine uses `musl`, so `pip install torch` there falls back to a source build
> and fails. `python:3.13-slim` is Debian, so the prebuilt wheels install
> cleanly.
>
> **Why `build-essential` in the API image?** `shapiq` has no prebuilt wheel for
> Python 3.13, so `uv sync` compiles it from source — which needs a C/C++
> compiler.

## Design choices (talking points)

- **Modular, one concern per image.** Serving and training never share a runtime,
  so each stays minimal and independently rebuildable.
- **Layer caching.** Dependencies are installed (`uv sync ... --no-install-project`)
  *before* source is copied. Editing `src/` reuses the cached dependency layer, so
  rebuilds take seconds instead of re-downloading torch.
- **Lean runtime.** `uv sync --no-dev` excludes test/lint/docs tooling from the
  shipped image.
- **Artifacts as volumes, not layers.** The ~270 MB model is DVC-tracked and
  mounted read-only at run time. The image carries code only; swapping models
  needs no rebuild.
- **Resilient startup + healthcheck.** If the model volume is missing the API
  stays up and reports `model_loaded: false` on `/health` (no crash loop). A
  `HEALTHCHECK` polls `/health` using only the Python stdlib.
- **Config via environment.** `MODEL_DIR`, `RISK_THRESHOLD`, `DEVICE`,
  `MAX_LENGTH` are read from the environment (12-factor), set in Compose.

## Usage

```bash
# --- Serving (default profile) ---
docker compose up api                 # build + run, http://localhost:8000
# or plain docker:
docker build -t shapiq-api:latest -f dockerfiles/api.dockerfile .
docker run -p 8000:8000 -v "$PWD/models:/app/models:ro" shapiq-api:latest

# --- Training ---
docker compose --profile train up train         # CPU
docker compose --profile gpu  up train-gpu      # CUDA host only

# --- invoke shortcuts ---
uv run invoke docker-build-api      # build the API image
uv run invoke docker-run-api        # run it with the model mounted
uv run invoke compose-up            # docker compose up --build api
```

## Architecture

```
                      docker compose
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                    ▼
   ┌─────────┐        ┌──────────┐        ┌────────────┐
   │   api   │        │  train   │        │ train-gpu  │
   │ :8000   │        │ (profile │        │ (profile   │
   │ FastAPI │        │  train)  │        │   gpu)     │
   └────┬────┘        └────┬─────┘        └─────┬──────┘
        │ ro                │ rw                 │ rw
        ▼                   ▼                    ▼
   ./models           ./data ./models ./reports ./configs   (host volumes)
```
