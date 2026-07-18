# Containerization

The project is containerized as **one image per MLOps concern**, orchestrated
with Docker Compose profiles. Code lives in images; data and model artifacts are
mounted as volumes, so the two have independent lifecycles.

## Images

| Image | Dockerfile | Final base | Purpose |
| ----- | ---------- | ---------- | ------- |
| `shapiq-api:latest` | `dockerfiles/api.dockerfile` | `python:3.13-slim` | Serve classifier + SHAPIQ + web UI |
| `shapiq-train:latest` | `dockerfiles/train.dockerfile` | `python:3.13-slim` | CPU training run |
| `shapiq-train-gpu:latest` | `dockerfiles/train.gpu.dockerfile` | `nvidia/cuda:12.4.1-base-ubuntu22.04` | CUDA 12.4 training run |

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
- **Explicit PyTorch variants.** API and CPU-training images install
  `--extra cpu`; the GPU image installs `--extra cu124`. This prevents Linux CI
  and CPU images from pulling CUDA libraries while keeping the GPU build pinned
  to CUDA 12.4.
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

The Compose training services mount `data/`, `models/`, `reports/`, and
`configs/` from the host. Because the processed data is already mounted, they
set `SKIP_DVC_PULL=true`. Their hardware selection is explicit:

| Compose service | `TRAIN_HARDWARE_PROFILE` | Hydra result |
| --------------- | --------------------------- | ------------ |
| `train` | `local` | one auto-selected device, 32-bit |
| `train-gpu` | `single_gpu` | one CUDA GPU, 16-bit mixed precision |

## Training entrypoint

Both training images share `dockerfiles/train-entrypoint.sh`. Its sequence is:

1. Select the Hydra profile from `TRAIN_HARDWARE_PROFILE` (default: `local`).
2. Pull the three processed splits with DVC unless `SKIP_DVC_PULL=true`.
3. Run `python -m shapiq_attribution.train hardware=<profile>` plus any
   additional Hydra overrides passed to the container.
4. If `PUSH_DVC_ON_SUCCESS=true`, commit the configured DVC stage and push its
   outputs.
5. If `DVC_METADATA_BUCKET` is set, upload `dvc.lock` and
   `reports/metrics.json` below `DVC_METADATA_PREFIX`.

The default commit stage is `train_vertex_ddp`. To protect its provenance, the
entrypoint refuses to commit that stage unless
`TRAIN_HARDWARE_PROFILE=ddp`. A Vertex DDP container therefore needs at least:

```bash
TRAIN_HARDWARE_PROFILE=ddp
PUSH_DVC_ON_SUCCESS=true
DVC_COMMIT_STAGE=train_vertex_ddp
```

For online W&B logging, provide `WANDB_API_KEY` directly or set
`WANDB_API_KEY_SECRET` to a full Secret Manager version resource such as
`projects/PROJECT_ID/secrets/SECRET_ID/versions/latest`.

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
