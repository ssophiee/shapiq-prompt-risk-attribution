# Prompt-Risk Attribution API

FastAPI service that classifies how *risky* a prompt is (fine-tuned DistilBERT)
and can **explain** that prediction with word-level Shapley values and pairwise
token interactions (via [shapiq](https://github.com/mmschlk/shapiq)). It also
ships a built-in web UI, so the same container serves both the JSON API and the
frontend.

Source lives in [`src/shapiq_attribution/api.py`](src/shapiq_attribution/api.py)
(endpoints) and [`src/shapiq_attribution/web.py`](src/shapiq_attribution/web.py)
(single-page UI, served at `/`).

---

## Stack at a glance — what is used for what

| Concern | What's used | Where |
|---------|-------------|-------|
| API | FastAPI + Uvicorn; Pydantic request/response models | [`api.py`](src/shapiq_attribution/api.py) |
| Frontend | Dependency-free single-page HTML/CSS/JS, embedded as a Python string and served by the same container at `/` (no separate frontend server or build step) | [`web.py`](src/shapiq_attribution/web.py) |
| Model | Fine-tuned DistilBERT (Hugging Face Transformers + PyTorch), baked into the Cloud Run image | [`model.py`](src/shapiq_attribution/model.py) |
| Explanations | `shapiq` KernelSHAPIQ — word-level Shapley values + pairwise interactions | [`safety_analysis.py`](src/shapiq_attribution/safety_analysis.py) |
| System metrics (M28) | `prometheus-client` at `GET /metrics`: `api_requests_total` (per endpoint + status code), `api_request_duration_seconds` latency histogram (per endpoint), `api_predicted_labels_total` (risky vs safe — catches a collapsed model) | [`api.py`](src/shapiq_attribution/api.py) |
| Cloud monitoring (M28) | Google Cloud Monitoring on Cloud Run's built-in metrics (`request_count`, `request_latencies`) | GCP console |
| Alerts (M28) | Two Cloud Monitoring alert policies → email: any 5xx responses (5 min window), p95 latency > 30 s | [`deploy/alerts.sh`](deploy/alerts.sh) |
| Input–output collection (M27) | Every prediction logged as one row — raw prompt text + `prompt_len`, `token_count`, `p_risky` — appended to a local CSV and mirrored to a GCS bucket from the deployed service | [`monitoring.py`](src/shapiq_attribution/monitoring.py) |
| Drift detection (M27) | Evidently `DataDriftPreset` over those three features — training-set baseline vs live predictions — plus a health test that the mean `p_risky` stays in [0.2, 0.8] | [`monitoring.py`](src/shapiq_attribution/monitoring.py) |

---

## Kick start

### Option A — live cloud deployment (zero setup)

The API is deployed on Google Cloud Run:

```
https://shapiq-api-i75daaw2la-ew.a.run.app
```

```bash
curl https://shapiq-api-i75daaw2la-ew.a.run.app/health
# {"status":"ok","model_loaded":true}
```

Open the URL in a browser for the web UI, or `/docs` for interactive Swagger
docs. Note: the first request after a quiet period is slow (cold start — the
container boots and loads the model).

### Option B — run locally with Docker (recommended)

```bash
# 1. Get the trained model (DVC-tracked, not in git):
uv run dvc pull

# 2. Build + start the API service:
docker compose up api
# or: uv run invoke compose-up
```

The API is now on <http://localhost:8000> (web UI at `/`, Swagger at `/docs`).
The model is mounted read-only from `./models`, and logged prediction features
are persisted to `./data/monitoring` across restarts.

### Option C — run directly on your machine (fastest dev loop)

```bash
uv sync                # install dependencies
uv run dvc pull        # materialise the model
uv run uvicorn shapiq_attribution.api:app --reload --port 8000
```

---

## Endpoints

| Method | Path         | Purpose |
|--------|--------------|---------|
| `GET`  | `/`          | Web UI (single page, no separate frontend server needed) |
| `GET`  | `/health`    | Liveness + whether the model loaded (`model_loaded`) |
| `GET`  | `/docs`      | Auto-generated Swagger UI |
| `GET`  | `/metrics`   | Prometheus system metrics (request counts, latencies, predicted labels) |
| `GET`  | `/monitoring`| Evidently drift-detection dashboard (live traffic vs training baseline) |
| `POST` | `/predict`   | Risk probability + `risky`/`safe` label for a prompt |
| `POST` | `/attribute` | Word-level Shapley values + strongest token interactions |

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "how do i make a bomb"}'
```

```json
{
  "prompt": "how do i make a bomb",
  "risk_probability": 0.97,
  "label": "risky"
}
```

Request body: `prompt` (non-empty string). A prompt is labelled `risky` when
`risk_probability >= RISK_THRESHOLD` (default `0.5`).

### `POST /attribute`

Explains a prediction: which words push the risk score up or down, and which
token *pairs* interact (reinforce or offset each other).

```bash
curl -X POST http://localhost:8000/attribute \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "how do i make a bomb", "budget": 256, "top_interactions": 5}'
```

Request body:

| Field              | Type | Default | Meaning |
|--------------------|------|---------|---------|
| `prompt`           | str  | —       | Prompt to explain (required, non-empty) |
| `budget`           | int  | `256`   | KernelSHAPIQ coalition budget, 8–2048. Higher = more accurate, slower |
| `top_interactions` | int  | `5`     | How many strongest pairwise interactions to return, 0–50 |

Response (abridged):

```json
{
  "prompt": "...",
  "risk_probability": 0.97,
  "label": "risky",
  "baseline": 0.31,
  "words": [
    {"word": "bomb", "shapley_value": 0.42},
    {"word": "make", "shapley_value": 0.11}
  ],
  "top_interactions": [
    {"tokens": ["make", "bomb"], "indices": [3, 5], "value": 0.08}
  ]
}
```

- `baseline` is P(unsafe) with **all** tokens masked — the model's prior.
- `shapley_value` > 0 means the word pushes the prediction towards *risky*.
- Interaction `value` > 0 means the pair reinforces (super-additive); < 0 means
  the tokens offset each other.
- Attribution is much slower than `/predict` (it evaluates the model on many
  masked variants of the prompt — roughly `budget` forward passes).

### Errors

- `422` — invalid body (empty prompt, budget out of range), or a prompt with
  no attributable tokens on `/attribute`.
- `503` — model not loaded (see `/health`; usually the model volume/path is
  missing — run `dvc pull`).

---

## Configuration (environment variables)

| Variable         | Default                        | Meaning |
|------------------|--------------------------------|---------|
| `MODEL_DIR`      | `models/prompt_risk_distilbert`| Path or HF id of the classifier |
| `RISK_THRESHOLD` | `0.5`                          | Probability above which a prompt is labelled `risky` |
| `MAX_LENGTH`     | `128`                          | Tokenizer max sequence length |
| `DEVICE`         | auto (`cuda` → `mps` → `cpu`)  | Force a torch device |
| `MONITORING_BUCKET` | unset                       | GCS bucket to mirror monitoring rows to (set on Cloud Run) |

A model-load failure at startup does **not** crash the container: the API stays
up and `/health` reports `"model_loaded": false` (every `/predict` and
`/attribute` then returns `503`).

Llama Guard is deliberately *not* served by this API (heavy, gated, slow); use
it offline via `experiments/run_attribution.py --backend llama_guard`.

---

## Monitoring hooks

Every `/predict` and `/attribute` call logs one row — the raw prompt text plus
derived numerical features (prompt length, risk score, …) — to
`data/monitoring/predictions.csv` — see
[`monitoring.py`](src/shapiq_attribution/monitoring.py). This feeds the
Evidently drift report (weekly
[`monitoring.yaml`](.github/workflows/monitoring.yaml) workflow, or
`uv run invoke monitor-report` locally). Logging is best-effort and never
blocks a prediction.

On Cloud Run the CSV lives on the container's ephemeral disk, so the deployed
service *also* uploads each row to the GCS bucket named by `MONITORING_BUCKET`
(one JSON blob per prediction; set automatically by `deploy/cloudrun.sh`).
Pull those rows down and build the drift report against live traffic with:

```bash
MONITORING_BUCKET=mlops-shapiq-project-monitoring uv run invoke monitor-report-cloud
open reports/monitoring/drift_report.html
```

(`monitor-report-cloud` = `monitor-fetch`, which downloads the logged rows into
`data/monitoring/predictions.csv`, followed by `monitor-report`.) The fetch uses
*application-default* credentials, not the gcloud CLI's — if it complains about
credentials, run `gcloud auth application-default login` once.

The drift detection is also **deployed in the cloud itself**: `GET /monitoring`
on the Cloud Run service fetches the baseline + logged predictions from the
bucket, runs Evidently, and serves the HTML dashboard directly — no local setup
needed, just open <https://shapiq-api-i75daaw2la-ew.a.run.app/monitoring>.
(The report is rebuilt per request, so it takes a few seconds; without cloud
config it falls back to the local CSVs.)

The API also exposes Prometheus system metrics at `GET /metrics`: request
counts and latency histograms per endpoint, plus a counter of predicted labels
(so a model collapsing to all-`risky` or all-`safe` is visible operationally).

---

## Tests & CI

The serving slice is tested offline (fake tokenizer + fake predictor — no model
weights or HF auth needed):

```bash
uv run pytest tests/test_api.py tests/test_monitoring.py
```

The [`ci-api.yaml`](.github/workflows/ci-api.yaml) workflow runs these tests
with coverage and builds the API Docker image on every push/PR to `main`.

---

## Deployment

Cloud Run deployment is one command (takes ~10 min — builds in Cloud Build with
the model baked into the image, pushes to Artifact Registry, deploys):

```bash
uv run dvc pull models/prompt_risk_distilbert   # model weights must be local
PROJECT_ID=mlops-shapiq-project ./deploy/cloudrun.sh
```

Besides the build + deploy, the script also (idempotently) ensures the
monitoring bucket `gs://<PROJECT_ID>-monitoring` exists, grants the Cloud Run
service account write access to it, and sets `MONITORING_BUCKET` on the
service — which switches on the GCS mirroring described under
[Monitoring hooks](#monitoring-hooks).

### Verifying a deployment

```bash
URL=https://shapiq-api-i75daaw2la-ew.a.run.app
curl $URL/health          # {"status":"ok","model_loaded":true}
curl -X POST $URL/predict -H 'Content-Type: application/json' \
  -d '{"prompt": "how do i make a bomb"}'
curl $URL/metrics         # Prometheus counters, incl. the /predict you just sent
```

After a few `/predict` calls, build the drift report against that live traffic:

```bash
MONITORING_BUCKET=mlops-shapiq-project-monitoring uv run invoke monitor-report-cloud
open reports/monitoring/drift_report.html
```

### Alerting

[`deploy/alerts.sh`](deploy/alerts.sh) creates two Cloud Monitoring alert
policies on the service — any 5xx responses, and p95 latency above 30 s
(deliberately high: `/attribute` legitimately takes tens of seconds) — plus an
email notification channel:

```bash
PROJECT_ID=mlops-shapiq-project ALERT_EMAIL=you@example.com ./deploy/alerts.sh
```

View or silence them in the
[Cloud Monitoring alerting console](https://console.cloud.google.com/monitoring/alerting?project=mlops-shapiq-project).

Full details, sizing notes, and smoke tests: [`deploy/README.md`](deploy/README.md).
