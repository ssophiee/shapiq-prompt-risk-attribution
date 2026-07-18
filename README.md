# SHAPIQ Attribution for Prompt-Risk Classification

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.13-blue.svg" alt="Python 3.13"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/code%20style-ruff-261230.svg" alt="Code style: ruff"></a>
  <a href="https://github.com/mmschlk/shapiq"><img src="https://img.shields.io/badge/built%20with-shapiq-orange.svg" alt="Built with shapiq"></a>
</p>

A prompt-risk classifier that not only scores how unsafe a prompt is, but explains the
score with Shapley interaction values — surfacing the tokens and token interactions
that drive risky and safe predictions.

<p align="center">
  <img src="reports/image.png" alt="Prompt-Risk Attribution web UI" width="650">
</p>

**Live demo:** <https://shapiq-api-i75daaw2la-ew.a.run.app>

## Overview

1. **Classification.** A fine-tuned DistilBERT model (or Llama Guard 3 1B) estimates
   `P(unsafe)` for an input prompt.
2. **Attribution.** `SafetyAnalysisGame`, a subclass of `shapiq.Game`, wraps the
   classifier so Shapley interaction values can attribute the prediction to
   individual tokens and their interactions.
3. **Serving.** A FastAPI service exposes both, plus a web UI for interactive
   exploration.

## Architecture

```mermaid
flowchart LR
    subgraph Training
        RAW[Safety benchmarks<br/>AdvBench / HarmBench / WildGuard] -->|preprocess + split| DVC[(DVC-tracked<br/>data + model)]
        DVC --> TRAIN[Train DistilBERT<br/>Hydra config]
        TRAIN --> WANDB[W&B experiment tracking]
        TRAIN --> BASE[baseline.csv<br/>feature snapshot]
    end

    subgraph CI/CD
        GH[GitHub Actions<br/>lint + tests] --> BUILD[Cloud Build<br/>Docker image]
        BUILD --> RUN[Cloud Run<br/>FastAPI service]
    end

    DVC -->|model baked into image| BUILD

    subgraph Serving & Monitoring
        USER((User)) -->|/predict, /attribute| RUN
        RUN -->|Prometheus /metrics| METRICS[System metrics]
        RUN -->|feature rows| GCS[(GCS monitoring bucket)]
        RUN --> CM[Cloud Monitoring<br/>5xx + latency alerts]
        GCS -->|fetch| EV[Evidently drift report]
        BASE --> EV
    end
```

## Quickstart

Requires Python 3.13 and [`uv`](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Pull DVC-tracked data and model artifacts (required before serving)
uv run dvc pull
```

Serve the API with Docker:

```bash
docker build -t shapiq-api:latest -f dockerfiles/api.dockerfile .
docker run -p 8000:8000 -v "$PWD/models:/app/models:ro" shapiq-api:latest
```

The web interface is then available at <http://localhost:8000>. See [API.md](API.md)
for the API endpoints and request/response schemas, and
[deploy/README.md](deploy/README.md) for the GCP deployment runbook.

## Monitoring

- **System metrics** — Prometheus counters and latency histograms at `/metrics`,
  plus the [Cloud Run metrics tab](https://console.cloud.google.com/run/detail/europe-west1/shapiq-api/metrics?project=mlops-shapiq-project).
- **Alerts** — Cloud Monitoring email policies for 5xx responses and p95 latency
  ([deploy/alerts.sh](deploy/alerts.sh)).
- **Data drift** — live predictions are logged to a GCS bucket and compared against
  the training baseline with Evidently at
  [/monitoring](https://shapiq-api-i75daaw2la-ew.a.run.app/monitoring).

## Dataset

Prompts come from public safety benchmarks (AdvBench, HarmBench, WildGuard),
normalized to a shared JSONL schema and split into train/val/test — all
DVC-tracked.

## Project structure

```text
├── configs/                  # Hydra configuration files
├── data/                     # Raw and processed datasets (DVC-tracked)
├── deploy/                   # Cloud Run deployment and alerting scripts
├── dockerfiles/              # Training and API Dockerfiles
├── docs/                     # MkDocs documentation
├── models/                   # Trained model artifacts (DVC-tracked)
├── notebooks/                # Exploratory notebooks
├── reports/                  # Metrics, reports, and figures
├── src/shapiq_attribution/   # Project package
├── tests/                    # Unit tests
├── pyproject.toml
└── tasks.py
```

## Tech stack

- **Model** — DistilBERT (Hugging Face Transformers), configured with Hydra
- **Explainability** — [shapiq](https://github.com/mmschlk/shapiq) Shapley interaction values
- **Data & model versioning** — DVC
- **Experiment tracking** — Weights & Biases
- **API** — FastAPI + Docker
- **CI/CD** — GitHub Actions + Cloud Build → Cloud Run
- **Monitoring** — Prometheus metrics, Cloud Monitoring alerts, Evidently drift reports
- **Quality** — pytest, ruff, mypy, pre-commit

## Development

```bash
uv run pytest tests/              # run the test suite
uv run ruff check . --fix         # lint and autofix
uv run ruff format .              # format
```

## License and acknowledgements

Released under the [MIT License](LICENSE). Built with
[shapiq](https://github.com/mmschlk/shapiq) and based on
[mlops_template](https://github.com/SkafteNicki/mlops_template).
