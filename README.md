# SHAPIQ Attribution for Prompt-Risk Classification

<p align="center">
  <a href="https://github.com/ssophiee/shapiq-prompt-risk-attribution/actions/workflows/ci-lint.yaml"><img src="https://github.com/ssophiee/shapiq-prompt-risk-attribution/actions/workflows/ci-lint.yaml/badge.svg" alt="Lint"></a>
  <a href="https://github.com/ssophiee/shapiq-prompt-risk-attribution/actions/workflows/ci-api.yaml"><img src="https://github.com/ssophiee/shapiq-prompt-risk-attribution/actions/workflows/ci-api.yaml/badge.svg" alt="API CI"></a>
  <a href="https://github.com/ssophiee/shapiq-prompt-risk-attribution/actions/workflows/ci-train.yaml"><img src="https://github.com/ssophiee/shapiq-prompt-risk-attribution/actions/workflows/ci-train.yaml/badge.svg" alt="Train CI"></a>
  <a href="https://ssophiee.github.io/shapiq-prompt-risk-attribution/"><img src="https://img.shields.io/badge/docs-MkDocs%20Material-526CFE.svg" alt="Documentation"></a>
</p>
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

<p align="center">
  <b><a href="https://shapiq-api-268593597387.europe-west1.run.app">Live demo</a></b>
  ·
  <b><a href="https://ssophiee.github.io/shapiq-prompt-risk-attribution/">Documentation</a></b>
  ·
  <b><a href="https://shapiq-api-268593597387.europe-west1.run.app/docs">API reference</a></b>
  ·
  <b><a href="https://shapiq-api-268593597387.europe-west1.run.app/monitoring">Drift dashboard</a></b>
</p>

## Overview

1. **Classification.** A fine-tuned DistilBERT model estimates `P(unsafe)` for an
   input prompt.
2. **Attribution.** `SafetyAnalysisGame`, a subclass of `shapiq.Game`, wraps the
   classifier so Shapley interaction values can attribute the prediction to
   individual tokens and their interactions.
3. **Serving.** A FastAPI service on Cloud Run exposes both, plus a web UI for
   interactive exploration — with drift detection and system monitoring built in.

## Architecture

![Architecture overview](reports/figures/architecture.png)

## Quickstart

The only prerequisite is [`uv`](https://docs.astral.sh/uv/) — it installs Python
and every dependency for you.

**1. Clone and install**

```bash
git clone git@github.com:ssophiee/shapiq-prompt-risk-attribution.git
cd shapiq-prompt-risk-attribution
uv sync
```

**2. Verify the setup** — the test suite runs fully offline, no data needed:

```bash
uv run pytest tests/
```

**3. Fetch the data and trained model** (requires access to the GCS bucket):

```bash
uv run dvc pull
```

**4. Start the app** and open <http://localhost:8000>:

```bash
uv run uvicorn shapiq_attribution.api:app --port 8000
```

For the full setup guide, Docker usage and cloud deployment, head to the
**[documentation](https://ssophiee.github.io/shapiq-prompt-risk-attribution/)**.

<!-- ## Monitoring

- **System metrics** — Prometheus counters and latency histograms at `/metrics`,
  plus the [Cloud Run metrics tab](https://console.cloud.google.com/run/detail/europe-west1/shapiq-api/metrics?project=mlops-shapiq-project).
- **Alerts** — Cloud Monitoring email policies for 5xx responses and p95 latency
  ([deploy/alerts.sh](deploy/alerts.sh)).
- **Data drift** — live predictions are logged to a GCS bucket and compared against
  the training baseline with Evidently at
  [/monitoring](https://shapiq-api-268593597387.europe-west1.run.app/monitoring). -->

## Dataset

The classifier is trained on prompts from five public safety benchmarks:

| Source | Contributes |
| --- | --- |
| [AdvBench](https://huggingface.co/datasets/walledai/AdvBench) | harmful instructions (risky) |
| [HarmBench](https://huggingface.co/datasets/walledai/HarmBench) | harmful behaviors (risky) |
| [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix) | prompts labeled harmful / unharmful |
| [ToxicChat](https://huggingface.co/datasets/lmsys/toxic-chat) | real user chats labeled for toxicity and jailbreaks |
| [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | prompts with safe / unsafe labels |

All sources are normalized to one JSONL schema (`prompt`, `label`, `source`),
deduplicated, and split into train/val/test. Data and the trained model are
versioned with DVC, stored on Google Cloud Storage.

## Project structure

```text
├── configs/                  # Hydra configs (train, sweep, hardware profiles)
├── data/                     # Raw and processed datasets (DVC-tracked)
├── deploy/                   # Cloud Run deployment and alerting scripts
├── dockerfiles/              # Training and API Dockerfiles
├── docs/                     # MkDocs documentation site
├── models/                   # Trained model artifacts (DVC-tracked)
├── notebooks/                # Exploratory notebooks
├── reports/                  # Metrics, reports, and figures
├── src/shapiq_attribution/   # Project package
├── tests/                    # Offline test suite
├── locustfile.py             # Load test
├── pyproject.toml
└── tasks.py                  # invoke task catalog
```

## Tech stack

- **Model** — DistilBERT (Hugging Face Transformers), trained with PyTorch Lightning, configured with Hydra
- **Explainability** — [shapiq](https://github.com/mmschlk/shapiq) Shapley interaction values
- **Data & model versioning** — DVC on Google Cloud Storage
- **Experiment tracking** — Weights & Biases (incl. Bayesian hyperparameter sweep)
- **API** — FastAPI + Docker, deployed on Cloud Run via Cloud Build
- **CI/CD** — GitHub Actions (lint, tests, image builds)
- **Monitoring** — Prometheus metrics, Cloud Monitoring alerts, Evidently drift reports, Locust load testing
- **Quality** — pytest, ruff, mypy, pre-commit

## Development

```bash
uv run invoke --list              # all project tasks
uv run pytest tests/              # run the test suite
uv run ruff check . --fix         # lint and autofix
uv run ruff format .              # format
uv run invoke serve-docs          # preview the documentation site
```

## License and acknowledgements

Released under the [MIT License](LICENSE). Built with
[shapiq](https://github.com/mmschlk/shapiq) and based on
[mlops_template](https://github.com/SkafteNicki/mlops_template).
