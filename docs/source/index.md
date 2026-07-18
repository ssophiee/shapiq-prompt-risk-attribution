# shapiq_attribution

**Prompt-risk classification with game-theoretic explanations.** A fine-tuned
DistilBERT classifier estimates the probability that a user prompt is unsafe,
and [shapiq](https://github.com/mmschlk/shapiq) explains every prediction by
treating the prompt's words as players in a cooperative game — computing
per-word Shapley values and pairwise interactions that show *which words made
the prompt risky, alone and in combination*.

## 🚀 Live demo

The service runs on Google Cloud Run:
**<https://shapiq-api-268593597387.europe-west1.run.app>**

Type a prompt in the web UI to get its risk score, or explore the
[interactive API docs](https://shapiq-api-268593597387.europe-west1.run.app/docs).

## ✨ Features

- **Prompt-risk classification** — DistilBERT fine-tuned on AdvBench,
  HarmBench, WildGuardMix, ToxicChat and BeaverTails
- **Explainable predictions** — per-word Shapley values and pairwise
  interactions via KernelSHAPIQ, served as an API endpoint
- **Web interface** — dependency-free single-page UI served by the same
  container
- **Reproducible pipeline** — uv-locked environment, DVC-versioned data on
  GCS, Hydra configuration, Dockerized training and serving
- **Experiment tracking** — PyTorch Lightning training with W&B logging and
  a Bayesian hyperparameter sweep
- **Production monitoring** — metrics, Evidently drift
  dashboard, and Cloud Monitoring email alerts

## ⚡ Quick start

```bash
git clone git@github.com:ssophiee/shapiq-prompt-risk-attribution.git
cd shapiq-prompt-risk-attribution
uv sync                # env + all locked dependencies
uv run pytest tests/   # offline test suite — no data or model needed
```

Continue with [Getting started](getting-started.md) for data access, serving
and Docker.

## 📊 At a glance

| | |
| --- | --- |
| Model | DistilBERT (sequence classification, 2 labels) |
| Training data | 5 public safety datasets, deduplicated |
| Test suite | 90 offline tests, ~10 s |
| Load test | 266 requests, 0 failures, `/predict` p99 520 ms |
| Serving | Cloud Run (2 vCPU / 2 GiB, scale-to-zero) |

## 🗺️ Where to go next

- [Getting started](getting-started.md) — full setup, data access, Docker
- [API](api.md) — endpoints and request/response examples
- [Training](training.md) — pipeline, hardware profiles, experiment tracking
- [Monitoring](monitoring.md) — drift detection, metrics, alerting
- [Project structure](structure.md) — repository layout
- [Code reference](reference.md) — generated from docstrings
