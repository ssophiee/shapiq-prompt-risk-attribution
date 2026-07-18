# Getting started

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) — manages Python and all dependencies
- For data/model access: a Google account with permission on the DVC bucket
  (ask a maintainer for an IAM grant), plus the
  [Google Cloud SDK](https://cloud.google.com/sdk)

## Environment

```bash
git clone git@github.com:ssophiee/shapiq-prompt-risk-attribution.git
cd shapiq-prompt-risk-attribution
uv sync
```

`uv sync` installs Python 3.13 and the exact dependency versions pinned in
`uv.lock` into `.venv`, including the dev tools. The test suite works
immediately — it is fully offline:

```bash
uv run pytest tests/
```

## Data and model

```bash
gcloud auth application-default login
uv run dvc pull
```

This fetches the DVC-tracked datasets and the trained model
(`models/prompt_risk_distilbert/`) from `gs://prompt_classifier_mlops`.

## Serve locally

```bash
uv run uvicorn shapiq_attribution.api:app --port 8000
```

Then open <http://localhost:8000> for the web UI, or `/docs` for Swagger.

## Docker

```bash
invoke docker-build-api
invoke docker-run-api        # mounts ./models read-only, serves on :8000
```

See
[`DOCKER.md`](https://github.com/ssophiee/shapiq-prompt-risk-attribution/blob/main/DOCKER.md)
for all images and
[`deploy/README.md`](https://github.com/ssophiee/shapiq-prompt-risk-attribution/blob/main/deploy/README.md)
for the Cloud Run deployment runbook
(`PROJECT_ID=<project> ./deploy/cloudrun.sh`).

## Pre-commit hooks (contributors)

```bash
uv run pre-commit install --install-hooks
uv run pre-commit install --hook-type pre-push
```

On commit: file hygiene + `ruff check --fix` + `ruff format`. On push: the
fast offline pytest subset.

## Useful commands

```bash
uv run invoke --list                          # full task catalog
uv run invoke test                            # tests + coverage
uv run invoke load-test                       # locust against the deployed API
uv run dvc repro prepare_dataset split_data   # rebuild the dataset
uv run ruff check . --fix && uv run ruff format .
```
