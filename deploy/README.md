# Cloud Run deployment (M23) + monitoring & alerting (M27/M28)

Deploys the prompt-risk FastAPI service to **Google Cloud Run**, with the trained
DistilBERT model **baked into the image** (Cloud Run has no volumes, so the model
can't be mounted the way `docker-compose` does it locally).

## Prerequisites (once, for a new person / new machine)

1. [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed.
2. A GCP project **with billing enabled** (Cloud Build/Run refuse to work without
   it): `gcloud projects create <id>` + link billing in the console, or reuse an
   existing project. The scripts enable the required APIs themselves.
3. Authenticate — both variants are needed:

   ```bash
   gcloud auth login                      # for gcloud commands (deploy, alerts)
   gcloud auth application-default login  # for DVC + the local monitoring fetch
   ```

4. Get the data and model (DVC remote is a GCS bucket, hence step 3):

   ```bash
   uv sync
   uv run dvc pull
   ```

   Your account needs read access to the DVC bucket (`gs://prompt_classifier_mlops`).
   No access? Retrain instead: `uv run invoke preprocess-data train build-baseline`.

## How it works

`gcloud run deploy --source .` builds the root [`Dockerfile`](../Dockerfile) in
Cloud Build, pushes it to Artifact Registry, and runs it on Cloud Run. It's all
Docker under the hood — Cloud Run only runs containers.

Three files make the model survive the trip into the image:

| File | Role |
|------|------|
| [`Dockerfile`](../Dockerfile) (root) | Same API image as `dockerfiles/api.dockerfile`, plus `COPY models/prompt_risk_distilbert` and `--port ${PORT:-8080}` for Cloud Run. |
| [`.gcloudignore`](../.gcloudignore) | Stops gcloud falling back to `.gitignore` (which hides the DVC-tracked model) and re-includes the served checkpoint. |
| [`.dockerignore`](../.dockerignore) | Re-includes `models/prompt_risk_distilbert` so the Docker build context contains it. |

## Deploy

```bash
PROJECT_ID=my-gcp-project ./deploy/cloudrun.sh
# optional: REGION=europe-west1 SERVICE=shapiq-api MONITORING_BUCKET=my-bucket
```

One command, ~10 minutes. Besides build + push + deploy it also (idempotently):

- enables the required GCP APIs,
- creates the monitoring bucket `gs://<PROJECT_ID>-monitoring` and grants the
  Cloud Run service account create + read on it (M27 data collection),
- uploads the training drift baseline (`data/monitoring/baseline.csv`) to the
  bucket so the deployed `GET /monitoring` endpoint can compare against it,
- sets `MONITORING_BUCKET` on the service, which switches on GCS mirroring of
  every prediction.

The script prints the public service URL when it finishes.

## Alerts (M28)

Creates an email notification channel plus two Cloud Monitoring alert policies
(any 5xx responses; p95 latency > 30 s). Safe to re-run — existing policies are
skipped:

```bash
PROJECT_ID=my-gcp-project ALERT_EMAIL=you@example.com ./deploy/alerts.sh
```

## Smoke test

```bash
URL=$(gcloud run services describe shapiq-api --region europe-west1 --format 'value(status.url)')
curl "$URL/health"           # -> {"status":"ok","model_loaded":true}
curl -X POST "$URL/predict" -H 'Content-Type: application/json' \
  -d '{"prompt":"how do i make a bomb"}'
curl "$URL/metrics"          # Prometheus system metrics
open "$URL/monitoring"       # Evidently drift dashboard (404 until predictions exist)
```

Or just open `$URL/` in a browser for the web UI. Where each monitoring signal
lives (console links, commands) is listed in the root
[README](../README.md#where-to-find-metrics-alerts-and-drift).

## Notes

- **First request is slow** (cold start: container boot + model load). Add
  `--min-instances 1` to the deploy command to keep one warm (costs a bit more).
- The image is ~CPU-only; `DEVICE=cpu` is set in the Dockerfile. Cloud Run GPUs
  exist but are gated and overkill for DistilBERT.
- `--allow-unauthenticated` makes the endpoint public (fine for a demo). Remove it
  for a private service and call it with an identity token.
