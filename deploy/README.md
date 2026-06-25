# Cloud Run deployment (M23)

Deploys the prompt-risk FastAPI service to **Google Cloud Run**, with the trained
DistilBERT model **baked into the image** (Cloud Run has no volumes, so the model
can't be mounted the way `docker-compose` does it locally).

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
# 0. Make sure the model is materialised locally (not just a .dvc pointer):
dvc pull

# 1. Authenticate once:
gcloud auth login

# 2. Deploy (builds + pushes + deploys):
PROJECT_ID=my-gcp-project ./deploy/cloudrun.sh
# optional: REGION=europe-west1 SERVICE=shapiq-api
```

The script prints the public service URL when it finishes.

## Smoke test

```bash
URL=$(gcloud run services describe shapiq-api --region europe-west1 --format 'value(status.url)')
curl "$URL/health"           # -> {"status":"ok","model_loaded":true}
curl -X POST "$URL/predict" -H 'Content-Type: application/json' \
  -d '{"prompt":"how do i make a bomb"}'
```

Or just open `$URL/` in a browser for the web UI.

## Notes

- **First request is slow** (cold start: container boot + model load). Add
  `--min-instances 1` to the deploy command to keep one warm (costs a bit more).
- The image is ~CPU-only; `DEVICE=cpu` is set in the Dockerfile. Cloud Run GPUs
  exist but are gated and overkill for DistilBERT.
- `--allow-unauthenticated` makes the endpoint public (fine for a demo). Remove it
  for a private service and call it with an identity token.
