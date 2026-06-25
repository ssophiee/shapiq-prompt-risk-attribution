#!/usr/bin/env bash
#
# Deploy the prompt-risk API to Google Cloud Run (M23).
#
# This wraps `gcloud run deploy --source .`, which:
#   1. Uploads the repo (filtered by .gcloudignore) to Cloud Build,
#   2. Builds the root ./Dockerfile (model baked in) in the cloud,
#   3. Pushes the image to Artifact Registry,
#   4. Deploys it as a Cloud Run service.
#
# No local `docker build` / `docker push` needed.
#
# Prerequisites (one-time):
#   - gcloud CLI installed and `gcloud auth login` done.
#   - The model present locally at models/prompt_risk_distilbert
#     (run `dvc pull` first if it's only a .dvc pointer).
#
# Usage:
#   PROJECT_ID=my-gcp-project ./deploy/cloudrun.sh
#   PROJECT_ID=my-gcp-project REGION=europe-west1 ./deploy/cloudrun.sh

set -euo pipefail

# ---- Configuration (override via env vars) ---------------------------------
PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID, e.g. PROJECT_ID=my-gcp-project ./deploy/cloudrun.sh}"
REGION="${REGION:-europe-west1}"
SERVICE="${SERVICE:-shapiq-api}"

echo ">> Project: $PROJECT_ID  Region: $REGION  Service: $SERVICE"

# ---- Enable the APIs this deploy needs (idempotent) ------------------------
echo ">> Enabling required APIs (run.googleapis.com, cloudbuild, artifactregistry)..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  --project "$PROJECT_ID"

# ---- Build from source + deploy --------------------------------------------
# Sizing notes:
#   --memory 2Gi --cpu 2 : torch + DistilBERT + shapiq attribution need headroom.
#   --timeout 300        : /attribute (KernelSHAPIQ) can take tens of seconds.
#   --port 8080          : matches ${PORT:-8080} in the Dockerfile CMD.
#   --allow-unauthenticated : public demo endpoint; drop it for a private service.
gcloud run deploy "$SERVICE" \
  --source . \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 3 \
  --allow-unauthenticated

echo ">> Done. Service URL:"
gcloud run services describe "$SERVICE" \
  --project "$PROJECT_ID" --region "$REGION" \
  --format 'value(status.url)'

echo ">> Smoke test once it's up:"
echo "   curl \$URL/health"
echo "   curl -X POST \$URL/predict -H 'Content-Type: application/json' -d '{\"prompt\":\"how do i make a bomb\"}'"
