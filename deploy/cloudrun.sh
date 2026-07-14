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
# Bucket the deployed API mirrors monitoring rows to (M27). Bucket names are
# globally unique, so prefix with the project id.
MONITORING_BUCKET="${MONITORING_BUCKET:-${PROJECT_ID}-monitoring}"

echo ">> Project: $PROJECT_ID  Region: $REGION  Service: $SERVICE  Bucket: $MONITORING_BUCKET"

# ---- Enable the APIs this deploy needs (idempotent) ------------------------
echo ">> Enabling required APIs (run.googleapis.com, cloudbuild, artifactregistry)..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  --project "$PROJECT_ID"

# ---- Monitoring bucket (M27, idempotent) ------------------------------------
# The API uploads one JSON blob per prediction here (ephemeral-disk-safe drift
# collection) and GET /monitoring reads everything back to build the Evidently
# report, so the Cloud Run runtime service account needs create + read/list.
echo ">> Ensuring monitoring bucket gs://$MONITORING_BUCKET exists..."
gcloud storage buckets describe "gs://$MONITORING_BUCKET" --project "$PROJECT_ID" >/dev/null 2>&1 \
  || gcloud storage buckets create "gs://$MONITORING_BUCKET" \
       --project "$PROJECT_ID" --location "$REGION" --uniform-bucket-level-access

PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
for role in roles/storage.objectCreator roles/storage.objectViewer; do
  gcloud storage buckets add-iam-policy-binding "gs://$MONITORING_BUCKET" \
    --member "serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role "$role" >/dev/null
done

# Upload the training-set drift baseline for GET /monitoring (data/ is excluded
# from the image, so the deployed report reads the baseline from the bucket).
BASELINE_CSV="data/monitoring/baseline.csv"
if [ ! -f "$BASELINE_CSV" ]; then
  echo "ERROR: $BASELINE_CSV not found — run 'uv run invoke build-baseline' (or dvc pull) first." >&2
  exit 1
fi
echo ">> Uploading drift baseline to gs://$MONITORING_BUCKET/baseline.csv..."
gcloud storage cp "$BASELINE_CSV" "gs://$MONITORING_BUCKET/baseline.csv"

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
  --set-env-vars "MONITORING_BUCKET=$MONITORING_BUCKET" \
  --allow-unauthenticated

echo ">> Done. Service URL:"
gcloud run services describe "$SERVICE" \
  --project "$PROJECT_ID" --region "$REGION" \
  --format 'value(status.url)'

echo ">> Smoke test once it's up:"
echo "   curl \$URL/health"
echo "   curl -X POST \$URL/predict -H 'Content-Type: application/json' -d '{\"prompt\":\"how do i make a bomb\"}'"
echo "   open \$URL/monitoring   # Evidently drift dashboard over live traffic"
