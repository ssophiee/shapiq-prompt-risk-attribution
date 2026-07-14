#!/usr/bin/env bash
#
# Create GCP Cloud Monitoring alert policies for the deployed API (M28).
#
# Sets up an email notification channel and two alert policies on the Cloud Run
# service, using Cloud Run's built-in metrics (no scraping infra needed):
#   1. Server errors: any 5xx responses over a 5-minute window.
#   2. Slow requests: p95 request latency above LATENCY_MS (default 30s —
#      /attribute legitimately takes tens of seconds, so the bar is high).
#
# Idempotent-ish: re-running creates duplicate policies, so it checks for an
# existing policy with the same display name first and skips it.
#
# Usage:
#   PROJECT_ID=mlops-shapiq-project ALERT_EMAIL=you@example.com ./deploy/alerts.sh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID, e.g. PROJECT_ID=mlops-shapiq-project ./deploy/alerts.sh}"
SERVICE="${SERVICE:-shapiq-api}"
ALERT_EMAIL="${ALERT_EMAIL:?Set ALERT_EMAIL, the address alerts are sent to}"
LATENCY_MS="${LATENCY_MS:-30000}"

echo ">> Project: $PROJECT_ID  Service: $SERVICE  Email: $ALERT_EMAIL"

gcloud services enable monitoring.googleapis.com --project "$PROJECT_ID"

# ---- Notification channel (reused if one already exists for this email) -----
CHANNEL=$(gcloud beta monitoring channels list --project "$PROJECT_ID" \
  --filter "type=email AND labels.email_address=$ALERT_EMAIL" \
  --format 'value(name)' | head -n 1)
if [ -z "$CHANNEL" ]; then
  echo ">> Creating email notification channel for $ALERT_EMAIL..."
  CHANNEL=$(gcloud beta monitoring channels create \
    --project "$PROJECT_ID" \
    --display-name "MLOps alerts ($ALERT_EMAIL)" \
    --type email \
    --channel-labels "email_address=$ALERT_EMAIL" \
    --format 'value(name)')
fi
echo ">> Notification channel: $CHANNEL"

# ---- Helper: create a policy from a JSON file unless it already exists ------
create_policy() {
  local display_name="$1" file="$2"
  local existing
  existing=$(gcloud alpha monitoring policies list --project "$PROJECT_ID" \
    --filter "displayName='$display_name'" --format 'value(name)' | head -n 1)
  if [ -n "$existing" ]; then
    echo ">> Policy '$display_name' already exists, skipping ($existing)"
    return
  fi
  echo ">> Creating policy '$display_name'..."
  gcloud alpha monitoring policies create --project "$PROJECT_ID" --policy-from-file "$file"
}

TMPDIR_ALERTS=$(mktemp -d)
trap 'rm -rf "$TMPDIR_ALERTS"' EXIT

# ---- Policy 1: any 5xx responses --------------------------------------------
cat > "$TMPDIR_ALERTS/errors.json" <<EOF
{
  "displayName": "[$SERVICE] 5xx server errors",
  "documentation": {
    "content": "Cloud Run service '$SERVICE' returned 5xx responses in the last 5 minutes. Check logs: gcloud run services logs read $SERVICE --project $PROJECT_ID",
    "mimeType": "text/markdown"
  },
  "combiner": "OR",
  "conditions": [
    {
      "displayName": "5xx request count > 0",
      "conditionThreshold": {
        "filter": "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"$SERVICE\" AND metric.type = \"run.googleapis.com/request_count\" AND metric.labels.response_code_class = \"5xx\"",
        "aggregations": [
          {
            "alignmentPeriod": "300s",
            "perSeriesAligner": "ALIGN_SUM",
            "crossSeriesReducer": "REDUCE_SUM"
          }
        ],
        "comparison": "COMPARISON_GT",
        "thresholdValue": 0,
        "duration": "0s",
        "trigger": { "count": 1 }
      }
    }
  ],
  "notificationChannels": ["$CHANNEL"]
}
EOF
create_policy "[$SERVICE] 5xx server errors" "$TMPDIR_ALERTS/errors.json"

# ---- Policy 2: p95 latency ---------------------------------------------------
cat > "$TMPDIR_ALERTS/latency.json" <<EOF
{
  "displayName": "[$SERVICE] p95 latency above ${LATENCY_MS}ms",
  "documentation": {
    "content": "p95 request latency on '$SERVICE' exceeded ${LATENCY_MS}ms for 5 minutes. The container may be overloaded or the model may be stuck.",
    "mimeType": "text/markdown"
  },
  "combiner": "OR",
  "conditions": [
    {
      "displayName": "p95 request latency",
      "conditionThreshold": {
        "filter": "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"$SERVICE\" AND metric.type = \"run.googleapis.com/request_latencies\"",
        "aggregations": [
          {
            "alignmentPeriod": "300s",
            "perSeriesAligner": "ALIGN_PERCENTILE_95",
            "crossSeriesReducer": "REDUCE_MAX"
          }
        ],
        "comparison": "COMPARISON_GT",
        "thresholdValue": $LATENCY_MS,
        "duration": "300s",
        "trigger": { "count": 1 }
      }
    }
  ],
  "notificationChannels": ["$CHANNEL"]
}
EOF
create_policy "[$SERVICE] p95 latency above ${LATENCY_MS}ms" "$TMPDIR_ALERTS/latency.json"

echo ">> Done. View policies: https://console.cloud.google.com/monitoring/alerting?project=$PROJECT_ID"
