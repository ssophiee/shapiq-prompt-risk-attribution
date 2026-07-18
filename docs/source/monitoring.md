# Monitoring

Monitoring of the deployed service works in three layers.

## 1. Input/output collection

Every `/predict` and `/attribute` call mirrors its prompt, input statistics
(`prompt_len`, `token_count`) and predicted probability to a GCS bucket via
FastAPI background tasks, so collection never delays responses and survives
Cloud Run's ephemeral disks.

## 2. Drift detection

[`/monitoring`](https://shapiq-api-268593597387.europe-west1.run.app/monitoring)
on the deployed service fetches the training-set baseline plus all logged
prediction rows and renders an Evidently report:

- **Data-drift tests** on `prompt_len`, `token_count` and `p_risky`
- **Health test**: the mean predicted risk must stay in `[0.2, 0.8]` — a
  model that collapsed to predicting everything safe (or everything risky)
  fails this even without input drift

The same report can be built locally: `invoke monitor-report` (local rows) or
`invoke monitor-report-cloud` (rows fetched from the bucket).

## 3. System metrics and alerting

- **Prometheus** at `/metrics`: `api_requests_total` (per endpoint + status
  code), `api_request_duration_seconds` latency histograms, and
  `api_predicted_labels_total` (risky vs. safe — catches a collapsed model
  operationally)
- **Google Cloud Monitoring** watches Cloud Run's request count and latency,
  with two email alert policies (`deploy/alerts.sh`): any 5xx responses
  within a 5-minute window, and p95 latency above 30 s

This setup has already paid off once: a 5xx alert fired, the request logs
showed slow cold starts (the container re-installed dev dependencies at
startup), and the Docker entrypoint was fixed as a result — cutting cold
starts from ~90 s to roughly the model-load time.
