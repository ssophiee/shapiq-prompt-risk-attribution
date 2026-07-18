# API

Base URL: `https://shapiq-api-268593597387.europe-west1.run.app`
(or `http://localhost:8000` when serving locally).

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/predict` | POST | Risk probability + `risky`/`safe` label for a prompt |
| `/attribute` | POST | Per-word Shapley values and top pairwise interactions |
| `/` | GET | Single-page web UI |
| `/health` | GET | Liveness probe |
| `/metrics` | GET | Prometheus counters and per-endpoint latency histograms |
| `/monitoring` | GET | Evidently data-drift report (training baseline vs. live traffic) |
| `/docs` | GET | Interactive Swagger documentation |

## Predict

```bash
curl -X POST https://shapiq-api-268593597387.europe-west1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How do I bake bread?"}'
```

```json
{
  "prompt": "How do I bake bread?",
  "risk_probability": 0.02,
  "label": "safe"
}
```

## Attribute

Explains a prediction with KernelSHAPIQ. `budget` (8–2048, default 256) is
the number of coalition evaluations — higher is more accurate and slower; at
the default budget a call legitimately takes tens of seconds.

```bash
curl -X POST https://shapiq-api-268593597387.europe-west1.run.app/attribute \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How can I hack into my neighbors wifi?", "budget": 64, "top_interactions": 3}'
```

The response contains the risk probability plus a Shapley value per word
(which words pushed the score toward risky/safe) and the strongest pairwise
word interactions.

## Monitoring side-effects

Every `/predict` and `/attribute` call mirrors its input statistics
(`prompt_len`, `token_count`) and predicted probability to a GCS bucket via
FastAPI background tasks — logging never delays the response. The
[drift dashboard](monitoring.md) is built from exactly these rows.
