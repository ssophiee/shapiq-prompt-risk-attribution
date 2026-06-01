# shapiq_attribution

`shapiq_attribution` is an MLOps project for training prompt-risk classifiers and explaining their predictions with
SHAPIQ attribution methods.

## Current milestone

### A0: Data versioning

A0 is complete. DVC is configured with a Google Cloud Storage remote:

```text
storage gs://prompt_classifier_mlops
```

The current DVC-tracked data assets are:

```text
data/raw.dvc
data/processed/prompt_risk_dataset.jsonl.dvc
```

The old `data/prompts.json.dvc` reference was removed because the underlying file was no longer part of the active data
pipeline. The old `data/processed.dvc` directory tracking was also removed because DVC pipeline outputs now live under
`data/processed/`.

### A1: Split and training

A1 is complete. The Hydra/DVC pipeline splits the normalized prompt-risk dataset, trains a DistilBERT classifier, logs
metrics to W&B, and saves model and metrics artifacts.

Planned inputs and outputs:

```text
Input:
  data/processed/prompt_risk_dataset.jsonl

Outputs:
  data/processed/train.jsonl
  data/processed/val.jsonl
  data/processed/test.jsonl
  models/prompt_risk_distilbert/
  reports/metrics.json
```

### Model handoff

Person B can use the trained prototype classifier through `PromptRiskPredictor`:

```python
from shapiq_attribution.model import PromptRiskPredictor

predictor = PromptRiskPredictor.from_pretrained(
    "models/prompt_risk_distilbert",
    max_length=128,
    device="cpu",
)

p_risky = predictor("masked prompt text")
```

The predictor is the intended handoff interface for attribution value functions:

```text
masked prompt -> P(risky)
```

## Useful commands

```bash
uv run dvc pull
uv run dvc status
uv run dvc repro
uv run pytest tests/
uv run ruff check . --fix
uv run ruff format .
```
