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
data/processed.dvc
```

The old `data/prompts.json.dvc` reference was removed because the underlying file was no longer part of the active data
pipeline.

### A1: Split and training

The next milestone is to split the normalized prompt-risk dataset into train, validation, and test files, then train the
first binary classifier.

Planned inputs and outputs:

```text
Input:
  data/processed/prompt_risk_dataset.jsonl

Outputs:
  data/processed/train.jsonl
  data/processed/val.jsonl
  data/processed/test.jsonl
  models/
  reports/metrics.json
```

## Useful commands

```bash
uv run dvc pull
uv run dvc status
uv run pytest tests/
uv run ruff check . --fix
uv run ruff format .
```
