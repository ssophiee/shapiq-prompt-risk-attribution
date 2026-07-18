# shapiq_attribution

`shapiq_attribution` is an MLOps project for training prompt-risk classifiers and explaining their predictions with
SHAPIQ attribution methods.

## Current milestone

### A0: Data versioning

A0 is complete. DVC is configured with a Google Cloud Storage remote:

```text
storage gs://prompt_classifier_mlops
```

The raw snapshot is tracked through a standalone DVC file:

```text
data/raw.dvc
```

Processed data, the served model, and metrics are declared as pipeline outputs
in `dvc.yaml` and recorded in `dvc.lock`; they do not have separate `.dvc`
files.

### A1: Split and training

A1 is complete. The Hydra/DVC pipeline splits the normalized prompt-risk
dataset, trains a DistilBERT classifier with PyTorch Lightning, logs metrics to
W&B, and saves model and metrics artifacts.

Current inputs and outputs:

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

Hydra provides three hardware profiles:

```text
local       one auto-selected device, 32-bit
single_gpu  one CUDA GPU, 16-bit mixed precision
ddp         two CUDA GPUs on one node, 16-bit mixed precision
```

Direct training defaults to `local`:

```bash
uv run python -m shapiq_attribution.train
uv run python -m shapiq_attribution.train hardware=single_gpu
```

The canonical DVC model producer is the two-GPU stage:

```bash
uv run dvc repro train_vertex_ddp
```

That command requires two CUDA GPUs because the stage explicitly selects
`hardware=ddp`.

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
uv sync --extra cpu
uv run dvc pull
uv run dvc status
uv run dvc repro prepare_dataset split_data
uv run dvc repro train_vertex_ddp  # requires two CUDA GPUs
uv run pytest tests/
uv run ruff check . --fix
uv run ruff format .
```
