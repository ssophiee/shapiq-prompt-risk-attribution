# Training

## Pipeline

The Hydra/DVC pipeline splits the normalized prompt-risk dataset, trains a
DistilBERT classifier with PyTorch Lightning, logs metrics to W&B, and saves
model and metrics artifacts:

```text
Input:
  data/processed/prompt_risk_dataset.jsonl

Outputs:
  data/processed/{train,val,test}.jsonl
  models/prompt_risk_distilbert/
  reports/metrics.json
```

The training code is organized Lightning-style: `lightning_module.py` (model,
loss, metrics), `lightning_data_module.py` (splits and dataloaders), and
`train.py` driving `lightning.Trainer` from the Hydra config
(`configs/train.yaml`).

## Hardware profiles

Hydra provides three hardware profiles (`configs/hardware/`):

```text
local       one auto-selected device, 32-bit
single_gpu  one CUDA GPU, 16-bit mixed precision
ddp         two CUDA GPUs on one node, 16-bit mixed precision (DDP)
```

```bash
uv run python -m shapiq_attribution.train                      # defaults to local
uv run python -m shapiq_attribution.train hardware=single_gpu
uv run dvc repro train_vertex_ddp                              # canonical producer; needs two CUDA GPUs
```

## Experiment tracking

Training logs to Weights & Biases; `configs/sweep.yaml` defines a Bayesian
hyperparameter sweep maximising `val/roc_auc`. In containers the W&B API key
can be injected from GCP Secret Manager (`WANDB_API_KEY_SECRET`), so no
credentials live in code or images.

## Data versioning

DVC with a Google Cloud Storage remote (`gs://prompt_classifier_mlops`). Raw
snapshots are tracked via `data/raw.dvc`; processed data, the served model
and metrics are pipeline outputs declared in `dvc.yaml` and recorded in
`dvc.lock`.

## Using the trained classifier from Python

```python
from shapiq_attribution.model import PromptRiskPredictor

predictor = PromptRiskPredictor.from_pretrained(
    "models/prompt_risk_distilbert",
    max_length=128,
    device="cpu",
)

p_risky = predictor("masked prompt text")  # masked prompt -> P(risky)
```

`PromptRiskPredictor` is also the value-function interface the attribution
game evaluates on masked prompts — see the [code reference](reference.md).
