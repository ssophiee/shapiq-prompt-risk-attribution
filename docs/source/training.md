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

## Profiling and optimized batching

Profiling is optional and selected through Hydra. Normal training uses
`profiling=disabled`; `profiling=simple` records Lightning action timings,
while `profiling=pytorch` records operator shapes, CPU time and memory:

```bash
uv run python -m shapiq_attribution.train profiling=simple
uv run python -m shapiq_attribution.train profiling=pytorch
```

A CPU baseline over 100 training batches showed that data loading used less
than 1% of fit time, while the forward and backward passes dominated. The
operator profile showed that every DistilBERT batch had shape
`[16, 128, 768]`. The 27,894 training prompts contain only 53.91 tokens on
average (median 21), so fixed padding wasted 57.88% of token positions.

Training now tokenizes complete batches, pads only to the longest sequence in
each batch, and uses deterministic length-grouped sampling. The sampler still
reshuffles every epoch and Lightning wraps it for distributed training. Over
the same 100 CPU batches, the mean padded width fell from 128 to 56.67 tokens
and the measured fit time fell from 62.527 s to 31.360 s:

| Measurement | Fixed padding | Optimized | Change |
| --- | ---: | ---: | ---: |
| Mean padded batch width | 128.00 | 56.67 | -55.73% |
| Total fit profiler time | 62.527 s | 31.360 s | -49.85% |
| Training-batch time | 59.750 s | 28.335 s | -52.58% |
| Mean `training_step` time | 205.37 ms | 90.26 ms | -56.05% |

Profiler traces and temporary checkpoints are written below
`outputs/profiling/`, which is ignored by Git.

## Data versioning

DVC with a Google Cloud Storage remote (`gs://prompt_classifier_mlops`). Raw
snapshots are tracked via `data/raw.dvc`; processed data, the served model
and metrics are pipeline outputs declared in `dvc.yaml` and recorded in
`dvc.lock`.

## Continuous data checks

The `cml-data` GitHub Actions workflow follows the course's continuous
machine-learning pattern. Pushes and pull requests that change `.dvc` files,
the `.dvc/` configuration, `dvc.yaml` or `dvc.lock` trigger the workflow. It:

1. authenticates to GCP with the `GCP_SA_KEY` repository secret;
2. pulls the DVC-tracked raw snapshots from Cloud Storage;
3. reproduces normalization and deterministic train/validation/test splitting;
4. runs the data and split tests;
5. generates a text-only Markdown report with class balance, source balance,
   duplicates and cross-split overlap checks;
6. uploads the report as a workflow artifact and updates a CML comment on pull
   requests.

The same integrity report can be generated locally:

```bash
uv run python -m shapiq_attribution.data_statistics --check
```

The command exits with status 1 if the splits do not reconstruct the processed
dataset, prompts leak between splits, a dataset is empty, or duplicate prompts
are present.

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
