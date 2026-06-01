# shapiq_attribution Work Plan

> Last updated: 2026-06-01

## Project direction

The project is no longer centered on chain-of-thought generation. The active direction is prompt-risk classification and
SHAPIQ attribution over prompts or prompt tokens. The project package is `shapiq_attribution` to avoid colliding with the
external `shapiq` library.

## Current status

| Area | Status | Notes |
|---|---|---|
| Package naming | Complete | Source package renamed to `src/shapiq_attribution/`. |
| DVC remote | Complete | Default remote is `storage gs://prompt_classifier_mlops`. |
| Data tracking | Complete | `data/raw.dvc` tracks raw snapshots; `data/processed/prompt_risk_dataset.jsonl.dvc` tracks the processed input dataset. |
| Old prompt file | Complete | Removed stale `data/prompts.json.dvc`; the actual file was missing and not part of the active pipeline. |
| A1 split | Complete | Hydra split step writes deterministic train/validation/test JSONL files. |
| A1 training | Complete | DistilBERT prompt-risk classifier trains with PyTorch and logs to W&B. |
| A1 DVC pipeline | Complete | `dvc.yaml` and `dvc.lock` reproduce split and training. |
| A1 artifacts | Complete | Model and metrics are pushed to the GCS DVC remote. |
| Evaluation quality | Needs follow-up | Current metrics are likely over-optimistic due to source/domain leakage in the random split. |

## Active data assets

```text
data/raw/
├── advbench.jsonl
├── harmbench.jsonl
└── wildguard_safe.jsonl

data/processed/
├── prompt_risk_dataset.jsonl
├── train.jsonl
├── val.jsonl
└── test.jsonl
```

The processed dataset combines risky prompts from AdvBench/HarmBench with safe prompts from WildGuard. The processed
input dataset is tracked as an individual DVC file. The train/validation/test files are DVC pipeline outputs generated
by the `split_data` stage.

## Current artifacts

```text
configs/train.yaml
dvc.yaml
dvc.lock
models/prompt_risk_distilbert/
reports/metrics.json
```

`models/prompt_risk_distilbert/` and `reports/metrics.json` are produced by the DVC training stage and pushed to Google
Cloud Storage through DVC. They should not be committed directly to Git.

## Milestones

### A0. Data versioning

| Task | Status | Notes |
|---|---|---|
| Install GCS support for DVC | Complete | `dvc-gs` is present in `pyproject.toml`. |
| Configure GCS remote | Complete | `.dvc/config` points to `gs://prompt_classifier_mlops`. |
| Track current raw data | Complete | `data/raw.dvc`. |
| Track processed input dataset | Complete | `data/processed/prompt_risk_dataset.jsonl.dvc`. |
| Remove stale prompt reference | Complete | `data/prompts.json.dvc` removed. |
| Resolve processed-output overlap | Complete | Removed `data/processed.dvc`; split files are now pipeline outputs. |
| Verify DVC status | Complete | `uv run dvc status` reports data and pipelines up to date. |

### A1. Split and training

| Task | Status | Notes |
|---|---|---|
| Add Hydra train config | Complete | `configs/train.yaml` controls data paths, model, training, output, and W&B. |
| Add split function and CLI | Complete | `uv run python -m shapiq_attribution.split_data`. |
| Write train/validation/test files | Complete | Outputs under `data/processed/`. |
| Add DistilBERT training script | Complete | `uv run python -m shapiq_attribution.train`. |
| Use PyTorch device config | Complete | `training.device` supports `auto`, `cpu`, `cuda`, and `mps`. |
| Add W&B logging | Complete | Training and validation/test metrics are logged to W&B. |
| Save model artifact | Complete | Output under `models/prompt_risk_distilbert/`. |
| Save metrics | Complete | Output at `reports/metrics.json`. |
| Add DVC pipeline stages | Complete | `split_data` and `train` stages in `dvc.yaml`. |
| Push artifacts to GCS | Complete | `uv run dvc push` pushed current A1 artifacts. |
| Add tests | Complete | Split determinism and training helper tests pass. |

## A1 command flow

```bash
uv run python -m shapiq_attribution.split_data
uv run python -m shapiq_attribution.train training.device=mps
uv run dvc repro
uv run dvc push
```

For a smoke run without online W&B syncing:

```bash
uv run python -m shapiq_attribution.train training.device=mps wandb.mode=offline training.epochs=1 training.batch_size=4
```

## Evaluation caveat

The current validation/test scores are perfect, but they should be treated as pipeline-validation evidence rather than
final model-quality evidence. The likely issue is source/domain leakage: AdvBench and HarmBench currently provide risky
examples, while WildGuard provides safe examples. A random stratified split can therefore let the model learn source or
style cues instead of robust prompt-risk semantics.

This does not block Person B from using the model for an attribution/API prototype, but it does mean the model is not
yet scientifically reliable.

## Person B handoff

Person B can now run:

```bash
uv run dvc pull
```

and use:

```text
models/prompt_risk_distilbert/
```

as a prototype classifier artifact for API and SHAPIQ attribution integration. The expected model interface is:

```text
prompt or masked prompt -> PromptRiskPredictor -> P(risky)
```

Person B should use `PromptRiskPredictor` from `shapiq_attribution.model`:

```python
from shapiq_attribution.model import PromptRiskPredictor

predictor = PromptRiskPredictor.from_pretrained(
    "models/prompt_risk_distilbert",
    max_length=128,
    device="cpu",
)

p_risky = predictor("masked prompt text")
```

For SHAPIQ, the value function can delegate directly to the predictor:

```python
def value_function(masked_prompt: str) -> float:
    return predictor(masked_prompt)
```

## Next steps

### A1.9. Improve evaluation quality

| Task | Status | Notes |
|---|---|---|
| Add source-wise metrics | Not started | Report metrics separately for AdvBench, HarmBench, and WildGuard. |
| Add source-aware split option | Not started | Avoid evaluating on examples too similar to training examples by source. |
| Add held-out benchmark evaluation | Not started | Example: train with AdvBench + WildGuard subset, evaluate on HarmBench. |
| Add calibration check | Not started | Inspect overconfidence; consider expected calibration error or reliability curves. |

### A2. Hyperparameter optimization

| Task | Status | Notes |
|---|---|---|
| Add `configs/sweep.yaml` | Not started | Define W&B sweep over learning rate, batch size, max length, weight decay, and epochs. |
| Make training sweep-compatible | Not started | Ensure Hydra overrides from W&B agent are cleanly supported. |
| Run W&B sweep agent | Not started | Use sweep metric such as validation F1 or ROC-AUC. |
| Select best config | Not started | Promote best hyperparameters back into `configs/train.yaml`. |
| Train final best model | Not started | Re-run final model through DVC after selecting the best config. |

### A3. Training infrastructure

| Task | Status | Notes |
|---|---|---|
| Update `dockerfiles/train.dockerfile` | Not started | Entrypoint should run `python -m shapiq_attribution.train`. |
| Local Docker smoke test | Not started | Build and run the training container with a short config override. |
| Add training CI workflow | Not started | Run tests and optionally validate Docker build. |
| Add cloud training job | Not started | Optional later step for longer training runs. |

## Shared conventions

- Use `uv` for dependency and command execution.
- Use Hydra for configurable split and training steps.
- Use W&B for experiment tracking.
- Use DVC for data, split outputs, model artifacts, and metrics.
- Keep source imports under `shapiq_attribution`.
- Reserve `import shapiq` for the external SHAPIQ library.
- Keep documentation updated when pipeline commands, data locations, or artifact locations change.
