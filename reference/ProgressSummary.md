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
| Data tracking | Complete | `data/raw.dvc` and `data/processed.dvc` track current datasets. |
| Old prompt file | Complete | Removed stale `data/prompts.json.dvc`; the actual file was missing and not part of the active pipeline. |
| A1 split | Next | Create deterministic train/validation/test splits from `prompt_risk_dataset.jsonl`. |
| A1 training | Next | Train first binary prompt-risk classifier and save metrics/model artifacts. |

## Active data assets

```text
data/raw/
├── advbench.jsonl
├── harmbench.jsonl
└── wildguard_safe.jsonl

data/processed/
└── prompt_risk_dataset.jsonl
```

The processed dataset combines risky prompts from AdvBench/HarmBench with safe prompts from WildGuard. The next training
pipeline should split `data/processed/prompt_risk_dataset.jsonl` into train, validation, and test files.

## Milestones

### A0. Data versioning

| Task | Status | Notes |
|---|---|---|
| Install GCS support for DVC | Complete | `dvc-gs` is present in `pyproject.toml`. |
| Configure GCS remote | Complete | `.dvc/config` points to `gs://prompt_classifier_mlops`. |
| Track current raw data | Complete | `data/raw.dvc`. |
| Track current processed data | Complete | `data/processed.dvc`. |
| Remove stale prompt reference | Complete | `data/prompts.json.dvc` removed. |
| Verify DVC status | Complete | `uv run dvc status` reported data and pipelines up to date. |

### A1. Split and training

| Task | Status | Notes |
|---|---|---|
| Add split function and CLI | Not started | Deterministic, seeded, preferably label-stratified. |
| Write train/validation/test files | Not started | Output under `data/processed/`. |
| Add baseline training script | Not started | Start with a robust baseline before larger models. |
| Save model artifact | Not started | Output under `models/`. |
| Save metrics | Not started | Output under `reports/metrics.json`. |
| Add DVC pipeline stages | Not started | Reproducible split and train stages in `dvc.yaml`. |
| Add tests | Not started | Split determinism, valid labels, training smoke test. |

## Proposed A1 command flow

```bash
uv run python -m shapiq_attribution.data split \
  --input-path data/processed/prompt_risk_dataset.jsonl \
  --output-dir data/processed \
  --seed 42

uv run python -m shapiq_attribution.train

uv run dvc add data/processed models reports
uv run dvc push
```

The exact commands may change once the split and training CLIs are implemented.

## Shared conventions

- Use `uv` for dependency and command execution.
- Use DVC for data and model artifacts.
- Keep source imports under `shapiq_attribution`.
- Reserve `import shapiq` for the external SHAPIQ library.
- Keep documentation updated when pipeline commands or data locations change.
