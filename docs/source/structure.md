# Project structure

```text
shapiq-prompt-risk-attribution/
├── src/shapiq_attribution/      # the Python package
│   ├── api.py                   # FastAPI app: /predict, /attribute, /monitoring, /metrics
│   ├── web.py                   # embedded single-page frontend (served at /)
│   ├── model.py                 # PromptRiskPredictor + load/save utilities
│   ├── data.py                  # dataset download, normalization, JSONL I/O, CLI
│   ├── split_data.py            # train/val/test splitting
│   ├── train.py                 # Hydra-driven Lightning training entrypoint
│   ├── lightning_module.py      # LightningModule (model, loss, metrics)
│   ├── lightning_data_module.py # LightningDataModule (splits, dataloaders)
│   ├── evaluate.py              # classification metrics
│   ├── game.py                  # shapiq game layer (CoTGame, ExactComputer)
│   ├── attribution.py           # coalition value function (stable ML core)
│   ├── safety_analysis.py       # token-level safety attribution game
│   ├── monitoring.py            # drift baseline, row collection, Evidently report
│   ├── pipeline.py              # natural-reasoning CoT attribution (research)
│   └── visualize.py             # attribution plots
├── tests/                       # 84 offline tests (stubbed models/tokenizers)
├── configs/                     # Hydra configs: train.yaml, sweep.yaml, hardware/
├── data/                        # DVC-tracked (raw snapshots, processed dataset)
├── models/                      # DVC-tracked trained model
├── dockerfiles/                 # api / train / train.gpu images
├── Dockerfile                   # Cloud Run image (model baked in)
├── deploy/                      # cloudrun.sh, alerts.sh, runbook
├── experiments/                 # attribution experiment runner (W&B)
├── docs/                        # this MkDocs site
├── reports/                     # exam report + figures
├── dvc.yaml / dvc.lock          # data pipeline definition
├── tasks.py                     # invoke task catalog
├── locustfile.py                # load test
└── pyproject.toml / uv.lock     # dependencies (uv)
```

## CI (GitHub Actions)

- `ci-lint.yaml` — ruff lint + format check
- `ci-api.yaml` / `ci-train.yaml` — tests + Docker image builds for the API
  and training images on every push/PR to `main`
- `monitoring.yaml` — drift-report tooling
