# shapiq_attribution

SHAPIQ-based attribution for prompt-risk classification. The project builds a DVC-tracked prompt-risk dataset, trains
a binary classifier, and uses Shapley interaction values to explain which prompt tokens or spans drive risky/safe
predictions.

## Current focus

- **A0 Data versioning:** complete. DVC is connected to Google Cloud Storage at `gs://prompt_classifier_mlops`.
- **A1 Training:** next. Split `data/processed/prompt_risk_dataset.jsonl` into train/validation/test sets and train the
  first prompt-risk classifier.

## Data

The current dataset is stored as normalized JSONL files:

```text
data/raw/
├── advbench.jsonl
├── harmbench.jsonl
└── wildguard_safe.jsonl

data/processed/
└── prompt_risk_dataset.jsonl
```

The raw and processed data directories are tracked with DVC:

```bash
uv run dvc pull
uv run dvc status
uv run dvc push
```

## Development

Install or update dependencies with `uv`:

```bash
uv sync
uv add <package-name>
```

Run tests:

```bash
uv run pytest tests/
```

Run linting and formatting:

```bash
uv run ruff check . --fix
uv run ruff format .
```

## Project structure

```text
├── configs/                  # Configuration files
├── data/                     # DVC-tracked raw and processed datasets
│   ├── processed/
│   └── raw/
├── dockerfiles/              # Training and API Dockerfiles
├── docs/                     # MkDocs documentation
├── models/                   # Trained model artifacts
├── notebooks/                # Exploratory notebooks
├── reference/                # Planning and research notes
├── reports/                  # Metrics, reports, and figures
├── src/
│   └── shapiq_attribution/   # Project package
├── tests/                    # Unit tests
├── pyproject.toml
├── tasks.py
└── uv.lock
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template), a cookiecutter template for MLOps
projects.
