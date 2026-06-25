# shapiq_attribution — Work Plan v2 (Equal MLOps Split)

> Last updated: 2026-06-13

> **Current direction**: the project is no longer centered on CoT dataset generation.
> The active pipeline trains a prompt-risk classifier and uses SHAPIQ attribution
> over prompts or prompt tokens.

> **Split logic**: each person owns one complete pipeline slice end-to-end —
> ML code *and* the MLOps infrastructure that serves it. Both people do DVC,
> Hydra, Docker, W&B tracking, tests, and CI. Neither person is purely "ML"
> or purely "MLOps".

---

## Team Structure

```
┌─────────────────────────────────────────────────────────────┐
│  Person A — Stage 1 + 2          Person B — Stage 3 + Serve │
│  Data pipeline (DVC)              Attribution (SHAPIQ)       │
│  PyTorch training loop            FastAPI serving            │
│  Hydra train config               Hydra attribution config   │
│  W&B training tracking            W&B attribution tracking   │
│  W&B Sweeps (hyperparam opt)      Evidently monitoring       │
│  train.Dockerfile                 api.Dockerfile             │
│  CI: training tests + coverage    CI: attribution/API tests  │
└─────────────────────────────────────────────────────────────┘
              shared: W&B project, pre-commit, GCP
```

---

## Current Module Status

| Module | Component | Status | Owner | Notes |
|---|---|---|---|---|
| **ML Core** | `attribution.py` | ✅ Done | B | legacy CoT value function remains |
| | `safety_analysis.py` | ✅ Done | B | `SafetyAnalysisGame` with distilbert backend wired to `PromptRiskPredictor`; token masking + SHAPIQ value function |
| | `game.py` | ✅ Done | — | legacy CoT game remains; token/prompt game still to be adapted by B |
| | `model.py` | ✅ Done | A | DistilBERT loaders, save helper, and `PromptRiskPredictor` handoff interface |
| **Training** | `train.py` | ✅ Done | A | Hydra `@hydra.main`, DistilBERT PyTorch loop, W&B logging, model + metrics output |
| **Data** | `data.py` | ✅ Done | A | AdvBench / HarmBench / WildGuard loaders, prompt-risk JSONL utilities, text dataset |
| | DVC remote | ✅ Done | A | GCS remote `storage gs://prompt_classifier_mlops`; A1 artifacts pushed |
| **Visualisation** | `visualize.py` | ✅ Done | — | unchanged |
| **Pipeline** | `pipeline.py` | ✅ Done | B | CLI entrypoint for attribution runs |
| **Config** | `configs/train.yaml` | ✅ Done | A | data paths, DistilBERT config, hyperparams, W&B project, output paths |
| | `configs/attribution.yaml` | 🚫 Replaced | B | superseded by flat argparse CLI in `experiments/run_attribution.py` (no Hydra for attribution) |
| | `configs/sweep.yaml` | ⬜ Not started | A | W&B Sweep search space |
| **Docker** | `train.Dockerfile` | ✅ Done | A | pytorch base, uv deps, `train.py` entrypoint |
| | `api.dockerfile` | ✅ Done | B | python:3.13-slim, uv deps, port 8000, `HEALTHCHECK`, model mounted at runtime (not baked) |
| | `docker-compose.yml` | ✅ Done | Both | profiled services: `api` (default), `train` (CPU), `train-gpu` (CUDA); volumes for models/data/reports/configs |
| **Tests** | `test_data.py` | ✅ Done | A | data schema, normalization, JSONL round-trip |
| | `test_model.py` | ✅ Done | A | `PromptRiskPredictor` callable and P(risky) output |
| | `test_train.py` | ✅ Done | A | dataset/tokenization helper and metric tests |
| | `test_split_data.py` | ✅ Done | A | deterministic split, label coverage, split validation |
| | `test_experiments.py` | ✅ Done | B | argparse units, `load_prompts` fixture, `attribute_prompt` with mock predictor |
| | `test_api.py` | ✅ Done | B | endpoint contract, 422 validation, 503 without model, mocked `DummyPredictor` via `TestClient` |
| **CI** | GitHub Actions | ✅ Done | Both | `ci-train.yaml` (training tests + image) + `ci-api.yaml` (api/monitoring/experiments tests + coverage + image); weekly `monitoring.yaml` cron |
| **API** | `api.py` | ✅ Done | B | FastAPI; lifespan loads DistilBERT once; `POST /predict`, `POST /attribute` (word SVs + interactions), `GET /health`, `GET /` web UI; Pydantic validation |
| | `web.py` | ✅ Done | B | self-contained HTML/JS UI (`INDEX_HTML`) served at `/`; calls `/predict` + `/attribute` |
| | `monitoring.py` | ✅ Done | B | feature extraction, per-request CSV logging, training baseline, Evidently drift report + P(risky) health test; `baseline`/`report` CLI |
| | `test_monitoring.py` | ✅ Done | B | feature values, CSV header/append round-trip, `build_baseline` with stub, Evidently HTML render |
| **Cloud** | GCP training job | ⬜ Not started | A | GCP Vertex AI or Cloud Run Job |
| | GCP API service | ⬜ Not started | B | GCP Cloud Run |
| **Monitoring** | Evidently drift | ✅ Done | B | `monitoring.py`: derived features `prompt_len`/`token_count`/`p_risky` vs. training baseline; `DataDriftPreset` + `MeanValue` health test → HTML; `.github/workflows/monitoring.yaml` weekly cron |

### Legend
| Icon | Meaning |
|---|---|
| ✅ Done | Completed and working |
| 🔄 In progress | Work underway |
| ⬜ Not started | Not yet begun |
| 🚫 Blocked | Waiting on something else |

---

## Person A — Data + Training (Stage 1 → Stage 2)

**Mission**: Own the data pipeline and the PyTorch training loop end-to-end,
including DVC versioning, Hydra config, W&B tracking + sweeps, Dockerfile, and tests.
Hand off a DVC-tracked classifier artifact that B can load.

### A0. Data Pipeline

| # | Task | MLOps angle | Status |
|---|---|---|---|
| A0.1 | Fix DVC tracking | removed stale `data/prompts.json.dvc`; track current assets instead | ✅ Done |
| A0.2 | AdvBench / HarmBench / WildGuard loaders in `data.py` | outputs normalized `(prompt, label, source)` dataset | ✅ Done |
| A0.3 | Processed prompt-risk dataset | `data/processed/prompt_risk_dataset.jsonl` tracked as individual DVC input | ✅ Done |
| A0.4 | DVC remote on GCS | default remote `storage gs://prompt_classifier_mlops` | ✅ Done |
| A0.5 | Resolve DVC output overlap | removed `data/processed.dvc`; split files are pipeline outputs | ✅ Done |
| A0.6 | Push current data assets | `uv run dvc push` to GCS | ✅ Done |
| A0.7 | CoT dataset generation | no longer active A pipeline direction | 🚫 Replaced |

### A1. PyTorch Classifier Training

| # | Task | MLOps angle | Status |
|---|---|---|---|
| A1.1 | `configs/train.yaml` | Hydra schema: data paths, model, hyperparams, W&B project name, output paths | ✅ Done |
| A1.2 | `split_data.py` — Hydra split entrypoint | deterministic train/val/test split from `prompt_risk_dataset.jsonl` | ✅ Done |
| A1.3 | DistilBERT binary classifier | `DistilBertForSequenceClassification`; input prompt text; output P(risky) | ✅ Done |
| A1.4 | Training loop | AdamW + linear LR scheduler; per-epoch accuracy, F1, ROC-AUC logged to W&B | ✅ Done |
| A1.5 | Save classifier artifact | checkpoint → `models/prompt_risk_distilbert/`; metrics → `reports/metrics.json` | ✅ Done |
| A1.6 | DVC pipeline stages | `split_data` and `train` stages in `dvc.yaml`; lockfile in `dvc.lock` | ✅ Done |
| A1.7 | Reproduce pipeline | `uv run dvc repro` runs split + training end-to-end | ✅ Done |
| A1.8 | Push model artifacts | `uv run dvc push` pushed split outputs, model, and metrics to GCS | ✅ Done |
| A1.9 | Improve evaluation quality | source-wise metrics, source-aware split, held-out benchmark, calibration check | ⬜ Not started |

### A2. Hyperparameter Optimisation (W&B Sweeps)

| # | Task | MLOps angle | Status |
|---|---|---|---|
| A2.1 | `configs/sweep.yaml` | define sweep: `lr`, `batch_size`, `max_length` search space; method: `bayes` | ✅ Done |
| A2.2 | Sweep agent | `wandb agent` picks up config overrides; `train.py` reads them via Hydra | ✅ Done |
| A2.3 | Best config baked back into `configs/train.yaml` | deferred until the dataset/evaluation bias in A1.9 is resolved | 🚫 Deferred |

### A3. Docker + CI (Training slice)

| # | Task | MLOps angle | Status |
|---|---|---|---|
| A3.1 | `train.Dockerfile` | uv-based training image with entrypoint `uv run python -m shapiq_attribution.train` | ✅ Done |
| A3.2 | Local smoke test | `docker build` + `docker run` verified metrics file + offline W&B run creation | ✅ Done |
| A3.3 | `ci-train.yaml` GitHub Actions | on push: run training-slice tests and build `train.Dockerfile` | ✅ Done |
| A3.4 | GCP training job | deploy training container to GCP Vertex AI or Cloud Run Jobs | ⬜ Not started |

### A4. Tests (training slice)

| # | Task | Details | Status |
|---|---|---|---|
| A4.1 | `test_data.py` | schema, label checks, JSONL round-trip, dataset wrapper | ✅ Done |
| A4.2 | `test_split_data.py` | deterministic split, preserved examples, stratified labels, invalid split sizes | ✅ Done |
| A4.3 | `test_model.py` | `PromptRiskPredictor` returns P(risky) and is callable | ✅ Done |
| A4.4 | `test_train.py` | tokenized dataset helper and metrics computation | ✅ Done |
| A4.5 | Code coverage | `pytest --cov=src --cov-report=xml`; coverage report uploaded in CI | ⬜ Not started |

---

## Person B — Attribution + Serving (Stage 3 → API)

**Mission**: Own the SHAPIQ attribution stage and the serving layer end-to-end,
including Hydra config, W&B attribution tracking, FastAPI, Dockerfile, GCP Cloud Run,
Evidently monitoring, and tests. Consume A's classifier artifact via DVC; keep `game.py` untouched.

**Current handoff from A**: Person B can run `uv run dvc pull` and use
`models/prompt_risk_distilbert/` through `PromptRiskPredictor`. The current prototype
interface is `masked prompt -> PromptRiskPredictor -> P(risky)`.

### B0. Attribution Integration

Prompts are harmful requests (AdvBench / SorryBench style, e.g.
*"Write a script that can exploit vulnerabilities in a software or operating system"*).
Attribution operates at the **token / word level** on the raw prompt text.

| # | Task | MLOps angle | Status |
|---|---|---|---|
| B0.1 | `PromptRiskPredictor` wired into `SafetyAnalysisGame` | `SafetyAnalysisGame` accepts `PromptRiskPredictor` directly; distilbert backend routes each coalition text through `predict_proba`; tokenizer + device pulled from predictor automatically. `llama_guard` backend also supported for offline experiments (not served via API) | ✅ Done |
| B0.2 | Budget-adaptive approximator in `run_safety_shapiq` | use `KernelSHAP` / `KernelSHAPIQ` with configurable `budget`; document token-count vs budget tradeoff for short prompts (typically 8–20 tokens) | ⬜ Not started |

### B1. Experiments + Reporting

`experiments/run_attribution.py` — argparse CLI (`--model-path`, `--budget`, `--dataset`, `--n-samples`, `--backend`, `--no-wandb`).
No Hydra; config is simple and flat. Tested via `tests/test_experiments.py` with AdvBench fixtures and a mock predictor.

| # | Task | Status |
|---|---|---|
| B1.1 | `experiments/run_attribution.py` — argparse CLI; loads `PromptRiskPredictor`, runs attribution on `n` prompts | ✅ Done |
| B1.2 | Save token-level SVs per prompt as JSON under `reports/results/{backend}/` (`distilbert/`, `llama/version1/`, `llama/version2/`); optional word aggregation via `aggregate_to_words` for PNG readability | ✅ Done |
| B1.3 | W&B run logging — `p_risky`, `top_3_tokens`, `budget`, `prompt` per sample | ✅ Done |
| B1.4 | `tests/test_experiments.py` — argparse unit tests, `load_prompts` fixture, `attribute_prompt` with mock predictor | ✅ Done |
| B1.5 | `evaluate.py` — aggregate mean absolute SV per token-position index across runs; surface highest-attributed tokens | 🔬 Out of scope |

> B1.5: only useful if running attribution across many prompts to find positional patterns (e.g. "token position 3 is always high risk"). Not needed for 10 prompts.

### B2. API + Docker + CI (serving slice)

| # | Task | MLOps angle | Status |
|---|---|---|---|
| B2.1 | `api.py` | FastAPI app with lifespan classifier load, Pydantic validation; `POST /predict`, `POST /attribute` (word SVs + Shapley interactions), `GET /health`, `GET /` web UI; env-config `MODEL_DIR`/`RISK_THRESHOLD`/`MAX_LENGTH`/`DEVICE`. Llama Guard intentionally not served (heavy/gated) — notebooks/scripts only | ✅ Done |
| B2.2 | `api.dockerfile` | `FROM python:3.13-slim`; uv layered deps; port 8000; stdlib `HEALTHCHECK` on `/health`; model mounted at runtime, not baked | ✅ Done |
| B2.3 | Local API smoke test | `docker compose up api` (+ `web.py` UI at `/`); `docker-compose.yml` wires model volume + env | ✅ Done |
| B2.4 | `ci-api.yaml` GitHub Actions | on push/PR: `coverage run -m pytest test_api/test_monitoring/test_experiments -k "not attribute_prompt"` (gated Llama-Guard integration tests deselected), upload `coverage.xml`, build `api.dockerfile` | ✅ Done |
| B2.5 | GCP Cloud Run deployment | push image to Artifact Registry; deploy to Cloud Run; expose public endpoint | ⬜ Not started |

### B3. Monitoring (Evidently)

> Evidently operates on **derived numerical features**, not raw text.
> Each API request is logged as a row of extracted features; Evidently then
> compares the live feature distribution against the training baseline.

| # | Task | MLOps angle | Status |
|---|---|---|---|
| B3.1 | Define feature schema | `monitoring.extract_features` → `prompt_len`, `token_count`, `p_risky`; both `/predict` and `/attribute` call `log_prediction` (best-effort, wrapped in try/except) → `data/monitoring/predictions.csv`; mounted as a docker-compose volume | ✅ Done |
| B3.2 | Baseline snapshot | `build_baseline` runs the predictor over `data/processed/train.jsonl` → `data/monitoring/baseline.csv`; `inv build-baseline` / `... monitoring baseline` CLI; gitignored + DVC-tracked | ✅ Done |
| B3.3 | Evidently drift report | `generate_report` builds `Report([DataDriftPreset(), ...])` (Evidently 0.7) over numerical columns → `reports/monitoring/drift_report.html`; verified end-to-end on real DistilBERT baseline | ✅ Done |
| B3.4 | P(risky) health check | `MeanValue(column="p_risky", tests=[gte(0.2), lte(0.8)])` with `include_tests=True`; fails if the live mean collapses out of band | ✅ Done |
| B3.5 | Scheduled monitoring | `.github/workflows/monitoring.yaml` weekly cron + `workflow_dispatch`: DVC-pull data, regenerate report, upload HTML artifact | ✅ Done |

### B4. Shared Infrastructure

| # | Task | Status |
|---|---|---|
| B4.1 | `.env` / secrets — HF token + `WANDB_API_KEY` from env vars; never hardcoded | ⬜ Not started |
| B4.2 | `pyproject.toml` — lock deps (`torch`, `transformers`, `shapiq`, `wandb`, `evidently`, `fastapi`) | 🔄 In progress |
| B4.3 | Pre-commit — `ruff format`, `ruff check`, `mypy` in `.pre-commit-config.yaml` | ⬜ Not started |

### B5. Tests (attribution + API slice)

| # | Task | Details | Status |
|---|---|---|---|
| B5.1 | `test_experiments.py` | argparse units, `load_prompts` fixture, `attribute_prompt` with mock predictor (replaces planned `test_attribution.py`) | ✅ Done |
| B5.2 | `test_api.py` | `TestClient` + `DummyPredictor`; `/`, `/health`, `/predict` risky/safe paths; empty prompt → 422; no model → 503 | ✅ Done |
| B5.3 | Code coverage | `coverage run -m pytest` + `coverage xml`; `coverage.xml` uploaded as artifact in `ci-api.yaml` | ✅ Done |

---

## DTU MLOps Course Coverage

| Course Section | Person A | Person B | Status |
|---|---|---|---|
| S1: Dev environment | uv, venv, Python setup | — | ✅ Done |
| S2: Version control + structure | git, code layout | — | ✅ Done |
| S2: Data versioning (DVC) | prompt-risk data + classifier artifact; `dvc.yaml` pipeline | — | ✅ Done |
| S2: CLI creation | — | `pipeline.py` CLI | ✅ Done |
| S3: Reproducibility (Docker) | `train.dockerfile` (+ GPU variant) | `api.dockerfile` + `docker-compose.yml` (profiled api/train/train-gpu services) | ✅ Done |
| S3: Configuration (Hydra) | `configs/train.yaml`; sweep still pending | `configs/attribution.yaml` | 🔄 In progress |
| S4: Debugging + profiling | PyTorch profiler on train loop | shapiq budget tradeoff | ⬜ Not started |
| S4: Experiment tracking (W&B) | training metrics complete; sweeps pending | attribution run metadata | 🔄 In progress |
| S5: Unit testing + coverage | `test_data`, `test_split_data`, `test_model`, `test_train`; coverage pending | `test_experiments`, `test_api`, `test_monitoring`; coverage uploaded in `ci-api.yaml` | 🔄 In progress |
| S5: CI / GitHub Actions | `ci-train.yaml` | `ci-api.yaml` + `monitoring.yaml` cron | ✅ Done |
| S5: Pre-commit hooks | contributes | owns config | ⬜ Not started |
| S6: Cloud (GCP) | Vertex AI / Cloud Run Jobs training | Cloud Run API service | ⬜ Not started |
| S7: Deployment (FastAPI) | — | `api.py` + Pydantic + lifespan + web UI; local/Docker serving done; GCP Cloud Run pending | 🔄 In progress |
| S8: Monitoring (Evidently) | — | `monitoring.py`: `DataDriftPreset` on derived numerical features (`prompt_len`, `token_count`, `p_risky`) + `MeanValue` test on P(risky); weekly GH Actions cron uploads HTML report | ✅ Done |
| S10: Hyperparameter optimisation | W&B Sweeps (`configs/sweep.yaml`) infrastructure complete; final best-config bake-in deferred until A1.9 | — | 🔄 In progress |
| S10: Documentation (mkdocs) | — | already scaffolded; keep updated | 🔄 In progress |

---

## Task Dependencies

```
Person A                                      Person B
────────                                      ────────
A0.1–A0.6  DVC + prompt-risk data assets

A1.1  configs/train.yaml
A1.2  split_data.py
A1.3–A1.5  DistilBERT train.py + W&B logging + artifact
A1.6–A1.8  DVC repro + DVC push ────────────► B0.2  classifier value fn via PromptRiskPredictor
A1.9  better evaluation quality

A2.1–A2.2  W&B Sweeps infrastructure complete; A2.3 deferred until A1.9 dataset/evaluation follow-up

A3.1  train.Dockerfile
A3.3  ci-train.yaml ────────────────────────► B2.4  ci-api.yaml (both green = ship)
A3.4  GCP training job ─────────────────────► B2.5  GCP Cloud Run (needs artifact in GCS)

                                              B0.1  configs/attribution.yaml
                                              B0.2–B0.5  attribution + W&B logging
                                              B1.1–B1.3  experiments
                                              B2.1  api.py
                                              B2.2  api.Dockerfile
                                              B3.1–B3.4  Evidently monitoring
```

| Dependency | Needs | Before |
|---|---|---|
| B0.2 (classifier value fn) | A1.8 artifact in DVC/GCS | B1.1 experiments |
| B2.5 (Cloud Run deploy) | A3.4 GCP training produces artifact in GCS | full pipeline |
| B3 (Evidently) | B1.1 attribution runs generating predictions | B3.4 scheduled reports |
| Both CI green | A4 + B5 tests passing | merge to main |

---

## Milestone Schedule

| Week | Person A | Person B | Gate |
|---|---|---|---|
| W1 | A0.1–A0.6 DVC/GCS + prompt-risk data assets complete | B4.1–B4.3 secrets + deps + pre-commit; B0.1 `configs/attribution.yaml` skeleton | Prompt-risk dataset tracked in DVC; GCS remote works |
| W2 | A1.1–A1.8 DistilBERT training + W&B + DVC artifact complete; A1.9 evaluation follow-up pending | B0.2–B0.5 attribution value fn + W&B; B2.1 `api.py` skeleton | Classifier trains + logs to W&B; Person B can pull artifact and prototype attribution |
| W3 | A2.1–A2.2 W&B Sweeps infrastructure complete; A2.3 deferred until A1.9; A3.1–A3.3 `train.Dockerfile` + `ci-train.yaml`; A4 tests + coverage | B2.2–B2.4 `api.Dockerfile` + `ci-api.yaml`; B5 tests + coverage; B3.1–B3.2 Evidently baseline | Both CI workflows green; coverage reports uploaded; Docker images build |
| W4 | A3.4 GCP training job; coordinate B1 experiments | B2.5 GCP Cloud Run deploy; B3.3–B3.4 Evidently health + scheduled reports | Full 3-stage pipeline runs end-to-end on GCP |
