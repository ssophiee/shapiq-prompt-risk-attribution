# shapiq-cot Attribution — Work Plan v2 (Equal MLOps Split)

> Last updated: 2026-05-19

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
| **ML Core** | `attribution.py` | ✅ Done | B | add `make_classifier_value_function` |
| | `game.py` | ✅ Done | — | untouched |
| | `model.py` | ✅ Done | A | add `load_classifier(path)` |
| **Training** | `train.py` | ⬜ Not started | A | Hydra `@hydra.main`, full PyTorch loop + W&B logging |
| **Data** | `data.py` | 🔄 In progress | A | add AdvBench / HarmBench loaders |
| | DVC remote | 🔄 In progress | A | `data/prompts.json.dvc` not yet committed |
| **Visualisation** | `visualize.py` | ✅ Done | — | unchanged |
| **Pipeline** | `pipeline.py` | ✅ Done | B | add `--value-fn classifier\|llm` |
| **Config** | `configs/train.yaml` | ⬜ Not started | A | hyperparams, W&B project, output paths |
| | `configs/attribution.yaml` | ⬜ Not started | B | `prompt_type`, `few_shot_type`, `budget`, `value_fn` |
| | `configs/sweep.yaml` | ⬜ Not started | A | W&B Sweep search space |
| **Docker** | `train.Dockerfile` | ⬜ Not started | A | pytorch base, uv deps, `train.py` entrypoint |
| | `api.Dockerfile` | ⬜ Not started | B | python:3.11-slim, API deps, port 8080 |
| **Tests** | `test_data.py` | ⬜ Not started | A | parametrized + coverage |
| | `test_model.py` | ⬜ Not started | A | dummy PyTorch model |
| | `test_train.py` | ⬜ Not started | A | 2-sample smoke test |
| | `test_attribution.py` | ⬜ Not started | B | fixtures, mock classifier, parametrized |
| | `test_api.py` | ⬜ Not started | B | endpoint contract, mock model |
| **CI** | GitHub Actions | 🔄 In progress | Both | split into `ci-train.yaml` + `ci-api.yaml` |
| **API** | `api.py` | ⬜ Not started | B | FastAPI `POST /attribute` with lifespan |
| **Cloud** | GCP training job | ⬜ Not started | A | GCP Vertex AI or Cloud Run Job |
| | GCP API service | ⬜ Not started | B | GCP Cloud Run |
| **Monitoring** | Evidently drift | ⬜ Not started | B | derived features: `prompt_len`, `step_count`, `avg_step_len`, `p_risky` vs. training baseline |

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
| A0.1 | Fix DVC tracking | `git add data/prompts.json.dvc` + commit + `dvc push` | 🔄 In progress |
| A0.2 | AdvBench / HarmBench loaders in `data.py` | outputs `(prompt, cot_steps, label)` dataset | ⬜ Not started |
| A0.3 | CoT dataset generation script | run `generate_cot` on benchmark prompts; save to DVC-tracked `data/cot_dataset.json` | ⬜ Not started |
| A0.4 | DVC pipeline stage for generation | `dvc run` step so dataset regeneration is reproducible with `dvc repro` | ⬜ Not started |
| A0.5 | Evaluate `parse_cot_steps` robustness | test on varied Qwen outputs; fix regex if needed | ⬜ Not started |

### A1. PyTorch Classifier Training

| # | Task | MLOps angle | Status |
|---|---|---|---|
| A1.1 | `configs/train.yaml` | Hydra schema: model, hyperparams, W&B project name, output paths | ⬜ Not started |
| A1.2 | `train.py` — Hydra entrypoint | `@hydra.main`; load data, instantiate model, run loop, call `wandb.log` | ⬜ Not started |
| A1.3 | DistilBERT binary classifier | `DistilBertForSequenceClassification`; input `[CLS] {cot_steps} [SEP]`; output P(risky) | ⬜ Not started |
| A1.4 | Training loop | AdamW + linear LR scheduler; per-epoch accuracy, F1, ROC-AUC logged to W&B | ⬜ Not started |
| A1.5 | Save + DVC-track classifier artifact | best checkpoint → `models/classifier/`; `dvc add` + push; log artifact to W&B run | ⬜ Not started |

### A2. Hyperparameter Optimisation (W&B Sweeps)

| # | Task | MLOps angle | Status |
|---|---|---|---|
| A2.1 | `configs/sweep.yaml` | define sweep: `lr`, `batch_size`, `max_length` search space; method: `bayes` | ⬜ Not started |
| A2.2 | Sweep agent | `wandb agent` picks up config overrides; `train.py` reads them via Hydra | ⬜ Not started |
| A2.3 | Best config baked back into `configs/train.yaml` | run final training with best params; save artifact | ⬜ Not started |

### A3. Docker + CI (Training slice)

| # | Task | MLOps angle | Status |
|---|---|---|---|
| A3.1 | `train.Dockerfile` | `FROM pytorch/pytorch`; install uv deps; entrypoint `uv run python -m shapiq_cot.train` | ⬜ Not started |
| A3.2 | Local smoke test | `docker build` + `docker run` → verify metrics file + W&B run created | ⬜ Not started |
| A3.3 | `ci-train.yaml` GitHub Actions | on push: `pytest tests/test_data.py tests/test_model.py tests/test_train.py --cov`; build `train.Dockerfile` | ⬜ Not started |
| A3.4 | GCP training job | deploy training container to GCP Vertex AI or Cloud Run Jobs | ⬜ Not started |

### A4. Tests (training slice)

| # | Task | Details | Status |
|---|---|---|---|
| A4.1 | `test_data.py` | shape + label checks; `@pytest.mark.parametrize` over dataset splits | ⬜ Not started |
| A4.2 | `test_model.py` | `load_classifier` returns callable; output ∈ [0,1]; `pytest.raises` for bad input | ⬜ Not started |
| A4.3 | `test_train.py` | 2-sample smoke: loop completes, metrics dict returned, no crash | ⬜ Not started |
| A4.4 | Code coverage | `pytest --cov=src --cov-report=xml`; coverage report uploaded in CI | ⬜ Not started |

---

## Person B — Attribution + Serving (Stage 3 → API)

**Mission**: Own the SHAPIQ attribution stage and the serving layer end-to-end,
including Hydra config, W&B attribution tracking, FastAPI, Dockerfile, GCP Cloud Run,
Evidently monitoring, and tests. Consume A's classifier artifact via DVC; keep `game.py` untouched.

### B0. Attribution Integration

| # | Task | MLOps angle | Status |
|---|---|---|---|
| B0.1 | `configs/attribution.yaml` | Hydra schema: `model_id`, `prompt_type`, `few_shot_type`, `budget`, `value_fn`, W&B project | ⬜ Not started |
| B0.2 | `make_classifier_value_function` in `attribution.py` | for each coalition: load present steps → classifier → P(risky) | ⬜ Not started |
| B0.3 | `--value-fn` flag in `pipeline.py` | wire Hydra config into pipeline; switch `llm` / `classifier` | ⬜ Not started |
| B0.4 | Approximate shapiq for long chains | `shapiq.PermutationSamplingSV` when `n > 8`; controlled by `budget:` in config | ⬜ Not started |
| B0.5 | W&B attribution logging | log per-run SVs, k-SII matrix, `prompt_type`, `few_shot_type` as W&B Table | ⬜ Not started |

### B1. Experiments + Reporting

| # | Task | Status |
|---|---|---|
| B1.1 | Experiment: jailbreak attribution on 10+ examples | ⬜ Not started |
| B1.2 | Save SVs + k-SII to `reports/figures/` as PNG + JSON per run | ⬜ Not started |
| B1.3 | `evaluate.py` — aggregate mean SV per step position across runs | ⬜ Not started |

### B2. API + Docker + CI (serving slice)

| # | Task | MLOps angle | Status |
|---|---|---|---|
| B2.1 | `api.py` skeleton | FastAPI `POST /attribute` with Pydantic input validation + lifespan startup (load classifier) | ⬜ Not started |
| B2.2 | `api.Dockerfile` | `FROM python:3.11-slim`; install API deps; expose port 8080 | ⬜ Not started |
| B2.3 | Local API smoke test | `docker build` + `docker run` + `curl POST /attribute` | ⬜ Not started |
| B2.4 | `ci-api.yaml` GitHub Actions | on push: `pytest tests/test_attribution.py tests/test_api.py --cov`; build `api.Dockerfile` | ⬜ Not started |
| B2.5 | GCP Cloud Run deployment | push image to Artifact Registry; deploy to Cloud Run; expose public endpoint | ⬜ Not started |

### B3. Monitoring (Evidently)

> Evidently operates on **derived numerical features**, not raw text.
> Each API request is logged as a row of extracted features; Evidently then
> compares the live feature distribution against the training baseline.

| # | Task | MLOps angle | Status |
|---|---|---|---|
| B3.1 | Define feature schema | extract per-request: `prompt_len`, `step_count`, `avg_step_len`, `p_risky`; log to `data/predictions.csv` | ⬜ Not started |
| B3.2 | Baseline snapshot | run same extraction on training set; save to `data/baseline.csv`; DVC-track both | ⬜ Not started |
| B3.3 | Evidently drift report | `DataDriftPreset` on numerical columns: `prompt_len`, `step_count`, `avg_step_len`; generate HTML report | ⬜ Not started |
| B3.4 | P(risky) health check | `TestSuite`: assert mean P(risky) ∈ [0.2, 0.8]; alert if distribution collapses to all-0 or all-1 | ⬜ Not started |
| B3.5 | Scheduled monitoring | GH Actions cron (or Cloud Scheduler) triggers Evidently report weekly on accumulated predictions | ⬜ Not started |

### B4. Shared Infrastructure

| # | Task | Status |
|---|---|---|
| B4.1 | `.env` / secrets — HF token + `WANDB_API_KEY` from env vars; never hardcoded | ⬜ Not started |
| B4.2 | `pyproject.toml` — lock deps (`torch`, `transformers`, `shapiq`, `wandb`, `evidently`, `fastapi`) | 🔄 In progress |
| B4.3 | Pre-commit — `ruff format`, `ruff check`, `mypy` in `.pre-commit-config.yaml` | ⬜ Not started |

### B5. Tests (attribution + API slice)

| # | Task | Details | Status |
|---|---|---|---|
| B5.1 | `test_attribution.py` | `parse_cot_steps` on 3 fixtures; `make_classifier_value_function` with mock; `@pytest.mark.parametrize` over coalition sizes | ⬜ Not started |
| B5.2 | `test_api.py` | endpoint contract with mock classifier; Pydantic validation errors return 422; response schema check | ⬜ Not started |
| B5.3 | Code coverage | `pytest --cov=src --cov-report=xml`; report uploaded in `ci-api.yaml` | ⬜ Not started |

---

## DTU MLOps Course Coverage

| Course Section | Person A | Person B | Status |
|---|---|---|---|
| S1: Dev environment | uv, venv, Python setup | — | ✅ Done |
| S2: Version control + structure | git, code layout | — | ✅ Done |
| S2: Data versioning (DVC) | CoT dataset + classifier artifact; `dvc run` pipeline | — | 🔄 In progress |
| S2: CLI creation | — | `pipeline.py` CLI | ✅ Done |
| S3: Reproducibility (Docker) | `train.Dockerfile` | `api.Dockerfile` | ⬜ Not started |
| S3: Configuration (Hydra) | `configs/train.yaml` + sweep | `configs/attribution.yaml` | ⬜ Not started |
| S4: Debugging + profiling | PyTorch profiler on train loop | shapiq budget tradeoff | ⬜ Not started |
| S4: Experiment tracking (W&B) | training metrics + sweeps | attribution run metadata | ⬜ Not started |
| S5: Unit testing + coverage | `test_data`, `test_model`, `test_train`; `--cov` | `test_attribution`, `test_api`; `--cov` | ⬜ Not started |
| S5: CI / GitHub Actions | `ci-train.yaml` | `ci-api.yaml` | 🔄 In progress |
| S5: Pre-commit hooks | contributes | owns config | ⬜ Not started |
| S6: Cloud (GCP) | Vertex AI / Cloud Run Jobs training | Cloud Run API service | ⬜ Not started |
| S7: Deployment (FastAPI) | — | `api.py` + Pydantic + lifespan | ⬜ Not started |
| S8: Monitoring (Evidently) | — | `DataDriftPreset` on derived numerical features (`prompt_len`, `step_count`, `avg_step_len`); `TestSuite` on P(risky) distribution | ⬜ Not started |
| S10: Hyperparameter optimisation | W&B Sweeps (`configs/sweep.yaml`) | — | ⬜ Not started |
| S10: Documentation (mkdocs) | — | already scaffolded; keep updated | 🔄 In progress |

---

## Task Dependencies

```
Person A                                      Person B
────────                                      ────────
A0.1  DVC fix
A0.2  data loaders
A0.3  CoT dataset ──────────────────────────► (B: dvc pull before B0.2)
A0.4  DVC pipeline stage

A1.1  configs/train.yaml
A1.2–A1.4  train.py + W&B logging
A1.5  classifier artifact (DVC) ────────────► B0.2  classifier value fn

A2.1–A2.3  W&B Sweeps → best config baked in

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
| B0.2 (classifier value fn) | A1.5 artifact in DVC | B1.1 experiments |
| B2.5 (Cloud Run deploy) | A3.4 GCP training produces artifact in GCS | full pipeline |
| B3 (Evidently) | B1.1 attribution runs generating predictions | B3.4 scheduled reports |
| Both CI green | A4 + B5 tests passing | merge to main |

---

## Milestone Schedule

| Week | Person A | Person B | Gate |
|---|---|---|---|
| W1 | A0.1 DVC fix, A0.2–A0.4 data loaders + CoT generation + DVC stage | B4.1–B4.3 secrets + deps + pre-commit; B0.1 `configs/attribution.yaml` skeleton | CoT dataset tracked in DVC; pre-commit green |
| W2 | A1.1–A1.5 full training loop + W&B logging + artifact | B0.2–B0.5 attribution value fn + W&B; B2.1 `api.py` skeleton | Classifier trains + logs to W&B; attribution runs with classifier value fn |
| W3 | A2.1–A2.3 W&B Sweeps; A3.1–A3.3 `train.Dockerfile` + `ci-train.yaml`; A4 tests + coverage | B2.2–B2.4 `api.Dockerfile` + `ci-api.yaml`; B5 tests + coverage; B3.1–B3.2 Evidently baseline | Both CI workflows green; coverage reports uploaded; Docker images build |
| W4 | A3.4 GCP training job; coordinate B1 experiments | B2.5 GCP Cloud Run deploy; B3.3–B3.4 Evidently health + scheduled reports | Full 3-stage pipeline runs end-to-end on GCP |
