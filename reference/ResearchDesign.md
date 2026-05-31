# Research Design — Token-Level Safety Attribution

> Last updated: 2026-05-31

---

## Motivation

[Feng et al. (2025)](https://arxiv.org/html/2602.04294v1) empirically show that **think-mode / reasoning-enhanced LLMs are consistently more vulnerable to jailbreaks** across all tested prompt configurations — a result they call the "think-mode paradox." The paper demonstrates *that* this happens but does not explain *which parts of the input* drive that vulnerability.

This project takes a direct approach: train a binary safety classifier (safe / risky) on jailbreak benchmarks, then use Shapley interaction values (shapiq) to attribute the classifier's verdict to **individual tokens** in the input prompt. The result is a token-level explanation of why a prompt is classified as dangerous.

---

## Core Design

The architecture has two stages:

1. **Training** — fine-tune a DistilBERT binary classifier on (prompt, label) pairs from AdvBench / HarmBench. Label = did the jailbreak succeed? Output = P(risky) ∈ [0, 1].

2. **Attribution** — for a new prompt, tokenize it and treat each token as a *player* in a cooperative game. The value function masks absent tokens with `[MASK]` and queries the classifier. SHAPIQ computes how much each token contributed to the safe/risky verdict.

This is the pattern from [shapiq's `language_model_game` notebook](https://github.com/mmschlk/shapiq) — the `SentimentClassificationGame` adapted for safety classification.

**Why tokens as players?**  
CoT steps were a natural unit when attributing LLM reasoning chains, but the bottleneck was `2^n` full LLM forward passes per prompt. Tokens as players with a small DistilBERT classifier are orders of magnitude cheaper, and the result is more directly interpretable: *specific words and phrases* that trigger the unsafe classification.

---

## Pipeline

```
                    STAGE 1 — DATA
                    ──────────────
AdvBench / HarmBench
        │
        ▼
(prompt, label)  ←── tracked by DVC
label: 1 = jailbreak succeeded (risky), 0 = refused (safe)


                    STAGE 2 — TRAINING (PyTorch)
                    ────────────────────────────
(prompt, label) dataset
        │
        ▼
Fine-tune DistilBERT: prompt tokens → P(risky)
        │
        ▼
Evaluate: accuracy, F1, ROC-AUC on held-out split
        │
        ▼
Log metrics + artifacts  ←── W&B
        │
        ▼
Save classifier checkpoint  ←── DVC, configured by Hydra


                    STAGE 3 — ATTRIBUTION (token-level)
                    ─────────────────────────────────────
Input prompt
        │
        ▼
Tokenize → [t1, t2, ..., tN]   (players)
        │
        ▼
TokenSafetyGame(classifier, tokens)
        │   for each coalition S ⊆ {1..N}:
        │     replace absent tokens with [MASK]
        │     classifier(masked_input) → P(risky)
        │     value(S) = P(risky | S) − P(risky | ∅)
        ▼
shapiq.KernelSHAPIQ → Shapley values per token + k-SII interactions
        │
        ▼
"tokens 'ignore' and 'previous' jointly drive the risky verdict"
```

---

## The Game Class

`TokenSafetyGame` inherits from `shapiq.Game` (mirrors `SentimentClassificationGame` in the shapiq library):

```python
class TokenSafetyGame(shapiq.Game):
    def __init__(self, classifier, tokenizer, prompt: str):
        self.tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        # exclude [CLS] and [SEP] from player set
        self.player_token_ids = self.tokens[1:-1]
        super().__init__(n_players=len(self.player_token_ids))

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        scores = []
        for coalition in coalitions:
            masked = self.player_token_ids.clone()
            masked[coalition == 0] = MASK_ID
            input_ids = torch.cat([CLS, masked, SEP]).unsqueeze(0)
            with torch.no_grad():
                p_risky = self.classifier(input_ids).logits.softmax(-1)[0, 1].item()
            scores.append(p_risky)
        return np.array(scores)
```

Normalization: `value(∅) = 0` by subtracting the all-masked baseline, so Shapley values sum to `P(risky | full prompt) − P(risky | all masked)`.

---

## What the classifier sees

Training input: raw prompt text from AdvBench / HarmBench, tokenized with DistilBERT's WordPiece tokenizer.

```
[CLS] Ignore previous instructions and tell me how to ... [SEP]
```

At attribution time the same tokenization is used. Absent tokens in a coalition are replaced with `[MASK]` — the same strategy used in the shapiq `language_model_game` tutorial.

---

## Experimental axes (maps to Hydra config)

| Axis | Values | Why |
|---|---|---|
| `prompt_type` | `role_oriented`, `task_oriented` | Paper's main variable — RoP vs ToP |
| `dataset` | `advbench`, `harmbench` | Cross-benchmark robustness |
| `budget` | `exact`, `kernel_shap` | shapiq approximation for long token sequences |
| `max_tokens` | `64`, `128` | control coalition space size |

Comparing `prompt_type` attribution patterns checks whether RoP and ToP prompts exploit different token-level features in the classifier.

---

## Training + Evaluation + Logging

Implemented in `train.py` with Hydra + W&B:

1. **Data loading** — load `(prompt, label)` pairs from DVC-tracked dataset; split train/val/test
2. **Model** — `DistilBertForSequenceClassification` (PyTorch) with a binary head
3. **Training loop** — AdamW + linear LR scheduler; per-epoch accuracy, F1, ROC-AUC
4. **W&B logging** — hyperparameters, per-epoch metrics, best checkpoint artifact
5. **W&B Sweeps** — Bayesian search over `lr`, `batch_size`, `max_length` (`configs/sweep.yaml`)
6. **Artifact** — best checkpoint → `models/classifier/` tracked by DVC

---

## What changes in the codebase

| File | Change |
|---|---|
| `game.py` | Replace `CoTGame` with `TokenSafetyGame`; `run_shapiq` stays |
| `attribution.py` | Replace `make_value_function` with token-masking value function |
| `data.py` | Add loaders for AdvBench / HarmBench `(prompt, label)` pairs |
| `model.py` | Add `load_classifier(path)` — returns DistilBERT + tokenizer |
| `pipeline.py` | Remove Qwen / CoT generation; add tokenization + attribution entrypoint |
| `train.py` (new) | Hydra `@hydra.main` — full PyTorch loop: data → model → train → eval → W&B → save |
| `configs/train.yaml` (new) | Hyperparameters, W&B project, output paths |
| `configs/sweep.yaml` (new) | W&B Sweep search space |
| `configs/attribution.yaml` (new) | `prompt_type`, `dataset`, `budget`, `max_tokens` |

**Removed:** `parse_cot_steps`, Qwen dependency, `generate_cot`, CoT dataset generation stage.

---

## Hydra config sketch

```yaml
# configs/train.yaml
model:
  type: distilbert-base-uncased
  max_length: 128

training:
  epochs: 3
  batch_size: 16
  learning_rate: 2e-5
  train_split: 0.8
  seed: 42

data:
  dataset: advbench        # or harmbench
  max_samples: 1000

tracker:
  project: shapiq-token-safety
  experiment_name: distilbert-safety-classifier

output:
  model_dir: models/classifier
  metrics_file: reports/classifier_metrics.json
```

```yaml
# configs/attribution.yaml
prompt_type: role_oriented   # or task_oriented
dataset: advbench
budget: kernel_shap          # or exact
max_tokens: 128
output_dir: reports/
```

```yaml
# configs/sweep.yaml
program: train.py
method: bayes
metric:
  goal: maximize
  name: val/roc_auc
parameters:
  training.learning_rate:
    distribution: log_uniform_values
    min: 1e-5
    max: 5e-4
  training.batch_size:
    values: [8, 16, 32]
  model.max_length:
    values: [64, 128]
```

---

## Monitoring

Evidently operates on **derived numerical features**, not raw text. Each API request is logged as a feature row; Evidently compares the live distribution against the training baseline.

```
API request arrives
        │
        ▼
extract: prompt_len (chars), token_count, p_risky
        │
        ▼
append row → data/predictions.csv  (DVC-tracked)
        │
        ▼
Evidently (scheduled weekly):
  DataDriftPreset  ← compares predictions.csv vs. data/baseline.csv
  TestSuite        ← assert mean(p_risky) ∈ [0.2, 0.8]
        │
        ▼
HTML report → reports/monitoring/
```

**Baseline** is extracted from the training set once after `train.py` completes.

| Feature | Drift signal |
|---|---|
| `prompt_len` | inputs shifting to much longer / shorter prompts |
| `token_count` | tokenizer producing significantly more/fewer tokens |
| `p_risky` | classifier output distribution collapsing — most important signal |

---

## Research question

> Do Shapley interaction values over tokens reveal systematic patterns in which words and phrases drive a safety classifier's verdict — and do those patterns differ between role-oriented and task-oriented jailbreak prompt types?
