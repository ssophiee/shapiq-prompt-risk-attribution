# Handover: Data and Training Pipeline

Status: 19 July 2026  
Repository: `ssophiee/shapiq-cot-attribution`  
Branch: `main`  
Last verified commit: `eb20ebb`

## Purpose of this document

This document summarizes the parts of the project that I implemented and maintained. It focuses on:

- data acquisition and preprocessing;
- data and model versioning with DVC;
- the DistilBERT prompt-risk classifier;
- the PyTorch Lightning training pipeline;
- Hydra configuration;
- Weights & Biases experiment tracking and the model registry;
- profiling and training optimizations;
- CPU, GPU, and distributed training;
- training Docker images;
- continuous integration and continuous machine learning;
- cloud training on Compute Engine and Vertex AI.

The components developed primarily by Sofiia—especially SHAPIQ attribution, FastAPI, the frontend, deployment, and
monitoring—are covered only where they interface with the data and training pipeline.

## Project overview

The project classifies prompts as safe or risky. A fine-tuned DistilBERT model estimates `P(risky)`, and SHAPIQ then
explains the prediction at word level.

The relevant flow is:

```text
Public prompt-safety datasets
        ↓
Normalization to a shared JSONL schema
        ↓
Deduplication
        ↓
Stratified train/validation/test split
        ↓
DistilBERT training with PyTorch Lightning
        ↓
Best model selected by validation ROC-AUC
        ↓
Hugging Face model + reports/metrics.json
        ↓
DVC/GCS and the W&B Model Registry
        ↓
FastAPI, SHAPIQ attribution, and Cloud Run
```

## Status at handover

| Area | Status |
| --- | --- |
| Git branch | `main`, synchronized with `origin/main` |
| Working tree | Clean before this handover file was added |
| Test suite | 100 of 100 tests passed |
| Test runtime | 9.20 seconds, with 34 warnings |
| Data integrity | `PASS` |
| Trained model available locally | Yes |
| Processed data available locally | Yes |
| DVC remote | `gs://prompt_classifier_mlops` |
| Final model | DistilBERT, versioned through DVC |
| Training modes | Local, single GPU, and DDP |
| W&B integration | Experiment tracking, sweep, and model registry |
| Continuous data validation | Implemented |
| Continuous model validation | Implemented |

The warnings in the verified test run came mainly from Evidently/Litestar, NumPy/SciPy, and Lightning DataLoader
recommendations. There were no test failures.

The coverage value documented in the exam report is 71%. It was not recalculated during this handover review.

## My contributions

I was primarily responsible for the data and training side of the project.

### Data pipeline

I implemented the acquisition, normalization, validation, and combination of the safety datasets in
`src/shapiq_attribution/data.py`.

The training data comes from:

- AdvBench;
- HarmBench;
- WildGuardMix;
- ToxicChat;
- BeaverTails.

XSTest can additionally be downloaded as a separate evaluation dataset. It is not included in the combined training
dataset.

All sources are normalized to the same schema:

```json
{
  "prompt": "Prompt text",
  "label": 1,
  "source": "advbench"
}
```

The labels are:

- `1`: risky;
- `0`: safe.

The normalization functions handle the different field names and source-specific label conventions. Prompts, labels,
and source names are validated both when constructing examples and when reading or writing JSONL files.

During dataset construction, prompts are deduplicated by `prompt.strip().lower()`. If the same prompt occurs in more
than one source, the first entry wins according to the current source order:

1. AdvBench
2. HarmBench
3. WildGuard
4. ToxicChat
5. BeaverTails

This ordering matters if two sources provide conflicting labels for the same normalized prompt.

### Current dataset

The current processed dataset contains 34,869 unique prompts:

| Split | Rows | Safe | Risky | Risky share |
| --- | ---: | ---: | ---: | ---: |
| All | 34,869 | 20,206 | 14,663 | 42.05% |
| Train | 27,894 | 16,164 | 11,730 | 42.05% |
| Validation | 3,485 | 2,020 | 1,465 | 42.04% |
| Test | 3,490 | 2,022 | 1,468 | 42.06% |

Distribution by source:

| Source | All | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| AdvBench | 520 | 416 | 52 | 52 |
| BeaverTails | 7,349 | 5,878 | 734 | 737 |
| HarmBench | 200 | 160 | 20 | 20 |
| ToxicChat | 4,965 | 3,972 | 496 | 497 |
| WildGuard | 21,835 | 17,468 | 2,183 | 2,184 |

The current integrity report confirms that:

- all four datasets are non-empty;
- train, validation, and test reconstruct the complete processed dataset exactly;
- no prompt occurs in more than one split;
- none of the datasets contains duplicate prompt rows.

### Dataset splitting

The splitting logic is in `src/shapiq_attribution/split_data.py`.

The current configuration uses:

- 80% training data;
- 10% validation data;
- 10% test data;
- seed `12`;
- stratification by the combination of `label` and `source`.

This preserves both class balance and source balance across the splits as closely as possible. The split is
deterministic: unchanged data, configuration, and seed produce the same output files.

## DVC and artifact versioning

I configured DVC to version the datasets, trained model, and final training metrics.

The default remote is named `storage` and points to:

```text
gs://prompt_classifier_mlops
```

The main files and artifacts are:

- `data/raw.dvc`: raw dataset snapshots;
- `dvc.yaml`: reproducible pipeline definition;
- `dvc.lock`: concrete content hashes and artifact versions;
- `models/prompt_risk_distilbert/`: exported model and tokenizer;
- `reports/metrics.json`: final validation and test metrics.

The DVC pipeline contains three stages:

```text
prepare_dataset
    ↓
split_data
    ↓
train_vertex_ddp
```

`train_vertex_ddp` is the canonical DVC producer of the final trained model and expects two CUDA GPUs. A normal local
training run writes the same model and metrics paths, but it should not be committed as the Vertex DDP stage.

Common commands:

```bash
uv run dvc pull
uv run dvc status
uv run dvc repro split_data
uv run dvc repro train_vertex_ddp
uv run dvc push
```

Local access to the GCS-backed DVC remote normally requires Google Application Default Credentials:

```bash
gcloud auth application-default login
```

## DistilBERT model interface

The model utilities are implemented in `src/shapiq_attribution/model.py`.

They cover:

- loading the Hugging Face tokenizer;
- loading `AutoModelForSequenceClassification`;
- saving the model and tokenizer;
- single-prompt probability prediction;
- batch probability prediction;
- CPU or GPU device selection.

The central serving interface is `PromptRiskPredictor`, which returns the probability assigned to the risky class:

```python
from shapiq_attribution.model import PromptRiskPredictor

predictor = PromptRiskPredictor.from_pretrained(
    "models/prompt_risk_distilbert",
    max_length=128,
    device="cpu",
)

risk_probability = predictor("How do I make a bomb?")
```

The SHAPIQ attribution code also consumes this interface. Changes to `PromptRiskPredictor` should therefore be checked
against the API and attribution tests as well as the model tests. The risky class is currently output index `1`.

## PyTorch Lightning training

I refactored the original training pipeline into PyTorch Lightning.

The code is divided into:

- `lightning_module.py`: model, loss, metrics, optimizer, and scheduler;
- `lightning_data_module.py`: datasets, tokenization, samplers, and DataLoaders;
- `train.py`: Hydra entrypoint and Lightning Trainer configuration.

### LightningModule

The Lightning module computes:

- accuracy;
- precision;
- recall;
- F1;
- ROC-AUC;
- loss.

Separate metric collections are maintained for validation and testing. Metric states and logged values are
synchronized across processes during distributed training.

The optimizer is AdamW with a linear step-wise learning-rate scheduler and no warmup.

### Checkpointing and outputs

The best checkpoint is selected by validation ROC-AUC:

```yaml
checkpoint:
  monitor: val/roc_auc
  mode: max
  save_top_k: 1
  save_last: link
```

After fitting, validation and testing are explicitly run against the best checkpoint. Only global rank zero writes:

- the exported Hugging Face model;
- the tokenizer;
- `reports/metrics.json`;
- the W&B model artifact when online artifact logging is enabled.

### Final model metrics

The currently DVC-versioned model reaches:

| Metric | Validation | Test |
| --- | ---: | ---: |
| Loss | 0.3508 | 0.3397 |
| Accuracy | 0.8460 | 0.8456 |
| F1 | 0.8143 | 0.8121 |
| Precision | 0.8260 | 0.8315 |
| Recall | 0.8029 | 0.7936 |
| ROC-AUC | 0.9283 | 0.9314 |

The exported model is approximately 256 MiB.

## Hydra configuration

The main training configuration is `configs/train.yaml`.

It contains:

- dataset paths and split parameters;
- model name and maximum sequence length;
- epochs, batch size, learning rate, and weight decay;
- hardware and distributed-training parameters;
- checkpoint configuration;
- output paths;
- W&B settings;
- profiling settings.

Hydra overrides can be supplied directly on the command line:

```bash
uv run python -m shapiq_attribution.train \
  hardware=single_gpu \
  training.learning_rate=3e-5 \
  training.batch_size=32
```

### Hardware profiles

The profiles in `configs/hardware/` are:

| Profile | Configuration |
| --- | --- |
| `local` | One automatically selected device, 32-bit precision |
| `single_gpu` | One CUDA GPU, 16-bit mixed precision |
| `ddp` | Two CUDA GPUs, DDP, 16-bit mixed precision |

Examples:

```bash
# Local training without W&B
uv run python -m shapiq_attribution.train wandb.mode=disabled

# One GPU
uv run python -m shapiq_attribution.train hardware=single_gpu

# Two GPUs with DDP
uv run python -m shapiq_attribution.train hardware=ddp
```

## Distributed training with DDP

### What DDP means in this project

DDP stands for PyTorch Distributed Data Parallel. Our setup uses **one machine with two CUDA GPUs**, not two separate
machines. Lightning starts one Python training process per GPU:

```text
Vertex AI worker or two-GPU host
├── rank 0 → GPU 0 → one complete DistilBERT replica
└── rank 1 → GPU 1 → one complete DistilBERT replica
```

Both processes execute the same training code, but they receive different portions of each epoch's data. Each process
performs its own forward and backward pass. During the backward pass, DDP averages the gradients across both processes
with an all-reduce operation. Both optimizers then apply the same synchronized gradients, so the two model replicas
remain identical after every optimizer step.

The high-level sequence for one training step is:

```text
                         full logical training step
                                      │
                   ┌──────────────────┴──────────────────┐
                   │                                     │
          rank 0 / GPU 0                        rank 1 / GPU 1
          receives batch A                      receives batch B
                   │                                     │
              forward pass                          forward pass
                   │                                     │
              local backward                       local backward
                   └──────── gradient all-reduce ────────┘
                                      │
                           averaged gradients
                                      │
                   both ranks run the optimizer step
                                      │
                         identical model parameters
```

DDP is data parallelism: it increases throughput by processing more examples at the same time. It does not split
DistilBERT itself across the GPUs, and it does not reduce the amount of GPU memory required for one model replica.
Each GPU must be able to hold the complete model, optimizer state, gradients, and its local batch.

### DDP Hydra profile

The complete project-specific DDP configuration is in `configs/hardware/ddp.yaml`:

```yaml
# @package training
accelerator: gpu
devices: 2
strategy: ddp
num_nodes: 1
precision: 16-mixed
num_workers: 4
pin_memory: true
persistent_workers: true
```

Each value has a specific role:

| Setting | Meaning |
| --- | --- |
| `accelerator: gpu` | Requires CUDA and selects Lightning's GPU accelerator |
| `devices: 2` | Starts two training processes and assigns one visible GPU to each process |
| `strategy: ddp` | Enables synchronous PyTorch Distributed Data Parallel |
| `num_nodes: 1` | Declares that both GPUs are attached to the same machine |
| `precision: 16-mixed` | Uses FP16 where safe and FP32 where needed through automatic mixed precision |
| `num_workers: 4` | Creates four DataLoader workers per DDP process, eight in total |
| `pin_memory: true` | Uses pinned host memory for faster CPU-to-GPU transfers |
| `persistent_workers: true` | Keeps DataLoader workers alive between epochs |

Hydra applies this file to the `training` section of `configs/train.yaml`. The model, data paths, optimizer settings,
checkpoint settings, and W&B settings continue to come from the main configuration. Only the hardware-related values
are replaced.

### How Lightning activates DDP

`src/shapiq_attribution/train.py` passes the resolved Hydra values directly to `lightning.Trainer`:

```python
trainer = L.Trainer(
    accelerator=str(cfg.training.accelerator),
    devices=int(cfg.training.devices),
    strategy=str(cfg.training.strategy),
    num_nodes=int(cfg.training.num_nodes),
    precision=cfg.training.precision,
    accumulate_grad_batches=int(cfg.training.accumulate_grad_batches),
    # remaining configuration omitted
)
```

With `hardware=ddp`, Lightning is responsible for:

- launching the two worker processes;
- assigning local ranks and GPUs;
- initializing the PyTorch distributed process group;
- broadcasting the initial model state;
- wrapping the model in `DistributedDataParallel`;
- synchronizing gradients after backward passes;
- coordinating checkpoint loading and loop transitions;
- shutting down the distributed process group at the end.

No manual calls to `torch.distributed.init_process_group`, `torchrun`, or gradient all-reduce are needed in our code.
The same Python entrypoint is used for local, single-GPU, and DDP training.

### Data distribution and sampling

`PromptRiskDataModule` constructs the same logical dataset on both ranks. Lightning's
`use_distributed_sampler=True` default detects the distributed strategy and injects a distributed sampler into every
DataLoader.

For the normal validation and test loaders, Lightning replaces the sequential sampler with a
`DistributedSampler`. For the training loader, it wraps our custom `LengthGroupedSampler` in a
`DistributedSamplerWrapper`. This preserves the length-based ordering while assigning a different subset of indices
to each rank.

The resulting behavior is:

- the training examples are deterministically shuffled and grouped by approximate token length;
- rank 0 and rank 1 receive different shards of that ordered index stream;
- both ranks process the same number of optimizer steps;
- dynamic padding still happens independently for the local batch on each GPU;
- the sampler seed is derived from the configured global seed;
- the sampling order changes deterministically between epochs.

This interaction is important because dynamic padding and length grouping still provide the profiling optimization
under DDP, while distributed sampling prevents both GPUs from processing the same full dataset.

### Local and global batch size

`training.batch_size` is a **per-device batch size**. The effective global batch size is:

```text
batch_size_per_device
× number_of_devices
× number_of_nodes
× accumulate_grad_batches
```

With the current defaults:

```text
16 × 2 × 1 × 1 = 32 examples per optimizer step
```

The global batch size is calculated in `train.py` and logged to W&B as `training.global_batch_size`.

This is a key difference from single-GPU training. Running the same configuration with `hardware=single_gpu` uses a
global batch size of 16, while `hardware=ddp` uses 32. If an experiment must keep the global batch size at 16, set the
per-device batch size to 8:

```bash
uv run python -m shapiq_attribution.train \
  hardware=ddp \
  training.batch_size=8
```

Gradient accumulation multiplies the global batch size again. For example, `batch_size=8`, two GPUs, and
`accumulate_grad_batches=2` also produce an effective global batch size of 32, but the optimizer updates only after two
local batches per rank.

### Mixed precision

The DDP profile uses `precision: 16-mixed`. Lightning enables automatic mixed precision and gradient scaling:

- suitable tensor operations run in FP16 for higher GPU throughput and lower activation memory;
- numerically sensitive operations remain in FP32;
- gradient scaling reduces the risk of FP16 underflow;
- the final exported model remains a normal Hugging Face model and does not require mixed-precision inference.

The CUDA training image installs the PyTorch CUDA 12.4 build through the `cu124` dependency extra. The host or Vertex
AI worker must expose compatible NVIDIA drivers and at least two visible GPUs.

### Distributed metrics and model selection

Each rank sees only its own validation and test shard, so local metrics would be incomplete. The Lightning module
therefore synchronizes metric state and logged values across ranks:

```python
self.log(..., sync_dist=True)
self.log_dict(metrics, sync_dist=True)
```

The synchronized values include:

- training loss;
- validation and test loss;
- accuracy;
- precision;
- recall;
- F1;
- ROC-AUC.

Checkpoint selection therefore uses the globally aggregated `val/roc_auc`, not the ROC-AUC of rank 0's local shard.
After fitting, both `trainer.validate(..., ckpt_path="best")` and `trainer.test(..., ckpt_path="best")` load and
evaluate the selected checkpoint under the same distributed strategy.

### Rank-zero side effects

All DDP ranks execute most of `train_model`, but only global rank zero is allowed to write final artifacts:

```python
if trainer.is_global_zero:
    save_prompt_risk_model(...)
    metrics_path.write_text(...)
    log_model_artifact(...)
```

This prevents both processes from writing the same model files, metrics file, or W&B artifact concurrently. After the
rank-zero writes, `trainer.strategy.barrier()` makes every rank wait until the final artifacts are complete before the
training process exits.

Lightning's checkpoint callback is also distributed-aware. The checkpoint represents the synchronized model state,
so it is not necessary to merge two separately trained models after the run.

### DVC integration

The canonical DVC stage is:

```yaml
train_vertex_ddp:
  cmd: uv run python -m shapiq_attribution.train hardware=ddp
```

It declares the three processed splits, training code, model code, hardware profiles, and main Hydra configuration as
dependencies. Its versioned outputs are:

```text
models/prompt_risk_distilbert/
reports/metrics.json
```

On a two-GPU machine with the data already restored, the complete DVC-controlled run is:

```bash
uv run dvc repro train_vertex_ddp
```

After a successful managed cloud run, the container entrypoint can commit and push the outputs:

```text
TRAIN_HARDWARE_PROFILE=ddp
PUSH_DVC_ON_SUCCESS=true
DVC_COMMIT_STAGE=train_vertex_ddp
```

`dockerfiles/train-entrypoint.sh` explicitly rejects `PUSH_DVC_ON_SUCCESS=true` for the `train_vertex_ddp` stage when
the active hardware profile is not `ddp`. This protects the provenance recorded in `dvc.lock`.

### Running a DDP smoke test

Before launching a full cloud job, a short run can verify GPU visibility, process startup, distributed sampling,
gradient synchronization, checkpointing, and validation:

```bash
nvidia-smi

uv run python -m shapiq_attribution.train \
  hardware=ddp \
  wandb.mode=disabled \
  training.epochs=1 \
  training.limit_train_batches=4 \
  training.limit_val_batches=2 \
  training.limit_test_batches=2
```

The command still requires both GPUs and all three processed JSONL splits, but completes much faster than a full
training job.

Useful signs in the Lightning output are:

- `GLOBAL_RANK: 0` and `GLOBAL_RANK: 1`;
- both CUDA devices are registered;
- the strategy is reported as DDP;
- both ranks complete the same number of steps;
- only one final model directory and metrics file are written.

### Running DDP in the GPU container

The GPU image supports DDP, but the host must expose both GPUs to the container. A direct Docker run can use:

```bash
docker build \
  --platform linux/amd64 \
  -t shapiq-train-gpu:latest \
  -f dockerfiles/train.gpu.dockerfile .

docker run --rm --gpus all \
  -e TRAIN_HARDWARE_PROFILE=ddp \
  -e SKIP_DVC_PULL=true \
  -e WANDB_MODE=offline \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  -v "$PWD/reports:/app/reports" \
  -v "$PWD/configs:/app/configs" \
  shapiq-train-gpu:latest
```

`SKIP_DVC_PULL=true` is appropriate when the processed data is mounted from the host. Without it, the container needs
GCP credentials and will pull the three processed splits from DVC before training.

The current `train-gpu` Docker Compose service is configured for `single_gpu` and reserves one GPU. It is intended for
single-GPU development. The two-GPU DDP path uses the same GPU image through Vertex AI or a direct Docker invocation
that exposes both GPUs.

### Vertex AI topology

The final training run used one Vertex AI worker with two attached Tesla T4 GPUs:

```text
Vertex AI Custom Job
└── 1 × n1-standard-16 worker
    ├── Tesla T4 GPU 0 → DDP rank 0
    └── Tesla T4 GPU 1 → DDP rank 1
```

This matches `num_nodes: 1` and `devices: 2`. It is single-node, multi-GPU DDP. A multi-node Vertex setup would need
additional worker replicas and a matching `num_nodes` configuration; that is a different topology from the one used
for the final model.

The Vertex container sequence is:

1. Vertex AI provisions the worker and exposes both GPUs.
2. The training entrypoint selects `TRAIN_HARDWARE_PROFILE=ddp`.
3. The entrypoint pulls the DVC-versioned train, validation, and test splits.
4. Lightning starts two local DDP ranks.
5. Both ranks train synchronously with mixed precision.
6. Global validation ROC-AUC selects the best checkpoint.
7. Rank 0 exports the Hugging Face model and `reports/metrics.json`.
8. The entrypoint commits and pushes the DVC outputs when configured.
9. Optional metadata upload copies `dvc.lock` and the metrics JSON to the configured metadata bucket.

### DDP component map

| Component | DDP responsibility |
| --- | --- |
| `configs/hardware/ddp.yaml` | Selects two GPUs, DDP, one node, mixed precision, and GPU DataLoader settings |
| `configs/train.yaml` | Supplies the per-device batch size, accumulation, seed, checkpoint, and W&B settings |
| `src/shapiq_attribution/train.py` | Creates the Lightning Trainer, calculates global batch size, and limits final writes to rank 0 |
| `src/shapiq_attribution/lightning_module.py` | Defines the replicated model, synchronized losses and metrics, optimizer, and scheduler |
| `src/shapiq_attribution/lightning_data_module.py` | Creates the datasets, dynamic collator, and length-grouped sampler |
| Lightning | Launches ranks, assigns GPUs, injects distributed samplers, synchronizes gradients, and coordinates loops |
| PyTorch DDP/NCCL | Performs GPU-to-GPU gradient communication |
| `dockerfiles/train.gpu.dockerfile` | Provides CUDA 12.4 and the matching PyTorch build |
| `dockerfiles/train-entrypoint.sh` | Selects the DDP profile, pulls data, starts training, and handles successful DVC publication |
| `dvc.yaml` | Defines the canonical `train_vertex_ddp` stage and its versioned dependencies and outputs |
| Vertex AI | Provisions and manages the one-worker, two-GPU cloud environment |

## Profiling and performance optimization

I integrated both Lightning Simple Profiling and PyTorch Profiling.

The configuration groups are in `configs/profiling/`:

- `disabled.yaml`;
- `simple.yaml`;
- `pytorch.yaml`.

Run them with:

```bash
uv run python -m shapiq_attribution.train profiling=simple
uv run python -m shapiq_attribution.train profiling=pytorch
```

The profiles showed that the original fixed padding length of 128 tokens wasted a substantial amount of computation.
The 27,894 training prompts contain 53.91 tokens on average, with a median of 21.

Based on these results, I implemented:

- dynamic padding to the longest sequence in each batch;
- tokenization at batch level;
- deterministic length-grouped batching;
- an epoch-aware `LengthGroupedSampler`.

Results from matched 100-batch CPU profiles:

| Measurement | Fixed padding | Optimized | Change |
| --- | ---: | ---: | ---: |
| Mean padded batch width | 128.00 | 56.67 | -55.73% |
| Total fit profiler time | 62.527 s | 31.360 s | -49.85% |
| Training-batch time | 59.750 s | 28.335 s | -52.58% |
| Mean training-step time | 205.37 ms | 90.26 ms | -56.05% |

Profiler traces are written below `outputs/profiling/`, which is ignored by Git.

## W&B experiment tracking

Training is tracked through Lightning's `WandbLogger`.

Each run records:

- the fully resolved Hydra configuration;
- the effective global batch size;
- training and validation loss;
- accuracy, precision, recall, F1, and ROC-AUC;
- the learning rate;
- the selected hardware profile;
- the model artifact and metrics when artifact logging is enabled.

The W&B API key can be supplied directly through `WANDB_API_KEY` or loaded in cloud environments from GCP Secret
Manager:

```text
WANDB_API_KEY_SECRET=projects/PROJECT_ID/secrets/SECRET_ID/versions/latest
```

### Hyperparameter sweep

`configs/sweep.yaml` defines a Bayesian sweep over:

- learning rate;
- batch size;
- weight decay;
- maximum sequence length;
- number of training epochs.

The optimization target is `val/roc_auc`.

## W&B Model Registry

The registry integration is implemented in `src/shapiq_attribution/model_registry.py`.

An online training run can publish the exported Hugging Face model and `reports/metrics.json` as a versioned W&B model
artifact. The artifact metadata also records:

- the Git commit;
- the SHA-256 hash of `dvc.lock`;
- validation and test metrics.

Newly published models are initially assigned the `candidate` alias.

An existing local model can be published without retraining:

```bash
WANDB_ENTITY=<team> \
WANDB_PROJECT=shapiq-attribution \
uv run python -m shapiq_attribution.model_registry publish
```

### Staging and validation

When the `staging` alias is assigned in W&B, the configured W&B automation sends a `repository_dispatch` event to
GitHub.

The `.github/workflows/stage-model.yaml` workflow:

1. downloads the exact W&B artifact version;
2. pulls the DVC-versioned test split;
3. selects a deterministic, approximately balanced subset of at most 512 examples;
4. checks the model structure;
5. runs batched CPU inference;
6. writes Markdown and JSON validation reports;
7. assigns the `production` alias only if all checks pass.

The validation requirements are:

- model configuration and weights are present;
- probabilities are finite and between 0 and 1;
- both classes are predicted;
- F1 is at least 0.70;
- ROC-AUC is at least 0.85.

## Docker and cloud training

I created separate CPU and GPU training images:

| Image | Dockerfile | Purpose |
| --- | --- | --- |
| `shapiq-train` | `dockerfiles/train.dockerfile` | CPU training |
| `shapiq-train-gpu` | `dockerfiles/train.gpu.dockerfile` | CUDA training |

The GPU image uses CUDA 12.4 and the `cu124` PyTorch extra. The CPU images and Linux CI use the `cpu` extra, preventing
CPU environments from installing the roughly 7 GB CUDA dependency stack.

Both images use `dockerfiles/train-entrypoint.sh`. The entrypoint:

1. selects the Hydra hardware profile;
2. pulls the DVC training splits when required;
3. starts the training process;
4. optionally commits and pushes successful DVC outputs;
5. can upload `dvc.lock` and `reports/metrics.json` to a separate metadata bucket.

Important environment variables:

| Variable | Purpose |
| --- | --- |
| `TRAIN_HARDWARE_PROFILE` | Selects `local`, `single_gpu`, or `ddp` |
| `SKIP_DVC_PULL` | Skips the DVC pull inside the container |
| `PUSH_DVC_ON_SUCCESS` | Commits and pushes successful training outputs |
| `DVC_COMMIT_STAGE` | Selects the DVC stage to commit |
| `DVC_METADATA_BUCKET` | Optional bucket for DVC and metric metadata |
| `DVC_METADATA_PREFIX` | Optional object prefix in that bucket |
| `WANDB_API_KEY_SECRET` | Secret Manager resource for the W&B credential |

The entrypoint refuses to commit `train_vertex_ddp` unless the active hardware profile is `ddp`. This prevents a local
or single-GPU model from being recorded as the Vertex DDP result.

### Cloud-training history

Cloud training was completed in two stages:

1. The earlier single-GPU training pipeline ran on a Compute Engine VM with one Tesla T4.
2. After the Lightning refactor, the final model was trained as a Vertex AI Custom Job on an `n1-standard-16` worker
   with two Tesla T4 GPUs.

The final run:

- pulled the DVC-versioned datasets from GCS;
- used both GPUs through Lightning DDP;
- used 16-bit mixed precision;
- logged configuration and metrics to W&B;
- versioned the resulting model and metrics through DVC.

## Continuous integration

The GitHub Actions workflows are under `.github/workflows/`.

### `ci-lint.yaml`

Runs on pushes and pull requests targeting `main`:

- `ruff check`;
- `ruff format --check`.

### `ci-train.yaml`

Checks the training side:

- data tests;
- split tests;
- model tests;
- Lightning training tests;
- CPU training image build.

### `ci-api.yaml`

Checks the serving side:

- API tests;
- monitoring tests;
- experiment and attribution test slice;
- coverage report;
- API image build.

### `cml-data.yaml`

Runs when DVC metadata or DVC pipeline definitions change.

The workflow:

1. authenticates to GCP with `GCP_SA_KEY`;
2. pulls the raw data snapshots;
3. reproduces preprocessing and splitting;
4. runs the data tests;
5. generates the Data Checker report;
6. uploads the report as a workflow artifact;
7. updates a CML comment on pull requests;
8. fails the job when an integrity check fails.

The same report can be generated locally:

```bash
uv run python -m shapiq_attribution.data_statistics --check
```

### `stage-model.yaml`

Validates a staged W&B artifact and assigns the `production` alias only after all validation gates pass.

Required GitHub Secrets:

- `WANDB_API_KEY`;
- `WANDB_ENTITY`;
- `WANDB_PROJECT`;
- `GCP_SA_KEY`.

## Local setup

### 1. Clone and install

```bash
git clone git@github.com:ssophiee/shapiq-cot-attribution.git
cd shapiq-cot-attribution
```

On macOS:

```bash
uv sync --frozen
```

On Linux, using CPU-only PyTorch:

```bash
uv sync --frozen --extra cpu
```

### 2. Run the tests

```bash
uv run pytest tests/
```

Expected handover status:

```text
100 passed
```

### 3. Restore the data and model

```bash
gcloud auth application-default login
uv run dvc pull
```

The following paths should then be present:

```text
data/processed/prompt_risk_dataset.jsonl
data/processed/train.jsonl
data/processed/val.jsonl
data/processed/test.jsonl
models/prompt_risk_distilbert/
reports/metrics.json
```

### 4. Start the application locally

```bash
uv run uvicorn shapiq_attribution.api:app --port 8000
```

Local endpoints:

- web interface: `http://localhost:8000`;
- Swagger UI: `http://localhost:8000/docs`;
- health check: `http://localhost:8000/health`.

## Common workflows

### Check data integrity

```bash
uv run python -m shapiq_attribution.data_statistics --check
```

### Rebuild the processed data

```bash
uv run dvc repro prepare_dataset
uv run dvc repro split_data
```

### Train locally without W&B

```bash
uv run python -m shapiq_attribution.train wandb.mode=disabled
```

### Train with offline W&B logging

```bash
uv run python -m shapiq_attribution.train wandb.mode=offline
```

### Train on one GPU

```bash
uv run python -m shapiq_attribution.train hardware=single_gpu
```

### Train with DDP

```bash
uv run python -m shapiq_attribution.train hardware=ddp
```

### Build the CPU training image

```bash
uv run invoke docker-build-train
```

### Build the GPU training image

```bash
uv run invoke docker-build-train-gpu
```

### Train through Docker Compose

```bash
docker compose --profile train up train
docker compose --profile gpu up train-gpu
```

### Run code-quality checks

```bash
uv run ruff check . --fix
uv run ruff format .
uv run pre-commit run --all-files
```

### Preview the documentation

```bash
uv run invoke serve-docs
```

## Required access

The credentials themselves are not stored in the repository and must be transferred separately.

### GitHub

- Write access to the repository
- Access to the Actions configuration and repository secrets
- Permission to start workflows manually

### GCP: data and training

- Access to the GCP project containing the DVC bucket
- Access to `gs://prompt_classifier_mlops`
- Vertex AI permissions where relevant
- Access to the W&B credential in Secret Manager
- Access to the service-account configuration used by GitHub Actions

### GCP: serving

According to the project documentation, the deployed service runs in the separate `mlops-shapiq-project` project.
This area was maintained primarily by Sofiia and has its own IAM permissions.

### Weights & Biases

- Access to the `shapiq-attribution` project
- Access to the Model Registry
- A W&B API key
- The configured W&B entity
- Access to Registry Automations and webhook settings

### Hugging Face

- `HF_TOKEN` when a source dataset does not permit anonymous access or is gated

## Interfaces with Sofiia's components

The training pipeline exports a standard Hugging Face model to:

```text
models/prompt_risk_distilbert/
```

The serving and attribution components use this location, or the path configured through `MODEL_DIR`.

The model is consumed through `PromptRiskPredictor`, particularly:

- `predict_proba(prompt)`;
- `predict_proba_batch(prompts)`;
- `__call__(prompt)`.

Changes to tokenization, output types, or label ordering can affect both the API and the attribution pipeline. The
risky class is output index `1`.

## Most important files

| File | Purpose |
| --- | --- |
| `src/shapiq_attribution/data.py` | Download, normalization, deduplication, validation, and JSONL I/O |
| `src/shapiq_attribution/split_data.py` | Deterministic stratified splitting |
| `src/shapiq_attribution/model.py` | Model loading, saving, and predictor interface |
| `src/shapiq_attribution/lightning_module.py` | Training, metrics, optimizer, and scheduler |
| `src/shapiq_attribution/lightning_data_module.py` | DataLoaders, dynamic padding, and length grouping |
| `src/shapiq_attribution/train.py` | Hydra and Lightning training entrypoint |
| `src/shapiq_attribution/data_statistics.py` | Continuous data-integrity reporting |
| `src/shapiq_attribution/model_registry.py` | Model publication, validation, and promotion |
| `configs/train.yaml` | Main training configuration |
| `configs/hardware/` | Local, single-GPU, and DDP profiles |
| `configs/profiling/` | Lightning and PyTorch profiling profiles |
| `configs/sweep.yaml` | W&B hyperparameter sweep |
| `dvc.yaml` | Reproducible data and training pipeline |
| `dvc.lock` | Concrete artifact versions and hashes |
| `dockerfiles/train-entrypoint.sh` | Container training entrypoint |
| `.github/workflows/cml-data.yaml` | Continuous data validation |
| `.github/workflows/stage-model.yaml` | Continuous staged-model validation |
| `docs/source/training.md` | Detailed training documentation |
| `DOCKER.md` | Container and entrypoint documentation |

## Summary

My part of the project provides a reproducible and tested data and training pipeline. Five public prompt-safety
datasets are normalized, deduplicated, split with source-and-label stratification, and versioned through DVC.
DistilBERT is trained with PyTorch Lightning and Hydra and can run locally, on one GPU, or with DDP.

Profiling led to dynamic padding and length-grouped batching, approximately halving the measured CPU training time in
the matched profiling experiment. The final model was trained on Vertex AI with two Tesla T4 GPUs and achieved a test
ROC-AUC of 0.9314 and a test F1 score of 0.8121.

Continuous data validation and staged-model validation provide automated quality gates around the two most important
ML artifacts: the dataset and the trained classifier.
