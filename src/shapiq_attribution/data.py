"""Data loading and normalization for prompt-risk classification."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

ADV_BENCH_DATASET = "walledai/AdvBench"
HARM_BENCH_DATASET = "walledai/HarmBench"
HARM_BENCH_CONFIG = "standard"
WILDGUARD_DATASET = "allenai/wildguardmix"
WILDGUARD_CONFIG = "wildguardtest"
BEAVERTAILS_DATASET = "PKU-Alignment/BeaverTails"
PKU_SAFE_RLHF_DATASET = "PKU-Alignment/PKU-SafeRLHF"
TOXIC_CHAT_DATASET = "lmsys/toxic-chat"
TOXIC_CHAT_CONFIG = "toxicchat0124"
XSTEST_DATASET = "walledai/XSTest"


class PromptRiskExample(TypedDict):
    """Normalized prompt-risk example used by Person A's training pipeline."""

    prompt: str
    label: int
    source: str


def make_prompt_risk_example(prompt: str, label: int, source: str) -> PromptRiskExample:
    """Create and validate a normalized prompt-risk example.

    Args:
        prompt: User prompt text.
        label: Binary label where 1 means risky and 0 means not risky.
        source: Dataset source name.

    Returns:
        A normalized prompt-risk example.

    Raises:
        ValueError: If prompt, label, or source is invalid.
    """
    normalized_prompt = prompt.strip()
    normalized_source = source.strip()
    if not normalized_prompt:
        raise ValueError("prompt must be a non-empty string")
    if label not in {0, 1}:
        raise ValueError("label must be 0 or 1")
    if not normalized_source:
        raise ValueError("source must be a non-empty string")
    return {"prompt": normalized_prompt, "label": label, "source": normalized_source}


def _first_text_value(row: Mapping[str, Any], fields: Sequence[str]) -> str | None:
    """Return the first non-empty text value from a row."""
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, Sequence) and not isinstance(value, str):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item
    return None


def _limit_examples(examples: list[PromptRiskExample], max_samples: int | None) -> list[PromptRiskExample]:
    """Limit examples when a maximum sample count is provided."""
    if max_samples is None:
        return examples
    return examples[:max_samples]


def normalize_advbench_rows(
    rows: Iterable[Mapping[str, Any]],
    max_samples: int | None = None,
) -> list[PromptRiskExample]:
    """Normalize AdvBench rows as risky prompt examples.

    Args:
        rows: Raw AdvBench rows.
        max_samples: Optional maximum number of examples to return.

    Returns:
        Prompt-risk examples with label 1.
    """
    examples = []
    prompt_fields = ("prompt", "goal", "instruction", "instructions", "content")
    for row in rows:
        prompt = _first_text_value(row, prompt_fields)
        if prompt is not None:
            examples.append(make_prompt_risk_example(prompt=prompt, label=1, source="advbench"))
    return _limit_examples(examples, max_samples)


def normalize_harmbench_rows(
    rows: Iterable[Mapping[str, Any]],
    max_samples: int | None = None,
) -> list[PromptRiskExample]:
    """Normalize HarmBench rows as risky prompt examples.

    Args:
        rows: Raw HarmBench rows.
        max_samples: Optional maximum number of examples to return.

    Returns:
        Prompt-risk examples with label 1.
    """
    examples = []
    prompt_fields = ("Behavior", "behavior", "prompt", "goal", "instruction")
    for row in rows:
        prompt = _first_text_value(row, prompt_fields)
        if prompt is not None:
            examples.append(make_prompt_risk_example(prompt=prompt, label=1, source="harmbench"))
    return _limit_examples(examples, max_samples)


def normalize_wildguard_safe_rows(
    rows: Iterable[Mapping[str, Any]],
    max_samples: int | None = None,
) -> list[PromptRiskExample]:
    """Normalize safe WildGuard rows as not-risky prompt examples.

    Args:
        rows: Raw WildGuardMix rows.
        max_samples: Optional maximum number of examples to return.

    Returns:
        Prompt-risk examples with label 0.
    """
    examples = []
    safe_labels = {"benign", "safe", "unharmful", "not harmful", "not_harmful"}
    for row in rows:
        prompt = _first_text_value(row, ("prompt",))
        prompt_harm_label = str(row.get("prompt_harm_label", "")).strip().lower()
        prompt_type = str(row.get("prompt_type", "")).strip().lower()
        if prompt is not None and (prompt_harm_label in safe_labels or prompt_type == "prompt_safe"):
            examples.append(make_prompt_risk_example(prompt=prompt, label=0, source="wildguard"))
    return _limit_examples(examples, max_samples)


def normalize_wildguard_prompt_rows(
    rows: Iterable[Mapping[str, Any]],
    max_samples: int | None = None,
) -> list[PromptRiskExample]:
    """Normalize WildGuardMix rows using prompt harmfulness labels."""
    examples = []
    for row in rows:
        prompt = _first_text_value(row, ("prompt",))
        label_value = str(row.get("prompt_harm_label", "")).strip().lower()
        if prompt is None or label_value not in {"harmful", "unharmful"}:
            continue
        label = 1 if label_value == "harmful" else 0
        examples.append(make_prompt_risk_example(prompt=prompt, label=label, source="wildguard"))
    return _limit_examples(examples, max_samples)


def normalize_toxic_chat_rows(
    rows: Iterable[Mapping[str, Any]],
    max_samples: int | None = None,
) -> list[PromptRiskExample]:
    """Normalize ToxicChat rows from toxicity and jailbreaking labels."""
    examples = []
    for row in rows:
        prompt = _first_text_value(row, ("user_input",))
        if prompt is None:
            continue
        toxicity = int(row.get("toxicity", 0) or 0)
        jailbreaking = int(row.get("jailbreaking", 0) or 0)
        label = 1 if toxicity == 1 or jailbreaking == 1 else 0
        examples.append(make_prompt_risk_example(prompt=prompt, label=label, source="toxicchat"))
    return _limit_examples(examples, max_samples)


def normalize_beavertails_rows(
    rows: Iterable[Mapping[str, Any]],
    max_samples: int | None = None,
) -> list[PromptRiskExample]:
    """Normalize BeaverTails rows using prompt and is_safe labels."""
    examples = []
    for row in rows:
        prompt = _first_text_value(row, ("prompt",))
        if prompt is None or "is_safe" not in row:
            continue
        label = 0 if bool(row["is_safe"]) else 1
        examples.append(make_prompt_risk_example(prompt=prompt, label=label, source="beavertails"))
    return _limit_examples(examples, max_samples)


def normalize_xstest_rows(
    rows: Iterable[Mapping[str, Any]],
    max_samples: int | None = None,
) -> list[PromptRiskExample]:
    """Normalize XSTest rows as an evaluation set."""
    examples = []
    for row in rows:
        prompt = _first_text_value(row, ("prompt",))
        label_value = str(row.get("label", row.get("type", ""))).strip().lower()
        if prompt is None:
            continue
        if "safe" in label_value and "unsafe" not in label_value:
            label = 0
        elif "unsafe" in label_value:
            label = 1
        else:
            continue
        examples.append(make_prompt_risk_example(prompt=prompt, label=label, source="xstest"))
    return _limit_examples(examples, max_samples)


def load_advbench(token: str | None = None, max_samples: int | None = None) -> list[PromptRiskExample]:
    """Load and normalize AdvBench from Hugging Face.

    Args:
        token: Optional Hugging Face token for gated dataset access.
        max_samples: Optional maximum number of examples to return.

    Returns:
        Normalized risky prompt examples.
    """
    rows = load_dataset(ADV_BENCH_DATASET, split="train", token=token)
    return normalize_advbench_rows(rows, max_samples=max_samples)


def load_harmbench(token: str | None = None, max_samples: int | None = None) -> list[PromptRiskExample]:
    """Load and normalize HarmBench from Hugging Face.

    Args:
        token: Optional Hugging Face token for gated dataset access.
        max_samples: Optional maximum number of examples to return.

    Returns:
        Normalized risky prompt examples.
    """
    rows = load_dataset(HARM_BENCH_DATASET, HARM_BENCH_CONFIG, split="train", token=token)
    return normalize_harmbench_rows(rows, max_samples=max_samples)


def load_wildguard_safe(token: str | None = None, max_samples: int | None = None) -> list[PromptRiskExample]:
    """Load and normalize safe WildGuardMix prompts from Hugging Face.

    Args:
        token: Optional Hugging Face token for gated dataset access.
        max_samples: Optional maximum number of examples to return.

    Returns:
        Normalized not-risky prompt examples.
    """
    rows = load_dataset(WILDGUARD_DATASET, WILDGUARD_CONFIG, split="test", token=token)
    return normalize_wildguard_safe_rows(rows, max_samples=max_samples)


def load_wildguard(token: str, split: str = "test") -> list[dict[str, Any]]:
    """Load raw WildGuardMix examples for compatibility with existing pipeline code."""
    ds = load_dataset(WILDGUARD_DATASET, WILDGUARD_CONFIG, token=token)
    return list(ds[split])


def load_wildguard_prompts(token: str | None = None, max_samples: int | None = None) -> list[PromptRiskExample]:
    """Load WildGuardMix prompt examples with harmful and unharmful labels."""
    rows = load_dataset(WILDGUARD_DATASET, "wildguardtrain", split="train", token=token)
    return normalize_wildguard_prompt_rows(rows, max_samples=max_samples)


def load_toxic_chat(token: str | None = None, max_samples: int | None = None) -> list[PromptRiskExample]:
    """Load ToxicChat prompt examples."""
    rows = load_dataset(TOXIC_CHAT_DATASET, TOXIC_CHAT_CONFIG, split="train", token=token)
    return normalize_toxic_chat_rows(rows, max_samples=max_samples)


def load_beavertails(token: str | None = None, max_samples: int | None = None) -> list[PromptRiskExample]:
    """Load BeaverTails prompt examples."""
    rows = load_dataset(BEAVERTAILS_DATASET, split="330k_train", token=token)
    return normalize_beavertails_rows(rows, max_samples=max_samples)


def load_xstest(token: str | None = None, max_samples: int | None = None) -> list[PromptRiskExample]:
    """Load XSTest examples for evaluation."""
    rows = load_dataset(XSTEST_DATASET, split="test", token=token)
    return normalize_xstest_rows(rows, max_samples=max_samples)


def get_harmful_examples(token: str, split: str = "test") -> list[dict[str, Any]]:
    """Return harmful raw WildGuardMix examples for compatibility with existing pipeline code."""
    examples = load_wildguard(token, split)
    return [ex for ex in examples if ex.get("prompt_harm_label") == "harmful" and ex.get("prompt")]


def write_prompt_risk_jsonl(examples: Iterable[PromptRiskExample], path: str | Path) -> None:
    """Write prompt-risk examples to a JSONL file.

    Args:
        examples: Normalized prompt-risk examples.
        path: Output JSONL path.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for example in examples:
            validated = make_prompt_risk_example(
                prompt=example["prompt"],
                label=example["label"],
                source=example["source"],
            )
            file.write(f"{json.dumps(validated, ensure_ascii=False)}\n")


def load_prompt_risk_jsonl(path: str | Path) -> list[PromptRiskExample]:
    """Load prompt-risk examples from a JSONL file.

    Args:
        path: Input JSONL path.

    Returns:
        Normalized prompt-risk examples.
    """
    examples = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                examples.append(
                    make_prompt_risk_example(
                        prompt=row["prompt"],
                        label=row["label"],
                        source=row["source"],
                    )
                )
            except KeyError as error:
                raise ValueError(f"missing required field on line {line_number}: {error.args[0]}") from error
    return examples


def prepare_raw_snapshots(
    output_dir: str | Path = "data/raw",
    hf_token: str | None = None,
    max_samples_per_source: int | None = None,
) -> dict[str, Path]:
    """Download and write raw normalized JSONL snapshots for DVC tracking.

    Args:
        output_dir: Directory for JSONL snapshots.
        hf_token: Optional Hugging Face token for gated dataset access.
        max_samples_per_source: Optional maximum examples per source.

    Returns:
        Mapping from source name to written JSONL path.
    """
    output_path = Path(output_dir)
    token = hf_token or os.environ.get("HF_TOKEN")
    snapshots = {
        "advbench": (
            load_advbench(token=token, max_samples=max_samples_per_source),
            output_path / "advbench.jsonl",
        ),
        "harmbench": (
            load_harmbench(token=token, max_samples=max_samples_per_source),
            output_path / "harmbench.jsonl",
        ),
        "wildguard": (
            load_wildguard_prompts(token=token, max_samples=25000),
            output_path / "wildguard.jsonl",
        ),
        "toxicchat": (
            load_toxic_chat(token=token, max_samples=10000),
            output_path / "toxicchat.jsonl",
        ),
        "beavertails": (
            load_beavertails(token=token, max_samples=12000),
            output_path / "beavertails.jsonl",
        ),
        "xstest": (
            load_xstest(token=token),
            output_path / "xstest_eval.jsonl",
        ),
    }
    written_paths = {}
    for source, (examples, path) in snapshots.items():
        write_prompt_risk_jsonl(examples, path)
        written_paths[source] = path
    return written_paths


def main(
    output_dir: str = "data/raw",
    hf_token: str | None = None,
    max_samples_per_source: int | None = None,
) -> None:
    """Create local raw JSONL snapshots for DVC tracking.

    Args:
        output_dir: Directory for JSONL snapshots.
        hf_token: Optional Hugging Face token for gated dataset access.
        max_samples_per_source: Optional maximum examples per source.
    """
    written_paths = prepare_raw_snapshots(
        output_dir=output_dir,
        hf_token=hf_token,
        max_samples_per_source=max_samples_per_source,
    )
    for source, path in written_paths.items():
        print(f"{source}: {path}")


def build_prompt_risk_dataset(raw_dir: str | Path, output_path: str | Path) -> None:
    """Build the processed prompt-risk dataset from local raw JSONL snapshots.

    Args:
        raw_dir: Directory containing raw normalized JSONL snapshots.
        output_path: Output path for the processed prompt-risk dataset.
    """
    raw_path = Path(raw_dir)
    examples = []
    seen_prompts = set()

    for filename in (
        "advbench.jsonl",
        "harmbench.jsonl",
        "wildguard.jsonl",
        "toxicchat.jsonl",
        "beavertails.jsonl",
    ):
        for example in load_prompt_risk_jsonl(raw_path / filename):
            prompt_key = example["prompt"].strip().lower()
            if prompt_key in seen_prompts:
                continue
            seen_prompts.add(prompt_key)
            examples.append(example)

    write_prompt_risk_jsonl(examples, output_path)


def cli() -> None:
    """Parse command-line arguments for data preparation tasks."""
    parser = argparse.ArgumentParser(description="Prompt-risk data utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshots_parser = subparsers.add_parser("raw-snapshots", help="Create local raw JSONL snapshots.")
    snapshots_parser.add_argument("--output-dir", default="data/raw")
    snapshots_parser.add_argument("--hf-token", default=None)
    snapshots_parser.add_argument("--max-samples-per-source", type=int, default=None)

    build_parser = subparsers.add_parser("build-dataset", help="Build processed dataset from raw snapshots.")
    build_parser.add_argument("--raw-dir", default="data/raw")
    build_parser.add_argument("--output-path", default="data/processed/prompt_risk_dataset.jsonl")

    args = parser.parse_args()
    if args.command == "raw-snapshots":
        main(
            output_dir=args.output_dir,
            hf_token=args.hf_token,
            max_samples_per_source=args.max_samples_per_source,
        )
    elif args.command == "build-dataset":
        build_prompt_risk_dataset(raw_dir=args.raw_dir, output_path=args.output_path)


class PromptRiskDataset(Dataset):
    """Torch dataset backed by a normalized prompt-risk JSONL file."""

    def __init__(self, path: str | Path) -> None:
        """Load examples from a JSONL file.

        Args:
            path: Input JSONL path.
        """
        self.examples = load_prompt_risk_jsonl(path)

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.examples)

    def __getitem__(self, index: int) -> PromptRiskExample:
        """Return one prompt-risk example by index."""
        return self.examples[index]


class PromptRiskTextDataset(Dataset):
    """Tokenized prompt-risk dataset for transformer training."""

    def __init__(self, examples: list[PromptRiskExample], tokenizer: Any, max_length: int) -> None:
        """Initialize the dataset.

        Args:
            examples: Prompt-risk examples.
            tokenizer: Hugging Face tokenizer.
            max_length: Maximum tokenized sequence length.
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one tokenized training example."""
        example = self.examples[index]
        encoded = self.tokenizer(
            example["prompt"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(example["label"], dtype=torch.long)
        return item


if __name__ == "__main__":
    cli()
