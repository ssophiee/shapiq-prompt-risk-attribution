"""Generate a Markdown quality report for DVC-tracked prompt-risk data."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from shapiq_attribution.data import PromptRiskExample, load_prompt_risk_jsonl

DEFAULT_DATASETS = {
    "all": Path("data/processed/prompt_risk_dataset.jsonl"),
    "train": Path("data/processed/train.jsonl"),
    "validation": Path("data/processed/val.jsonl"),
    "test": Path("data/processed/test.jsonl"),
}


def _example_key(example: PromptRiskExample) -> tuple[str, int, str]:
    """Return the fields that uniquely identify a normalized example."""
    return example["prompt"], example["label"], example["source"]


def _prompt_set(examples: Sequence[PromptRiskExample]) -> set[str]:
    """Return unique prompt texts from a dataset."""
    return {example["prompt"] for example in examples}


def _format_percentage(count: int, total: int) -> str:
    """Format a count as a percentage of a non-empty dataset."""
    return f"{count / total:.2%}" if total else "n/a"


def build_data_statistics_report(
    dataset_paths: Mapping[str, str | Path] = DEFAULT_DATASETS,
) -> tuple[str, list[str]]:
    """Build a Markdown data report and return any integrity issues.

    Args:
        dataset_paths: Mapping containing ``all``, ``train``, ``validation`` and
            ``test`` JSONL paths.

    Returns:
        Markdown report and a list of integrity issues.

    Raises:
        ValueError: If a required dataset name is missing.
    """
    required_names = set(DEFAULT_DATASETS)
    missing_names = required_names.difference(dataset_paths)
    if missing_names:
        raise ValueError(f"missing dataset paths: {', '.join(sorted(missing_names))}")

    datasets = {name: load_prompt_risk_jsonl(path) for name, path in dataset_paths.items() if name in required_names}
    issues: list[str] = []

    for name, examples in datasets.items():
        if not examples:
            issues.append(f"`{name}` is empty")

    split_names = ("train", "validation", "test")
    split_examples = [example for name in split_names for example in datasets[name]]
    if Counter(map(_example_key, datasets["all"])) != Counter(map(_example_key, split_examples)):
        issues.append("train, validation and test do not exactly reconstruct the full processed dataset")

    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1 :]:
            overlap = _prompt_set(datasets[left_name]).intersection(_prompt_set(datasets[right_name]))
            if overlap:
                issues.append(f"{left_name} and {right_name} share {len(overlap)} prompt(s)")

    for name, examples in datasets.items():
        duplicate_count = len(examples) - len(_prompt_set(examples))
        if duplicate_count:
            issues.append(f"`{name}` contains {duplicate_count} duplicate prompt row(s)")

    status = "PASS" if not issues else "FAIL"
    lines = [
        "# Data Checker",
        "",
        f"**Integrity status: {status}**",
        "",
        "## Dataset summary",
        "",
        "| Dataset | Rows | Safe | Risky | Risky share | Unique prompts | Duplicate rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("all", *split_names):
        examples = datasets[name]
        label_counts = Counter(example["label"] for example in examples)
        unique_prompts = len(_prompt_set(examples))
        lines.append(
            f"| {name} | {len(examples)} | {label_counts[0]} | {label_counts[1]} | "
            f"{_format_percentage(label_counts[1], len(examples))} | {unique_prompts} | "
            f"{len(examples) - unique_prompts} |"
        )

    sources = sorted({example["source"] for examples in datasets.values() for example in examples})
    lines.extend(
        [
            "",
            "## Source distribution",
            "",
            "| Source | All | Train | Validation | Test |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    source_counts = {name: Counter(example["source"] for example in examples) for name, examples in datasets.items()}
    for source in sources:
        lines.append(
            f"| {source} | {source_counts['all'][source]} | {source_counts['train'][source]} | "
            f"{source_counts['validation'][source]} | {source_counts['test'][source]} |"
        )

    lines.extend(["", "## Integrity checks", ""])
    if issues:
        lines.extend(f"- ❌ {issue}" for issue in issues)
    else:
        lines.extend(
            [
                "- ✅ All four datasets are non-empty.",
                "- ✅ Train, validation and test exactly reconstruct the processed dataset.",
                "- ✅ No prompt occurs in more than one split.",
                "- ✅ No dataset contains duplicate prompt rows.",
            ]
        )

    return "\n".join(lines) + "\n", issues


def main() -> None:
    """Print the Markdown data report and optionally fail on integrity issues."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-path", type=Path, default=DEFAULT_DATASETS["all"])
    parser.add_argument("--train-path", type=Path, default=DEFAULT_DATASETS["train"])
    parser.add_argument("--validation-path", type=Path, default=DEFAULT_DATASETS["validation"])
    parser.add_argument("--test-path", type=Path, default=DEFAULT_DATASETS["test"])
    parser.add_argument("--check", action="store_true", help="Exit with status 1 when an integrity check fails.")
    args = parser.parse_args()

    report, issues = build_data_statistics_report(
        {
            "all": args.all_path,
            "train": args.train_path,
            "validation": args.validation_path,
            "test": args.test_path,
        }
    )
    print(report, end="")
    if args.check and issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
