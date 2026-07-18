"""Tests for the continuous-data Markdown report."""

from pathlib import Path

from shapiq_attribution.data import PromptRiskExample, write_prompt_risk_jsonl
from shapiq_attribution.data_statistics import build_data_statistics_report


def _write_datasets(
    tmp_path: Path,
    train: list[PromptRiskExample],
    validation: list[PromptRiskExample],
    test: list[PromptRiskExample],
) -> dict[str, Path]:
    """Write a complete set of processed dataset fixtures."""
    paths = {
        "all": tmp_path / "all.jsonl",
        "train": tmp_path / "train.jsonl",
        "validation": tmp_path / "validation.jsonl",
        "test": tmp_path / "test.jsonl",
    }
    write_prompt_risk_jsonl(train + validation + test, paths["all"])
    write_prompt_risk_jsonl(train, paths["train"])
    write_prompt_risk_jsonl(validation, paths["validation"])
    write_prompt_risk_jsonl(test, paths["test"])
    return paths


def test_data_statistics_report_summarizes_valid_splits(tmp_path: Path) -> None:
    """Test Markdown statistics and successful integrity checks."""
    paths = _write_datasets(
        tmp_path,
        train=[
            {"prompt": "safe train", "label": 0, "source": "source-a"},
            {"prompt": "risky train", "label": 1, "source": "source-b"},
        ],
        validation=[{"prompt": "safe validation", "label": 0, "source": "source-a"}],
        test=[{"prompt": "risky test", "label": 1, "source": "source-b"}],
    )

    report, issues = build_data_statistics_report(paths)

    assert issues == []
    assert "**Integrity status: PASS**" in report
    assert "| all | 4 | 2 | 2 | 50.00% | 4 | 0 |" in report
    assert "| source-a | 2 | 1 | 1 | 0 |" in report


def test_data_statistics_report_detects_split_overlap(tmp_path: Path) -> None:
    """Test that repeated prompts across splits fail the integrity report."""
    shared = {"prompt": "shared prompt", "label": 0, "source": "source-a"}
    paths = _write_datasets(
        tmp_path,
        train=[shared],
        validation=[shared],
        test=[{"prompt": "test prompt", "label": 1, "source": "source-b"}],
    )

    report, issues = build_data_statistics_report(paths)

    assert "**Integrity status: FAIL**" in report
    assert "train and validation share 1 prompt(s)" in issues
    assert "`all` contains 1 duplicate prompt row(s)" in issues


def test_data_statistics_report_requires_all_dataset_paths(tmp_path: Path) -> None:
    """Test that incomplete report configuration is rejected."""
    try:
        build_data_statistics_report({"all": tmp_path / "all.jsonl"})
    except ValueError as error:
        assert "missing dataset paths" in str(error)
    else:
        raise AssertionError("missing dataset paths should fail")
