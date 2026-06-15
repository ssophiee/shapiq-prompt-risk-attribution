import pytest
from shapiq_attribution.data import (
    PromptRiskDataset,
    load_prompt_risk_jsonl,
    make_prompt_risk_example,
    normalize_advbench_rows,
    normalize_harmbench_rows,
    normalize_wildguard_safe_rows,
    write_prompt_risk_jsonl,
)


def test_make_prompt_risk_example_returns_target_schema() -> None:
    """Test that normalized examples use the prompt-risk schema."""
    example = make_prompt_risk_example(prompt="How do I bake bread?", label=0, source="fixture")

    assert example == {
        "prompt": "How do I bake bread?",
        "label": 0,
        "source": "fixture",
    }


@pytest.mark.parametrize("label", [-1, 2, "risky"])
def test_make_prompt_risk_example_rejects_invalid_labels(label: object) -> None:
    """Test that labels must be binary integers."""
    with pytest.raises(ValueError, match="label"):
        make_prompt_risk_example(prompt="hello", label=label, source="fixture")  # type: ignore[arg-type]


def test_make_prompt_risk_example_rejects_empty_prompts() -> None:
    """Test that empty prompts are rejected."""
    with pytest.raises(ValueError, match="prompt"):
        make_prompt_risk_example(prompt=" ", label=0, source="fixture")


def test_normalize_advbench_rows_labels_examples_as_risky() -> None:
    """Test that AdvBench examples are normalized as risky prompts."""
    rows = [{"prompt": "harmful request"}, {"goal": "another harmful request"}, {"prompt": ""}]

    examples = normalize_advbench_rows(rows)

    assert examples == [
        {"prompt": "harmful request", "label": 1, "source": "advbench"},
        {"prompt": "another harmful request", "label": 1, "source": "advbench"},
    ]


def test_normalize_harmbench_rows_labels_examples_as_risky() -> None:
    """Test that HarmBench examples are normalized as risky prompts."""
    rows = [{"Behavior": "harmful behavior"}, {"behavior": "second harmful behavior"}, {"behavior": ""}]

    examples = normalize_harmbench_rows(rows)

    assert examples == [
        {"prompt": "harmful behavior", "label": 1, "source": "harmbench"},
        {"prompt": "second harmful behavior", "label": 1, "source": "harmbench"},
    ]


def test_normalize_wildguard_safe_rows_labels_safe_prompts_as_not_risky() -> None:
    """Test that safe WildGuard prompts are normalized as not risky."""
    rows = [
        {"prompt": "Tell me a joke.", "prompt_harm_label": "unharmful"},
        {"prompt": "Give me a recipe.", "prompt_harm_label": "safe"},
        {"prompt": "harmful request", "prompt_harm_label": "harmful"},
        {"prompt": "", "prompt_harm_label": "unharmful"},
    ]

    examples = normalize_wildguard_safe_rows(rows)

    assert examples == [
        {"prompt": "Tell me a joke.", "label": 0, "source": "wildguard"},
        {"prompt": "Give me a recipe.", "label": 0, "source": "wildguard"},
    ]


def test_prompt_risk_jsonl_round_trip(tmp_path) -> None:
    """Test writing and reading prompt-risk JSONL snapshots."""
    path = tmp_path / "examples.jsonl"
    examples = [
        make_prompt_risk_example(prompt="safe prompt", label=0, source="fixture"),
        make_prompt_risk_example(prompt="risky prompt", label=1, source="fixture"),
    ]

    write_prompt_risk_jsonl(examples, path)

    assert load_prompt_risk_jsonl(path) == examples


def test_prompt_risk_dataset_returns_examples(tmp_path) -> None:
    """Test the local JSONL dataset wrapper."""
    path = tmp_path / "examples.jsonl"
    examples = [make_prompt_risk_example(prompt="safe prompt", label=0, source="fixture")]
    write_prompt_risk_jsonl(examples, path)

    dataset = PromptRiskDataset(path)

    assert len(dataset) == 1
    assert dataset[0] == examples[0]

def test_build_prompt_risk_dataset_combines_raw_snapshots(tmp_path) -> None:
    """Test building the processed dataset from raw snapshots."""
    from shapiq_attribution.data import build_prompt_risk_dataset

    raw_dir = tmp_path / "raw"
    output_path = tmp_path / "processed" / "prompt_risk_dataset.jsonl"

    write_prompt_risk_jsonl(
        [make_prompt_risk_example(prompt="adv risky", label=1, source="advbench")],
        raw_dir / "advbench.jsonl",
    )
    write_prompt_risk_jsonl(
        [make_prompt_risk_example(prompt="harm risky", label=1, source="harmbench")],
        raw_dir / "harmbench.jsonl",
    )
    write_prompt_risk_jsonl(
        [make_prompt_risk_example(prompt="wild safe", label=0, source="wildguard")],
        raw_dir / "wildguard.jsonl",
    )

    write_prompt_risk_jsonl(
    [make_prompt_risk_example(prompt="toxic risky", label=1, source="toxicchat")],
    raw_dir / "toxicchat.jsonl",
    )   
    write_prompt_risk_jsonl(
    [make_prompt_risk_example(prompt="beaver safe", label=0, source="beavertails")],
    raw_dir / "beavertails.jsonl",
    )

    build_prompt_risk_dataset(raw_dir=raw_dir, output_path=output_path)

    assert load_prompt_risk_jsonl(output_path) == [
        {"prompt": "adv risky", "label": 1, "source": "advbench"},
        {"prompt": "harm risky", "label": 1, "source": "harmbench"},
        {"prompt": "wild safe", "label": 0, "source": "wildguard"},
        {"prompt": "toxic risky", "label": 1, "source": "toxicchat"},
        {"prompt": "beaver safe", "label": 0, "source": "beavertails"},
    ]