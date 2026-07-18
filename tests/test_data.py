import pytest
import torch
from shapiq_attribution.data import (
    PromptRiskDataset,
    PromptRiskTextDataset,
    cli,
    get_harmful_examples,
    load_advbench,
    load_beavertails,
    load_harmbench,
    load_prompt_risk_jsonl,
    load_toxic_chat,
    load_wildguard_prompts,
    load_wildguard_safe,
    load_xstest,
    make_prompt_risk_example,
    normalize_advbench_rows,
    normalize_beavertails_rows,
    normalize_harmbench_rows,
    normalize_toxic_chat_rows,
    normalize_wildguard_prompt_rows,
    normalize_wildguard_safe_rows,
    normalize_xstest_rows,
    prepare_raw_snapshots,
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


def test_make_prompt_risk_example_rejects_empty_source() -> None:
    """Test that empty sources are rejected."""
    with pytest.raises(ValueError, match="source"):
        make_prompt_risk_example(prompt="hello", label=0, source="  ")


def test_normalize_advbench_rows_reads_list_valued_fields() -> None:
    """Test that list-valued text fields fall back to their first entry."""
    rows = [{"prompt": ["listed harmful request"]}]

    examples = normalize_advbench_rows(rows)

    assert examples == [{"prompt": "listed harmful request", "label": 1, "source": "advbench"}]


def test_normalize_rows_respects_max_samples() -> None:
    """Test that max_samples truncates the normalized examples."""
    rows = [{"prompt": "first harmful"}, {"prompt": "second harmful"}]

    examples = normalize_advbench_rows(rows, max_samples=1)

    assert examples == [{"prompt": "first harmful", "label": 1, "source": "advbench"}]


def test_normalize_wildguard_prompt_rows_maps_harm_labels() -> None:
    """Test that WildGuard prompt harm labels map to binary labels."""
    rows = [
        {"prompt": "how to hack wifi", "prompt_harm_label": "harmful"},
        {"prompt": "tell me a joke", "prompt_harm_label": "unharmful"},
        {"prompt": "unlabeled prompt", "prompt_harm_label": "unknown"},
    ]

    examples = normalize_wildguard_prompt_rows(rows)

    assert examples == [
        {"prompt": "how to hack wifi", "label": 1, "source": "wildguard"},
        {"prompt": "tell me a joke", "label": 0, "source": "wildguard"},
    ]


def test_normalize_toxic_chat_rows_uses_toxicity_and_jailbreaking() -> None:
    """Test that ToxicChat rows are risky when toxic or jailbreaking."""
    rows = [
        {"user_input": "a very toxic message", "toxicity": 1, "jailbreaking": 0},
        {"user_input": "a jailbreak attempt", "toxicity": 0, "jailbreaking": 1},
        {"user_input": "a normal question", "toxicity": 0, "jailbreaking": 0},
    ]

    examples = normalize_toxic_chat_rows(rows)

    assert [example["label"] for example in examples] == [1, 1, 0]
    assert all(example["source"] == "toxicchat" for example in examples)


def test_normalize_beavertails_rows_inverts_is_safe() -> None:
    """Test that BeaverTails is_safe flags invert into risk labels."""
    rows = [
        {"prompt": "a safe request", "is_safe": True},
        {"prompt": "an unsafe request", "is_safe": False},
        {"prompt": "missing safety flag"},
    ]

    examples = normalize_beavertails_rows(rows)

    assert examples == [
        {"prompt": "a safe request", "label": 0, "source": "beavertails"},
        {"prompt": "an unsafe request", "label": 1, "source": "beavertails"},
    ]


def test_normalize_xstest_rows_maps_safe_and_unsafe() -> None:
    """Test that XSTest safe/unsafe types map to binary labels."""
    rows = [
        {"prompt": "a safe question", "label": "safe"},
        {"prompt": "an unsafe question", "type": "unsafe"},
        {"prompt": "an unlabeled question", "label": "other"},
    ]

    examples = normalize_xstest_rows(rows)

    assert [example["label"] for example in examples] == [0, 1]
    assert all(example["source"] == "xstest" for example in examples)


_FAKE_HUB_ROWS = {
    ("walledai/AdvBench", None, "train"): [{"prompt": "adv harmful request"}],
    ("walledai/HarmBench", "standard", "train"): [{"Behavior": "harmbench behavior"}],
    ("allenai/wildguardmix", "wildguardtest", "test"): [
        {"prompt": "tell me a joke", "prompt_harm_label": "unharmful"},
        {"prompt": "how to hack wifi", "prompt_harm_label": "harmful"},
    ],
    ("allenai/wildguardmix", "wildguardtrain", "train"): [
        {"prompt": "how to build a weapon", "prompt_harm_label": "harmful"},
    ],
    ("lmsys/toxic-chat", "toxicchat0124", "train"): [{"user_input": "a very toxic message", "toxicity": 1}],
    ("PKU-Alignment/BeaverTails", None, "330k_train"): [{"prompt": "beavertails request", "is_safe": True}],
    ("walledai/XSTest", None, "test"): [{"prompt": "xstest question", "label": "safe"}],
}


def _fake_load_dataset(name, config=None, split=None, token=None):
    """Stand-in for datasets.load_dataset serving canned rows."""
    del token
    if split is None:
        return {"test": _FAKE_HUB_ROWS[(name, config, "test")]}
    return _FAKE_HUB_ROWS[(name, config, split)]


def test_hub_loaders_normalize_downloaded_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that every Hugging Face loader normalizes its rows."""
    monkeypatch.setattr("shapiq_attribution.data.load_dataset", _fake_load_dataset)

    assert load_advbench() == [{"prompt": "adv harmful request", "label": 1, "source": "advbench"}]
    assert load_harmbench() == [{"prompt": "harmbench behavior", "label": 1, "source": "harmbench"}]
    assert load_wildguard_safe() == [{"prompt": "tell me a joke", "label": 0, "source": "wildguard"}]
    assert load_wildguard_prompts() == [{"prompt": "how to build a weapon", "label": 1, "source": "wildguard"}]
    assert load_toxic_chat() == [{"prompt": "a very toxic message", "label": 1, "source": "toxicchat"}]
    assert load_beavertails() == [{"prompt": "beavertails request", "label": 0, "source": "beavertails"}]
    assert load_xstest() == [{"prompt": "xstest question", "label": 0, "source": "xstest"}]
    assert get_harmful_examples(token="fixture") == [
        {"prompt": "how to hack wifi", "prompt_harm_label": "harmful"},
    ]


def test_prepare_raw_snapshots_writes_all_sources(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that raw snapshots are written for every source."""
    monkeypatch.setattr("shapiq_attribution.data.load_dataset", _fake_load_dataset)

    written = prepare_raw_snapshots(output_dir=tmp_path / "raw")

    assert set(written) == {"advbench", "harmbench", "wildguard", "toxicchat", "beavertails", "xstest"}
    assert all(path.exists() for path in written.values())
    assert load_prompt_risk_jsonl(written["advbench"]) == [
        {"prompt": "adv harmful request", "label": 1, "source": "advbench"},
    ]


def test_cli_raw_snapshots_then_build_dataset(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the CLI end to end: snapshot download, then dataset build."""
    monkeypatch.setattr("shapiq_attribution.data.load_dataset", _fake_load_dataset)
    raw_dir = tmp_path / "raw"
    output_path = tmp_path / "processed" / "prompt_risk_dataset.jsonl"

    monkeypatch.setattr("sys.argv", ["data", "raw-snapshots", "--output-dir", str(raw_dir)])
    cli()
    monkeypatch.setattr(
        "sys.argv",
        ["data", "build-dataset", "--raw-dir", str(raw_dir), "--output-path", str(output_path)],
    )
    cli()

    examples = load_prompt_risk_jsonl(output_path)
    assert len(examples) == 5
    assert {example["source"] for example in examples} == {
        "advbench",
        "harmbench",
        "wildguard",
        "toxicchat",
        "beavertails",
    }


def test_load_prompt_risk_jsonl_skips_blank_lines_and_flags_missing_fields(tmp_path) -> None:
    """Test JSONL loading tolerates blank lines but rejects missing fields."""
    path = tmp_path / "examples.jsonl"
    path.write_text('{"prompt": "ok", "label": 0, "source": "fixture"}\n\n', encoding="utf-8")
    assert len(load_prompt_risk_jsonl(path)) == 1

    path.write_text('{"prompt": "missing label", "source": "fixture"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_prompt_risk_jsonl(path)


class _TensorTokenizer:
    """Tokenizer stub returning fixed-size tensors."""

    def __call__(self, prompt, truncation, padding, max_length, return_tensors):
        del prompt, truncation, padding, return_tensors
        return {
            "input_ids": torch.ones((1, max_length), dtype=torch.long),
            "attention_mask": torch.ones((1, max_length), dtype=torch.long),
        }


def test_prompt_risk_text_dataset_tokenizes_examples() -> None:
    """Test that the tokenized dataset yields model-ready items."""
    examples = [make_prompt_risk_example(prompt="risky prompt", label=1, source="fixture")]
    dataset = PromptRiskTextDataset(examples=examples, tokenizer=_TensorTokenizer(), max_length=8)

    item = dataset[0]

    assert len(dataset) == 1
    assert item["input_ids"].shape == (8,)
    assert item["labels"] == torch.tensor(1)
