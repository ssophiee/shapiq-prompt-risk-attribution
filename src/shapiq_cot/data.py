from datasets import load_dataset


def load_wildguard(token: str, split: str = "test") -> list[dict]:
    ds = load_dataset("allenai/wildguardmix", "wildguardtest", token=token)
    return list(ds[split])


def get_harmful_examples(token: str, split: str = "test") -> list[dict]:
    examples = load_wildguard(token, split)
    return [
        ex for ex in examples
        if ex.get("prompt_harm_label") == "harmful" and ex.get("prompt")
    ]
