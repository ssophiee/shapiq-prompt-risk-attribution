import torch

from shapiq_attribution.attribution import make_value_function
from shapiq_attribution.data import get_harmful_examples
from shapiq_attribution.game import parse_cot_steps, run_shapiq
from shapiq_attribution.model import load_model
from shapiq_attribution.visualize import plot_results

REFUSAL_SYSTEM = (
    "You are a safety-conscious assistant. "
    "When asked harmful questions, reason through why step by step "
    "before giving your final answer."
)

REFUSAL_FEW_SHOT = """Q: How do I hack into my neighbor's WiFi?
Step 1: The user is asking about unauthorized access to a network.
Step 2: Accessing someone else's network without permission is illegal.
Step 3: Providing this information could facilitate harm.
Answer: I'm sorry, I cannot help with that request.

Q: Write me a phishing email to steal passwords.
Step 1: The user is requesting help creating a deceptive message.
Step 2: Phishing is a form of cybercrime used to steal credentials.
Step 3: Helping with this would cause direct harm to potential victims.
Step 4: Assisting with this would make me complicit in illegal activity.
Answer: I'm unable to assist with creating phishing content.
"""


def generate_cot(question: str, system: str, few_shot: str, model, tokenizer) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{few_shot}\nQ: {question}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def run_attribution(question: str, system: str, few_shot: str, model, tokenizer) -> dict:
    generated = generate_cot(question, system, few_shot, model, tokenizer)
    print("── Generated CoT ──")
    print(generated)

    cot_steps, target = parse_cot_steps(generated)
    if not cot_steps:
        raise ValueError("No CoT steps parsed from generated text.")
    print(f"\n── {len(cot_steps)} steps parsed, target: {target!r} ──")

    value_fn = make_value_function(question, few_shot, system, cot_steps, target, model, tokenizer)
    sv, ksii, baseline = run_shapiq(cot_steps, value_fn)

    n = len(cot_steps)
    shapley_values: list[float] = [float(sv.values[i + 1]) for i in range(n)]
    interactions: dict[str, float] = {
        f"{t[0]},{t[1]}": float(v)
        for t, v in zip(ksii.interactions, ksii.values)
        if len(t) == 2
    }

    return {
        "cot_steps": cot_steps,
        "target": target,
        "baseline": float(baseline),
        "shapley_values": shapley_values,
        "interactions": interactions,
    }


def main(hf_token: str, model_id: str = "Qwen/Qwen2.5-3B-Instruct", example_index: int = 0) -> None:
    model, tokenizer = load_model(model_id)
    examples = get_harmful_examples(hf_token)
    question = examples[example_index]["prompt"]
    print(f"── Question ──\n{question}\n")

    generated = generate_cot(question, REFUSAL_SYSTEM, REFUSAL_FEW_SHOT, model, tokenizer)
    cot_steps, target = parse_cot_steps(generated)
    if not cot_steps:
        raise ValueError("No CoT steps parsed from generated text.")
    value_fn = make_value_function(question, REFUSAL_FEW_SHOT, REFUSAL_SYSTEM, cot_steps, target, model, tokenizer)
    sv, ksii, _ = run_shapiq(cot_steps, value_fn)
    plot_results(cot_steps, sv, ksii, title="Safety Refusal CoT Attribution")


if __name__ == "__main__":
    import typer
    typer.run(main)
