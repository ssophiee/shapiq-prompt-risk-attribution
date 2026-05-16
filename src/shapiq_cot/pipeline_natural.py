"""Natural-reasoning pipeline for Qwen2.5-3B without few-shot prompting.

Players are paragraph-level chunks of the model's own CoT output —
no step format is imposed, so the reasoning is genuinely the model's own.

Differences from pipeline.py:
  - No few_shot argument anywhere
  - build_prompt_natural uses double-newline joins (paragraph boundaries)
  - parse_natural_steps splits on blank lines, falls back to sentences
  - Last paragraph is treated as the target/conclusion
"""

import re

import numpy as np
import torch

from shapiq_cot.game import run_shapiq
from shapiq_cot.model import load_model
from shapiq_cot.visualize import plot_results

NATURAL_SYSTEM = (
    "You are a thoughtful assistant. "
    "Think through your reasoning carefully before giving a final answer. "
    "Write your thoughts naturally, then end with your conclusion."
)


def generate_cot_natural(
    question: str,
    model,
    tokenizer,
    system: str = NATURAL_SYSTEM,
    max_new_tokens: int = 300,
) -> str:
    """Generate CoT without few-shot — model's own natural reasoning."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def parse_natural_steps(generated_text: str, min_chars: int = 15) -> tuple[list[str], str]:
    """Split natural CoT into paragraph-level players.

    Strategy:
      1. Split on blank lines (double newline) → paragraph chunks
      2. If only one chunk, fall back to sentence splitting
      3. Last chunk becomes the target/conclusion; the rest are players

    Returns:
        steps:  reasoning paragraphs (all but the last)
        target: concluding paragraph (used as the Shapiq value target)
    """
    chunks = [c.strip() for c in re.split(r"\n{2,}", generated_text)]
    chunks = [c for c in chunks if len(c) >= min_chars]

    if len(chunks) >= 2:
        return chunks[:-1], chunks[-1]

    # Fallback: sentence-level split on .  !  ?
    sentences = re.split(r"(?<=[.!?])\s+", generated_text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) >= min_chars]
    if len(sentences) >= 2:
        return sentences[:-1], sentences[-1]

    return [generated_text.strip()], ""


def build_prompt_natural(
    question: str,
    system: str,
    present_steps: list[str],
    tokenizer,
) -> str:
    """Construct a prompt with selected reasoning chunks, no few-shot context."""
    reasoning = "\n\n".join(present_steps) if present_steps else "(no reasoning provided)"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
        {"role": "assistant", "content": reasoning},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


@torch.no_grad()
def compute_answer_logprob_natural(prompt: str, target: str, model, tokenizer) -> float:
    """Mean log-probability of target tokens given prompt as context."""
    if not target:
        return 0.0
    full_ids = tokenizer(prompt + target, return_tensors="pt").input_ids.to(model.device)
    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    logits = model(full_ids).logits
    log_probs = torch.log_softmax(logits[0], dim=-1)
    answer_ids = full_ids[0, prompt_len:]
    if len(answer_ids) == 0:
        return 0.0
    token_lp = log_probs[prompt_len - 1 : prompt_len - 1 + len(answer_ids)]
    token_lp = token_lp[torch.arange(len(answer_ids)), answer_ids]
    return token_lp.mean().item()


def make_value_function_natural(
    question: str,
    system: str,
    cot_steps: list[str],
    target: str,
    model,
    tokenizer,
):
    """Returns a coalition value function for CoTGame (no few-shot version)."""

    def value_fn(coalitions: np.ndarray) -> np.ndarray:
        scores = []
        for coalition in coalitions:
            present = [cot_steps[i] for i, m in enumerate(coalition) if m]
            prompt = build_prompt_natural(question, system, present, tokenizer)
            scores.append(compute_answer_logprob_natural(prompt, target, model, tokenizer))
        return np.array(scores)

    return value_fn


def run_attribution_natural(
    question: str,
    model,
    tokenizer,
    system: str = NATURAL_SYSTEM,
    max_new_tokens: int = 300,
) -> dict:
    """Full natural-reasoning attribution pipeline. Reuse this from notebooks.

    Returns a dict with:
        cot_steps      – list of reasoning paragraph strings (players)
        target         – concluding paragraph (attribution target)
        baseline       – empty-coalition log-prob score
        shapley_values – list of per-player Shapley values
        interactions   – dict of pairwise k-SII scores  {"i,j": float}
        sv             – raw shapiq InteractionValues object (for plotting)
        ksii           – raw shapiq InteractionValues object (for plotting)
    """
    generated = generate_cot_natural(question, model, tokenizer, system, max_new_tokens)
    print("── Natural CoT ──")
    print(generated)

    cot_steps, target = parse_natural_steps(generated)
    if not cot_steps:
        raise ValueError("No reasoning steps parsed. Try a longer max_new_tokens.")
    print(f"\n── {len(cot_steps)} players parsed ──")
    print(f"Target: {target[:120]!r}")

    value_fn = make_value_function_natural(question, system, cot_steps, target, model, tokenizer)
    sv, ksii, baseline = run_shapiq(cot_steps, value_fn)

    n = len(cot_steps)
    shapley_values = [float(sv.values[i + 1]) for i in range(n)]
    interactions = {
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
        "sv": sv,
        "ksii": ksii,
    }


def generate_cot_qwen3(
    question: str,
    model,
    tokenizer,
    system: str = NATURAL_SYSTEM,
    max_new_tokens: int = 512,
) -> str:
    """Generate with Qwen3 thinking mode enabled — outputs <think>...</think> + answer."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def parse_qwen3_thinking(generated_text: str, min_chars: int = 15) -> tuple[list[str], str]:
    """Parse Qwen3 <think>...</think> output into paragraph-level players.

    Content inside <think> is split into paragraphs (players).
    Text after </think> is the target/final answer.
    Falls back to parse_natural_steps if no <think> block is found.

    Returns:
        steps:  paragraph chunks from inside the thinking block
        target: final answer text (after </think>)
    """
    think_match = re.search(r"<think>(.*?)</think>(.*)", generated_text, re.DOTALL)
    if not think_match:
        return parse_natural_steps(generated_text, min_chars)

    thinking_block = think_match.group(1).strip()
    target = think_match.group(2).strip()

    # Remove special tokens that may appear in the target
    target = re.sub(r"<\|[^|]+\|>", "", target).strip()

    chunks = [c.strip() for c in re.split(r"\n{2,}", thinking_block)]
    chunks = [c for c in chunks if len(c) >= min_chars]

    if not chunks:
        # Thinking block present but nearly empty — fall back to sentence split
        sentences = re.split(r"(?<=[.!?])\s+", thinking_block)
        chunks = [s.strip() for s in sentences if len(s.strip()) >= min_chars]

    if not chunks:
        raise ValueError("Thinking block is empty. Try a longer max_new_tokens.")

    if not target:
        # Model cut off before answering — use last thinking chunk as target
        target = chunks.pop()

    return chunks, target


def run_attribution_qwen3(
    question: str,
    model,
    tokenizer,
    system: str = NATURAL_SYSTEM,
    max_new_tokens: int = 512,
) -> dict:
    """Attribution pipeline for Qwen3 thinking models.

    Players are paragraphs from the <think> block.
    Target is the final answer after </think>.

    Returns same dict shape as run_attribution_natural (including sv/ksii for plotting).
    """
    generated = generate_cot_qwen3(question, model, tokenizer, system, max_new_tokens)
    print("── Qwen3 Raw Output ──")
    print(generated)

    cot_steps, target = parse_qwen3_thinking(generated)
    print(f"\n── {len(cot_steps)} players parsed from <think> block ──")
    print(f"Target: {target[:120]!r}")

    value_fn = make_value_function_natural(question, system, cot_steps, target, model, tokenizer)
    sv, ksii, baseline = run_shapiq(cot_steps, value_fn)

    n = len(cot_steps)
    shapley_values = [float(sv.values[i + 1]) for i in range(n)]
    interactions = {
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
        "sv": sv,
        "ksii": ksii,
    }


def main(
    question: str = "Why is the sky blue?",
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
) -> None:
    model, tokenizer = load_model(model_id)
    print(f"── Question ──\n{question}\n")
    result = run_attribution_natural(question, model, tokenizer)
    plot_results(result["cot_steps"], result["sv"], result["ksii"], title="Natural CoT Attribution")


if __name__ == "__main__":
    import typer

    typer.run(main)
