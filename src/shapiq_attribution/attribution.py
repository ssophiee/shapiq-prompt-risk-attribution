"""Stable ML core — input/output contract must not change.

build_prompt:           (str, str, str, list[str], tokenizer) -> str
compute_answer_logprob: (str, str, model, tokenizer)          -> float
make_value_function:    (...) -> callable[[np.ndarray], np.ndarray]
"""

import numpy as np
import torch


def build_prompt(
    question: str,
    few_shot: str,
    system: str,
    present_steps: list[str],
    tokenizer,
) -> str:
    reasoning = "\n".join(present_steps) if present_steps else "(no reasoning provided)"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{few_shot}\nQ: {question}"},
        {"role": "assistant", "content": reasoning},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


@torch.no_grad()
def compute_answer_logprob(prompt: str, target: str, model, tokenizer) -> float:
    """Mean log-probability of target tokens given prompt as context."""
    full_ids = tokenizer(prompt + target, return_tensors="pt").input_ids.to(model.device)
    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    logits = model(full_ids).logits
    log_probs = torch.log_softmax(logits[0], dim=-1)
    answer_ids = full_ids[0, prompt_len:]
    token_lp = log_probs[prompt_len - 1 : prompt_len - 1 + len(answer_ids)]
    token_lp = token_lp[torch.arange(len(answer_ids)), answer_ids]
    return token_lp.mean().item()


def make_value_function(
    question: str,
    few_shot: str,
    system: str,
    cot_steps: list[str],
    target_answer: str,
    model,
    tokenizer,
):
    """Returns a coalition value function for use with CoTGame."""

    def value_fn(coalitions: np.ndarray) -> np.ndarray:
        scores = []
        for coalition in coalitions:
            present = [cot_steps[i] for i, m in enumerate(coalition) if m]
            prompt = build_prompt(question, few_shot, system, present, tokenizer)
            scores.append(compute_answer_logprob(prompt, target_answer, model, tokenizer))
        return np.array(scores)

    return value_fn
