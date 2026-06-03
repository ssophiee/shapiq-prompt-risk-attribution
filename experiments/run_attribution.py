"""Token-level safety attribution experiment runner.

Runs SafetyAnalysisGame on a set of prompts from AdvBench or SorryBench,
saves per-prompt JSON + PNG results, and logs a summary run to W&B.

Usage::

    python experiments/run_attribution.py \
        --model-path models/prompt_risk_distilbert/ \
        --dataset advbench \
        --n-samples 10 \
        --budget 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Token-level safety attribution over a prompt dataset")
    parser.add_argument("--model-path", required=True, help="Path to trained PromptRiskPredictor checkpoint")
    parser.add_argument("--budget", type=int, default=256, help="SHAPIQ coalition sample budget")
    parser.add_argument(
        "--dataset",
        default="advbench",
        choices=["advbench", "sorrybench"],
        help="Source dataset",
    )
    parser.add_argument("--n-samples", type=int, default=10, help="Number of prompts to attribute")
    parser.add_argument(
        "--backend",
        default="distilbert",
        choices=["distilbert", "llama_guard"],
        help="Classifier backend for SafetyAnalysisGame",
    )
    parser.add_argument("--output-dir", default="reports/figures", help="Directory for JSON + PNG outputs")
    parser.add_argument("--wandb-project", default="shapiq-attribution", help="W&B project name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    return parser.parse_args(argv)


def load_prompts(dataset: str, n: int, data_dir: Path = Path("data/raw")) -> list[str]:
    path = data_dir / f"{dataset}.jsonl"
    prompts: list[str] = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            prompts.append(row["prompt"])
            if len(prompts) >= n:
                break
    return prompts


def attribute_prompt(
    prompt: str,
    predictor,
    budget: int,
    backend: str = "distilbert",
    model_name: str = "meta-llama/Llama-Guard-3-1B",
) -> dict:
    from shapiq_attribution.safety_analysis import (
        SafetyAnalysisGame,
        aggregate_to_words,
        run_safety_shapiq,
    )

    if predictor is not None:
        game = SafetyAnalysisGame(prompt, model=predictor, backend=backend)
    else:
        game = SafetyAnalysisGame(prompt, model_name=model_name, backend=backend)
    sv, ksii = run_safety_shapiq(game, budget=budget)

    p_risky = predictor.predict_proba(prompt) if predictor is not None else float(game._baseline)
    token_names = game.token_names
    sv_values = [float(sv[(j,)]) for j in range(len(token_names))]
    top_3 = sorted(zip(token_names, sv_values), key=lambda x: abs(x[1]), reverse=True)[:3]

    # Optional: aggregate subword tokens to readable words for logging
    word_sv, word_names = aggregate_to_words(sv, token_names)

    return {
        "prompt": prompt,
        "p_risky": p_risky,
        "tokens": token_names,
        "shapley_values": sv_values,
        "top_3_tokens": [t for t, _ in top_3],
        "words": word_names,
        "word_shapley_values": [float(word_sv[(j,)]) for j in range(len(word_names))],
        "budget": budget,
    }


def run(args: argparse.Namespace) -> list[dict]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "distilbert":
        from shapiq_attribution.model import PromptRiskPredictor
        predictor = PromptRiskPredictor.from_pretrained(args.model_path)
    else:
        # llama_guard: SafetyAnalysisGame loads the model itself via model_name
        predictor = None

    prompts = load_prompts(args.dataset, args.n_samples)

    wb_run = None
    if not args.no_wandb:
        import wandb
        wb_run = wandb.init(project=args.wandb_project, config=vars(args))

    results = []
    for i, prompt in enumerate(prompts):
        result = attribute_prompt(prompt, predictor, args.budget, backend=args.backend, model_name=args.model_path)
        result["dataset"] = args.dataset
        results.append(result)

        out_path = output_dir / f"{args.dataset}_{i:03d}.json"
        out_path.write_text(json.dumps(result, indent=2))

        if wb_run is not None:
            wb_run.log({
                "sample_idx": i,
                "prompt": prompt,
                "p_risky": result["p_risky"],
                "top_3_tokens": str(result["top_3_tokens"]),
                "budget": args.budget,
            })

    if wb_run is not None:
        wb_run.finish()

    return results


if __name__ == "__main__":
    run(parse_args())
