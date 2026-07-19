"""Shapiq game layer — this is where players/coalition logic lives and will evolve."""

import re

import numpy as np
import shapiq


class CoTGame(shapiq.Game):
    def __init__(self, n_players: int, value_fn):
        self._value_fn = value_fn
        super().__init__(n_players=n_players, normalize=False)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        return self._value_fn(coalitions)


def run_shapiq(cot_steps: list[str], value_fn) -> tuple:
    """Run ExactComputer for SV (order=1) and k-SII (order=2).

    Returns:
        sv:       first-order Shapley values
        ksii:     pairwise k-SII interactions
        baseline: empty-coalition score
    """
    n = len(cot_steps)
    baseline = value_fn(np.zeros((1, n), dtype=bool))[0]
    game = CoTGame(n_players=n, value_fn=value_fn)
    game.normalization_value = baseline
    computer = shapiq.ExactComputer(n_players=n, game=game)
    sv = computer(index="SV", order=1)
    ksii = computer(index="k-SII", order=2)
    return sv, ksii, baseline


def parse_cot_steps(generated_text: str) -> tuple[list[str], str]:
    """Extract 'Step N: ...' lines and the Answer line from generated text.

    Returns:
        steps:  list of reasoning step strings (Answer line excluded)
        target: the answer string (empty string if not found)
    """
    all_steps = [s for s in re.findall(r"Step \d+:.*", generated_text) if not re.match(r"Step \d+:\s*\.{0,3}\s*$", s)]
    steps = [s for s in all_steps if "Answer:" not in s]
    answer_match = re.search(r"Answer:(.*)", generated_text)
    target = answer_match.group(1).strip() if answer_match else ""
    return steps, target
