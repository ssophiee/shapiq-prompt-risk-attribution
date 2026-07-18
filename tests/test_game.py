"""Tests for the shapiq game layer."""

import numpy as np
from shapiq_attribution.game import CoTGame, parse_cot_steps, run_shapiq


def test_parse_cot_steps_extracts_steps_and_answer() -> None:
    """Test that step lines and the answer line are parsed out."""
    text = "Step 1: read the prompt\nStep 2: assess the risk\nAnswer: risky"

    steps, target = parse_cot_steps(text)

    assert steps == ["Step 1: read the prompt", "Step 2: assess the risk"]
    assert target == "risky"


def test_parse_cot_steps_skips_placeholder_steps() -> None:
    """Test that empty 'Step N: ...' placeholder lines are dropped."""
    text = "Step 1: real reasoning\nStep 2: ...\nStep 3:\nAnswer: safe"

    steps, target = parse_cot_steps(text)

    assert steps == ["Step 1: real reasoning"]
    assert target == "safe"


def test_parse_cot_steps_without_answer_returns_empty_target() -> None:
    """Test that a missing answer line yields an empty target."""
    steps, target = parse_cot_steps("Step 1: only reasoning, no conclusion")

    assert steps == ["Step 1: only reasoning, no conclusion"]
    assert target == ""


def _additive_value_fn(weights: np.ndarray, baseline: float):
    """Build a value function where each player contributes its weight."""

    def value_fn(coalitions: np.ndarray) -> np.ndarray:
        return baseline + coalitions.astype(float) @ weights

    return value_fn


def test_cot_game_evaluates_value_function() -> None:
    """Test that the game delegates coalition values to the value function."""
    weights = np.array([1.0, 2.0, 3.0])
    game = CoTGame(n_players=3, value_fn=_additive_value_fn(weights, baseline=0.5))

    coalitions = np.array([[False, False, False], [True, True, True]])
    values = game.value_function(coalitions)

    assert values[0] == 0.5
    assert values[1] == 6.5


def test_run_shapiq_recovers_additive_contributions() -> None:
    """Test that Shapley values of an additive game equal the per-step weights."""
    steps = ["Step 1: a", "Step 2: b", "Step 3: c"]
    weights = np.array([0.1, 0.5, -0.2])
    value_fn = _additive_value_fn(weights, baseline=0.3)

    sv, ksii, baseline = run_shapiq(steps, value_fn)

    assert baseline == 0.3
    for player, weight in enumerate(weights):
        assert np.isclose(sv[(player,)], weight)
    # An additive game has no pairwise interactions.
    for i in range(3):
        for j in range(i + 1, 3):
            assert np.isclose(ksii[(i, j)], 0.0)
