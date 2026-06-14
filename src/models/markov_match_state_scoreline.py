from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from src.models.favorite_blowout_mixture_scoreline import (
    favorite_bucket,
    normalize_win_probability,
    parse_scoreline,
    top_scorelines,
)


STATES = ("normal", "open_game", "favorite_pressure", "blowout")
STATE_TO_INDEX = {state: index for index, state in enumerate(STATES)}


@dataclass(frozen=True)
class MatchStateConfig:
    """Research-only V3 Markov match-state scoreline configuration.

    This config affects only pre-match scoreline simulation. It never modifies
    calibrated W/D/L probabilities, team ratings, Elo, roster layers, or
    tournament simulation logic.
    """

    favorite_goal_multipliers: Mapping[str, float] = field(
        default_factory=lambda: {
            "normal": 1.00,
            "open_game": 1.15,
            "favorite_pressure": 1.35,
            "blowout": 2.00,
        }
    )
    underdog_goal_multipliers: Mapping[str, float] = field(
        default_factory=lambda: {
            "normal": 1.00,
            "open_game": 1.15,
            "favorite_pressure": 0.90,
            "blowout": 0.75,
        }
    )
    initial_pressure_by_bucket: Mapping[str, float] = field(
        default_factory=lambda: {
            "balanced": 0.00,
            "slight_favorite": 0.01,
            "clear_favorite": 0.03,
            "heavy_favorite": 0.07,
            "extreme_mismatch": 0.12,
        }
    )
    initial_open_by_bucket: Mapping[str, float] = field(
        default_factory=lambda: {
            "balanced": 0.04,
            "slight_favorite": 0.05,
            "clear_favorite": 0.06,
            "heavy_favorite": 0.06,
            "extreme_mismatch": 0.05,
        }
    )
    max_goals: int = 12
    transition_open_scale: float = 1.0
    transition_pressure_scale: float = 1.0
    transition_blowout_scale: float = 1.0
    late_control_strength: float = 0.0
    underdog_resistance_strength: float = 0.0


def _state_initial_probabilities(bucket: str, config: MatchStateConfig) -> np.ndarray:
    pressure = float(config.initial_pressure_by_bucket.get(bucket, 0.0))
    open_game = float(config.initial_open_by_bucket.get(bucket, 0.04))
    blowout = 0.0
    normal = max(0.0, 1.0 - pressure - open_game - blowout)
    probs = np.array([normal, open_game, pressure, blowout], dtype=float)
    return probs / probs.sum()


def _transition_probabilities(
    current_state: int,
    bucket: str,
    favorite_margin: int,
    step_index: int,
    n_steps: int,
    lambda_imbalance: float,
    config: MatchStateConfig,
) -> np.ndarray:
    """Return conservative transition probabilities from current simulated state.

    The transition rules use only pre-match structure and simulated path state.
    No actual in-game or post-match data enters this research model.
    """
    late = step_index >= int(n_steps * 0.65)
    control_phase = step_index >= int(n_steps * 0.60)
    resistance_phase = step_index >= int(n_steps * 0.55)
    early = step_index <= int(n_steps * 0.35)
    mismatch = bucket in {"heavy_favorite", "extreme_mismatch"}
    clear_or_more = bucket in {"clear_favorite", "heavy_favorite", "extreme_mismatch"}

    probs = np.zeros(4, dtype=float)
    if current_state == STATE_TO_INDEX["normal"]:
        p_open = (0.04 + (0.03 if late and abs(favorite_margin) <= 1 else 0.0)) * config.transition_open_scale
        p_pressure = ((0.015 if clear_or_more else 0.004) + 0.025 * lambda_imbalance) * config.transition_pressure_scale
        if favorite_margin >= 1:
            p_pressure += (0.02 if clear_or_more else 0.006) * config.transition_pressure_scale
        p_blowout = 0.0
        probs[:] = [1.0 - p_open - p_pressure - p_blowout, p_open, p_pressure, p_blowout]
    elif current_state == STATE_TO_INDEX["open_game"]:
        p_normal = 0.18 if not late else 0.10
        p_pressure = ((0.04 if clear_or_more else 0.012) + 0.035 * lambda_imbalance) * config.transition_pressure_scale
        if favorite_margin >= 1:
            p_pressure += 0.025 * config.transition_pressure_scale
        p_blowout = (0.004 if mismatch and favorite_margin >= 2 and not late else 0.0) * config.transition_blowout_scale
        probs[:] = [p_normal, 1.0 - p_normal - p_pressure - p_blowout, p_pressure, p_blowout]
    elif current_state == STATE_TO_INDEX["favorite_pressure"]:
        p_normal = 0.08 if favorite_margin <= 0 else 0.035
        p_open = 0.07 if favorite_margin <= 1 else 0.035
        p_open *= config.transition_open_scale
        p_blowout = 0.0
        if mismatch and favorite_margin >= 2 and not late:
            p_blowout = 0.035 + 0.05 * lambda_imbalance
        elif clear_or_more and favorite_margin >= 3 and early:
            p_blowout = 0.025 + 0.03 * lambda_imbalance
        if late and favorite_margin < 2:
            p_blowout *= 0.25
        p_blowout *= config.transition_blowout_scale
        probs[:] = [p_normal, p_open, 1.0 - p_normal - p_open - p_blowout, p_blowout]
    else:
        p_pressure = 0.10 if favorite_margin < 3 else 0.04
        p_open = 0.02 if favorite_margin < 2 else 0.0
        p_normal = 0.02 if favorite_margin <= 1 else 0.0
        probs[:] = [p_normal, p_open, p_pressure, 1.0 - p_normal - p_open - p_pressure]

    # Conservative research knobs. A leading favorite often controls the match
    # instead of continually opening it up, and an underdog that survives past
    # 50-60 minutes should have lower future collapse probability. These use
    # simulated pre-match paths only, never actual in-game events.
    if config.late_control_strength and control_phase and favorite_margin >= 2 and not mismatch:
        damp = max(0.0, 1.0 - config.late_control_strength)
        probs[STATE_TO_INDEX["open_game"]] *= damp
        probs[STATE_TO_INDEX["blowout"]] *= damp
    if config.underdog_resistance_strength and resistance_phase and favorite_margin < 2:
        damp = max(0.0, 1.0 - config.underdog_resistance_strength)
        probs[STATE_TO_INDEX["favorite_pressure"]] *= damp
        probs[STATE_TO_INDEX["blowout"]] *= damp

    probs = np.clip(probs, 0.0, 1.0)
    total = probs.sum()
    if total <= 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return probs / total


def _sample_next_states(
    rng: np.random.Generator,
    states: np.ndarray,
    bucket: str,
    favorite_margin: np.ndarray,
    step_index: int,
    n_steps: int,
    lambda_imbalance: float,
    config: MatchStateConfig,
) -> np.ndarray:
    next_states = states.copy()
    draws = rng.random(len(states))
    for state_index in range(len(STATES)):
        state_mask = states == state_index
        if not state_mask.any():
            continue
        margins = favorite_margin[state_mask]
        for margin_value in np.unique(margins):
            mask_indices = np.where(state_mask)[0][margins == margin_value]
            probs = _transition_probabilities(
                current_state=state_index,
                bucket=bucket,
                favorite_margin=int(margin_value),
                step_index=step_index,
                n_steps=n_steps,
                lambda_imbalance=lambda_imbalance,
                config=config,
            )
            cumulative = np.cumsum(probs)
            next_states[mask_indices] = np.searchsorted(cumulative, draws[mask_indices], side="right")
    return next_states


def _scoreline_result(scoreline: str) -> str:
    goals_a, goals_b = parse_scoreline(scoreline)
    if goals_a > goals_b:
        return "team_a_win"
    if goals_a < goals_b:
        return "team_b_win"
    return "draw"


def _tail_probabilities(
    scoreline_grid: Mapping[str, float],
    favorite_side: str,
) -> dict[str, float]:
    favorite_4 = favorite_5 = margin_4 = total_5 = btts = over25 = over35 = 0.0
    for scoreline, probability in scoreline_grid.items():
        goals_a, goals_b = parse_scoreline(scoreline)
        fav_goals = goals_a if favorite_side == "team_a" else goals_b
        dog_goals = goals_b if favorite_side == "team_a" else goals_a
        if fav_goals >= 4:
            favorite_4 += probability
        if fav_goals >= 5:
            favorite_5 += probability
        if fav_goals - dog_goals >= 4:
            margin_4 += probability
        if goals_a + goals_b >= 5:
            total_5 += probability
        if goals_a + goals_b >= 3:
            over25 += probability
        if goals_a + goals_b >= 4:
            over35 += probability
        if goals_a > 0 and goals_b > 0:
            btts += probability
    return {
        "over_2_5_probability": over25,
        "over_3_5_probability": over35,
        "btts_probability": btts,
        "favorite_scores_4_plus_probability": favorite_4,
        "favorite_scores_5_plus_probability": favorite_5,
        "margin_4_plus_probability": margin_4,
        "total_goals_5_plus_probability": total_5,
    }


def simulate_markov_scoreline_distribution(
    lambda_a: float,
    lambda_b: float,
    team_a_win_pct: float,
    draw_pct: float,
    team_b_win_pct: float,
    n_sims: int = 20000,
    step_minutes: int = 10,
    random_seed: int = 42,
    config: MatchStateConfig | None = None,
) -> dict[str, object]:
    """Simulate a pre-match Markov regime scoreline distribution.

    This is a research-only scoreline simulator. It marginalizes over possible
    latent match paths before kickoff and never uses actual in-game events.
    """
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")
    if step_minutes <= 0 or 90 % step_minutes != 0:
        raise ValueError("step_minutes must be a positive divisor of 90")
    if lambda_a < 0 or lambda_b < 0:
        raise ValueError("lambdas must be non-negative")
    config = config or MatchStateConfig()
    rng = np.random.default_rng(random_seed)

    p_team_a_win = normalize_win_probability(team_a_win_pct)
    p_team_b_win = normalize_win_probability(team_b_win_pct)
    normalize_win_probability(draw_pct)
    favorite_is_a = p_team_a_win >= p_team_b_win
    favorite_side = "team_a" if favorite_is_a else "team_b"
    favorite_win_prob = max(p_team_a_win, p_team_b_win)
    raw_total_lambda = lambda_a + lambda_b
    lambda_imbalance = abs(lambda_a - lambda_b) / max(raw_total_lambda, 1e-9)
    bucket = favorite_bucket(favorite_win_prob, lambda_imbalance)

    n_steps = 90 // step_minutes
    state_probs = _state_initial_probabilities(bucket, config)
    states = rng.choice(len(STATES), size=n_sims, p=state_probs)
    score_a = np.zeros(n_sims, dtype=int)
    score_b = np.zeros(n_sims, dtype=int)
    state_visits = np.zeros((n_sims, len(STATES)), dtype=int)

    fav_base_lambda = lambda_a if favorite_is_a else lambda_b
    dog_base_lambda = lambda_b if favorite_is_a else lambda_a
    fav_step_base = fav_base_lambda / n_steps
    dog_step_base = dog_base_lambda / n_steps

    fav_multipliers = np.array([config.favorite_goal_multipliers[state] for state in STATES])
    dog_multipliers = np.array([config.underdog_goal_multipliers[state] for state in STATES])

    for step_index in range(n_steps):
        fav_score = score_a if favorite_is_a else score_b
        dog_score = score_b if favorite_is_a else score_a
        favorite_margin = fav_score - dog_score
        states = _sample_next_states(
            rng=rng,
            states=states,
            bucket=bucket,
            favorite_margin=favorite_margin,
            step_index=step_index,
            n_steps=n_steps,
            lambda_imbalance=lambda_imbalance,
            config=config,
        )
        state_visits[np.arange(n_sims), states] += 1
        fav_goals = rng.poisson(fav_step_base * fav_multipliers[states])
        dog_goals = rng.poisson(dog_step_base * dog_multipliers[states])
        if favorite_is_a:
            score_a += fav_goals
            score_b += dog_goals
        else:
            score_a += dog_goals
            score_b += fav_goals

    scorelines, counts = np.unique(
        np.char.add(np.char.add(score_a.astype(str), "-"), score_b.astype(str)),
        return_counts=True,
    )
    scoreline_grid = {
        str(scoreline): float(count) / float(n_sims)
        for scoreline, count in zip(scorelines, counts)
    }
    top5 = top_scorelines(scoreline_grid, top_n=5)
    state_visit_rates = {
        state: float(state_visits[:, index].sum() / (n_sims * n_steps))
        for index, state in enumerate(STATES)
    }
    blowout_path_probability = float((state_visits[:, STATE_TO_INDEX["blowout"]] > 0).mean())
    tail = _tail_probabilities(scoreline_grid, favorite_side=favorite_side)
    return {
        "scoreline_probabilities": scoreline_grid,
        "top_1_scoreline": top5[0]["scoreline"],
        "top_3_scorelines": top5[:3],
        "top_5_scorelines": top5,
        "favorite_side": favorite_side,
        "favorite_win_prob": favorite_win_prob,
        "favorite_bucket": bucket,
        "lambda_imbalance": lambda_imbalance,
        "state_visit_rates": state_visit_rates,
        "blowout_path_probability": blowout_path_probability,
        **tail,
    }
