from __future__ import annotations

import math
import random
from typing import Mapping


EXTRA_TIME_GOAL_SCALE = 30.0 / 90.0
EXTRA_TIME_CAUTION_FACTOR = 0.72
EXTRA_TIME_CANDIDATES = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]


def outcome_from_goals(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "team_a_win"
    if goals_b > goals_a:
        return "team_b_win"
    return "draw"


def scoreline_text(goals_a: int, goals_b: int) -> str:
    return f"{goals_a}-{goals_b}"


def parse_scoreline(scoreline: str) -> tuple[int, int]:
    left, right = str(scoreline).split("-", maxsplit=1)
    return int(left), int(right)


def most_likely_scoreline(scoreline_probabilities: Mapping[str, float]) -> str:
    return max(scoreline_probabilities.items(), key=lambda item: item[1])[0]


def poisson_pmf(goal_count: int, lam: float) -> float:
    return math.exp(-lam) * math.pow(lam, goal_count) / math.factorial(goal_count)


def extra_time_increment_probabilities(
    lambda_a: float,
    lambda_b: float,
    goal_scale: float = EXTRA_TIME_GOAL_SCALE,
    caution_factor: float = EXTRA_TIME_CAUTION_FACTOR,
) -> dict[str, float]:
    """Return compact ET increment probabilities for display/simulation only.

    This layer is conditional on a 90-minute draw. It does not change the core
    90-minute W/D/L model or scoreline distribution.
    """
    et_lambda_a = max(float(lambda_a) * goal_scale * caution_factor, 1e-9)
    et_lambda_b = max(float(lambda_b) * goal_scale * caution_factor, 1e-9)
    raw = {
        scoreline_text(goals_a, goals_b): poisson_pmf(goals_a, et_lambda_a)
        * poisson_pmf(goals_b, et_lambda_b)
        for goals_a, goals_b in EXTRA_TIME_CANDIDATES
    }
    total = sum(raw.values())
    return {scoreline: probability / total for scoreline, probability in raw.items()}


def likely_extra_time_increment(lambda_a: float, lambda_b: float) -> str:
    probabilities = extra_time_increment_probabilities(lambda_a, lambda_b)
    return max(probabilities.items(), key=lambda item: item[1])[0]


def sample_extra_time_increment(lambda_a: float, lambda_b: float, rng: random.Random) -> str:
    probabilities = extra_time_increment_probabilities(lambda_a, lambda_b)
    threshold = rng.random()
    cumulative = 0.0
    last_scoreline = "0-0"
    for scoreline, probability in probabilities.items():
        cumulative += probability
        last_scoreline = scoreline
        if threshold <= cumulative:
            return scoreline
    return last_scoreline


def penalty_win_probability_team_a(rating_a: float, rating_b: float) -> float:
    """Return a conservative rating-based penalty edge capped at 45%-55%."""
    rating_gap = float(rating_a) - float(rating_b)
    probability = 0.50 + max(-0.05, min(0.05, rating_gap / 4000.0))
    return max(0.45, min(0.55, probability))


def add_scoreline(base_scoreline: str, increment_scoreline: str) -> str:
    goals_a, goals_b = parse_scoreline(base_scoreline)
    inc_a, inc_b = parse_scoreline(increment_scoreline)
    return scoreline_text(goals_a + inc_a, goals_b + inc_b)


def winner_from_result(result: str, team_a: str, team_b: str) -> str | None:
    if result == "team_a_win":
        return team_a
    if result == "team_b_win":
        return team_b
    return None


def predicted_knockout_resolution(
    team_a: str,
    team_b: str,
    prediction: Mapping[str, object],
    rating_a: float,
    rating_b: float,
) -> dict[str, object]:
    """Build display fields for 90/120/penalty knockout resolution.

    The 90-minute scoreline comes from the existing scoreline model. Extra time
    and penalties are only applied when that 90-minute display scoreline is tied.
    """
    scoreline_probabilities = prediction["scoreline_probabilities"]
    predicted_90min_score = most_likely_scoreline(scoreline_probabilities)
    goals_a_90, goals_b_90 = parse_scoreline(predicted_90min_score)
    result_90 = outcome_from_goals(goals_a_90, goals_b_90)
    p_pen_a = penalty_win_probability_team_a(rating_a, rating_b)
    p_pen_b = 1.0 - p_pen_a

    if result_90 != "draw":
        final_winner = winner_from_result(result_90, team_a, team_b)
        return {
            "predicted_90min_score": predicted_90min_score,
            "predicted_90min_result": result_90,
            "predicted_120min_score": predicted_90min_score,
            "predicted_120min_result": result_90,
            "extra_time_goal_increment": None,
            "penalty_winner": None,
            "penalty_win_prob_team_a": round(p_pen_a * 100.0, 1),
            "penalty_win_prob_team_b": round(p_pen_b * 100.0, 1),
            "final_winner": final_winner,
            "match_resolution": "90min",
            "final_advancing_team": final_winner,
        }

    increment = likely_extra_time_increment(
        float(prediction["lambda_a"]),
        float(prediction["lambda_b"]),
    )
    predicted_120min_score = add_scoreline(predicted_90min_score, increment)
    goals_a_120, goals_b_120 = parse_scoreline(predicted_120min_score)
    result_120 = outcome_from_goals(goals_a_120, goals_b_120)
    if result_120 != "draw":
        final_winner = winner_from_result(result_120, team_a, team_b)
        penalty_winner = None
        resolution = "extra_time"
    else:
        penalty_winner = team_a if p_pen_a >= p_pen_b else team_b
        final_winner = penalty_winner
        resolution = "penalties"

    return {
        "predicted_90min_score": predicted_90min_score,
        "predicted_90min_result": result_90,
        "predicted_120min_score": predicted_120min_score,
        "predicted_120min_result": result_120,
        "extra_time_goal_increment": increment,
        "penalty_winner": penalty_winner,
        "penalty_win_prob_team_a": round(p_pen_a * 100.0, 1),
        "penalty_win_prob_team_b": round(p_pen_b * 100.0, 1),
        "final_winner": final_winner,
        "match_resolution": resolution,
        "final_advancing_team": final_winner,
    }


def simulate_knockout_resolution(
    team_a: str,
    team_b: str,
    goals_a_90: int,
    goals_b_90: int,
    prediction: Mapping[str, object],
    rating_a: float,
    rating_b: float,
    rng: random.Random,
) -> dict[str, object]:
    """Resolve a sampled knockout match through ET and penalties if required."""
    result_90 = outcome_from_goals(goals_a_90, goals_b_90)
    p_pen_a = penalty_win_probability_team_a(rating_a, rating_b)
    p_pen_b = 1.0 - p_pen_a
    score_90 = scoreline_text(goals_a_90, goals_b_90)
    if result_90 != "draw":
        winner = winner_from_result(result_90, team_a, team_b)
        loser = team_b if winner == team_a else team_a
        return {
            "winner": winner,
            "loser": loser,
            "match_resolution": "90min",
            "predicted_90min_score": score_90,
            "predicted_120min_score": score_90,
            "extra_time_goal_increment": None,
            "penalty_winner": None,
            "penalty_win_prob_team_a": round(p_pen_a * 100.0, 1),
            "penalty_win_prob_team_b": round(p_pen_b * 100.0, 1),
        }

    increment = sample_extra_time_increment(
        float(prediction["lambda_a"]),
        float(prediction["lambda_b"]),
        rng=rng,
    )
    score_120 = add_scoreline(score_90, increment)
    goals_a_120, goals_b_120 = parse_scoreline(score_120)
    result_120 = outcome_from_goals(goals_a_120, goals_b_120)
    if result_120 != "draw":
        winner = winner_from_result(result_120, team_a, team_b)
        loser = team_b if winner == team_a else team_a
        return {
            "winner": winner,
            "loser": loser,
            "match_resolution": "extra_time",
            "predicted_90min_score": score_90,
            "predicted_120min_score": score_120,
            "extra_time_goal_increment": increment,
            "penalty_winner": None,
            "penalty_win_prob_team_a": round(p_pen_a * 100.0, 1),
            "penalty_win_prob_team_b": round(p_pen_b * 100.0, 1),
        }

    penalty_winner = team_a if rng.random() < p_pen_a else team_b
    loser = team_b if penalty_winner == team_a else team_a
    return {
        "winner": penalty_winner,
        "loser": loser,
        "match_resolution": "penalties",
        "predicted_90min_score": score_90,
        "predicted_120min_score": score_120,
        "extra_time_goal_increment": increment,
        "penalty_winner": penalty_winner,
        "penalty_win_prob_team_a": round(p_pen_a * 100.0, 1),
        "penalty_win_prob_team_b": round(p_pen_b * 100.0, 1),
    }
