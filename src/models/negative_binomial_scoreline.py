from __future__ import annotations

import math
from typing import Mapping


ScorelineGrid = dict[str, float]


def _negative_binomial_pmf(goals: int, mean_goals: float, dispersion_k: float) -> float:
    """Return NB probability with mean ``mean_goals`` and dispersion ``k``.

    This uses the common football-style parameterization where variance is
    ``mu + mu^2 / k``. Larger ``k`` approaches a Poisson distribution, while
    smaller values allow more over-dispersion in displayed scorelines.
    """
    if goals < 0:
        return 0.0
    if mean_goals < 0:
        raise ValueError("mean goals must be non-negative")
    if dispersion_k <= 0:
        raise ValueError("dispersion_k must be positive")
    if mean_goals == 0:
        return 1.0 if goals == 0 else 0.0

    k = float(dispersion_k)
    probability = k / (k + mean_goals)
    log_combination = math.lgamma(goals + k) - math.lgamma(k) - math.lgamma(goals + 1)
    return math.exp(
        log_combination
        + k * math.log(probability)
        + goals * math.log(1.0 - probability)
    )


def _implied_result(scoreline: str) -> str:
    goals_a, goals_b = scoreline.split("-")
    goals_a = int(goals_a)
    goals_b = int(goals_b)
    if goals_a > goals_b:
        return "team_a_win"
    if goals_a < goals_b:
        return "team_b_win"
    return "draw"


def negative_binomial_scoreline_grid(
    lambda_a: float,
    lambda_b: float,
    dispersion_k: float = 12,
    max_goals: int = 8,
) -> ScorelineGrid:
    """Generate a normalized Negative Binomial scoreline display grid."""
    if max_goals < 0:
        raise ValueError("max_goals must be non-negative")

    raw: ScorelineGrid = {}
    for goals_a in range(max_goals + 1):
        p_a = _negative_binomial_pmf(goals_a, lambda_a, dispersion_k)
        for goals_b in range(max_goals + 1):
            raw[f"{goals_a}-{goals_b}"] = p_a * _negative_binomial_pmf(
                goals_b,
                lambda_b,
                dispersion_k,
            )

    total = sum(raw.values())
    if total <= 0:
        raise ValueError("Negative Binomial scoreline grid must have positive mass")
    return {scoreline: probability / total for scoreline, probability in raw.items()}


def get_top_scorelines(
    scoreline_grid: Mapping[str, float],
    top_n: int = 3,
) -> list[dict[str, float | str]]:
    """Return the top scorelines with probabilities and implied W/D/L result."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    return [
        {
            "scoreline": scoreline,
            "probability": float(probability),
            "implied_result": _implied_result(scoreline),
        }
        for scoreline, probability in sorted(
            scoreline_grid.items(),
            key=lambda item: (-item[1], item[0]),
        )[:top_n]
    ]


def scoreline_entropy(scoreline_grid: Mapping[str, float]) -> float:
    """Return Shannon entropy for the normalized display scoreline grid."""
    return -sum(
        probability * math.log(probability)
        for probability in scoreline_grid.values()
        if probability > 0
    )
