from __future__ import annotations

import math
from typing import Mapping


class ScorelineGrid(dict[str, float]):
    """Normalized NB scoreline probabilities plus display-selection metadata."""

    def __init__(
        self,
        *args,
        lambda_a: float | None = None,
        lambda_b: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b


def adjust_lambdas_for_display_scoreline(
    lambda_a: float,
    lambda_b: float,
    aggressiveness: float = 1.0,
) -> tuple[float, float]:
    """Return display-only lambdas for Negative Binomial scoreline selection.

    Official match W/D/L probabilities are calibrated elsewhere. This helper
    only makes the displayed scoreline grid a little more expressive in clear
    mismatches, so a strong favorite can surface 3-0 or 3-1 candidates more
    often without changing the outcome probability model.
    """
    if lambda_a < 0 or lambda_b < 0:
        raise ValueError("mean goals must be non-negative")
    if aggressiveness < 0:
        raise ValueError("aggressiveness must be non-negative")
    if aggressiveness == 0:
        return lambda_a, lambda_b

    total_lambda = lambda_a + lambda_b
    if total_lambda <= 0:
        return lambda_a, lambda_b

    favorite_gap = abs(lambda_a - lambda_b) / max(total_lambda, 1e-9)

    # Display-only boost: balanced games barely move, while clear mismatches
    # get a capped total-goal lift. This must not be used as official xG.
    mismatch_boost = min(0.35, 0.45 * favorite_gap) * aggressiveness
    adjusted_total = total_lambda + mismatch_boost

    # Shift a small amount of display goal share toward the favorite. The caps
    # avoid turning strong-vs-strong or weak-vs-weak games into unrealistic
    # blowouts.
    favorite_share_boost = min(0.06, 0.10 * favorite_gap) * aggressiveness
    favorite_share_boost = min(0.08, favorite_share_boost)
    original_share_a = lambda_a / total_lambda
    if lambda_a >= lambda_b:
        adjusted_share_a = min(0.88, original_share_a + favorite_share_boost)
    else:
        adjusted_share_a = max(0.12, original_share_a - favorite_share_boost)

    display_lambda_a = adjusted_total * adjusted_share_a
    display_lambda_b = adjusted_total * (1.0 - adjusted_share_a)
    return display_lambda_a, display_lambda_b


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
    max_goals: int = 10,
    aggressiveness: float = 0.0,
) -> ScorelineGrid:
    """Generate a normalized Negative Binomial scoreline display grid.

    Raw NB scoreline probabilities are separate from official W/D/L
    probabilities. The optional lambda aggressiveness argument is retained for
    backward compatibility, but the default path keeps the NB grid based on the
    incoming lambdas and lets display selection handle conservatism.
    """
    if max_goals < 0:
        raise ValueError("max_goals must be non-negative")
    display_lambda_a, display_lambda_b = adjust_lambdas_for_display_scoreline(
        lambda_a,
        lambda_b,
        aggressiveness=aggressiveness,
    )

    raw: dict[str, float] = {}
    for goals_a in range(max_goals + 1):
        p_a = _negative_binomial_pmf(goals_a, display_lambda_a, dispersion_k)
        for goals_b in range(max_goals + 1):
            raw[f"{goals_a}-{goals_b}"] = p_a * _negative_binomial_pmf(
                goals_b,
                display_lambda_b,
                dispersion_k,
            )

    total = sum(raw.values())
    if total <= 0:
        raise ValueError("Negative Binomial scoreline grid must have positive mass")
    return ScorelineGrid(
        {scoreline: probability / total for scoreline, probability in raw.items()},
        lambda_a=lambda_a,
        lambda_b=lambda_b,
    )


def _parse_scoreline(scoreline: str) -> tuple[int, int]:
    goals_a, goals_b = scoreline.split("-")
    return int(goals_a), int(goals_b)


def _raw_sorted_scorelines(
    scoreline_grid: Mapping[str, float],
) -> list[tuple[str, float]]:
    return sorted(scoreline_grid.items(), key=lambda item: (-item[1], item[0]))


def _display_utility(
    scoreline: str,
    probability: float,
    lambda_a: float,
    lambda_b: float,
    mode: str,
) -> float:
    goals_a, goals_b = _parse_scoreline(scoreline)
    total_goals = goals_a + goals_b
    margin = abs(goals_a - goals_b)
    total_lambda = lambda_a + lambda_b
    lambda_diff = lambda_a - lambda_b
    abs_diff = abs(lambda_diff)
    imbalance = abs_diff / max(total_lambda, 1e-9)
    favorite_is_a = lambda_a >= lambda_b
    favorite_wins = goals_a > goals_b if favorite_is_a else goals_b > goals_a
    underdog_goals = goals_b if favorite_is_a else goals_a

    multiplier = 1.35 if mode == "aggressive" else 1.0
    utility = math.log(probability + 1e-12)

    # This utility affects displayed scoreline selection only. It does not
    # change the raw NB grid and does not touch calibrated W/D/L probabilities.
    if imbalance < 0.10:
        if goals_a == goals_b:
            utility += 0.03
    else:
        if favorite_wins:
            utility += multiplier * min(0.16, 0.22 * imbalance)
        elif goals_a == goals_b:
            utility -= multiplier * min(0.12, 0.18 * imbalance)
        else:
            utility -= multiplier * min(0.45, 0.65 * imbalance)

    if favorite_wins and imbalance >= 0.25:
        if margin >= 2:
            utility += multiplier * min(0.24, 0.38 * imbalance)
        elif margin == 1:
            utility += multiplier * min(0.07, 0.12 * imbalance)

    expected_total_floor = round(total_lambda)
    if favorite_wins and 2 <= total_goals <= 4 and total_goals >= expected_total_floor:
        utility += multiplier * min(0.10, 0.06 + 0.08 * imbalance)

    # Guardrails: high displays must still be genuinely competitive in the NB
    # grid. These penalties keep 5+ totals and huge margins from appearing
    # unless the raw distribution already supports them.
    if total_goals >= 5:
        penalty_base = 0.22 if total_lambda >= 3.2 else 0.36
        utility -= penalty_base * (total_goals - 4)
    if margin >= 4 and imbalance < 0.55:
        utility -= 0.35 * (margin - 3)
    if underdog_goals >= 4:
        utility -= 0.60 + 0.20 * (underdog_goals - 4)

    return utility


def select_display_scoreline_from_grid(
    scoreline_grid: Mapping[str, float],
    lambda_a: float,
    lambda_b: float,
    mode: str = "balanced",
    top_n_candidates: int = 10,
) -> dict[str, float | str | list[dict[str, float | str]]]:
    """Select a displayed scoreline from a plausible NB candidate set.

    ``mode='mode'`` returns the raw NB mode. ``balanced`` is the default display
    behavior: candidates must remain close to the raw mode probability, then a
    utility function mildly rewards scorelines that better express a clear
    favorite's edge. This is display-only and never changes official outcome
    probabilities.
    """
    if mode not in {"mode", "balanced", "aggressive"}:
        raise ValueError("mode must be one of: mode, balanced, aggressive")
    if top_n_candidates <= 0:
        raise ValueError("top_n_candidates must be positive")

    sorted_scorelines = _raw_sorted_scorelines(scoreline_grid)
    if not sorted_scorelines:
        raise ValueError("scoreline_grid must not be empty")

    raw_mode_scoreline, raw_mode_probability = sorted_scorelines[0]
    total_lambda = lambda_a + lambda_b
    imbalance = abs(lambda_a - lambda_b) / max(total_lambda, 1e-9)
    top_5_scorelines = [
        {
            "scoreline": scoreline,
            "probability": float(probability),
            "implied_result": _implied_result(scoreline),
        }
        for scoreline, probability in sorted_scorelines[:5]
    ]

    if mode == "mode":
        return {
            "scoreline": raw_mode_scoreline,
            "probability": float(raw_mode_probability),
            "implied_result": _implied_result(raw_mode_scoreline),
            "raw_mode_scoreline": raw_mode_scoreline,
            "raw_mode_probability": float(raw_mode_probability),
            "selected_display_scoreline": raw_mode_scoreline,
            "selected_display_probability": float(raw_mode_probability),
            "display_selection_mode": mode,
            "display_imbalance": float(imbalance),
            "top_5_scorelines": top_5_scorelines,
        }

    threshold = 0.70 if mode == "balanced" else 0.50
    candidates = [
        (scoreline, probability)
        for scoreline, probability in sorted_scorelines[:top_n_candidates]
        if probability >= raw_mode_probability * threshold
    ]
    if not candidates:
        candidates = [(raw_mode_scoreline, raw_mode_probability)]

    selected_scoreline, selected_probability = max(
        candidates,
        key=lambda item: (
            _display_utility(item[0], item[1], lambda_a, lambda_b, mode),
            item[1],
            item[0],
        ),
    )
    return {
        "scoreline": selected_scoreline,
        "probability": float(selected_probability),
        "implied_result": _implied_result(selected_scoreline),
        "raw_mode_scoreline": raw_mode_scoreline,
        "raw_mode_probability": float(raw_mode_probability),
        "selected_display_scoreline": selected_scoreline,
        "selected_display_probability": float(selected_probability),
        "display_selection_mode": mode,
        "display_imbalance": float(imbalance),
        "top_5_scorelines": top_5_scorelines,
    }


def get_top_scorelines(
    scoreline_grid: Mapping[str, float],
    top_n: int = 3,
    mode: str = "mode",
    top_n_candidates: int = 10,
    lambda_a: float | None = None,
    lambda_b: float | None = None,
) -> list[dict[str, float | str]]:
    """Return displayed scorelines with probabilities and implied W/D/L result.

    The production default is ``mode='mode'``: the first row is the raw NB
    mode. V2.1/V2.2 selector modes remain available for diagnostics only. In
    all modes, this function affects displayed scorelines only and never
    changes calibrated W/D/L probabilities.
    """
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    resolved_lambda_a = lambda_a if lambda_a is not None else getattr(scoreline_grid, "lambda_a", None)
    resolved_lambda_b = lambda_b if lambda_b is not None else getattr(scoreline_grid, "lambda_b", None)
    raw_top = [
        {
            "scoreline": scoreline,
            "probability": float(probability),
            "implied_result": _implied_result(scoreline),
        }
        for scoreline, probability in _raw_sorted_scorelines(scoreline_grid)
    ]
    if (
        mode == "mode"
        or resolved_lambda_a is None
        or resolved_lambda_b is None
    ):
        return raw_top[:top_n]

    selected = select_display_scoreline_from_grid(
        scoreline_grid,
        lambda_a=float(resolved_lambda_a),
        lambda_b=float(resolved_lambda_b),
        mode=mode,
        top_n_candidates=top_n_candidates,
    )
    remaining = [item for item in raw_top if item["scoreline"] != selected["scoreline"]]
    return [selected, *remaining][:top_n]


def scoreline_entropy(scoreline_grid: Mapping[str, float]) -> float:
    """Return Shannon entropy for the normalized display scoreline grid."""
    return -sum(
        probability * math.log(probability)
        for probability in scoreline_grid.values()
        if probability > 0
    )
