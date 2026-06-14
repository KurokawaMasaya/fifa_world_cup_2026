from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.models.negative_binomial_scoreline import (
    ScorelineGrid,
    _negative_binomial_pmf,
)


@dataclass(frozen=True)
class BlowoutMixtureParams:
    """Research-only V2.3 scoreline-tail parameters.

    These parameters affect only the displayed/research scoreline grid. They do
    not change calibrated W/D/L probabilities, team ratings, Elo, roster
    adjustments, or tournament simulation logic.
    """

    normal_k: float = 12.0
    blowout_k: float = 6.0
    blowout_lambda_multiplier: float = 1.75
    blowout_lambda_add: float = 0.75
    max_blowout_lambda_fav: float = 6.0
    underdog_blowout_lambda_multiplier: float = 0.82
    min_underdog_blowout_lambda: float = 0.15
    p_favorite_weight: float = 0.22
    p_imbalance_weight: float = 0.08
    p_rating_weight: float = 0.03
    p_multiplier: float = 1.0


@dataclass(frozen=True)
class GatedBlowoutParams(BlowoutMixtureParams):
    """Research-only V2.4 gated blowout mixture parameters."""

    blowout_k_factor: float = 1.0
    use_favorite_dominance_gate: bool = True
    use_lambda_imbalance_gate: bool = True
    use_favorite_scoring_capacity_gate: bool = True
    use_underdog_suppression_gate: bool = True
    use_motivation_gate: bool = True
    motivation_factor: float = 1.0
    favorite_win_prob_threshold: float = 0.58
    favorite_lambda_threshold: float = 1.65
    lambda_imbalance_threshold: float = 0.25
    favorite_scoring_capacity_power: float = 1.0


def normalize_win_probability(probability: float) -> float:
    """Accept either 0-1 probabilities or 0-100 percentage inputs."""
    probability = float(probability)
    return probability / 100.0 if probability > 1.0 else probability


def favorite_bucket(favorite_win_prob: float, lambda_imbalance: float) -> str:
    """Classify favorite-vs-underdog structure from pre-match features only."""
    favorite_win_prob = normalize_win_probability(favorite_win_prob)
    structure = max(favorite_win_prob, 0.50 + 0.50 * float(lambda_imbalance))
    if structure < 0.52:
        return "balanced"
    if structure < 0.62:
        return "slight_favorite"
    if structure < 0.74:
        return "clear_favorite"
    if structure < 0.86:
        return "heavy_favorite"
    return "extreme_mismatch"


def blowout_probability(
    favorite_win_prob: float,
    lambda_imbalance: float,
    rating_gap: float | None = None,
    params: BlowoutMixtureParams | None = None,
) -> tuple[float, str]:
    """Estimate latent pre-match probability of a blowout/collapse state.

    The result is deliberately bucket-capped. A strong team beating a weak team
    5-0 is a structurally plausible tail event, but this research layer should
    not globally inflate scorelines or make blowouts top predictions by force.
    """
    favorite_win_prob = normalize_win_probability(favorite_win_prob)
    params = params or BlowoutMixtureParams()
    lambda_imbalance = max(0.0, min(1.0, float(lambda_imbalance)))
    bucket = favorite_bucket(favorite_win_prob, lambda_imbalance)
    caps = {
        "balanced": (0.00, 0.02),
        "slight_favorite": (0.00, 0.04),
        "clear_favorite": (0.03, 0.10),
        "heavy_favorite": (0.08, 0.20),
        "extreme_mismatch": (0.15, 0.30),
    }
    floor, cap = caps[bucket]
    rating_signal = 0.0
    if rating_gap is not None:
        rating_signal = max(0.0, min(1.0, abs(float(rating_gap)) / 600.0))

    # Pre-match structural signal only: favorite probability, lambda imbalance,
    # and optional rating gap. No current score, minute, red card, or final
    # result information is allowed here.
    raw = (
        floor
        + params.p_favorite_weight * max(0.0, favorite_win_prob - 0.58)
        + params.p_imbalance_weight * lambda_imbalance
        + params.p_rating_weight * rating_signal
    ) * params.p_multiplier
    return max(floor, min(cap, raw)), bucket


def bucket_cap_for_favorite_bucket(bucket: str) -> float:
    return {
        "balanced": 0.02,
        "slight_favorite": 0.04,
        "clear_favorite": 0.10,
        "heavy_favorite": 0.20,
        "extreme_mismatch": 0.30,
    }[bucket]


def _soft_gate(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        raise ValueError("upper must be greater than lower")
    return max(0.0, min(1.0, (float(value) - lower) / (upper - lower)))


def blowout_gate_components(
    favorite_win_prob: float,
    lambda_imbalance: float,
    base_lambda_fav: float,
    base_lambda_dog: float,
    rating_gap: float | None = None,
    motivation_factor: float = 1.0,
    favorite_win_prob_threshold: float = 0.58,
    favorite_lambda_threshold: float = 1.65,
    lambda_imbalance_threshold: float = 0.25,
    favorite_scoring_capacity_power: float = 1.0,
) -> dict[str, float | str]:
    """Compute V2.4 pre-match structural gate components.

    All inputs are pre-match features. Motivation is neutral unless real
    pre-match motivation data is supplied by a caller.
    """
    favorite_win_prob = normalize_win_probability(favorite_win_prob)
    favorite_dominance_gate = _soft_gate(favorite_win_prob, favorite_win_prob_threshold, 0.82)
    lambda_imbalance_gate = _soft_gate(lambda_imbalance, lambda_imbalance_threshold, 0.62)
    favorite_scoring_capacity_gate = _soft_gate(base_lambda_fav, favorite_lambda_threshold, 2.65)
    if favorite_scoring_capacity_power != 1.0:
        favorite_scoring_capacity_gate = favorite_scoring_capacity_gate ** favorite_scoring_capacity_power
    underdog_suppression_gate = 1.0 - _soft_gate(base_lambda_dog, 0.65, 1.25)
    motivation_gate = max(0.0, min(1.0, float(motivation_factor)))

    reasons = []
    if favorite_win_prob < favorite_win_prob_threshold:
        reasons.append(f"favorite_win_prob_below_{favorite_win_prob_threshold:.2f}")
        favorite_dominance_gate *= 0.20
    if base_lambda_fav < favorite_lambda_threshold:
        reasons.append(f"base_lambda_favorite_below_{favorite_lambda_threshold:.2f}")
        favorite_scoring_capacity_gate *= 0.25
    if lambda_imbalance < lambda_imbalance_threshold:
        reasons.append(f"lambda_imbalance_below_{lambda_imbalance_threshold:.2f}")
        lambda_imbalance_gate *= 0.25
    if base_lambda_dog > 1.20:
        reasons.append("underdog_lambda_not_suppressed")

    return {
        "favorite_dominance_gate": favorite_dominance_gate,
        "lambda_imbalance_gate": lambda_imbalance_gate,
        "favorite_scoring_capacity_gate": favorite_scoring_capacity_gate,
        "underdog_suppression_gate": underdog_suppression_gate,
        "motivation_gate": motivation_gate,
        "gate_suppression_reason": ";".join(reasons) if reasons else "none",
    }


def gated_blowout_probability(
    favorite_win_prob: float,
    lambda_imbalance: float,
    base_lambda_fav: float,
    base_lambda_dog: float,
    rating_gap: float | None = None,
    params: GatedBlowoutParams | None = None,
) -> dict[str, float | str]:
    """Return V2.4 gated blowout probability and diagnostics."""
    params = params or GatedBlowoutParams()
    p_raw, bucket = blowout_probability(
        favorite_win_prob=favorite_win_prob,
        lambda_imbalance=lambda_imbalance,
        rating_gap=rating_gap,
        params=params,
    )
    components = blowout_gate_components(
        favorite_win_prob=favorite_win_prob,
        lambda_imbalance=lambda_imbalance,
        base_lambda_fav=base_lambda_fav,
        base_lambda_dog=base_lambda_dog,
        rating_gap=rating_gap,
        motivation_factor=params.motivation_factor,
        favorite_win_prob_threshold=params.favorite_win_prob_threshold,
        favorite_lambda_threshold=params.favorite_lambda_threshold,
        lambda_imbalance_threshold=params.lambda_imbalance_threshold,
        favorite_scoring_capacity_power=params.favorite_scoring_capacity_power,
    )
    active_gates = []
    for enabled, name in [
        (params.use_favorite_dominance_gate, "favorite_dominance_gate"),
        (params.use_lambda_imbalance_gate, "lambda_imbalance_gate"),
        (params.use_favorite_scoring_capacity_gate, "favorite_scoring_capacity_gate"),
        (params.use_underdog_suppression_gate, "underdog_suppression_gate"),
        (params.use_motivation_gate, "motivation_gate"),
    ]:
        if enabled:
            active_gates.append(float(components[name]))
    blowout_gate = 1.0
    for gate in active_gates:
        blowout_gate *= gate

    cap = bucket_cap_for_favorite_bucket(bucket)
    p_final = max(0.0, min(cap, p_raw * blowout_gate * params.blowout_k_factor))
    return {
        "p_blowout_raw": p_raw,
        "blowout_gate": blowout_gate,
        "blowout_k": params.blowout_k_factor,
        "p_blowout_final": p_final,
        "bucket_cap": cap,
        "favorite_bucket": bucket,
        **components,
    }


def _normalize_grid(raw: dict[str, float], lambda_a: float, lambda_b: float) -> ScorelineGrid:
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("Scoreline grid must have positive probability mass")
    return ScorelineGrid(
        {scoreline: probability / total for scoreline, probability in raw.items()},
        lambda_a=lambda_a,
        lambda_b=lambda_b,
    )


def _nb_independent_grid(
    lambda_a: float,
    lambda_b: float,
    dispersion_k: float,
    max_goals: int,
) -> ScorelineGrid:
    raw: dict[str, float] = {}
    for goals_a in range(max_goals + 1):
        p_a = _negative_binomial_pmf(goals_a, lambda_a, dispersion_k)
        for goals_b in range(max_goals + 1):
            raw[f"{goals_a}-{goals_b}"] = p_a * _negative_binomial_pmf(
                goals_b,
                lambda_b,
                dispersion_k,
            )
    return _normalize_grid(raw, lambda_a=lambda_a, lambda_b=lambda_b)


def favorite_blowout_mixture_scoreline_grid(
    lambda_a: float,
    lambda_b: float,
    p_team_a_win: float,
    p_draw: float,
    p_team_b_win: float,
    rating_gap: float | None = None,
    max_goals: int = 10,
    params: BlowoutMixtureParams | None = None,
) -> tuple[ScorelineGrid, dict[str, float | str]]:
    """Return research-only V2.3 latent blowout-mixture scoreline grid.

    The normal state is the stable V2.0 Negative Binomial scoreline grid. The
    blowout state thickens only the favorite's right tail for structurally
    mismatched pre-match setups. W/D/L probabilities are supplied only as
    pre-match features for estimating ``p_blowout``; they are not overwritten.
    """
    if max_goals < 0:
        raise ValueError("max_goals must be non-negative")
    if lambda_a < 0 or lambda_b < 0:
        raise ValueError("lambdas must be non-negative")
    params = params or BlowoutMixtureParams()

    p_team_a_win = normalize_win_probability(p_team_a_win)
    p_team_b_win = normalize_win_probability(p_team_b_win)
    p_draw = normalize_win_probability(p_draw)
    favorite_is_a = p_team_a_win >= p_team_b_win
    favorite_win_prob = max(p_team_a_win, p_team_b_win)
    base_lambda_fav = lambda_a if favorite_is_a else lambda_b
    base_lambda_dog = lambda_b if favorite_is_a else lambda_a
    raw_total_lambda = lambda_a + lambda_b
    lambda_imbalance = abs(lambda_a - lambda_b) / max(raw_total_lambda, 1e-9)
    p_blowout, bucket = blowout_probability(
        favorite_win_prob=favorite_win_prob,
        lambda_imbalance=lambda_imbalance,
        rating_gap=rating_gap,
        params=params,
    )

    normal_grid = _nb_independent_grid(
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        dispersion_k=params.normal_k,
        max_goals=max_goals,
    )

    blowout_lambda_fav = min(
        params.max_blowout_lambda_fav,
        base_lambda_fav * params.blowout_lambda_multiplier + params.blowout_lambda_add,
    )
    blowout_lambda_dog = max(
        params.min_underdog_blowout_lambda,
        base_lambda_dog * params.underdog_blowout_lambda_multiplier,
    )
    blowout_lambda_a = blowout_lambda_fav if favorite_is_a else blowout_lambda_dog
    blowout_lambda_b = blowout_lambda_dog if favorite_is_a else blowout_lambda_fav
    blowout_grid = _nb_independent_grid(
        lambda_a=blowout_lambda_a,
        lambda_b=blowout_lambda_b,
        dispersion_k=params.blowout_k,
        max_goals=max_goals,
    )

    mixed = {
        scoreline: (1.0 - p_blowout) * normal_grid[scoreline]
        + p_blowout * blowout_grid[scoreline]
        for scoreline in normal_grid
    }
    grid = _normalize_grid(mixed, lambda_a=lambda_a, lambda_b=lambda_b)
    metadata = {
        "favorite_side": "team_a" if favorite_is_a else "team_b",
        "favorite_win_prob": favorite_win_prob,
        "favorite_bucket": bucket,
        "base_lambda_fav": base_lambda_fav,
        "base_lambda_dog": base_lambda_dog,
        "p_blowout": p_blowout,
        "blowout_lambda_fav": blowout_lambda_fav,
        "blowout_lambda_dog": blowout_lambda_dog,
        "lambda_imbalance": lambda_imbalance,
        "raw_total_lambda": raw_total_lambda,
        "normal_k": params.normal_k,
        "blowout_k": params.blowout_k,
        "p_team_a_win": p_team_a_win,
        "p_draw": p_draw,
        "p_team_b_win": p_team_b_win,
    }
    return grid, metadata


def gated_favorite_blowout_mixture_scoreline_grid(
    lambda_a: float,
    lambda_b: float,
    p_team_a_win: float,
    p_draw: float,
    p_team_b_win: float,
    rating_gap: float | None = None,
    max_goals: int = 10,
    params: GatedBlowoutParams | None = None,
) -> tuple[ScorelineGrid, dict[str, float | str]]:
    """Return research-only V2.4 gated blowout-mixture scoreline grid."""
    params = params or GatedBlowoutParams()
    p_team_a_win = normalize_win_probability(p_team_a_win)
    p_team_b_win = normalize_win_probability(p_team_b_win)
    p_draw = normalize_win_probability(p_draw)
    favorite_is_a = p_team_a_win >= p_team_b_win
    favorite_win_prob = max(p_team_a_win, p_team_b_win)
    base_lambda_fav = lambda_a if favorite_is_a else lambda_b
    base_lambda_dog = lambda_b if favorite_is_a else lambda_a
    raw_total_lambda = lambda_a + lambda_b
    lambda_imbalance = abs(lambda_a - lambda_b) / max(raw_total_lambda, 1e-9)
    gate = gated_blowout_probability(
        favorite_win_prob=favorite_win_prob,
        lambda_imbalance=lambda_imbalance,
        base_lambda_fav=base_lambda_fav,
        base_lambda_dog=base_lambda_dog,
        rating_gap=rating_gap,
        params=params,
    )

    normal_grid = _nb_independent_grid(
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        dispersion_k=params.normal_k,
        max_goals=max_goals,
    )
    blowout_lambda_fav = min(
        params.max_blowout_lambda_fav,
        base_lambda_fav * params.blowout_lambda_multiplier + params.blowout_lambda_add,
    )
    blowout_lambda_dog = max(
        params.min_underdog_blowout_lambda,
        base_lambda_dog * params.underdog_blowout_lambda_multiplier,
    )
    blowout_lambda_a = blowout_lambda_fav if favorite_is_a else blowout_lambda_dog
    blowout_lambda_b = blowout_lambda_dog if favorite_is_a else blowout_lambda_fav
    blowout_grid = _nb_independent_grid(
        lambda_a=blowout_lambda_a,
        lambda_b=blowout_lambda_b,
        dispersion_k=params.blowout_k,
        max_goals=max_goals,
    )
    p_final = float(gate["p_blowout_final"])
    mixed = {
        scoreline: (1.0 - p_final) * normal_grid[scoreline]
        + p_final * blowout_grid[scoreline]
        for scoreline in normal_grid
    }
    grid = _normalize_grid(mixed, lambda_a=lambda_a, lambda_b=lambda_b)
    metadata = {
        "favorite_side": "team_a" if favorite_is_a else "team_b",
        "favorite_win_prob": favorite_win_prob,
        "base_lambda_fav": base_lambda_fav,
        "base_lambda_dog": base_lambda_dog,
        "blowout_lambda_fav": blowout_lambda_fav,
        "blowout_lambda_dog": blowout_lambda_dog,
        "lambda_imbalance": lambda_imbalance,
        "raw_total_lambda": raw_total_lambda,
        "normal_k": params.normal_k,
        "blowout_state_k": params.blowout_k,
        "p_team_a_win": p_team_a_win,
        "p_draw": p_draw,
        "p_team_b_win": p_team_b_win,
        **gate,
    }
    return grid, metadata


def sorted_scorelines(scoreline_grid: Mapping[str, float]) -> list[tuple[str, float]]:
    return sorted(scoreline_grid.items(), key=lambda item: (-item[1], item[0]))


def top_scorelines(scoreline_grid: Mapping[str, float], top_n: int = 5) -> list[dict[str, object]]:
    rows = []
    for scoreline, probability in sorted_scorelines(scoreline_grid)[:top_n]:
        goals_a, goals_b = parse_scoreline(scoreline)
        if goals_a > goals_b:
            result = "team_a_win"
        elif goals_a < goals_b:
            result = "team_b_win"
        else:
            result = "draw"
        rows.append(
            {
                "scoreline": scoreline,
                "probability": float(probability),
                "implied_result": result,
            }
        )
    return rows


def parse_scoreline(scoreline: str) -> tuple[int, int]:
    goals_a, goals_b = scoreline.split("-")
    return int(goals_a), int(goals_b)


def tail_metrics(
    scoreline_grid: Mapping[str, float],
    favorite_side: str,
) -> dict[str, float]:
    """Return blowout-tail probabilities for the favorite in the grid."""
    p_fav_4_plus = 0.0
    p_fav_5_plus = 0.0
    p_total_5_plus = 0.0
    p_margin_4_plus = 0.0
    p_margin_5_plus = 0.0
    for scoreline, probability in scoreline_grid.items():
        goals_a, goals_b = parse_scoreline(scoreline)
        fav_goals = goals_a if favorite_side == "team_a" else goals_b
        dog_goals = goals_b if favorite_side == "team_a" else goals_a
        margin = fav_goals - dog_goals
        if fav_goals >= 4:
            p_fav_4_plus += probability
        if fav_goals >= 5:
            p_fav_5_plus += probability
        if goals_a + goals_b >= 5:
            p_total_5_plus += probability
        if margin >= 4:
            p_margin_4_plus += probability
        if margin >= 5:
            p_margin_5_plus += probability
    return {
        "p_favorite_scores_4_plus": p_fav_4_plus,
        "p_favorite_scores_5_plus": p_fav_5_plus,
        "p_total_goals_5_plus": p_total_5_plus,
        "p_margin_4_plus": p_margin_4_plus,
        "p_margin_5_plus": p_margin_5_plus,
        "tail_risk_index": (
            0.30 * p_fav_4_plus
            + 0.25 * p_fav_5_plus
            + 0.20 * p_total_5_plus
            + 0.15 * p_margin_4_plus
            + 0.10 * p_margin_5_plus
        ),
    }
