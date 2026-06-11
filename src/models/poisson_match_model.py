from __future__ import annotations

import json
import math
import argparse
from pathlib import Path
from typing import Mapping

import pandas as pd

from src.config.model_config import (
    draw_calibration_kwargs,
    load_model_config,
    poisson_parameter_kwargs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RATINGS_PATH = PROJECT_ROOT / "data" / "processed" / "team_ratings_world_cup_elo.csv"
CALIBRATED_RATINGS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "team_ratings_world_cup_elo_calibrated.csv"
)
DEFAULT_MATCHES_PATH = PROJECT_ROOT / "data" / "processed" / "games_result_clean.csv"
DEFAULT_RATING_COL = "anchored_final_strength"
FALLBACK_RATING_COL = "world_cup_elo_rating"
Scoreline = tuple[int, int]


def poisson_pmf(k: int, lam: float) -> float:
    """Return P(X = k) for X ~ Poisson(lam).

    The Poisson model is used here because football score counts are small,
    discrete, non-negative events. The module later combines two independent
    Poisson goal distributions, one for each team.
    """
    if lam < 0:
        raise ValueError("lam must be non-negative")
    if k < 0:
        return 0.0
    if int(k) != k:
        raise ValueError("k must be an integer")
    if lam == 0:
        return 1.0 if k == 0 else 0.0

    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def estimate_base_goals(matches_df: pd.DataFrame, start_date: str = "2014-01-01") -> float:
    """Estimate one-team baseline goals from cleaned historical match data.

    The model uses adjusted Elo strength to shift expected goals up/down rather
    than relying on raw team goal averages. Raw goals are noisy in football:
    opponent strength, red cards, and match state can distort scorelines. The
    historical average supplies only the global scoring environment.
    """
    required_columns = {"date", "home_score", "away_score"}
    missing = required_columns - set(matches_df.columns)
    if missing:
        raise ValueError(f"matches_df is missing columns: {sorted(missing)}")

    matches = matches_df.copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches["home_score"] = pd.to_numeric(matches["home_score"], errors="coerce")
    matches["away_score"] = pd.to_numeric(matches["away_score"], errors="coerce")
    matches = matches.dropna(subset=["date", "home_score", "away_score"])
    matches = matches.loc[matches["date"] >= pd.Timestamp(start_date)]

    if matches.empty:
        raise ValueError("No completed matches available for base-goals estimate")

    average_total_goals = (matches["home_score"] + matches["away_score"]).mean()
    return float(average_total_goals / 2.0)


def expected_goals_from_strength(
    strength_a: float,
    strength_b: float,
    advantage_a: float = 0,
    mapping_mode: str = "total_share",
    base_total_goals: float = 2.65,
    share_scale: float = 250,
    mismatch_total_bonus: float = 0.80,
    mismatch_scale: float = 600,
    base_goals: float = 1.35,
    scale: float = 800,
    lambda_min: float = 0.15,
    lambda_max: float = 4.50,
) -> dict:
    """Convert adjusted Elo strengths into expected goals for both teams.

    The default ``total_share`` mode separates two ideas that were previously
    tangled together: the stronger team's share of total goals and the match's
    total expected goals. This keeps win probability responsive to strength
    while preventing every mismatch from exploding into unrealistic xG.

    ``exp_symmetric`` keeps the older symmetric exponential mapping for model
    comparison and debugging.
    """
    if lambda_min < 0 or lambda_max <= lambda_min:
        raise ValueError("lambda bounds must satisfy 0 <= lambda_min < lambda_max")

    strength_diff = strength_a + advantage_a - strength_b

    if mapping_mode == "total_share":
        if base_total_goals <= 0:
            raise ValueError("base_total_goals must be positive")
        if share_scale <= 0:
            raise ValueError("share_scale must be positive")
        if mismatch_total_bonus < 0:
            raise ValueError("mismatch_total_bonus must be non-negative")
        if mismatch_scale <= 0:
            raise ValueError("mismatch_scale must be positive")

        goal_share_a = 1.0 / (1.0 + math.exp(-strength_diff / share_scale))
        goal_share_b = 1.0 - goal_share_a
        total_expected_goals = base_total_goals + min(
            mismatch_total_bonus,
            abs(strength_diff) / mismatch_scale,
        )
        lambda_a = total_expected_goals * goal_share_a
        lambda_b = total_expected_goals * goal_share_b
    elif mapping_mode == "exp_symmetric":
        if base_goals <= 0:
            raise ValueError("base_goals must be positive")
        if scale <= 0:
            raise ValueError("scale must be positive")

        lambda_a = base_goals * math.exp(strength_diff / scale)
        lambda_b = base_goals * math.exp(-strength_diff / scale)
        total_expected_goals = lambda_a + lambda_b
        goal_share_a = lambda_a / total_expected_goals
        goal_share_b = lambda_b / total_expected_goals
    else:
        raise ValueError("mapping_mode must be 'total_share' or 'exp_symmetric'")

    lambda_a = min(lambda_max, max(lambda_min, lambda_a))
    lambda_b = min(lambda_max, max(lambda_min, lambda_b))
    clipped_total = lambda_a + lambda_b
    if clipped_total > 0:
        goal_share_a = lambda_a / clipped_total
        goal_share_b = lambda_b / clipped_total
        total_expected_goals = clipped_total

    return {
        "strength_a": strength_a,
        "strength_b": strength_b,
        "strength_diff": strength_diff,
        "goal_share_a": goal_share_a,
        "goal_share_b": goal_share_b,
        "total_expected_goals": total_expected_goals,
        "lambda_a": lambda_a,
        "lambda_b": lambda_b,
        "base_total_goals": base_total_goals,
        "share_scale": share_scale,
        "mismatch_total_bonus": mismatch_total_bonus,
        "mismatch_scale": mismatch_scale,
        "mapping_mode": mapping_mode,
        "base_goals": base_goals,
        "scale": scale,
    }


def scoreline_probabilities(
    lambda_a: float,
    lambda_b: float,
    max_goals: int = 8,
) -> dict[Scoreline, float]:
    """Enumerate normalized scoreline probabilities from 0-0 to max_goals-max_goals.

    The model assumes team goal counts are conditionally independent given the
    two expected-goals values. Because the infinite Poisson tail is truncated at
    max_goals, probabilities are normalized afterward so the returned finite
    scoreline grid sums to 1.
    """
    if max_goals < 0:
        raise ValueError("max_goals must be non-negative")

    raw_probs: dict[Scoreline, float] = {}
    for goals_a in range(max_goals + 1):
        p_a = poisson_pmf(goals_a, lambda_a)
        for goals_b in range(max_goals + 1):
            raw_probs[(goals_a, goals_b)] = p_a * poisson_pmf(goals_b, lambda_b)

    total_probability = sum(raw_probs.values())
    if total_probability <= 0:
        raise ValueError("Truncated scoreline probabilities sum to zero")

    return {
        scoreline: probability / total_probability
        for scoreline, probability in raw_probs.items()
    }


def _parse_scoreline_key(scoreline: Scoreline | str) -> Scoreline:
    if isinstance(scoreline, tuple):
        return scoreline
    goals_a, goals_b = scoreline.split("-")
    return int(goals_a), int(goals_b)


def outcome_probabilities(score_probs: Mapping[Scoreline | str, float]) -> dict[str, float]:
    """Aggregate scoreline probabilities into win/draw/loss probabilities."""
    team_a_win = 0.0
    draw = 0.0
    team_b_win = 0.0

    for scoreline, probability in score_probs.items():
        goals_a, goals_b = _parse_scoreline_key(scoreline)
        if goals_a > goals_b:
            team_a_win += probability
        elif goals_a < goals_b:
            team_b_win += probability
        else:
            draw += probability

    return {
        "team_a_win": team_a_win,
        "draw": draw,
        "team_b_win": team_b_win,
    }


def apply_draw_boost_to_outcomes(
    outcome_probs: Mapping[str, float],
    strength_diff: float,
    draw_boost_max: float = 0.0,
    draw_boost_scale: float = 300.0,
) -> dict[str, float]:
    """Apply tuned draw calibration while preserving a valid W/D/L distribution."""
    if draw_boost_max < 0:
        raise ValueError("draw_boost_max must be non-negative")
    if draw_boost_scale <= 0:
        raise ValueError("draw_boost_scale must be positive")

    team_a_win = float(outcome_probs["team_a_win"])
    draw = float(outcome_probs["draw"])
    team_b_win = float(outcome_probs["team_b_win"])
    draw_boost = draw_boost_max * math.exp(-abs(strength_diff) / draw_boost_scale)
    boosted_draw = min(0.45, draw + draw_boost)
    old_decisive = team_a_win + team_b_win
    new_decisive = 1.0 - boosted_draw

    if old_decisive <= 0:
        boosted = {
            "team_a_win": new_decisive / 2.0,
            "draw": boosted_draw,
            "team_b_win": new_decisive / 2.0,
        }
    else:
        boosted = {
            "team_a_win": team_a_win / old_decisive * new_decisive,
            "draw": boosted_draw,
            "team_b_win": team_b_win / old_decisive * new_decisive,
        }

    total = sum(boosted.values())
    if total <= 0:
        raise ValueError("Draw-calibrated probabilities must sum to a positive value")
    return {key: value / total for key, value in boosted.items()}


def most_likely_scoreline(
    score_probs: Mapping[Scoreline | str, float],
) -> tuple[Scoreline, float]:
    """Return the scoreline with the highest probability and its probability."""
    scoreline, probability = max(score_probs.items(), key=lambda item: item[1])
    return _parse_scoreline_key(scoreline), probability


def top_scorelines(
    score_probs: Mapping[Scoreline | str, float],
    limit: int = 5,
) -> list[dict[str, float | str]]:
    """Return the highest-probability scorelines."""
    rows = []
    for scoreline, probability in sorted(
        score_probs.items(), key=lambda item: item[1], reverse=True
    )[:limit]:
        goals_a, goals_b = _parse_scoreline_key(scoreline)
        rows.append({"scoreline": f"{goals_a}-{goals_b}", "probability": probability})
    return rows


def predict_from_expected_goals(
    lambda_a: float,
    lambda_b: float,
    team_a: str = "Team A",
    team_b: str = "Team B",
    max_goals: int = 8,
) -> dict:
    """Predict a match from manually supplied expected goals.

    This manual expected-goals entry point is kept for debugging and sanity
    checks. The complete model should usually call predict_match_from_strength
    or predict_match_from_ratings so expected goals are estimated from team
    strength ratings.
    """
    score_probs = scoreline_probabilities(lambda_a, lambda_b, max_goals=max_goals)
    outcome_probs = outcome_probabilities(score_probs)
    likely_scoreline, likely_probability = most_likely_scoreline(score_probs)
    top_five = top_scorelines(score_probs, limit=5)

    return {
        "team_a": team_a,
        "team_b": team_b,
        "lambda_a": lambda_a,
        "lambda_b": lambda_b,
        "p_team_a_win": outcome_probs["team_a_win"],
        "p_draw": outcome_probs["draw"],
        "p_team_b_win": outcome_probs["team_b_win"],
        "most_likely_scoreline": f"{likely_scoreline[0]}-{likely_scoreline[1]}",
        "most_likely_scoreline_probability": likely_probability,
        "top_5_scorelines": top_five,
        "scoreline_probabilities": {
            f"{goals_a}-{goals_b}": probability
            for (goals_a, goals_b), probability in score_probs.items()
        },
    }


def predict_match_from_expected_goals(
    team_a: str,
    team_b: str,
    lambda_a: float,
    lambda_b: float,
    max_goals: int = 8,
) -> dict:
    """Backward-compatible wrapper for manual expected-goals debugging."""
    return predict_from_expected_goals(
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        team_a=team_a,
        team_b=team_b,
        max_goals=max_goals,
    )


def predict_from_strength(
    team_a: str,
    team_b: str,
    strength_a: float,
    strength_b: float,
    advantage_a: float = 0,
    base_goals: float = 1.35,
    scale: float = 800,
    mapping_mode: str = "total_share",
    base_total_goals: float = 2.65,
    share_scale: float = 250,
    mismatch_total_bonus: float = 0.80,
    mismatch_scale: float = 600,
    lambda_min: float = 0.15,
    lambda_max: float = 4.50,
    draw_boost_max: float = 0.0,
    draw_boost_scale: float = 300.0,
    max_goals: int = 8,
) -> dict:
    """Predict scoreline and W/D/L probabilities from team strength ratings.

    Workflow:
    raw match result -> context-adjusted performance signal -> adjusted Elo
    rating -> expected goals -> Poisson scoreline distribution -> W/D/L
    probability.

    This function starts from the adjusted Elo strength step. It deliberately
    uses adjusted Elo rather than raw goals because Elo already summarizes
    opponent-adjusted performance, tournament context, recency, and reliability
    signals from the rating pipeline.
    """
    expected_goals = expected_goals_from_strength(
        strength_a=strength_a,
        strength_b=strength_b,
        advantage_a=advantage_a,
        mapping_mode=mapping_mode,
        base_total_goals=base_total_goals,
        share_scale=share_scale,
        mismatch_total_bonus=mismatch_total_bonus,
        mismatch_scale=mismatch_scale,
        base_goals=base_goals,
        scale=scale,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
    )
    prediction = predict_from_expected_goals(
        lambda_a=expected_goals["lambda_a"],
        lambda_b=expected_goals["lambda_b"],
        team_a=team_a,
        team_b=team_b,
        max_goals=max_goals,
    )
    if draw_boost_max:
        boosted_outcomes = apply_draw_boost_to_outcomes(
            {
                "team_a_win": prediction["p_team_a_win"],
                "draw": prediction["p_draw"],
                "team_b_win": prediction["p_team_b_win"],
            },
            strength_diff=expected_goals["strength_diff"],
            draw_boost_max=draw_boost_max,
            draw_boost_scale=draw_boost_scale,
        )
        prediction["p_team_a_win"] = boosted_outcomes["team_a_win"]
        prediction["p_draw"] = boosted_outcomes["draw"]
        prediction["p_team_b_win"] = boosted_outcomes["team_b_win"]
    prediction.update(expected_goals)
    prediction["draw_boost_max"] = draw_boost_max
    prediction["draw_boost_scale"] = draw_boost_scale
    return prediction


def predict_match_from_strength(
    team_a: str,
    team_b: str,
    strength_a: float,
    strength_b: float,
    advantage_a: float = 0,
    base_goals: float = 1.35,
    scale: float = 800,
    mapping_mode: str = "total_share",
    base_total_goals: float = 2.65,
    share_scale: float = 250,
    mismatch_total_bonus: float = 0.80,
    mismatch_scale: float = 600,
    lambda_min: float = 0.15,
    lambda_max: float = 4.50,
    draw_boost_max: float = 0.0,
    draw_boost_scale: float = 300.0,
    max_goals: int = 8,
) -> dict:
    """Backward-compatible wrapper for strength-based match prediction."""
    return predict_from_strength(
        team_a=team_a,
        team_b=team_b,
        strength_a=strength_a,
        strength_b=strength_b,
        advantage_a=advantage_a,
        base_goals=base_goals,
        scale=scale,
        mapping_mode=mapping_mode,
        base_total_goals=base_total_goals,
        share_scale=share_scale,
        mismatch_total_bonus=mismatch_total_bonus,
        mismatch_scale=mismatch_scale,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        draw_boost_max=draw_boost_max,
        draw_boost_scale=draw_boost_scale,
        max_goals=max_goals,
    )


def _lookup_rating(team_name: str, ratings_df: pd.DataFrame, rating_col: str) -> float:
    required_columns = {"team_name", rating_col}
    missing_columns = required_columns - set(ratings_df.columns)
    if missing_columns:
        raise ValueError(f"ratings_df is missing columns: {sorted(missing_columns)}")

    matches = ratings_df.loc[ratings_df["team_name"] == team_name, rating_col]
    if matches.empty:
        available = ", ".join(ratings_df["team_name"].dropna().head(8).astype(str))
        raise ValueError(
            f"Team '{team_name}' was not found in ratings_df. "
            f"Available examples: {available}"
        )

    return float(matches.iloc[0])


def predict_from_ratings(
    team_a: str,
    team_b: str,
    ratings_df: pd.DataFrame,
    rating_col: str = "final_rating",
    advantage_a: float = 0,
    base_goals: float = 1.35,
    scale: float = 800,
    mapping_mode: str = "total_share",
    base_total_goals: float = 2.65,
    share_scale: float = 250,
    mismatch_total_bonus: float = 0.80,
    mismatch_scale: float = 600,
    lambda_min: float = 0.15,
    lambda_max: float = 4.50,
    draw_boost_max: float = 0.0,
    draw_boost_scale: float = 300.0,
    max_goals: int = 8,
) -> dict:
    """Look up team ratings in a DataFrame and predict the match.

    ratings_df must contain team_name and rating_col. A clear ValueError is
    raised if either team is absent, which keeps data issues visible instead of
    silently using a default rating.
    """
    strength_a = _lookup_rating(team_a, ratings_df, rating_col)
    strength_b = _lookup_rating(team_b, ratings_df, rating_col)
    return predict_from_strength(
        team_a=team_a,
        team_b=team_b,
        strength_a=strength_a,
        strength_b=strength_b,
        advantage_a=advantage_a,
        base_goals=base_goals,
        scale=scale,
        mapping_mode=mapping_mode,
        base_total_goals=base_total_goals,
        share_scale=share_scale,
        mismatch_total_bonus=mismatch_total_bonus,
        mismatch_scale=mismatch_scale,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        draw_boost_max=draw_boost_max,
        draw_boost_scale=draw_boost_scale,
        max_goals=max_goals,
    )


def predict_match_from_ratings(
    team_a: str,
    team_b: str,
    ratings_df: pd.DataFrame,
    rating_col: str = "final_rating",
    advantage_a: float = 0,
    base_goals: float = 1.35,
    scale: float = 800,
    mapping_mode: str = "total_share",
    base_total_goals: float = 2.65,
    share_scale: float = 250,
    mismatch_total_bonus: float = 0.80,
    mismatch_scale: float = 600,
    lambda_min: float = 0.15,
    lambda_max: float = 4.50,
    draw_boost_max: float = 0.0,
    draw_boost_scale: float = 300.0,
    max_goals: int = 8,
) -> dict:
    """Backward-compatible wrapper for ratings-table prediction."""
    return predict_from_ratings(
        team_a=team_a,
        team_b=team_b,
        ratings_df=ratings_df,
        rating_col=rating_col,
        advantage_a=advantage_a,
        base_goals=base_goals,
        scale=scale,
        mapping_mode=mapping_mode,
        base_total_goals=base_total_goals,
        share_scale=share_scale,
        mismatch_total_bonus=mismatch_total_bonus,
        mismatch_scale=mismatch_scale,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        draw_boost_max=draw_boost_max,
        draw_boost_scale=draw_boost_scale,
        max_goals=max_goals,
    )


def predict_match_poisson(
    team_a: str,
    team_b: str,
    strength_a: float,
    strength_b: float,
    base_goals: float = 1.35,
    scale: float = 800,
    advantage_a: float = 0,
    mapping_mode: str = "total_share",
    base_total_goals: float = 2.65,
    share_scale: float = 250,
    mismatch_total_bonus: float = 0.80,
    mismatch_scale: float = 600,
    lambda_min: float = 0.15,
    lambda_max: float = 4.50,
    draw_boost_max: float = 0.0,
    draw_boost_scale: float = 300.0,
    max_goals: int = 8,
) -> dict:
    """Backward-compatible wrapper for strength-based match prediction."""
    return predict_from_strength(
        team_a=team_a,
        team_b=team_b,
        strength_a=strength_a,
        strength_b=strength_b,
        advantage_a=advantage_a,
        base_goals=base_goals,
        scale=scale,
        mapping_mode=mapping_mode,
        base_total_goals=base_total_goals,
        share_scale=share_scale,
        mismatch_total_bonus=mismatch_total_bonus,
        mismatch_scale=mismatch_scale,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        draw_boost_max=draw_boost_max,
        draw_boost_scale=draw_boost_scale,
        max_goals=max_goals,
    )


def _prompt_text(prompt: str, default: str) -> str:
    value = input(f"{prompt} [{default}]: ").strip()
    return value or default


def _prompt_float(prompt: str, default: float) -> float:
    value = input(f"{prompt} [{default}]: ").strip()
    return float(value) if value else default


def _load_default_ratings(mode: str = "default") -> pd.DataFrame:
    config = load_model_config(mode)
    configured_ratings_path = config.get("rating_source_path")
    ratings_path = (
        PROJECT_ROOT / configured_ratings_path
        if configured_ratings_path
        else DEFAULT_RATINGS_PATH
    )
    if not ratings_path.exists() and not DEFAULT_RATINGS_PATH.exists() and not CALIBRATED_RATINGS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find ratings file: {ratings_path}. "
            "Run src/ratings/build_elo_ratings.py or src/ratings/build_player_impacted_strength.py first."
        )

    if not ratings_path.exists():
        ratings_path = DEFAULT_RATINGS_PATH
    ratings_df = pd.read_csv(ratings_path)
    if DEFAULT_RATING_COL not in ratings_df.columns and CALIBRATED_RATINGS_PATH.exists():
        calibrated_df = pd.read_csv(CALIBRATED_RATINGS_PATH)
        if DEFAULT_RATING_COL in calibrated_df.columns:
            ratings_path = CALIBRATED_RATINGS_PATH
            ratings_df = calibrated_df

    configured_rating_col = config.get("rating_col", DEFAULT_RATING_COL)
    rating_col = configured_rating_col if configured_rating_col in ratings_df.columns else FALLBACK_RATING_COL
    required_columns = {"team_name", rating_col}
    missing_columns = required_columns - set(ratings_df.columns)
    if missing_columns:
        raise ValueError(
            f"{ratings_path} is missing columns: {sorted(missing_columns)}"
        )
    ratings_df.attrs["rating_col"] = rating_col
    ratings_df.attrs["ratings_path"] = str(ratings_path)
    return ratings_df


def _load_default_base_goals() -> float:
    if not DEFAULT_MATCHES_PATH.exists():
        raise FileNotFoundError(
            f"Could not find cleaned matches file: {DEFAULT_MATCHES_PATH}. "
            "Run src/data/clean_results.py first."
        )

    matches_df = pd.read_csv(DEFAULT_MATCHES_PATH)
    return estimate_base_goals(matches_df)


def _load_default_base_total_goals() -> float:
    return float(load_model_config()["base_total_goals"])


def compare_prediction_sensitivity(
    ratings_df: pd.DataFrame,
    rating_col: str = DEFAULT_RATING_COL,
    matchups: list[tuple[str, str]] | None = None,
    base_total_goals: float = 2.65,
    share_scale: float = 250,
    mismatch_total_bonus: float = 0.80,
    mismatch_scale: float = 600,
) -> pd.DataFrame:
    """Compare xG and W/D/L sensitivity for representative matchups."""
    if matchups is None:
        matchups = [
            ("Spain", "France"),
            ("France", "South Africa"),
            ("Germany", "Curacao"),
        ]

    rows = []
    for team_a, team_b in matchups:
        prediction = predict_from_ratings(
            team_a=team_a,
            team_b=team_b,
            ratings_df=ratings_df,
            rating_col=rating_col,
            base_total_goals=base_total_goals,
            share_scale=share_scale,
            mismatch_total_bonus=mismatch_total_bonus,
            mismatch_scale=mismatch_scale,
        )
        rows.append(
            {
                "team_a": team_a,
                "team_b": team_b,
                "strength_a": prediction["strength_a"],
                "strength_b": prediction["strength_b"],
                "strength_diff": prediction["strength_diff"],
                "lambda_a": prediction["lambda_a"],
                "lambda_b": prediction["lambda_b"],
                "total_expected_goals": prediction["total_expected_goals"],
                "p_team_a_win": prediction["p_team_a_win"],
                "p_draw": prediction["p_draw"],
                "p_team_b_win": prediction["p_team_b_win"],
                "top_5_scorelines": prediction["top_5_scorelines"],
            }
        )
    return pd.DataFrame(rows)


def _print_top_scorelines(score_probs: Mapping[str, float], limit: int = 10) -> None:
    print(f"\nTop {limit} scorelines")
    for scoreline, probability in sorted(
        score_probs.items(), key=lambda item: item[1], reverse=True
    )[:limit]:
        print(f"  {scoreline}: {probability:.2%}")


def main() -> None:
    """Run a small interactive prediction demo from PyCharm or terminal."""
    parser = argparse.ArgumentParser(description="Interactive CupCast match prediction demo.")
    parser.add_argument(
        "--mode",
        default="default",
        choices=["default", "v2", "experimental", "test"],
        help="Model parameter mode.",
    )
    args = parser.parse_args()

    print("Poisson match prediction demo")
    print("Press Enter to keep the default values.")
    print("Enter team names; ratings and base goals are loaded automatically.")

    ratings_df = _load_default_ratings(args.mode)
    config = load_model_config(args.mode)
    rating_col = ratings_df.attrs["rating_col"]
    model_kwargs = {
        **poisson_parameter_kwargs(config),
        **draw_calibration_kwargs(config),
    }
    print(f"Ratings file: {ratings_df.attrs['ratings_path']}")
    print(f"Rating column: {rating_col}")
    print(f"Parameter config: {config['parameter_config_path']}")
    print(f"Matches file: {DEFAULT_MATCHES_PATH}\n")

    team_a = _prompt_text("Team A", "Spain")
    team_b = _prompt_text("Team B", "France")

    prediction = predict_from_ratings(
        team_a=team_a,
        team_b=team_b,
        ratings_df=ratings_df,
        rating_col=rating_col,
        **model_kwargs,
    )
    summary = {k: v for k, v in prediction.items() if k != "scoreline_probabilities"}

    print("\nPrediction summary")
    print(json.dumps(summary, indent=2))
    _print_top_scorelines(prediction["scoreline_probabilities"])


if __name__ == "__main__":
    main()
