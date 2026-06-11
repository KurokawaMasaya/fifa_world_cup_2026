from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from src.models.poisson_match_model import outcome_probabilities
from src.models.v2_club_form_adjustment import (
    DEFAULT_CLUB_FORM_FEATURES_PATH,
    adjust_probabilities_for_club_form,
    load_team_club_form_features,
)
from src.models.v2_superstar_adjustment import (
    DEFAULT_SUPERSTAR_FEATURES_PATH,
    adjust_probabilities_for_superstar_impact,
    load_superstar_features,
)
from src.models.v2_uncertainty_adjustment import (
    DEFAULT_SQUAD_VALUES_PATH,
    adjust_probabilities_for_uncertainty,
    load_squad_uncertainty_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBABILITY_TOLERANCE = 1e-9


def is_v2_probability_stack_enabled(config: Mapping[str, object]) -> bool:
    """Return whether a config should apply V2 post-processing layers."""
    if bool(config.get("use_v2_probability_stack", False)):
        return True
    model_version = str(config.get("model_version", "")).lower()
    return model_version.startswith("v2")


def _resolve_path(path_value: object, fallback: Path) -> Path:
    if not path_value:
        return fallback
    path = Path(str(path_value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_v2_feature_context(config: Mapping[str, object]) -> dict[str, object]:
    """Load V2 feature tables once so simulations can reuse them efficiently."""
    squad_path = _resolve_path(
        config.get("squad_values_file") or config.get("squad_features_path"),
        DEFAULT_SQUAD_VALUES_PATH,
    )
    superstar_path = _resolve_path(
        config.get("superstar_features_file"),
        DEFAULT_SUPERSTAR_FEATURES_PATH,
    )
    club_form_path = _resolve_path(
        config.get("club_form_features_file"),
        DEFAULT_CLUB_FORM_FEATURES_PATH,
    )

    squad_features = load_squad_uncertainty_features(squad_path)
    superstar_features = load_superstar_features(superstar_path)
    club_form_features = load_team_club_form_features(club_form_path)
    return {
        "squad_features": dict(
            zip(
                squad_features["team_name"].astype(str),
                pd.to_numeric(squad_features["volatility_score"], errors="coerce").fillna(0.0),
            )
        ),
        "superstar_features": {
            str(row["team"]): {
                "superstar_score": float(row["superstar_score"]),
                "top_player_name": ""
                if pd.isna(row["top_player_name"])
                else str(row["top_player_name"]),
            }
            for _, row in superstar_features.iterrows()
        },
        "club_form_features": {
            str(row["team"]): {
                "club_form_signal": float(row["club_form_signal"]),
                "club_form_data_coverage": float(row["club_form_data_coverage"]),
            }
            for _, row in club_form_features.iterrows()
        },
    }


def v2_parameter_kwargs(config: Mapping[str, object]) -> dict[str, float]:
    return {
        "max_shift": float(config.get("max_shift", config.get("v2_uncertainty_max_shift", 0.015))),
        "volatility_weight": float(
            config.get("volatility_weight", config.get("v2_volatility_weight", 0.010))
        ),
        "stability_edge_weight": float(
            config.get(
                "stability_edge_weight",
                config.get("v2_stability_edge_weight", 0.008),
            )
        ),
        "max_star_shift": float(config.get("max_star_shift", 0.025)),
        "star_weight": float(config.get("star_weight", 0.006)),
        "max_club_form_shift": float(config.get("max_club_form_shift", 0.025)),
        "club_form_weight": float(config.get("club_form_weight", 0.008)),
    }


def apply_v2_probability_stack(
    team_a: str,
    team_b: str,
    p_team_a_win: float,
    p_draw: float,
    p_team_b_win: float,
    config: Mapping[str, object],
    feature_context: Mapping[str, object],
) -> dict[str, object]:
    """Apply V2 uncertainty, superstar, and club-form layers to W/D/L probabilities."""
    params = v2_parameter_kwargs(config)
    uncertainty = adjust_probabilities_for_uncertainty(
        p_team_a_win=p_team_a_win,
        p_draw=p_draw,
        p_team_b_win=p_team_b_win,
        team_a=team_a,
        team_b=team_b,
        squad_features_df=feature_context["squad_features"],
        max_shift=params["max_shift"],
        volatility_weight=params["volatility_weight"],
        stability_edge_weight=params["stability_edge_weight"],
    )
    superstar = adjust_probabilities_for_superstar_impact(
        p_team_a_win=uncertainty["adjusted_p_team_a_win"],
        p_draw=uncertainty["adjusted_p_draw"],
        p_team_b_win=uncertainty["adjusted_p_team_b_win"],
        team_a=team_a,
        team_b=team_b,
        superstar_features_df=feature_context["superstar_features"],
        max_star_shift=params["max_star_shift"],
        star_weight=params["star_weight"],
    )
    club_form = adjust_probabilities_for_club_form(
        p_team_a_win=superstar["star_adjusted_p_team_a_win"],
        p_draw=superstar["star_adjusted_p_draw"],
        p_team_b_win=superstar["star_adjusted_p_team_b_win"],
        team_a=team_a,
        team_b=team_b,
        team_club_form_features_df=feature_context["club_form_features"],
        max_club_form_shift=params["max_club_form_shift"],
        club_form_weight=params["club_form_weight"],
    )
    return {
        "v2_p_team_a_win": club_form["club_form_adjusted_p_team_a_win"],
        "v2_p_draw": club_form["club_form_adjusted_p_draw"],
        "v2_p_team_b_win": club_form["club_form_adjusted_p_team_b_win"],
        "uncertainty": uncertainty,
        "superstar": superstar,
        "club_form": club_form,
    }


def outcome_for_scoreline(scoreline: tuple[int, int] | str) -> str:
    if isinstance(scoreline, str):
        goals_a, goals_b = scoreline.split("-")
        goals_a = int(goals_a)
        goals_b = int(goals_b)
    else:
        goals_a, goals_b = scoreline
    if goals_a > goals_b:
        return "team_a_win"
    if goals_a < goals_b:
        return "team_b_win"
    return "draw"


def rescale_scoreline_probabilities_to_outcomes(
    scoreline_probabilities: Mapping[tuple[int, int] | str, float],
    target_outcomes: Mapping[str, float],
) -> dict[str, float]:
    """Rescale scoreline probabilities so their W/D/L totals equal target outcomes.

    V2 adjusts W/D/L probabilities rather than expected goals. For simulation,
    this preserves the Poisson scoreline shape within each outcome bucket while
    making sampled match outcomes follow the newest V2 probabilities.
    """
    raw_outcomes = outcome_probabilities(scoreline_probabilities)
    ratios = {}
    for outcome in ["team_a_win", "draw", "team_b_win"]:
        raw_probability = raw_outcomes[outcome]
        ratios[outcome] = (
            float(target_outcomes[outcome]) / raw_probability
            if raw_probability > 0
            else 0.0
        )

    adjusted = {}
    for scoreline, probability in scoreline_probabilities.items():
        key = scoreline if isinstance(scoreline, str) else f"{scoreline[0]}-{scoreline[1]}"
        adjusted[key] = float(probability) * ratios[outcome_for_scoreline(scoreline)]

    total = sum(adjusted.values())
    if total <= 0:
        raise ValueError("V2 scoreline probabilities must sum to a positive value")
    adjusted = {scoreline: probability / total for scoreline, probability in adjusted.items()}
    adjusted_outcomes = outcome_probabilities(adjusted)
    max_error = max(
        abs(adjusted_outcomes[outcome] - float(target_outcomes[outcome]))
        for outcome in ["team_a_win", "draw", "team_b_win"]
    )
    if max_error > PROBABILITY_TOLERANCE:
        raise ValueError("V2 scoreline rescaling failed to match target W/D/L probabilities")
    return adjusted
