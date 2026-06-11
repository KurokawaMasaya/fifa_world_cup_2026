from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQUAD_VALUES_PATH = PROJECT_ROOT / "data" / "raw" / "squad_values.csv"
PROBABILITY_TOLERANCE = 1e-9


def _team_column(squad_values: pd.DataFrame) -> str:
    if "team" in squad_values.columns:
        return "team"
    if "team_name" in squad_values.columns:
        return "team_name"
    raise ValueError("squad_values must contain either 'team' or 'team_name'")


def _zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (numeric - numeric.mean()) / std


def build_squad_uncertainty_features(squad_values: pd.DataFrame) -> pd.DataFrame:
    """Build star-dependence features without changing team strength.

    Higher concentration means more of a squad's value is tied up in a small
    number of players. V2 uses that as an uncertainty signal, not as extra team
    strength, to avoid double-counting quality already captured by V1 ratings.
    """
    team_col = _team_column(squad_values)
    required = {team_col, "squad_market_value_eur", "top_5_value_eur"}
    missing = required - set(squad_values.columns)
    if missing:
        raise ValueError(f"squad_values is missing required columns: {sorted(missing)}")

    features = squad_values.copy()
    features["team_name"] = features[team_col].astype(str)
    features["squad_market_value_eur"] = pd.to_numeric(
        features["squad_market_value_eur"], errors="coerce"
    )
    features["top_5_value_eur"] = pd.to_numeric(features["top_5_value_eur"], errors="coerce")
    features["depth_concentration"] = (
        features["top_5_value_eur"] / features["squad_market_value_eur"]
    ).replace([float("inf"), -float("inf")], pd.NA)

    if "top_10_value_eur" in features.columns:
        features["top_10_value_eur"] = pd.to_numeric(
            features["top_10_value_eur"], errors="coerce"
        )
        features["top10_concentration"] = (
            features["top_10_value_eur"] / features["squad_market_value_eur"]
        ).replace([float("inf"), -float("inf")], pd.NA)
    else:
        features["top10_concentration"] = pd.NA

    features["volatility_score"] = _zscore(features["depth_concentration"]).fillna(0.0)
    return features[
        [
            "team_name",
            "squad_market_value_eur",
            "top_5_value_eur",
            "depth_concentration",
            "top10_concentration",
            "volatility_score",
        ]
    ].drop_duplicates("team_name", keep="first")


def load_squad_uncertainty_features(
    path: Path = DEFAULT_SQUAD_VALUES_PATH,
) -> pd.DataFrame:
    return build_squad_uncertainty_features(pd.read_csv(path))


def _feature_lookup(squad_features_df: pd.DataFrame | dict[str, float]) -> dict[str, float]:
    if isinstance(squad_features_df, dict):
        return squad_features_df
    if "volatility_score" not in squad_features_df.columns:
        squad_features_df = build_squad_uncertainty_features(squad_features_df)
    return dict(
        zip(
            squad_features_df["team_name"].astype(str),
            pd.to_numeric(squad_features_df["volatility_score"], errors="coerce").fillna(0.0),
        )
    )


def _clip_and_normalize(probabilities: dict[str, float]) -> dict[str, float]:
    clipped = {key: min(1.0, max(0.0, float(value))) for key, value in probabilities.items()}
    total = sum(clipped.values())
    if total <= 0:
        raise ValueError("Adjusted probabilities must sum to a positive value")
    return {key: value / total for key, value in clipped.items()}


def adjust_probabilities_for_uncertainty(
    p_team_a_win: float,
    p_draw: float,
    p_team_b_win: float,
    team_a: str,
    team_b: str,
    squad_features_df: pd.DataFrame,
    max_shift: float = 0.035,
    volatility_weight: float = 0.010,
    stability_edge_weight: float = 0.008,
) -> dict[str, float | bool | str]:
    """Post-process V1 probabilities using squad volatility features.

    V2 never changes team strength. It only moves a small amount of probability
    mass to reduce overconfidence when a favorite is star-dependent or when both
    teams are volatile.
    """
    if max_shift < 0 or volatility_weight < 0 or stability_edge_weight < 0:
        raise ValueError("V2 uncertainty weights must be non-negative")

    base = _clip_and_normalize(
        {
            "team_a_win": p_team_a_win,
            "draw": p_draw,
            "team_b_win": p_team_b_win,
        }
    )
    lookup = _feature_lookup(squad_features_df)
    missing = team_a not in lookup or team_b not in lookup
    volatility_a = float(lookup.get(team_a, 0.0))
    volatility_b = float(lookup.get(team_b, 0.0))
    avg_volatility = (volatility_a + volatility_b) / 2.0
    volatility_gap = volatility_a - volatility_b

    favorite = "team_a" if base["team_a_win"] >= base["team_b_win"] else "team_b"
    underdog_key = "team_b_win" if favorite == "team_a" else "team_a_win"
    favorite_key = "team_a_win" if favorite == "team_a" else "team_b_win"
    favorite_volatility = volatility_a if favorite == "team_a" else volatility_b
    underdog_volatility = volatility_b if favorite == "team_a" else volatility_a

    adjusted = dict(base)
    shift = 0.0
    reason = "neutral_uncertainty"

    both_volatile_shift = min(max_shift, volatility_weight * max(0.0, avg_volatility))
    if both_volatile_shift > 0:
        shift = both_volatile_shift
        adjusted[favorite_key] -= shift
        adjusted["draw"] += shift * 0.55
        adjusted[underdog_key] += shift * 0.45
        reason = "both_teams_volatile_shift_from_favorite"

    stability_edge = underdog_volatility - favorite_volatility
    instability_edge = favorite_volatility - underdog_volatility
    edge_shift = min(max_shift, stability_edge_weight * max(stability_edge, instability_edge, 0.0))
    if edge_shift > 0:
        if stability_edge > 0:
            shift = edge_shift
            draw_take = min(adjusted["draw"], shift * 0.55)
            underdog_take = min(adjusted[underdog_key], shift * 0.45)
            adjusted["draw"] -= draw_take
            adjusted[underdog_key] -= underdog_take
            adjusted[favorite_key] += draw_take + underdog_take
            reason = "favorite_more_stable_shift_to_favorite"
        elif instability_edge > 0:
            shift = edge_shift
            adjusted[favorite_key] -= shift
            adjusted["draw"] += shift * 0.55
            adjusted[underdog_key] += shift * 0.45
            reason = "favorite_more_volatile_shift_from_favorite"

    adjusted = _clip_and_normalize(adjusted)
    probability_sum = (
        adjusted["team_a_win"] + adjusted["draw"] + adjusted["team_b_win"]
    )
    if abs(probability_sum - 1.0) > PROBABILITY_TOLERANCE:
        raise ValueError("V2 adjusted probabilities must sum to 1")

    return {
        "adjusted_p_team_a_win": adjusted["team_a_win"],
        "adjusted_p_draw": adjusted["draw"],
        "adjusted_p_team_b_win": adjusted["team_b_win"],
        "uncertainty_shift": min(max_shift, abs(shift)),
        "adjustment_reason": reason,
        "volatility_a": volatility_a,
        "volatility_b": volatility_b,
        "volatility_gap": volatility_gap,
        "missing_squad_feature_flag": missing,
    }
