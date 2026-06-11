from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUPERSTAR_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "superstar_features.csv"
PROBABILITY_TOLERANCE = 1e-9


def load_superstar_features(path: Path = DEFAULT_SUPERSTAR_FEATURES_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def _clip_and_normalize(probabilities: dict[str, float]) -> dict[str, float]:
    clipped = {key: min(1.0, max(0.0, float(value))) for key, value in probabilities.items()}
    total = sum(clipped.values())
    if total <= 0:
        raise ValueError("Superstar-adjusted probabilities must sum to a positive value")
    return {key: value / total for key, value in clipped.items()}


def _lookup_features(superstar_features_df: pd.DataFrame | dict[str, dict]) -> dict[str, dict]:
    if isinstance(superstar_features_df, dict):
        return superstar_features_df
    required = {"team", "superstar_score", "top_player_name"}
    missing = required - set(superstar_features_df.columns)
    if missing:
        raise ValueError(f"superstar_features_df is missing columns: {sorted(missing)}")
    rows = {}
    for _, row in superstar_features_df.iterrows():
        score = pd.to_numeric(row["superstar_score"], errors="coerce")
        score = 0.0 if pd.isna(score) else float(score)
        rows[str(row["team"])] = {
            "superstar_score": score,
            "top_player_name": "" if pd.isna(row["top_player_name"]) else str(row["top_player_name"]),
        }
    return rows


def adjust_probabilities_for_superstar_impact(
    p_team_a_win: float,
    p_draw: float,
    p_team_b_win: float,
    team_a: str,
    team_b: str,
    superstar_features_df: pd.DataFrame,
    max_star_shift: float = 0.040,
    star_weight: float = 0.012,
) -> dict[str, float | bool | str]:
    """Apply a small post-V1/V2 probability nudge for exceptional top players."""
    if max_star_shift < 0 or star_weight < 0:
        raise ValueError("Superstar adjustment parameters must be non-negative")

    lookup = _lookup_features(superstar_features_df)
    missing = team_a not in lookup or team_b not in lookup
    a_features = lookup.get(team_a, {"superstar_score": 0.0, "top_player_name": ""})
    b_features = lookup.get(team_b, {"superstar_score": 0.0, "top_player_name": ""})
    superstar_a = float(a_features["superstar_score"])
    superstar_b = float(b_features["superstar_score"])
    star_edge = superstar_a - superstar_b

    adjusted = {
        "team_a_win": float(p_team_a_win),
        "draw": float(p_draw),
        "team_b_win": float(p_team_b_win),
    }
    shift = 0.0
    reason = "no_superstar_edge"

    if abs(star_edge) >= 0.25:
        shift = min(max_star_shift, star_weight * abs(star_edge))
        if star_edge > 0:
            adjusted["team_a_win"] += shift
            adjusted["draw"] -= shift * 0.55
            adjusted["team_b_win"] -= shift * 0.45
            reason = "team_a_superstar_edge"
        else:
            adjusted["team_b_win"] += shift
            adjusted["draw"] -= shift * 0.55
            adjusted["team_a_win"] -= shift * 0.45
            reason = "team_b_superstar_edge"

    adjusted = _clip_and_normalize(adjusted)
    probability_sum = adjusted["team_a_win"] + adjusted["draw"] + adjusted["team_b_win"]
    if abs(probability_sum - 1.0) > PROBABILITY_TOLERANCE:
        raise ValueError("Superstar-adjusted probabilities must sum to 1")

    return {
        "star_adjusted_p_team_a_win": adjusted["team_a_win"],
        "star_adjusted_p_draw": adjusted["draw"],
        "star_adjusted_p_team_b_win": adjusted["team_b_win"],
        "superstar_shift": shift,
        "superstar_edge": star_edge,
        "superstar_adjustment_reason": reason,
        "superstar_a": superstar_a,
        "superstar_b": superstar_b,
        "top_player_a": a_features["top_player_name"],
        "top_player_b": b_features["top_player_name"],
        "missing_superstar_feature_flag": missing,
    }
