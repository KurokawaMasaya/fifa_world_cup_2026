from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLUB_FORM_FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "team_club_form_features.csv"
)
PROBABILITY_TOLERANCE = 1e-9


def load_team_club_form_features(
    path: Path = DEFAULT_CLUB_FORM_FEATURES_PATH,
) -> pd.DataFrame:
    return pd.read_csv(path)


def _clip_and_normalize(probabilities: dict[str, float]) -> dict[str, float]:
    clipped = {key: min(1.0, max(0.0, float(value))) for key, value in probabilities.items()}
    total = sum(clipped.values())
    if total <= 0:
        raise ValueError("Club-form-adjusted probabilities must sum to a positive value")
    return {key: value / total for key, value in clipped.items()}


def _lookup_features(team_club_form_features_df: pd.DataFrame | dict[str, dict]) -> dict[str, dict]:
    if isinstance(team_club_form_features_df, dict):
        return team_club_form_features_df
    required = {"team", "club_form_signal", "club_form_data_coverage"}
    missing = required - set(team_club_form_features_df.columns)
    if missing:
        raise ValueError(f"team_club_form_features_df is missing columns: {sorted(missing)}")

    rows: dict[str, dict] = {}
    for _, row in team_club_form_features_df.iterrows():
        signal = pd.to_numeric(row["club_form_signal"], errors="coerce")
        coverage = pd.to_numeric(row["club_form_data_coverage"], errors="coerce")
        rows[str(row["team"])] = {
            "club_form_signal": 0.0 if pd.isna(signal) else float(signal),
            "club_form_data_coverage": 0.0 if pd.isna(coverage) else float(coverage),
        }
    return rows


def adjust_probabilities_for_club_form(
    p_team_a_win: float,
    p_draw: float,
    p_team_b_win: float,
    team_a: str,
    team_b: str,
    team_club_form_features_df: pd.DataFrame,
    max_club_form_shift: float = 0.025,
    club_form_weight: float = 0.008,
) -> dict[str, float | bool | str]:
    """Apply a small V2 probability nudge for recent club form.

    Club form is a short-window activity and attacking-output signal for squad
    players. It deliberately adjusts only W/D/L probabilities; it does not
    change national-team strength ratings.
    """
    if max_club_form_shift < 0 or club_form_weight < 0:
        raise ValueError("Club form adjustment parameters must be non-negative")

    lookup = _lookup_features(team_club_form_features_df)
    missing = team_a not in lookup or team_b not in lookup
    a_features = lookup.get(
        team_a, {"club_form_signal": 0.0, "club_form_data_coverage": 0.0}
    )
    b_features = lookup.get(
        team_b, {"club_form_signal": 0.0, "club_form_data_coverage": 0.0}
    )
    score_a = float(a_features["club_form_signal"])
    score_b = float(b_features["club_form_signal"])
    club_form_edge = score_a - score_b

    adjusted = {
        "team_a_win": float(p_team_a_win),
        "draw": float(p_draw),
        "team_b_win": float(p_team_b_win),
    }
    shift = 0.0
    reason = "no_club_form_edge"

    if abs(club_form_edge) >= 0.25:
        shift = min(max_club_form_shift, club_form_weight * abs(club_form_edge))
        if club_form_edge > 0:
            adjusted["team_a_win"] += shift
            adjusted["draw"] -= shift * 0.50
            adjusted["team_b_win"] -= shift * 0.50
            reason = "team_a_club_form_edge"
        else:
            adjusted["team_b_win"] += shift
            adjusted["draw"] -= shift * 0.50
            adjusted["team_a_win"] -= shift * 0.50
            reason = "team_b_club_form_edge"

    adjusted = _clip_and_normalize(adjusted)
    probability_sum = adjusted["team_a_win"] + adjusted["draw"] + adjusted["team_b_win"]
    if abs(probability_sum - 1.0) > PROBABILITY_TOLERANCE:
        raise ValueError("Club-form-adjusted probabilities must sum to 1")

    return {
        "club_form_adjusted_p_team_a_win": adjusted["team_a_win"],
        "club_form_adjusted_p_draw": adjusted["draw"],
        "club_form_adjusted_p_team_b_win": adjusted["team_b_win"],
        "club_form_shift": shift,
        "club_form_edge": club_form_edge,
        "club_form_adjustment_reason": reason,
        "club_form_score_a": score_a,
        "club_form_score_b": score_b,
        "club_form_data_coverage_a": float(a_features["club_form_data_coverage"]),
        "club_form_data_coverage_b": float(b_features["club_form_data_coverage"]),
        "missing_club_form_feature_flag": missing,
    }
