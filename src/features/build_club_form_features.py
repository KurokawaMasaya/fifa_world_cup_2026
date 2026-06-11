from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREFERRED_PLAYER_VALUES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "player_values_standardized.csv"
)
FALLBACK_PLAYER_VALUES_PATH = (
    PROJECT_ROOT / "output" / "diagnostics" / "player_values_standardized.csv"
)
APPEARANCES_PATH = PROJECT_ROOT / "data" / "raw" / "appearances.csv"
GAMES_PATH = PROJECT_ROOT / "data" / "raw" / "games.csv"
PLAYER_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "player_club_form_features.csv"
TEAM_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "team_club_form_features.csv"
TEAM_NAME_OVERRIDES = {
    "Bosnia And Herzegovina": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo",
    "Curaçao": "Curacao",
    "Côte D'Ivoire": "Cote d'Ivoire",
    "IR Iran": "IR Iran",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
    "USA": "USA",
}


def resolve_player_values_path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    if PREFERRED_PLAYER_VALUES_PATH.exists():
        return PREFERRED_PLAYER_VALUES_PATH
    if FALLBACK_PLAYER_VALUES_PATH.exists():
        return FALLBACK_PLAYER_VALUES_PATH
    raise FileNotFoundError(
        "Could not find player_values_standardized.csv in data/processed or output/diagnostics"
    )


def zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (numeric - numeric.mean()) / std


def position_group(position: str) -> str:
    value = str(position).upper()
    if value in {"FW", "MF", "DF", "GK"}:
        return value
    if "ATTACK" in value or "FORWARD" in value or "WINGER" in value:
        return "FW"
    if "MIDFIELD" in value:
        return "MF"
    if "DEF" in value or "BACK" in value:
        return "DF"
    if "GOAL" in value or "KEEPER" in value:
        return "GK"
    return value[:2] if value else "UNK"


def standardize_team_name(team_name: str) -> str:
    value = str(team_name).strip()
    return TEAM_NAME_OVERRIDES.get(value, value)


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    clean = pd.DataFrame({"value": values, "weight": weights}).dropna()
    clean = clean.loc[clean["weight"] > 0]
    if clean.empty:
        return float(pd.to_numeric(values, errors="coerce").mean())
    return float((clean["value"] * clean["weight"]).sum() / clean["weight"].sum())


def build_player_club_form_features(
    player_values_path: Path | None = None,
    appearances_path: Path = APPEARANCES_PATH,
    games_path: Path = GAMES_PATH,
    recent_window_days: int = 180,
    minimum_minutes_threshold: float = 2.0,
) -> pd.DataFrame:
    """Compute recent club form for World Cup squad players.

    Club form is treated as a short-window signal for player activity and
    attacking output. It is not a team-strength rating and is only intended for
    small V2 probability post-processing.
    """
    values_path = resolve_player_values_path(player_values_path)
    values = pd.read_csv(values_path)
    required_values = {"team", "player_name", "position", "player_id", "market_value_in_eur"}
    missing = required_values - set(values.columns)
    if missing:
        raise ValueError(f"{values_path} is missing required columns: {sorted(missing)}")

    squad = values.copy()
    squad["team"] = squad["team"].map(standardize_team_name)
    squad["tm_player_id"] = pd.to_numeric(squad["player_id"], errors="coerce")
    squad["market_value_eur"] = pd.to_numeric(
        squad["market_value_in_eur"], errors="coerce"
    ).fillna(0.0)
    squad["position_group"] = squad["position"].map(position_group)
    squad = squad.loc[squad["tm_player_id"].notna()].copy()
    squad["tm_player_id"] = squad["tm_player_id"].astype("int64")

    player_ids = set(squad["tm_player_id"])
    appearances = pd.read_csv(
        appearances_path,
        usecols=[
            "game_id",
            "player_id",
            "date",
            "goals",
            "assists",
            "minutes_played",
        ],
        parse_dates=["date"],
    )
    appearances = appearances.loc[appearances["player_id"].isin(player_ids)].copy()

    games = pd.read_csv(
        games_path,
        usecols=["game_id", "date", "competition_type"],
        parse_dates=["date"],
    )
    reference_date = games["date"].max()
    cutoff_date = reference_date - pd.Timedelta(days=recent_window_days)
    club_games = games.loc[
        games["competition_type"].fillna("").ne("national_team_competition"),
        ["game_id", "date"],
    ]
    recent_appearances = appearances.merge(
        club_games, on="game_id", how="inner", suffixes=("_appearance", "_game")
    )
    recent_appearances = recent_appearances.loc[
        recent_appearances["date_game"].between(cutoff_date, reference_date)
    ].copy()

    recent = (
        recent_appearances.groupby("player_id", as_index=False)
        .agg(
            recent_matches=("game_id", "nunique"),
            recent_minutes=("minutes_played", "sum"),
            recent_goals=("goals", "sum"),
            recent_assists=("assists", "sum"),
        )
        .rename(columns={"player_id": "tm_player_id"})
    )

    output = squad.merge(recent, on="tm_player_id", how="left")
    for column in ["recent_matches", "recent_minutes", "recent_goals", "recent_assists"]:
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)

    denominator_90s = (output["recent_minutes"] / 90.0).clip(lower=minimum_minutes_threshold)
    output["goals_per_90"] = output["recent_goals"] / denominator_90s
    output["assists_per_90"] = output["recent_assists"] / denominator_90s
    output["goal_contributions_per_90"] = (
        output["recent_goals"] + output["recent_assists"]
    ) / denominator_90s

    output["match_fitness_score"] = zscore(np.log1p(output["recent_minutes"]))
    output["attacking_form_score"] = zscore(output["goal_contributions_per_90"]).fillna(0.0)
    attacking_position = output["position_group"].isin(["FW", "MF"])
    output["club_form_score"] = 0.0
    output.loc[attacking_position, "club_form_score"] = (
        0.65 * output.loc[attacking_position, "match_fitness_score"]
        + 0.35 * output.loc[attacking_position, "attacking_form_score"]
    )
    output.loc[~attacking_position, "club_form_score"] = (
        0.90 * output.loc[~attacking_position, "match_fitness_score"]
        + 0.10 * output.loc[~attacking_position, "attacking_form_score"]
    )
    output["club_form_score"] = output["club_form_score"].clip(-2.5, 2.5).fillna(0.0)
    output["recent_activity_low"] = output["recent_minutes"] < 180
    output["club_form_reference_date"] = reference_date.date().isoformat()
    output["recent_window_days"] = recent_window_days

    columns = [
        "team",
        "player_name",
        "tm_player_id",
        "position",
        "position_group",
        "market_value_eur",
        "recent_matches",
        "recent_minutes",
        "recent_goals",
        "recent_assists",
        "goals_per_90",
        "assists_per_90",
        "goal_contributions_per_90",
        "match_fitness_score",
        "attacking_form_score",
        "club_form_score",
        "recent_activity_low",
        "club_form_reference_date",
        "recent_window_days",
    ]
    return output[columns]


def build_team_club_form_features(player_features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for team, group in player_features.groupby("team", sort=True):
        group = group.copy()
        group["market_value_eur"] = pd.to_numeric(
            group["market_value_eur"], errors="coerce"
        ).fillna(0.0)
        ranked = group.sort_values("market_value_eur", ascending=False)
        top5 = ranked.head(5)
        top10 = ranked.head(10)
        top11 = ranked.head(11)
        squad_size = len(group)
        coverage = float(group["recent_matches"].gt(0).sum() / squad_size) if squad_size else 0.0
        top_player = ranked.iloc[0] if not ranked.empty else None
        rows.append(
            {
                "team": team,
                "avg_club_form_score": float(group["club_form_score"].mean()),
                "value_weighted_club_form_score": weighted_average(
                    group["club_form_score"], group["market_value_eur"]
                ),
                "top5_value_weighted_club_form_score": weighted_average(
                    top5["club_form_score"], top5["market_value_eur"]
                ),
                "top11_value_weighted_club_form_score": weighted_average(
                    top11["club_form_score"], top11["market_value_eur"]
                ),
                "star_player_club_form_score": (
                    float(top_player["club_form_score"]) if top_player is not None else 0.0
                ),
                "inactive_high_value_players": int(top10["recent_activity_low"].sum()),
                "club_form_data_coverage": coverage,
                "low_club_form_data_coverage": coverage < 0.60,
                "club_form_signal": weighted_average(
                    top11["club_form_score"], top11["market_value_eur"]
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build World Cup squad club-form features.")
    parser.add_argument("--player-values", type=Path, default=None)
    parser.add_argument("--appearances", type=Path, default=APPEARANCES_PATH)
    parser.add_argument("--games", type=Path, default=GAMES_PATH)
    parser.add_argument("--recent-window-days", type=int, default=180)
    parser.add_argument("--minimum-minutes-threshold", type=float, default=2.0)
    parser.add_argument("--player-output", type=Path, default=PLAYER_OUTPUT_PATH)
    parser.add_argument("--team-output", type=Path, default=TEAM_OUTPUT_PATH)
    args = parser.parse_args()

    player_features = build_player_club_form_features(
        player_values_path=args.player_values,
        appearances_path=args.appearances,
        games_path=args.games,
        recent_window_days=args.recent_window_days,
        minimum_minutes_threshold=args.minimum_minutes_threshold,
    )
    team_features = build_team_club_form_features(player_features)
    args.player_output.parent.mkdir(parents=True, exist_ok=True)
    args.team_output.parent.mkdir(parents=True, exist_ok=True)
    player_features.to_csv(args.player_output, index=False)
    team_features.to_csv(args.team_output, index=False)

    print(f"Saved player club-form features to {args.player_output}")
    print(f"Saved team club-form features to {args.team_output}")
    print(f"Teams: {len(team_features)}")
    print("Lowest coverage teams:")
    print(
        team_features.sort_values("club_form_data_coverage")
        [["team", "club_form_data_coverage", "club_form_signal"]]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
