from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREFERRED_PLAYER_VALUES_PATH = PROJECT_ROOT / "data" / "processed" / "player_values_standardized.csv"
FALLBACK_PLAYER_VALUES_PATH = PROJECT_ROOT / "output" / "diagnostics" / "player_values_standardized.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "superstar_features.csv"


def resolve_player_values_path(path: Path | None = None) -> Path:
    if path is not None:
        return path if path.is_absolute() else PROJECT_ROOT / path
    if PREFERRED_PLAYER_VALUES_PATH.exists():
        return PREFERRED_PLAYER_VALUES_PATH
    if FALLBACK_PLAYER_VALUES_PATH.exists():
        return FALLBACK_PLAYER_VALUES_PATH
    raise FileNotFoundError(
        f"Could not find {PREFERRED_PLAYER_VALUES_PATH} or {FALLBACK_PLAYER_VALUES_PATH}"
    )


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, pd.NA)
    return (numerator / denominator).replace([float("inf"), -float("inf")], pd.NA)


def _zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (numeric - numeric.mean()) / std


def _log_positive(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric > 0).map(lambda value: math.log(value) if pd.notna(value) else pd.NA)


def build_superstar_features(player_values: pd.DataFrame) -> pd.DataFrame:
    required = {"team_name", "player_name", "market_value_in_eur"}
    missing = required - set(player_values.columns)
    if missing:
        raise ValueError(f"Player values file is missing columns: {sorted(missing)}")

    players = player_values.copy()
    players["team"] = players["team_name"].astype(str)
    players["market_value_eur"] = pd.to_numeric(
        players["market_value_in_eur"], errors="coerce"
    )
    valid_values = players.loc[players["market_value_eur"].notna()].copy()
    if valid_values.empty:
        raise ValueError("No player market values available for superstar features")

    top_idx = valid_values.groupby("team")["market_value_eur"].idxmax()
    top_players = valid_values.loc[top_idx, ["team", "player_name", "market_value_eur"]].rename(
        columns={
            "player_name": "top_player_name",
            "market_value_eur": "top_player_value_eur",
        }
    )
    grouped = valid_values.groupby("team")["market_value_eur"]
    aggregates = grouped.agg(
        squad_market_value_eur="sum",
        median_player_value_eur="median",
        avg_player_value_eur="mean",
        max_player_value_eur="max",
    ).reset_index()
    top_5 = (
        valid_values.sort_values(["team", "market_value_eur"], ascending=[True, False])
        .groupby("team")
        .head(5)
        .groupby("team")["market_value_eur"]
        .sum()
        .reset_index(name="top_5_value_eur")
    )

    output = aggregates.merge(top_players, on="team", how="left").merge(top_5, on="team", how="left")
    output["top_player_value_eur"] = output["max_player_value_eur"]
    output["top_1_value_share"] = _safe_divide(
        output["top_player_value_eur"], output["squad_market_value_eur"]
    )
    output["top_1_to_team_median_ratio"] = _safe_divide(
        output["top_player_value_eur"], output["median_player_value_eur"]
    )
    output["top_1_to_team_average_ratio"] = _safe_divide(
        output["top_player_value_eur"], output["avg_player_value_eur"]
    )
    output["top_1_to_top_5_ratio"] = _safe_divide(
        output["top_player_value_eur"], output["top_5_value_eur"]
    )

    z_log_top1_value = _zscore(_log_positive(output["top_player_value_eur"]))
    z_top1_to_median = _zscore(_log_positive(output["top_1_to_team_median_ratio"]))
    z_top1_share = _zscore(output["top_1_value_share"])
    output["superstar_score"] = (
        0.50 * z_log_top1_value + 0.30 * z_top1_to_median + 0.20 * z_top1_share
    ).clip(-2.5, 2.5)
    output["superstar_flag"] = output["superstar_score"] >= 1.25

    columns = [
        "team",
        "max_player_value_eur",
        "top_player_name",
        "top_player_value_eur",
        "squad_market_value_eur",
        "median_player_value_eur",
        "avg_player_value_eur",
        "top_5_value_eur",
        "top_1_value_share",
        "top_1_to_team_median_ratio",
        "top_1_to_team_average_ratio",
        "top_1_to_top_5_ratio",
        "superstar_score",
        "superstar_flag",
    ]
    return output[columns].sort_values("superstar_score", ascending=False).reset_index(drop=True)


def print_report(features: pd.DataFrame, player_values: pd.DataFrame) -> None:
    missing_values = player_values.loc[
        pd.to_numeric(player_values.get("market_value_in_eur"), errors="coerce").isna()
    ]
    print("\nTop 15 teams by superstar_score")
    print(
        features.head(15)[
            ["team", "top_player_name", "top_player_value_eur", "superstar_score", "superstar_flag"]
        ].to_string(index=False)
    )
    print("\nBottom 10 teams by superstar_score")
    print(
        features.tail(10)[
            ["team", "top_player_name", "top_player_value_eur", "superstar_score", "superstar_flag"]
        ].to_string(index=False)
    )
    print("\nTeams with missing player values")
    if missing_values.empty:
        print("None")
    else:
        print(missing_values.groupby("team_name").size().sort_values(ascending=False).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build superstar features from player market values.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    input_path = resolve_player_values_path(args.input)
    player_values = pd.read_csv(input_path)
    features = build_superstar_features(player_values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    print(f"Loaded player values from {input_path}")
    print(f"Saved superstar features to {args.output}")
    print_report(features, player_values)


if __name__ == "__main__":
    main()
