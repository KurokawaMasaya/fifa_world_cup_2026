from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_PATH = PROCESSED_DIR / "team_ratings_world_cup_elo.csv"
OUTPUT_PATH = PROCESSED_DIR / "team_ratings_world_cup_elo_calibrated.csv"
RATING_COL = "world_cup_elo_rating"
CALIBRATED_COL = "calibrated_world_cup_elo_rating"

SELECTED_MATCHUPS = [
    ("Spain", "France"),
    ("Germany", "Curacao"),
    ("France", "South Africa"),
    ("Brazil", "Haiti"),
    ("Argentina", "New Zealand"),
]


def resolve_center(ratings: pd.Series, center: str) -> float:
    if center == "mean":
        return float(ratings.mean())
    return float(center)


def calibrate_ratings(
    ratings_df: pd.DataFrame,
    rating_col: str = RATING_COL,
    spread_factor: float = 1.20,
    center: str = "mean",
) -> pd.DataFrame:
    """Expand or contract rating spread without changing team order."""
    if rating_col not in ratings_df.columns:
        raise ValueError(f"Missing rating column: {rating_col}")

    calibration_center = resolve_center(ratings_df[rating_col], center)
    output = ratings_df.copy()
    output[CALIBRATED_COL] = (
        calibration_center
        + spread_factor * (output[rating_col] - calibration_center)
    ).round(3)
    output["spread_factor"] = spread_factor
    output["calibration_center"] = round(calibration_center, 3)
    return output


def rating_difference(
    ratings_df: pd.DataFrame,
    team_a: str,
    team_b: str,
    rating_col: str,
) -> float:
    ratings = ratings_df.set_index("team_name")[rating_col]
    missing = [team for team in [team_a, team_b] if team not in ratings.index]
    if missing:
        raise ValueError(f"Missing teams in ratings table: {missing}")
    return float(ratings.loc[team_a] - ratings.loc[team_b])


def ranking_order(ratings_df: pd.DataFrame, rating_col: str) -> list[str]:
    return ratings_df.sort_values(rating_col, ascending=False)["team_name"].tolist()


def print_rating_report(ratings_df: pd.DataFrame) -> None:
    ratings = ratings_df[RATING_COL]
    print("World Cup Elo rating spread")
    print("===========================")
    print(f"min:  {ratings.min():.3f}")
    print(f"max:  {ratings.max():.3f}")
    print(f"mean: {ratings.mean():.3f}")
    print(f"std:  {ratings.std(ddof=1):.3f}")

    print("\nTop 10 teams")
    print(
        ratings_df.sort_values(RATING_COL, ascending=False)
        .head(10)[["team_name", RATING_COL]]
        .to_string(index=False)
    )

    print("\nBottom 10 teams")
    print(
        ratings_df.sort_values(RATING_COL, ascending=True)
        .head(10)[["team_name", RATING_COL]]
        .to_string(index=False)
    )

    print("\nSelected rating differences")
    for team_a, team_b in SELECTED_MATCHUPS:
        diff = rating_difference(ratings_df, team_a, team_b, RATING_COL)
        print(f"{team_a} vs {team_b}: {diff:.3f}")


def print_calibration_checks(
    ratings_df: pd.DataFrame,
    calibrated_df: pd.DataFrame,
) -> None:
    print("\nCalibration checks")
    print("==================")
    germany_curacao_before = abs(
        rating_difference(ratings_df, "Germany", "Curacao", RATING_COL)
    )
    germany_curacao_after = abs(
        rating_difference(calibrated_df, "Germany", "Curacao", CALIBRATED_COL)
    )
    spain_france_before = abs(
        rating_difference(ratings_df, "Spain", "France", RATING_COL)
    )
    spain_france_after = abs(
        rating_difference(calibrated_df, "Spain", "France", CALIBRATED_COL)
    )
    rankings_unchanged = ranking_order(ratings_df, RATING_COL) == ranking_order(
        calibrated_df, CALIBRATED_COL
    )

    print(
        "Germany vs Curacao diff increased: "
        f"{germany_curacao_before:.3f} -> {germany_curacao_after:.3f}"
    )
    print(
        "Spain vs France diff remains moderate: "
        f"{spain_france_before:.3f} -> {spain_france_after:.3f}"
    )
    print(f"Rankings unchanged: {rankings_unchanged}")

    if germany_curacao_after <= germany_curacao_before:
        raise AssertionError("Germany vs Curacao diff did not increase")
    if spain_france_after > 120:
        raise AssertionError("Spain vs France diff is no longer moderate")
    if not rankings_unchanged:
        raise AssertionError("Calibration changed ranking order")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose and calibrate rating spread.")
    parser.add_argument("--spread-factor", type=float, default=1.20)
    parser.add_argument("--center", default="mean")
    args = parser.parse_args()

    ratings_df = pd.read_csv(INPUT_PATH)
    print_rating_report(ratings_df)

    calibrated_df = calibrate_ratings(
        ratings_df,
        spread_factor=args.spread_factor,
        center=args.center,
    )
    output_columns = [
        "team_name",
        RATING_COL,
        CALIBRATED_COL,
        "spread_factor",
        "calibration_center",
    ]
    calibrated_df[output_columns].to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote calibrated ratings to {OUTPUT_PATH}")
    print_calibration_checks(ratings_df, calibrated_df)


if __name__ == "__main__":
    main()
