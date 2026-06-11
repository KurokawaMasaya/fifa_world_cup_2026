from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.model_config import (  # noqa: E402
    draw_calibration_kwargs,
    load_model_config,
    poisson_parameter_kwargs,
)
from src.models.poisson_match_model import (  # noqa: E402
    DEFAULT_RATINGS_PATH,
    predict_from_ratings,
)
from src.models.v2_probability_stack import (  # noqa: E402
    apply_v2_probability_stack,
    is_v2_probability_stack_enabled,
    load_v2_feature_context,
)
from src.simulation.group_stage_simulator import load_group_stage_fixtures  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "model_params_default.json"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "output" / "predictions" / "group_stage_prediction_confidence.csv"
)
EXPECTED_GROUP_STAGE_MATCHES = 72
PROBABILITY_TOLERANCE = 1e-9
VALID_OUTCOMES = {"team_a_win", "draw", "team_b_win"}
VALID_COVERS = {
    "team_a_win",
    "draw",
    "team_b_win",
    "team_a_win_or_draw",
    "draw_or_team_b_win",
    "all_outcomes",
}
FINAL_COLUMNS = [
    "match_id",
    "group",
    "team_a",
    "team_b",
    "strength_a",
    "strength_b",
    "strength_diff",
    "lambda_a",
    "lambda_b",
    "p_team_a_win",
    "p_draw",
    "p_team_b_win",
    "main_pick",
    "main_pick_probability",
    "most_likely_scoreline",
    "most_likely_scoreline_probability",
    "scoreline_implied_result",
    "draw_watch",
    "strong_draw_watch",
    "volatile_match",
    "outcome_scoreline_disagreement",
    "suggested_cover",
    "risk_note",
    "model_version",
    "rating_col",
    "use_v2_probability_stack",
    "player_impact_layers",
    "squad_values_file",
    "superstar_features_file",
    "club_form_features_file",
]


def resolve_project_path(path_value: str | None, fallback: Path) -> Path:
    if not path_value:
        return fallback
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_ratings_from_config(config: dict) -> pd.DataFrame:
    ratings_path = resolve_project_path(config.get("rating_source_path"), DEFAULT_RATINGS_PATH)
    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Rating file not found: {ratings_path}. "
            "Run the appropriate rating builder before predicting group-stage results."
        )

    ratings = pd.read_csv(ratings_path)
    rating_col = config.get("rating_col")
    required_columns = {"team_name", rating_col}
    missing = required_columns - set(ratings.columns)
    if missing:
        raise ValueError(f"{ratings_path} is missing required columns: {sorted(missing)}")
    return ratings


def main_pick_from_probabilities(
    p_team_a_win: float,
    p_draw: float,
    p_team_b_win: float,
) -> tuple[str, float]:
    """Pick strictly from calibrated W/D/L probabilities, never scorelines."""
    probabilities = {
        "team_a_win": p_team_a_win,
        "draw": p_draw,
        "team_b_win": p_team_b_win,
    }
    main_pick = max(probabilities, key=probabilities.get)
    return main_pick, float(probabilities[main_pick])


def scoreline_implied_result(scoreline: str) -> str:
    goals_a, goals_b = [int(value) for value in scoreline.split("-")]
    if goals_a > goals_b:
        return "team_a_win"
    if goals_a < goals_b:
        return "team_b_win"
    return "draw"


def suggested_cover_for_match(
    main_pick: str,
    draw_watch: bool,
    volatile_match: bool,
) -> str:
    if volatile_match and draw_watch:
        return "all_outcomes"
    if main_pick == "draw":
        return "draw"
    if main_pick == "team_a_win" and draw_watch:
        return "team_a_win_or_draw"
    if main_pick == "team_b_win" and draw_watch:
        return "draw_or_team_b_win"
    return main_pick


def risk_note_for_match(
    team_a: str,
    team_b: str,
    main_pick: str,
    scoreline_result: str,
    draw_watch: bool,
    volatile_match: bool,
) -> str:
    if main_pick == "team_a_win" and scoreline_result == "draw":
        return f"Outcome pick favors {team_a}, but most likely scoreline is draw."
    if main_pick == "team_b_win" and scoreline_result == "draw":
        return f"Outcome pick favors {team_b}, but most likely scoreline is draw."
    if volatile_match:
        return "Low confidence; all three outcomes are close."
    if draw_watch:
        return "High draw probability; treat this as a draw-watch match."
    return "Clear favorite; low draw risk."


def validate_fixtures(fixtures: pd.DataFrame) -> None:
    if len(fixtures) != EXPECTED_GROUP_STAGE_MATCHES:
        raise ValueError(
            f"Expected {EXPECTED_GROUP_STAGE_MATCHES} group-stage matches, found {len(fixtures)}"
        )
    if fixtures[["home_team", "away_team", "group_letter"]].isna().any().any():
        raise ValueError("Group-stage fixtures contain missing teams or group labels")
    if not fixtures["match_label"].str.startswith("Group ").all():
        raise ValueError("Non-group-stage match found in group-stage fixture set")


def validate_predictions(predictions: pd.DataFrame, fixtures: pd.DataFrame, ratings: pd.DataFrame, rating_col: str) -> None:
    if len(predictions) != EXPECTED_GROUP_STAGE_MATCHES:
        raise ValueError(
            f"Expected {EXPECTED_GROUP_STAGE_MATCHES} predictions, found {len(predictions)}"
        )
    if predictions[["team_a", "team_b"]].isna().any().any():
        raise ValueError("Predictions contain missing teams")

    rating_lookup = set(ratings.loc[ratings[rating_col].notna(), "team_name"])
    missing_ratings = sorted(
        (set(predictions["team_a"]) | set(predictions["team_b"])) - rating_lookup
    )
    if missing_ratings:
        raise ValueError(f"Missing ratings for teams: {missing_ratings}")

    probability_sum = (
        predictions["p_team_a_win"] + predictions["p_draw"] + predictions["p_team_b_win"]
    )
    bad_probability_rows = predictions.loc[
        (probability_sum - 1.0).abs() > PROBABILITY_TOLERANCE,
        ["match_id", "team_a", "team_b"],
    ]
    if not bad_probability_rows.empty:
        raise ValueError(
            "Probabilities do not sum to 1 for matches: "
            f"{bad_probability_rows.to_dict(orient='records')}"
        )

    if set(predictions["match_id"]) != set(fixtures["id"]):
        raise ValueError("Prediction match IDs do not match the group-stage fixture IDs")

    invalid_picks = set(predictions["main_pick"]) - VALID_OUTCOMES
    if invalid_picks:
        raise ValueError(f"Invalid main_pick values: {sorted(invalid_picks)}")

    invalid_scoreline_results = set(predictions["scoreline_implied_result"]) - VALID_OUTCOMES
    if invalid_scoreline_results:
        raise ValueError(
            f"Invalid scoreline_implied_result values: {sorted(invalid_scoreline_results)}"
        )

    invalid_covers = set(predictions["suggested_cover"]) - VALID_COVERS
    if invalid_covers:
        raise ValueError(f"Invalid suggested_cover values: {sorted(invalid_covers)}")


def predict_group_stage_results(config: dict) -> pd.DataFrame:
    fixtures = load_group_stage_fixtures()
    validate_fixtures(fixtures)

    ratings = load_ratings_from_config(config)
    rating_col = config["rating_col"]
    poisson_kwargs = poisson_parameter_kwargs(config)
    draw_kwargs = draw_calibration_kwargs(config)
    calibrated_model_kwargs = {**poisson_kwargs, **draw_kwargs}
    use_v2_stack = is_v2_probability_stack_enabled(config)
    v2_feature_context = load_v2_feature_context(config) if use_v2_stack else None

    rows = []
    for _, fixture in fixtures.iterrows():
        calibrated_prediction = predict_from_ratings(
            team_a=fixture["home_team"],
            team_b=fixture["away_team"],
            ratings_df=ratings,
            rating_col=rating_col,
            **calibrated_model_kwargs,
        )
        p_team_a_win = calibrated_prediction["p_team_a_win"]
        p_draw = calibrated_prediction["p_draw"]
        p_team_b_win = calibrated_prediction["p_team_b_win"]
        if use_v2_stack:
            v2_adjustment = apply_v2_probability_stack(
                team_a=fixture["home_team"],
                team_b=fixture["away_team"],
                p_team_a_win=p_team_a_win,
                p_draw=p_draw,
                p_team_b_win=p_team_b_win,
                config=config,
                feature_context=v2_feature_context,
            )
            p_team_a_win = v2_adjustment["v2_p_team_a_win"]
            p_draw = v2_adjustment["v2_p_draw"]
            p_team_b_win = v2_adjustment["v2_p_team_b_win"]
        main_pick, main_pick_probability = main_pick_from_probabilities(
            p_team_a_win=p_team_a_win,
            p_draw=p_draw,
            p_team_b_win=p_team_b_win,
        )
        scoreline_result = scoreline_implied_result(
            calibrated_prediction["most_likely_scoreline"]
        )
        draw_watch = p_draw >= 0.26
        strong_draw_watch = p_draw >= 0.29
        volatile_match = main_pick_probability < 0.45
        suggested_cover = suggested_cover_for_match(
            main_pick=main_pick,
            draw_watch=draw_watch,
            volatile_match=volatile_match,
        )
        risk_note = risk_note_for_match(
            team_a=fixture["home_team"],
            team_b=fixture["away_team"],
            main_pick=main_pick,
            scoreline_result=scoreline_result,
            draw_watch=draw_watch,
            volatile_match=volatile_match,
        )
        rows.append(
            {
                "match_id": int(fixture["id"]),
                "group": fixture["group_letter"],
                "team_a": fixture["home_team"],
                "team_b": fixture["away_team"],
                "strength_a": calibrated_prediction["strength_a"],
                "strength_b": calibrated_prediction["strength_b"],
                "strength_diff": calibrated_prediction["strength_diff"],
                "lambda_a": calibrated_prediction["lambda_a"],
                "lambda_b": calibrated_prediction["lambda_b"],
                "p_team_a_win": p_team_a_win,
                "p_draw": p_draw,
                "p_team_b_win": p_team_b_win,
                "main_pick": main_pick,
                "main_pick_probability": main_pick_probability,
                "most_likely_scoreline": calibrated_prediction["most_likely_scoreline"],
                "most_likely_scoreline_probability": calibrated_prediction[
                    "most_likely_scoreline_probability"
                ],
                "scoreline_implied_result": scoreline_result,
                "draw_watch": draw_watch,
                "strong_draw_watch": strong_draw_watch,
                "volatile_match": volatile_match,
                "outcome_scoreline_disagreement": main_pick != scoreline_result,
                "suggested_cover": suggested_cover,
                "risk_note": risk_note,
                "model_version": config.get("model_version"),
                "rating_col": rating_col,
                "use_v2_probability_stack": use_v2_stack,
                "player_impact_layers": str(config.get("player_impact_layers", [])),
                "squad_values_file": config.get("squad_values_file"),
                "superstar_features_file": config.get("superstar_features_file"),
                "club_form_features_file": config.get("club_form_features_file"),
            }
        )

    output = pd.DataFrame(rows)[FINAL_COLUMNS]
    validate_predictions(output, fixtures=fixtures, ratings=ratings, rating_col=rating_col)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict 2026 World Cup group-stage match W/D/L probabilities."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Model parameter JSON config. Defaults to config/model_params_default.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV output path for group-stage match predictions.",
    )
    args = parser.parse_args()

    config = load_model_config(config_path=args.config)
    predictions = predict_group_stage_results(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Saved {len(predictions)} group-stage predictions to {args.output}")
    print("\nGroup-stage prediction summary")
    print(f"Total matches: {len(predictions)}")
    print(f"Mean p_team_a_win: {predictions['p_team_a_win'].mean():.4f}")
    print(f"Mean p_draw: {predictions['p_draw'].mean():.4f}")
    print(f"Mean p_team_b_win: {predictions['p_team_b_win'].mean():.4f}")
    print("Main pick distribution:")
    print(predictions["main_pick"].value_counts().to_string())
    print("Scoreline-implied result distribution:")
    print(predictions["scoreline_implied_result"].value_counts().to_string())
    print(f"Average calibrated draw probability: {predictions['p_draw'].mean():.4f}")
    print(f"Matches with p_draw >= 0.23: {int((predictions['p_draw'] >= 0.23).sum())}")
    print(f"Matches with p_draw >= 0.28: {int((predictions['p_draw'] >= 0.28).sum())}")
    print(f"Draw-watch matches: {int(predictions['draw_watch'].sum())}")
    print(f"Strong draw-watch matches: {int(predictions['strong_draw_watch'].sum())}")
    print(f"Volatile matches: {int(predictions['volatile_match'].sum())}")
    print(
        "Outcome/scoreline disagreements: "
        f"{int(predictions['outcome_scoreline_disagreement'].sum())}"
    )


if __name__ == "__main__":
    main()
