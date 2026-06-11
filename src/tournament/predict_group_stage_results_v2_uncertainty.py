from __future__ import annotations

import argparse
import json
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
from src.models.poisson_match_model import DEFAULT_RATINGS_PATH, predict_from_ratings  # noqa: E402
from src.models.v2_uncertainty_adjustment import (  # noqa: E402
    DEFAULT_SQUAD_VALUES_PATH,
    adjust_probabilities_for_uncertainty,
    load_squad_uncertainty_features,
)
from src.models.v2_superstar_adjustment import (  # noqa: E402
    DEFAULT_SUPERSTAR_FEATURES_PATH,
    adjust_probabilities_for_superstar_impact,
    load_superstar_features,
)
from src.models.v2_club_form_adjustment import (  # noqa: E402
    DEFAULT_CLUB_FORM_FEATURES_PATH,
    adjust_probabilities_for_club_form,
    load_team_club_form_features,
)
from src.models.negative_binomial_scoreline import (  # noqa: E402
    get_top_scorelines,
    negative_binomial_scoreline_grid,
    scoreline_entropy,
)
from src.simulation.group_stage_simulator import load_group_stage_fixtures  # noqa: E402
from src.tournament.predict_group_stage_results import (  # noqa: E402
    EXPECTED_GROUP_STAGE_MATCHES,
    PROBABILITY_TOLERANCE,
    VALID_OUTCOMES,
    main_pick_from_probabilities,
    scoreline_implied_result,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "output" / "predictions" / "group_stage_predictions_v2_uncertainty.csv"
)
TUNED_OUTPUT_PATH = (
    PROJECT_ROOT / "output" / "predictions" / "group_stage_predictions_v2_uncertainty_tuned.csv"
)
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
    "v1_p_team_a_win",
    "v1_p_draw",
    "v1_p_team_b_win",
    "uncertainty_p_team_a_win",
    "uncertainty_p_draw",
    "uncertainty_p_team_b_win",
    "superstar_p_team_a_win",
    "superstar_p_draw",
    "superstar_p_team_b_win",
    "v2_p_team_a_win",
    "v2_p_draw",
    "v2_p_team_b_win",
    "superstar_score_a",
    "superstar_score_b",
    "superstar_edge",
    "superstar_shift",
    "top_player_a",
    "top_player_b",
    "superstar_adjustment_reason",
    "club_form_score_a",
    "club_form_score_b",
    "club_form_edge",
    "club_form_shift",
    "club_form_adjustment_reason",
    "club_form_data_coverage_a",
    "club_form_data_coverage_b",
    "most_likely_scoreline",
    "most_likely_scoreline_probability",
    "scoreline_implied_result",
    "nb_top_scoreline_1",
    "nb_top_scoreline_1_probability",
    "nb_top_scoreline_1_result",
    "nb_top_scoreline_2",
    "nb_top_scoreline_2_probability",
    "nb_top_scoreline_2_result",
    "nb_top_scoreline_3",
    "nb_top_scoreline_3_probability",
    "nb_top_scoreline_3_result",
    "nb_top_scorelines_json",
    "nb_scoreline_entropy",
    "nb_dispersion_k",
    "outcome_model",
    "scoreline_model",
    "draw_watch",
    "strong_draw_watch",
    "volatile_match",
    "outcome_scoreline_disagreement",
    "risk_note",
    "model_version",
    "rating_col",
    "max_shift",
    "volatility_weight",
    "stability_edge_weight",
    "max_star_shift",
    "star_weight",
    "max_club_form_shift",
    "club_form_weight",
]


def resolve_project_path(path_value: str | None, fallback: Path) -> Path:
    if not path_value:
        return fallback
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_v1_ratings(config: dict) -> pd.DataFrame:
    ratings_path = resolve_project_path(config.get("rating_source_path"), DEFAULT_RATINGS_PATH)
    ratings = pd.read_csv(ratings_path)
    required = {"team_name", config["rating_col"]}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"{ratings_path} is missing required columns: {sorted(missing)}")
    return ratings


def uncertainty_parameters(config: dict) -> dict[str, float]:
    return {
        "max_shift": float(
            config.get("max_shift", config.get("v2_uncertainty_max_shift", 0.035))
        ),
        "volatility_weight": float(
            config.get("volatility_weight", config.get("v2_volatility_weight", 0.010))
        ),
        "stability_edge_weight": float(
            config.get(
                "stability_edge_weight",
                config.get("v2_stability_edge_weight", 0.008),
            )
        ),
    }


def superstar_parameters(config: dict) -> dict[str, float]:
    return {
        "max_star_shift": float(config.get("max_star_shift", 0.040)),
        "star_weight": float(config.get("star_weight", 0.012)),
    }


def club_form_parameters(config: dict) -> dict[str, float]:
    return {
        "max_club_form_shift": float(config.get("max_club_form_shift", 0.025)),
        "club_form_weight": float(config.get("club_form_weight", 0.008)),
    }


def validate_predictions(predictions: pd.DataFrame) -> None:
    v1_sum = (
        predictions["v1_p_team_a_win"]
        + predictions["v1_p_draw"]
        + predictions["v1_p_team_b_win"]
    )
    v2_sum = (
        predictions["v2_p_team_a_win"]
        + predictions["v2_p_draw"]
        + predictions["v2_p_team_b_win"]
    )
    if (v1_sum - 1.0).abs().max() > PROBABILITY_TOLERANCE:
        raise ValueError("V1 probabilities must sum to 1")
    if (v2_sum - 1.0).abs().max() > PROBABILITY_TOLERANCE:
        raise ValueError("V2 probabilities must sum to 1")
    if predictions["nb_top_scoreline_1"].isna().any():
        raise ValueError("Negative Binomial top scoreline display is missing")
    if not predictions["outcome_model"].eq("calibrated_v2").all():
        raise ValueError("outcome_model must be calibrated_v2")
    if not predictions["scoreline_model"].eq("negative_binomial_display").all():
        raise ValueError("scoreline_model must be negative_binomial_display")
    if set(predictions["scoreline_implied_result"]) - VALID_OUTCOMES:
        raise ValueError("scoreline_implied_result contains invalid values")
    if predictions[["team_a", "team_b"]].isna().any().any():
        raise ValueError("Predictions contain missing teams")
    required_numeric = [
        "strength_a",
        "strength_b",
        "v1_p_team_a_win",
        "v1_p_draw",
        "v1_p_team_b_win",
        "uncertainty_p_team_a_win",
        "uncertainty_p_draw",
        "uncertainty_p_team_b_win",
        "superstar_p_team_a_win",
        "superstar_p_draw",
        "superstar_p_team_b_win",
        "v2_p_team_a_win",
        "v2_p_draw",
        "v2_p_team_b_win",
    ]
    if predictions[required_numeric].isna().any().any():
        raise ValueError("Predictions contain missing ratings or probabilities")
    if len(predictions) != EXPECTED_GROUP_STAGE_MATCHES:
        raise ValueError(
            f"Expected {EXPECTED_GROUP_STAGE_MATCHES} group-stage matches, "
            f"found {len(predictions)}"
        )


def risk_note_from_probabilities(
    team_a: str,
    team_b: str,
    outcome_argmax: str,
    scoreline_result: str,
    draw_watch: bool,
    volatile_match: bool,
) -> str:
    if scoreline_result == "draw" and outcome_argmax == "team_a_win":
        return (
            "Most likely scoreline is draw, but total win probability slightly "
            f"favors {team_a}."
        )
    if scoreline_result == "draw" and outcome_argmax == "team_b_win":
        return (
            "Most likely scoreline is draw, but total win probability slightly "
            f"favors {team_b}."
        )
    if volatile_match:
        return "Low confidence; all three outcome probabilities are close."
    if draw_watch:
        return "High draw probability; treat this as a draw-watch match."
    return "Clear favorite; low draw risk."


def predict_group_stage_v2_uncertainty(
    v1_config: dict,
    v2_config: dict,
    squad_features_path: Path,
    superstar_features_path: Path = DEFAULT_SUPERSTAR_FEATURES_PATH,
    club_form_features_path: Path = DEFAULT_CLUB_FORM_FEATURES_PATH,
    nb_dispersion_k: float = 12,
) -> pd.DataFrame:
    fixtures = load_group_stage_fixtures()
    if len(fixtures) != EXPECTED_GROUP_STAGE_MATCHES:
        print(
            "WARNING: Expected "
            f"{EXPECTED_GROUP_STAGE_MATCHES} group-stage matches, found {len(fixtures)}"
        )

    ratings = load_v1_ratings(v1_config)
    rating_col = v1_config["rating_col"]
    squad_features = load_squad_uncertainty_features(squad_features_path)
    params = uncertainty_parameters(v2_config)
    star_params = superstar_parameters(v2_config)
    club_params = club_form_parameters(v2_config)
    superstar_features = load_superstar_features(superstar_features_path)
    club_form_features = load_team_club_form_features(club_form_features_path)
    model_kwargs = {
        **poisson_parameter_kwargs(v1_config),
        **draw_calibration_kwargs(v1_config),
    }

    rows = []
    for _, fixture in fixtures.iterrows():
        v1_prediction = predict_from_ratings(
            team_a=fixture["home_team"],
            team_b=fixture["away_team"],
            ratings_df=ratings,
            rating_col=rating_col,
            **model_kwargs,
        )
        adjustment = adjust_probabilities_for_uncertainty(
            p_team_a_win=v1_prediction["p_team_a_win"],
            p_draw=v1_prediction["p_draw"],
            p_team_b_win=v1_prediction["p_team_b_win"],
            team_a=fixture["home_team"],
            team_b=fixture["away_team"],
            squad_features_df=squad_features,
            **params,
        )
        superstar_adjustment = adjust_probabilities_for_superstar_impact(
            p_team_a_win=adjustment["adjusted_p_team_a_win"],
            p_draw=adjustment["adjusted_p_draw"],
            p_team_b_win=adjustment["adjusted_p_team_b_win"],
            team_a=fixture["home_team"],
            team_b=fixture["away_team"],
            superstar_features_df=superstar_features,
            **star_params,
        )
        club_form_adjustment = adjust_probabilities_for_club_form(
            p_team_a_win=superstar_adjustment["star_adjusted_p_team_a_win"],
            p_draw=superstar_adjustment["star_adjusted_p_draw"],
            p_team_b_win=superstar_adjustment["star_adjusted_p_team_b_win"],
            team_a=fixture["home_team"],
            team_b=fixture["away_team"],
            team_club_form_features_df=club_form_features,
            **club_params,
        )
        outcome_argmax, outcome_argmax_probability = main_pick_from_probabilities(
            club_form_adjustment["club_form_adjusted_p_team_a_win"],
            club_form_adjustment["club_form_adjusted_p_draw"],
            club_form_adjustment["club_form_adjusted_p_team_b_win"],
        )
        nb_grid = negative_binomial_scoreline_grid(
            lambda_a=v1_prediction["lambda_a"],
            lambda_b=v1_prediction["lambda_b"],
            dispersion_k=nb_dispersion_k,
            max_goals=8,
        )
        nb_sum = sum(nb_grid.values())
        if abs(nb_sum - 1.0) > PROBABILITY_TOLERANCE:
            raise ValueError("Negative Binomial scoreline grid must sum to 1")
        nb_top_scorelines = get_top_scorelines(nb_grid, top_n=3)
        scoreline_result = str(nb_top_scorelines[0]["implied_result"])
        draw_watch = club_form_adjustment["club_form_adjusted_p_draw"] >= 0.26
        strong_draw_watch = club_form_adjustment["club_form_adjusted_p_draw"] >= 0.29
        volatile_match = outcome_argmax_probability < 0.45
        risk_note = risk_note_from_probabilities(
            team_a=fixture["home_team"],
            team_b=fixture["away_team"],
            outcome_argmax=outcome_argmax,
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
                "strength_a": v1_prediction["strength_a"],
                "strength_b": v1_prediction["strength_b"],
                "strength_diff": v1_prediction["strength_diff"],
                "lambda_a": v1_prediction["lambda_a"],
                "lambda_b": v1_prediction["lambda_b"],
                "v1_p_team_a_win": v1_prediction["p_team_a_win"],
                "v1_p_draw": v1_prediction["p_draw"],
                "v1_p_team_b_win": v1_prediction["p_team_b_win"],
                "uncertainty_p_team_a_win": adjustment["adjusted_p_team_a_win"],
                "uncertainty_p_draw": adjustment["adjusted_p_draw"],
                "uncertainty_p_team_b_win": adjustment["adjusted_p_team_b_win"],
                "superstar_p_team_a_win": superstar_adjustment["star_adjusted_p_team_a_win"],
                "superstar_p_draw": superstar_adjustment["star_adjusted_p_draw"],
                "superstar_p_team_b_win": superstar_adjustment["star_adjusted_p_team_b_win"],
                "v2_p_team_a_win": club_form_adjustment[
                    "club_form_adjusted_p_team_a_win"
                ],
                "v2_p_draw": club_form_adjustment["club_form_adjusted_p_draw"],
                "v2_p_team_b_win": club_form_adjustment[
                    "club_form_adjusted_p_team_b_win"
                ],
                "superstar_score_a": superstar_adjustment["superstar_a"],
                "superstar_score_b": superstar_adjustment["superstar_b"],
                "superstar_edge": superstar_adjustment["superstar_edge"],
                "superstar_shift": superstar_adjustment["superstar_shift"],
                "top_player_a": superstar_adjustment["top_player_a"],
                "top_player_b": superstar_adjustment["top_player_b"],
                "superstar_adjustment_reason": superstar_adjustment[
                    "superstar_adjustment_reason"
                ],
                "club_form_score_a": club_form_adjustment["club_form_score_a"],
                "club_form_score_b": club_form_adjustment["club_form_score_b"],
                "club_form_edge": club_form_adjustment["club_form_edge"],
                "club_form_shift": club_form_adjustment["club_form_shift"],
                "club_form_adjustment_reason": club_form_adjustment[
                    "club_form_adjustment_reason"
                ],
                "club_form_data_coverage_a": club_form_adjustment[
                    "club_form_data_coverage_a"
                ],
                "club_form_data_coverage_b": club_form_adjustment[
                    "club_form_data_coverage_b"
                ],
                "most_likely_scoreline": v1_prediction["most_likely_scoreline"],
                "most_likely_scoreline_probability": v1_prediction[
                    "most_likely_scoreline_probability"
                ],
                "scoreline_implied_result": scoreline_result,
                "nb_top_scoreline_1": nb_top_scorelines[0]["scoreline"],
                "nb_top_scoreline_1_probability": nb_top_scorelines[0]["probability"],
                "nb_top_scoreline_1_result": nb_top_scorelines[0]["implied_result"],
                "nb_top_scoreline_2": nb_top_scorelines[1]["scoreline"],
                "nb_top_scoreline_2_probability": nb_top_scorelines[1]["probability"],
                "nb_top_scoreline_2_result": nb_top_scorelines[1]["implied_result"],
                "nb_top_scoreline_3": nb_top_scorelines[2]["scoreline"],
                "nb_top_scoreline_3_probability": nb_top_scorelines[2]["probability"],
                "nb_top_scoreline_3_result": nb_top_scorelines[2]["implied_result"],
                "nb_top_scorelines_json": json.dumps(nb_top_scorelines),
                "nb_scoreline_entropy": scoreline_entropy(nb_grid),
                "nb_dispersion_k": nb_dispersion_k,
                "outcome_model": "calibrated_v2",
                "scoreline_model": "negative_binomial_display",
                "draw_watch": draw_watch,
                "strong_draw_watch": strong_draw_watch,
                "volatile_match": volatile_match,
                "outcome_scoreline_disagreement": outcome_argmax != scoreline_result,
                "risk_note": risk_note,
                "model_version": v2_config.get("model_version", "v2_uncertainty"),
                "rating_col": rating_col,
                "max_shift": params["max_shift"],
                "volatility_weight": params["volatility_weight"],
                "stability_edge_weight": params["stability_edge_weight"],
                "max_star_shift": star_params["max_star_shift"],
                "star_weight": star_params["star_weight"],
                "max_club_form_shift": club_params["max_club_form_shift"],
                "club_form_weight": club_params["club_form_weight"],
            }
        )

    output = pd.DataFrame(rows)[FINAL_COLUMNS]
    validate_predictions(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate group-stage predictions with V2 uncertainty adjustment."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--squad-values", type=Path, default=DEFAULT_SQUAD_VALUES_PATH)
    parser.add_argument("--superstar-features", type=Path, default=DEFAULT_SUPERSTAR_FEATURES_PATH)
    parser.add_argument("--club-form-features", type=Path, default=DEFAULT_CLUB_FORM_FEATURES_PATH)
    parser.add_argument("--nb-dispersion-k", type=float, default=12)
    args = parser.parse_args()

    v1_config = load_model_config("default")
    v2_config = (
        load_model_config(config_path=args.config)
        if args.config is not None
        else load_model_config("v2")
    )
    output_path = args.output
    if output_path is None:
        output_path = (
            TUNED_OUTPUT_PATH
            if "tuned" in str(v2_config.get("model_version", ""))
            else DEFAULT_OUTPUT_PATH
        )
    predictions = predict_group_stage_v2_uncertainty(
        v1_config=v1_config,
        v2_config=v2_config,
        squad_features_path=args.squad_values,
        superstar_features_path=args.superstar_features,
        club_form_features_path=args.club_form_features,
        nb_dispersion_k=args.nb_dispersion_k,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)

    print(f"Saved {len(predictions)} V2 uncertainty predictions to {output_path}")
    print(
        "Parameters: "
        f"max_shift={predictions['max_shift'].iloc[0]}, "
        f"volatility_weight={predictions['volatility_weight'].iloc[0]}, "
        f"stability_edge_weight={predictions['stability_edge_weight'].iloc[0]}"
    )
    print(
        "Superstar parameters: "
        f"max_star_shift={predictions['max_star_shift'].iloc[0]}, "
        f"star_weight={predictions['star_weight'].iloc[0]}"
    )
    print(
        "Club-form parameters: "
        f"max_club_form_shift={predictions['max_club_form_shift'].iloc[0]}, "
        f"club_form_weight={predictions['club_form_weight'].iloc[0]}"
    )
    print(f"Mean V1 draw probability: {predictions['v1_p_draw'].mean():.4f}")
    print(f"Mean V2 draw probability: {predictions['v2_p_draw'].mean():.4f}")
    print("Scoreline-implied result distribution:")
    print(predictions["scoreline_implied_result"].value_counts().to_string())
    print("Negative Binomial top-scoreline distribution:")
    print(predictions["nb_top_scoreline_1"].value_counts().to_string())
    print(
        "Matches where Negative Binomial top scoreline is 1-1: "
        f"{int(predictions['nb_top_scoreline_1'].eq('1-1').sum())}"
    )
    print(f"Draw-watch matches: {int(predictions['draw_watch'].sum())}")
    print(f"Outcome/scoreline disagreements: {int(predictions['outcome_scoreline_disagreement'].sum())}")


if __name__ == "__main__":
    main()
