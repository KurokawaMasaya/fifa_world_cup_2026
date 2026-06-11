from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.evaluate_match_predictions import (
    EPSILON,
    actual_outcome,
    build_prediction_row,
    load_clean_matches,
    normalize_probabilities,
    resolve_matches_path,
    summarize_predictions,
    train_ratings_until,
    update_ratings_after_match,
)
from src.models.poisson_match_model import (
    expected_goals_from_strength,
    outcome_probabilities,
    scoreline_probabilities,
)
from src.ratings.build_elo_ratings import HOME_ADVANTAGE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
TUNING_RESULTS_PATH = OUTPUT_DIR / "parameter_tuning_results.csv"
BEST_PARAMETERS_PATH = PROCESSED_DIR / "best_parameters.json"
FINAL_TEST_OUTPUT_PATH = OUTPUT_DIR / "final_test_evaluation_after_tuning.csv"
MODEL_VERSION = "parameter_tuning_validation"

BASE_TOTAL_GOALS_GRID = [2.45, 2.55, 2.65, 2.75]
SHARE_SCALE_GRID = [220, 250, 280, 320]
MISMATCH_TOTAL_BONUS_GRID = [0.5, 0.7, 0.9]
MISMATCH_SCALE_GRID = [500, 650, 800]
DRAW_BOOST_MAX_GRID = [0.00, 0.015, 0.025, 0.035, 0.045]
DRAW_BOOST_SCALE_GRID = [100, 150, 220, 300]


def apply_draw_boost(
    probabilities: dict[str, float],
    strength_diff: float,
    draw_boost_max: float,
    draw_boost_scale: float,
) -> dict[str, float]:
    """Boost draw probability in close matches and renormalize win probabilities."""
    if draw_boost_scale <= 0:
        raise ValueError("draw_boost_scale must be positive")
    draw_boost = draw_boost_max * math.exp(-abs(strength_diff) / draw_boost_scale)
    boosted_draw = min(0.45, probabilities["draw"] + draw_boost)
    old_non_draw = probabilities["home_win"] + probabilities["away_win"]
    new_non_draw = 1.0 - boosted_draw
    if old_non_draw <= 0 or new_non_draw < 0:
        raise ValueError("Invalid non-draw probability after draw boost")

    boosted = {
        "home_win": probabilities["home_win"] * (new_non_draw / old_non_draw),
        "draw": boosted_draw,
        "away_win": probabilities["away_win"] * (new_non_draw / old_non_draw),
    }
    boosted = normalize_probabilities(boosted)
    total = sum(boosted.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Probabilities do not sum to 1: {total}")
    if any(value < -1e-12 or value > 1 + 1e-12 for value in boosted.values()):
        raise ValueError(f"Invalid probabilities: {boosted}")
    return boosted


def parameter_grid() -> list[dict]:
    keys = [
        "base_total_goals",
        "share_scale",
        "mismatch_total_bonus",
        "mismatch_scale",
        "draw_boost_max",
        "draw_boost_scale",
    ]
    return [
        dict(zip(keys, values))
        for values in itertools.product(
            BASE_TOTAL_GOALS_GRID,
            SHARE_SCALE_GRID,
            MISMATCH_TOTAL_BONUS_GRID,
            MISMATCH_SCALE_GRID,
            DRAW_BOOST_MAX_GRID,
            DRAW_BOOST_SCALE_GRID,
        )
    ]


def predict_period(
    matches: pd.DataFrame,
    train_start: pd.Timestamp,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp | None,
    params: dict,
) -> pd.DataFrame:
    """Predict a period with ratings trained only before period_start."""
    ratings = train_ratings_until(matches, train_start, period_start)
    period_matches = matches.loc[matches["date"] >= period_start].copy()
    if period_end is not None:
        period_matches = period_matches.loc[period_matches["date"] <= period_end]
    if period_matches.empty:
        raise ValueError("No matches available for evaluation period")

    rows = []
    for _, row in period_matches.iterrows():
        home_strength = ratings[row["home_team"]]
        away_strength = ratings[row["away_team"]]
        advantage_home = 0.0 if row["neutral"] else HOME_ADVANTAGE
        strength_diff = home_strength + advantage_home - away_strength
        expected_goals = expected_goals_from_strength(
            strength_a=home_strength,
            strength_b=away_strength,
            advantage_a=advantage_home,
            mapping_mode="total_share",
            base_total_goals=params["base_total_goals"],
            share_scale=params["share_scale"],
            mismatch_total_bonus=params["mismatch_total_bonus"],
            mismatch_scale=params["mismatch_scale"],
        )
        score_probs = scoreline_probabilities(
            expected_goals["lambda_a"],
            expected_goals["lambda_b"],
        )
        outcome_probs = outcome_probabilities(score_probs)
        probabilities = {
            "home_win": outcome_probs["team_a_win"],
            "draw": outcome_probs["draw"],
            "away_win": outcome_probs["team_b_win"],
        }
        probabilities = apply_draw_boost(
            probabilities=probabilities,
            strength_diff=strength_diff,
            draw_boost_max=params["draw_boost_max"],
            draw_boost_scale=params["draw_boost_scale"],
        )
        actual = actual_outcome(row["home_score"], row["away_score"])
        rows.append(
            build_prediction_row(
                model_version=MODEL_VERSION,
                row=row,
                probabilities=probabilities,
                actual=actual,
                lambda_home=expected_goals["lambda_a"],
                lambda_away=expected_goals["lambda_b"],
                strength_diff=strength_diff,
            )
        )
        update_ratings_after_match(ratings, row)
    return pd.DataFrame(rows)


def prepare_period_contexts(
    matches: pd.DataFrame,
    train_start: pd.Timestamp,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp | None,
) -> list[dict]:
    """Precompute leakage-free rating contexts before each period match."""
    ratings = train_ratings_until(matches, train_start, period_start)
    period_matches = matches.loc[matches["date"] >= period_start].copy()
    if period_end is not None:
        period_matches = period_matches.loc[period_matches["date"] <= period_end]
    if period_matches.empty:
        raise ValueError("No matches available for evaluation period")

    contexts = []
    for _, row in period_matches.iterrows():
        home_strength = ratings[row["home_team"]]
        away_strength = ratings[row["away_team"]]
        advantage_home = 0.0 if row["neutral"] else HOME_ADVANTAGE
        contexts.append(
            {
                "row": row,
                "home_strength": home_strength,
                "away_strength": away_strength,
                "advantage_home": advantage_home,
                "strength_diff": home_strength + advantage_home - away_strength,
                "actual": actual_outcome(row["home_score"], row["away_score"]),
            }
        )
        update_ratings_after_match(ratings, row)
    return contexts


def predict_from_contexts(contexts: list[dict], params: dict) -> pd.DataFrame:
    """Evaluate one parameter set against precomputed match contexts."""
    rows = []
    for context in contexts:
        expected_goals = expected_goals_from_strength(
            strength_a=context["home_strength"],
            strength_b=context["away_strength"],
            advantage_a=context["advantage_home"],
            mapping_mode="total_share",
            base_total_goals=params["base_total_goals"],
            share_scale=params["share_scale"],
            mismatch_total_bonus=params["mismatch_total_bonus"],
            mismatch_scale=params["mismatch_scale"],
        )
        score_probs = scoreline_probabilities(
            expected_goals["lambda_a"],
            expected_goals["lambda_b"],
        )
        outcome_probs = outcome_probabilities(score_probs)
        probabilities = {
            "home_win": outcome_probs["team_a_win"],
            "draw": outcome_probs["draw"],
            "away_win": outcome_probs["team_b_win"],
        }
        probabilities = apply_draw_boost(
            probabilities=probabilities,
            strength_diff=context["strength_diff"],
            draw_boost_max=params["draw_boost_max"],
            draw_boost_scale=params["draw_boost_scale"],
        )
        rows.append(
            build_prediction_row(
                model_version=MODEL_VERSION,
                row=context["row"],
                probabilities=probabilities,
                actual=context["actual"],
                lambda_home=expected_goals["lambda_a"],
                lambda_away=expected_goals["lambda_b"],
                strength_diff=context["strength_diff"],
            )
        )
    return pd.DataFrame(rows)


def evaluate_parameter_set(
    contexts: list[dict],
    params: dict,
) -> dict:
    predictions = predict_from_contexts(contexts=contexts, params=params)
    summary = summarize_predictions(predictions, model_version=MODEL_VERSION)
    actual_draw_rate = predictions["actual_outcome"].eq("draw").mean()
    predicted_draw_rate = predictions["p_draw"].mean()
    predicted_pick_draw_rate = predictions["predicted_outcome"].eq("draw").mean()
    return {
        **params,
        "status": "ok",
        "error": "",
        "n_matches": summary["n_matches"],
        "accuracy": summary["accuracy"],
        "mean_brier_score": summary["mean_brier_score"],
        "mean_log_loss": summary["mean_log_loss"],
        "mean_actual_outcome_probability": summary["mean_actual_outcome_probability"],
        "predicted_draw_rate": predicted_draw_rate,
        "predicted_pick_draw_rate": predicted_pick_draw_rate,
        "actual_draw_rate": actual_draw_rate,
        "draw_calibration_error": abs(predicted_draw_rate - actual_draw_rate),
        "goal_mae": summary["goal_mae"],
        "goal_rmse": summary["goal_rmse"],
    }


def run_tuning(
    matches: pd.DataFrame,
    train_start: str = "2014-01-01",
    validation_start: str = "2022-01-01",
    validation_end: str = "2023-12-31",
) -> pd.DataFrame:
    rows = []
    train_start_ts = pd.Timestamp(train_start)
    validation_start_ts = pd.Timestamp(validation_start)
    validation_end_ts = pd.Timestamp(validation_end)
    contexts = prepare_period_contexts(
        matches=matches,
        train_start=train_start_ts,
        period_start=validation_start_ts,
        period_end=validation_end_ts,
    )

    for params in parameter_grid():
        try:
            rows.append(
                evaluate_parameter_set(
                    contexts=contexts,
                    params=params,
                )
            )
        except Exception as exc:  # Keep grid search robust and auditable.
            rows.append(
                {
                    **params,
                    "status": "skipped",
                    "error": str(exc),
                    "n_matches": 0,
                    "accuracy": pd.NA,
                    "mean_brier_score": pd.NA,
                    "mean_log_loss": pd.NA,
                    "mean_actual_outcome_probability": pd.NA,
                    "predicted_draw_rate": pd.NA,
                    "predicted_pick_draw_rate": pd.NA,
                    "actual_draw_rate": pd.NA,
                    "draw_calibration_error": pd.NA,
                    "goal_mae": pd.NA,
                    "goal_rmse": pd.NA,
                }
            )

    results = pd.DataFrame(rows)
    valid = results["status"].eq("ok")
    results = pd.concat(
        [
            results.loc[valid].sort_values(
                ["mean_log_loss", "mean_brier_score", "draw_calibration_error"],
                ascending=[True, True, True],
            ),
            results.loc[~valid],
        ],
        ignore_index=True,
    )
    return results


def best_parameters_from_results(results: pd.DataFrame) -> dict:
    valid = results.loc[results["status"].eq("ok")]
    if valid.empty:
        raise ValueError("No valid parameter combinations found")
    row = valid.iloc[0]
    keys = [
        "base_total_goals",
        "share_scale",
        "mismatch_total_bonus",
        "mismatch_scale",
        "draw_boost_max",
        "draw_boost_scale",
    ]
    return {key: float(row[key]) for key in keys}


def evaluate_final_test(
    matches: pd.DataFrame,
    params: dict,
    train_start: str = "2014-01-01",
    test_start: str = "2024-01-01",
) -> pd.DataFrame:
    predictions = predict_period(
        matches=matches,
        train_start=pd.Timestamp(train_start),
        period_start=pd.Timestamp(test_start),
        period_end=None,
        params=params,
    )
    summary = summarize_predictions(predictions, model_version=MODEL_VERSION)
    actual_draw_rate = predictions["actual_outcome"].eq("draw").mean()
    predicted_draw_rate = predictions["p_draw"].mean()
    predicted_pick_draw_rate = predictions["predicted_outcome"].eq("draw").mean()
    row = {
        **params,
        **summary,
        "predicted_draw_rate": predicted_draw_rate,
        "predicted_pick_draw_rate": predicted_pick_draw_rate,
        "actual_draw_rate": actual_draw_rate,
        "draw_calibration_error": abs(predicted_draw_rate - actual_draw_rate),
    }
    return pd.DataFrame([row])


def save_best_parameters(best_params: dict, output_path: Path = BEST_PARAMETERS_PATH) -> None:
    output = {
        "model_version": MODEL_VERSION,
        "selection_period": {
            "train_start": "2014-01-01",
            "validation_start": "2022-01-01",
            "validation_end": "2023-12-31",
            "test_start": "2024-01-01",
        },
        "parameters": best_params,
    }
    output_path.write_text(json.dumps(output, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune CupCast match prediction parameters.")
    parser.add_argument("--train-start", default="2014-01-01")
    parser.add_argument("--validation-start", default="2022-01-01")
    parser.add_argument("--validation-end", default="2023-12-31")
    parser.add_argument("--test-start", default="2024-01-01")
    args = parser.parse_args()

    matches_path = resolve_matches_path()
    matches = load_clean_matches(matches_path)
    results = run_tuning(
        matches=matches,
        train_start=args.train_start,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(TUNING_RESULTS_PATH, index=False)

    best_params = best_parameters_from_results(results)
    save_best_parameters(best_params)
    final_test = evaluate_final_test(
        matches=matches,
        params=best_params,
        train_start=args.train_start,
        test_start=args.test_start,
    )
    final_test.to_csv(FINAL_TEST_OUTPUT_PATH, index=False)

    print(f"Loaded cleaned matches from {matches_path}")
    print(f"Saved tuning results to {TUNING_RESULTS_PATH}")
    print(f"Saved best parameters to {BEST_PARAMETERS_PATH}")
    print(f"Saved final test evaluation to {FINAL_TEST_OUTPUT_PATH}")
    print("\nBest parameters")
    print(json.dumps(best_params, indent=2))
    print("\nFinal test evaluation")
    print(final_test.to_string(index=False))


if __name__ == "__main__":
    main()
