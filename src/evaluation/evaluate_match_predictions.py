from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.poisson_match_model import (
    apply_draw_boost_to_outcomes,
    expected_goals_from_strength,
    outcome_probabilities,
    scoreline_probabilities,
)
from src.config.model_config import (
    draw_calibration_kwargs,
    load_model_config,
    metadata_columns,
    output_path,
    poisson_parameter_kwargs,
)
from src.ratings.build_elo_ratings import (
    BASE_K,
    FAVORITE_MIN_CONVINCING_WIN_REWARD,
    HOME_ADVANTAGE,
    INITIAL_RATING,
    MAX_RATING_CHANGE,
    actual_scores,
    expected_score,
    margin_multiplier,
    tournament_multiplier,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MATCHES_CLEAN_PATH = PROCESSED_DIR / "matches_clean.csv"
FALLBACK_MATCHES_CLEAN_PATH = PROCESSED_DIR / "games_result_clean.csv"
PREDICTIONS_OUTPUT_PATH = PROCESSED_DIR / "evaluation_predictions_default.csv"
SUMMARY_OUTPUT_PATH = PROCESSED_DIR / "evaluation_summary_default.csv"
CALIBRATION_OUTPUT_PATH = PROCESSED_DIR / "calibration_table_default.csv"
MODEL_COMPARISON_OUTPUT_PATH = PROCESSED_DIR / "evaluation_model_comparison_default.csv"
FAVORITE_CALIBRATION_OUTPUT_PATH = PROCESSED_DIR / "evaluation_favorite_calibration_default.csv"
DRAW_CALIBRATION_OUTPUT_PATH = PROCESSED_DIR / "evaluation_draw_calibration_default.csv"
STRENGTH_DIFF_OUTPUT_PATH = PROCESSED_DIR / "evaluation_by_strength_diff_default.csv"

FINAL_MODEL_NAME = "final_model"
RANDOM_UNIFORM_NAME = "random_uniform"
HISTORICAL_FREQUENCY_NAME = "historical_frequency"
ELO_ONLY_NAME = "elo_only"
EPSILON = 1e-15


def resolve_matches_path() -> Path:
    if MATCHES_CLEAN_PATH.exists():
        return MATCHES_CLEAN_PATH
    if FALLBACK_MATCHES_CLEAN_PATH.exists():
        return FALLBACK_MATCHES_CLEAN_PATH
    raise FileNotFoundError(
        f"Could not find {MATCHES_CLEAN_PATH} or {FALLBACK_MATCHES_CLEAN_PATH}"
    )


def load_clean_matches(path: Path | None = None) -> pd.DataFrame:
    path = resolve_matches_path() if path is None else path
    matches = pd.read_csv(path)
    required_columns = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
    }
    missing = required_columns - set(matches.columns)
    if missing:
        raise ValueError(f"Cleaned matches are missing columns: {sorted(missing)}")

    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches["home_score"] = pd.to_numeric(matches["home_score"], errors="coerce")
    matches["away_score"] = pd.to_numeric(matches["away_score"], errors="coerce")
    matches = matches.dropna(
        subset=["date", "home_team", "away_team", "home_score", "away_score"]
    )
    matches["home_score"] = matches["home_score"].astype(int)
    matches["away_score"] = matches["away_score"].astype(int)
    matches["neutral"] = matches["neutral"].astype(bool)
    return matches.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


def actual_outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "home_win"
    if home_score < away_score:
        return "away_win"
    return "draw"


def predicted_outcome(probabilities: dict[str, float]) -> str:
    return max(probabilities, key=probabilities.get)


def normalize_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    total = sum(probabilities.values())
    if total <= 0:
        raise ValueError("Probabilities must sum to a positive value")
    return {key: value / total for key, value in probabilities.items()}


def update_ratings_after_match(ratings: defaultdict[str, float], row: pd.Series) -> None:
    """Update ratings only after a match prediction is already recorded."""
    home_team = row["home_team"]
    away_team = row["away_team"]
    home_before = ratings[home_team]
    away_before = ratings[away_team]
    home_for_expectation = home_before + (0.0 if row["neutral"] else HOME_ADVANTAGE)

    expected_home = expected_score(home_for_expectation, away_before)
    expected_away = 1.0 - expected_home
    actual_home, actual_away = actual_scores(row["home_score"], row["away_score"])
    goal_diff = int(row["home_score"] - row["away_score"])

    k_effective = (
        BASE_K
        * tournament_multiplier(row["tournament"])
        * margin_multiplier(goal_diff, home_before, away_before)
    )
    delta_home = k_effective * (actual_home - expected_home)

    if home_before > away_before and goal_diff >= 2 and actual_home > expected_home:
        delta_home = max(delta_home, FAVORITE_MIN_CONVINCING_WIN_REWARD)
    elif away_before > home_before and goal_diff <= -2 and actual_away > expected_away:
        delta_home = min(delta_home, -FAVORITE_MIN_CONVINCING_WIN_REWARD)

    delta_home = max(-MAX_RATING_CHANGE, min(MAX_RATING_CHANGE, delta_home))
    ratings[home_team] += delta_home
    ratings[away_team] -= delta_home


def train_ratings_until(
    matches: pd.DataFrame,
    train_start: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> defaultdict[str, float]:
    ratings: defaultdict[str, float] = defaultdict(lambda: INITIAL_RATING)
    training_matches = matches.loc[
        (matches["date"] >= train_start) & (matches["date"] < cutoff)
    ]
    for _, row in training_matches.iterrows():
        update_ratings_after_match(ratings, row)
    return ratings


def training_matches_before(
    matches: pd.DataFrame,
    train_start: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    return matches.loc[(matches["date"] >= train_start) & (matches["date"] < cutoff)]


def estimate_train_base_total_goals(
    matches: pd.DataFrame,
    train_start: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> float:
    training_matches = training_matches_before(matches, train_start, cutoff)
    if training_matches.empty:
        raise ValueError("No training matches available to estimate base goals")
    return float((training_matches["home_score"] + training_matches["away_score"]).mean())


def historical_outcome_probabilities(
    matches: pd.DataFrame,
    train_start: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> dict[str, float]:
    training_matches = training_matches_before(matches, train_start, cutoff)
    if training_matches.empty:
        return {"home_win": 1 / 3, "draw": 1 / 3, "away_win": 1 / 3}
    outcomes = training_matches.apply(
        lambda row: actual_outcome(row["home_score"], row["away_score"]),
        axis=1,
    )
    frequencies = outcomes.value_counts(normalize=True)
    return {
        "home_win": float(frequencies.get("home_win", 0.0)),
        "draw": float(frequencies.get("draw", 0.0)),
        "away_win": float(frequencies.get("away_win", 0.0)),
    }


def elo_only_probabilities(
    home_strength: float,
    away_strength: float,
    advantage_home: float,
    historical_draw_probability: float,
) -> dict[str, float]:
    """Convert Elo expected score to W/D/L using the training draw rate."""
    expected_home = expected_score(home_strength + advantage_home, away_strength)
    draw_probability = max(0.05, min(0.40, historical_draw_probability))
    decisive_probability = 1.0 - draw_probability
    return {
        "home_win": decisive_probability * expected_home,
        "draw": draw_probability,
        "away_win": decisive_probability * (1.0 - expected_home),
    }


def score_prediction(
    probabilities: dict[str, float],
    actual: str,
) -> tuple[float, float, float, str, bool]:
    classes = ["home_win", "draw", "away_win"]
    predicted = predicted_outcome(probabilities)
    correct = predicted == actual
    brier = sum(
        (probabilities[klass] - (1.0 if klass == actual else 0.0)) ** 2
        for klass in classes
    )
    actual_probability = max(EPSILON, min(1.0, probabilities[actual]))
    log_loss = -math.log(actual_probability)
    return brier, log_loss, actual_probability, predicted, correct


def build_prediction_row(
    model_version: str,
    row: pd.Series,
    probabilities: dict[str, float],
    actual: str,
    lambda_home: float | None,
    lambda_away: float | None,
    strength_diff: float,
    model_config: dict | None = None,
) -> dict:
    brier, log_loss, actual_probability, predicted, correct = score_prediction(
        probabilities,
        actual,
    )
    output = {
        "model_version": model_version,
        "date": row["date"].strftime("%Y-%m-%d"),
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "home_score": int(row["home_score"]),
        "away_score": int(row["away_score"]),
        "p_home_win": probabilities["home_win"],
        "p_draw": probabilities["draw"],
        "p_away_win": probabilities["away_win"],
        "predicted_outcome": predicted,
        "actual_outcome": actual,
        "correct": correct,
        "brier_score": brier,
        "log_loss": log_loss,
        "actual_outcome_probability": actual_probability,
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "actual_home_goals": int(row["home_score"]),
        "actual_away_goals": int(row["away_score"]),
        "strength_diff": strength_diff,
    }
    if model_config is not None:
        output.update(metadata_columns(model_config))
        output["model_version"] = model_version
    return output


def evaluate_predictions(
    matches: pd.DataFrame,
    train_start: str = "2014-01-01",
    test_start: str = "2022-01-01",
    test_end: str | None = None,
    model_config: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate leakage-free rolling predictions for baselines and final model."""
    model_config = load_model_config() if model_config is None else model_config
    train_start_ts = pd.Timestamp(train_start)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end) if test_end else None

    ratings = train_ratings_until(matches, train_start_ts, test_start_ts)
    historical_probs = historical_outcome_probabilities(matches, train_start_ts, test_start_ts)
    evaluation_matches = matches.loc[matches["date"] >= test_start_ts].copy()
    if test_end_ts is not None:
        evaluation_matches = evaluation_matches.loc[evaluation_matches["date"] <= test_end_ts]
    if evaluation_matches.empty:
        raise ValueError("No test matches found for the requested evaluation period")

    rows = []
    for _, row in evaluation_matches.iterrows():
        home_strength = ratings[row["home_team"]]
        away_strength = ratings[row["away_team"]]
        advantage_home = 0.0 if row["neutral"] else HOME_ADVANTAGE
        strength_diff = home_strength + advantage_home - away_strength

        expected_goals = expected_goals_from_strength(
            strength_a=home_strength,
            strength_b=away_strength,
            advantage_a=advantage_home,
            **poisson_parameter_kwargs(model_config),
        )
        score_probs = scoreline_probabilities(
            expected_goals["lambda_a"],
            expected_goals["lambda_b"],
        )
        outcome_probs = outcome_probabilities(score_probs)
        outcome_probs = apply_draw_boost_to_outcomes(
            outcome_probs,
            strength_diff=expected_goals["strength_diff"],
            **draw_calibration_kwargs(model_config),
        )
        final_probabilities = {
            "home_win": outcome_probs["team_a_win"],
            "draw": outcome_probs["draw"],
            "away_win": outcome_probs["team_b_win"],
        }
        actual = actual_outcome(row["home_score"], row["away_score"])
        model_inputs = [
            (
                RANDOM_UNIFORM_NAME,
                {"home_win": 1 / 3, "draw": 1 / 3, "away_win": 1 / 3},
                None,
                None,
            ),
            (HISTORICAL_FREQUENCY_NAME, historical_probs, None, None),
            (
                ELO_ONLY_NAME,
                elo_only_probabilities(
                    home_strength=home_strength,
                    away_strength=away_strength,
                    advantage_home=advantage_home,
                    historical_draw_probability=historical_probs["draw"],
                ),
                None,
                None,
            ),
            (
                FINAL_MODEL_NAME,
                final_probabilities,
                expected_goals["lambda_a"],
                expected_goals["lambda_b"],
            ),
        ]
        for model_version, probabilities, lambda_home, lambda_away in model_inputs:
            rows.append(
                build_prediction_row(
                    model_version=model_version,
                    row=row,
                    probabilities=normalize_probabilities(probabilities),
                    actual=actual,
                    lambda_home=lambda_home,
                    lambda_away=lambda_away,
                    strength_diff=strength_diff,
                    model_config=(
                        model_config if model_version == FINAL_MODEL_NAME else None
                    ),
                )
            )

        update_ratings_after_match(ratings, row)

    predictions = pd.DataFrame(rows)
    final_predictions = predictions.loc[predictions["model_version"] == FINAL_MODEL_NAME].copy()
    summary = build_summary(
        final_predictions,
        model_version=model_config["model_version"],
        model_config=model_config,
    )
    model_comparison = build_model_comparison(predictions, model_config=model_config)
    favorite_calibration = build_calibration_table(final_predictions)
    draw_calibration = build_draw_calibration(final_predictions)
    strength_diff = build_strength_diff_diagnostics(final_predictions)
    for output in (favorite_calibration, draw_calibration, strength_diff):
        for key, value in metadata_columns(model_config).items():
            output[key] = value
    return (
        predictions,
        summary,
        model_comparison,
        favorite_calibration,
        draw_calibration,
        strength_diff,
    )


def summarize_predictions(
    predictions: pd.DataFrame,
    model_version: str,
    model_config: dict | None = None,
) -> dict:
    if predictions["lambda_home"].notna().all() and predictions["lambda_away"].notna().all():
        home_error = predictions["lambda_home"] - predictions["actual_home_goals"]
        away_error = predictions["lambda_away"] - predictions["actual_away_goals"]
        absolute_errors = pd.concat([home_error.abs(), away_error.abs()], ignore_index=True)
        squared_errors = pd.concat([home_error ** 2, away_error ** 2], ignore_index=True)
        goal_mae = absolute_errors.mean()
        goal_rmse = math.sqrt(squared_errors.mean())
    else:
        goal_mae = pd.NA
        goal_rmse = pd.NA
    output = {
        "model_version": model_version,
        "n_matches": len(predictions),
        "accuracy": predictions["correct"].mean(),
        "mean_brier_score": predictions["brier_score"].mean(),
        "mean_log_loss": predictions["log_loss"].mean(),
        "mean_actual_outcome_probability": predictions[
            "actual_outcome_probability"
        ].mean(),
        "goal_mae": goal_mae,
        "goal_rmse": goal_rmse,
    }
    if model_config is not None:
        output.update(metadata_columns(model_config))
        output["model_version"] = model_version
    return output


def build_summary(
    predictions: pd.DataFrame,
    model_version: str | None = None,
    model_config: dict | None = None,
) -> pd.DataFrame:
    if model_config is None:
        model_config = load_model_config()
    if model_version is None:
        model_version = model_config["model_version"]
    return pd.DataFrame(
        [
            summarize_predictions(
                predictions,
                model_version=model_version,
                model_config=model_config,
            )
        ]
    )


def build_model_comparison(
    predictions: pd.DataFrame,
    model_config: dict | None = None,
) -> pd.DataFrame:
    rows = []
    for model_version, subset in predictions.groupby("model_version", sort=False):
        rows.append(
            summarize_predictions(
                subset,
                model_version=model_version,
                model_config=model_config if model_version == FINAL_MODEL_NAME else None,
            )
        )
    return pd.DataFrame(rows)


def build_calibration_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bins = [
        (0.40, 0.50),
        (0.50, 0.60),
        (0.60, 0.70),
        (0.70, 0.80),
        (0.80, 0.90),
        (0.90, 1.00),
    ]
    eval_df = predictions.copy()
    eval_df["favorite_side"] = eval_df.apply(
        lambda row: "home_win" if row["p_home_win"] >= row["p_away_win"] else "away_win",
        axis=1,
    )
    eval_df["favorite_win_probability"] = eval_df[["p_home_win", "p_away_win"]].max(axis=1)
    eval_df["favorite_won"] = eval_df["actual_outcome"] == eval_df["favorite_side"]

    for lower, upper in bins:
        if upper == 1.00:
            in_bin = eval_df["favorite_win_probability"].between(lower, upper, inclusive="both")
        else:
            in_bin = (eval_df["favorite_win_probability"] >= lower) & (
                eval_df["favorite_win_probability"] < upper
            )
        subset = eval_df.loc[in_bin]
        rows.append(
            {
                "favorite_win_probability_bin": f"{lower:.2f}-{upper:.2f}",
                "n_matches": len(subset),
                "avg_predicted_favorite_win_prob": (
                    subset["favorite_win_probability"].mean() if not subset.empty else pd.NA
                ),
                "actual_favorite_win_rate": (
                    subset["favorite_won"].mean() if not subset.empty else pd.NA
                ),
            }
        )
    return pd.DataFrame(rows)


def build_draw_calibration(predictions: pd.DataFrame) -> pd.DataFrame:
    draw_actual = predictions["actual_outcome"].eq("draw").astype(float)
    draw_error = predictions["p_draw"] - draw_actual
    return pd.DataFrame(
        [
            {
                "avg_predicted_draw_probability": predictions["p_draw"].mean(),
                "actual_draw_rate": draw_actual.mean(),
                "draw_brier_component": (draw_error ** 2).mean(),
            }
        ]
    )


def build_strength_diff_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    bins = [
        (0, 100, "0-100"),
        (100, 200, "100-200"),
        (200, 300, "200-300"),
        (300, 500, "300-500"),
        (500, float("inf"), "500+"),
    ]
    eval_df = predictions.copy()
    eval_df["abs_strength_diff"] = eval_df["strength_diff"].abs()
    eval_df["favorite_side"] = eval_df.apply(
        lambda row: "home_win" if row["p_home_win"] >= row["p_away_win"] else "away_win",
        axis=1,
    )
    eval_df["favorite_win_probability"] = eval_df[["p_home_win", "p_away_win"]].max(axis=1)
    eval_df["favorite_won"] = eval_df["actual_outcome"] == eval_df["favorite_side"]

    rows = []
    for lower, upper, label in bins:
        if math.isinf(upper):
            subset = eval_df.loc[eval_df["abs_strength_diff"] >= lower]
        else:
            subset = eval_df.loc[
                (eval_df["abs_strength_diff"] >= lower)
                & (eval_df["abs_strength_diff"] < upper)
            ]
        rows.append(
            {
                "strength_diff_bin": label,
                "n_matches": len(subset),
                "accuracy": subset["correct"].mean() if not subset.empty else pd.NA,
                "brier_score": subset["brier_score"].mean() if not subset.empty else pd.NA,
                "log_loss": subset["log_loss"].mean() if not subset.empty else pd.NA,
                "avg_predicted_favorite_win_prob": (
                    subset["favorite_win_probability"].mean() if not subset.empty else pd.NA
                ),
                "actual_favorite_win_rate": (
                    subset["favorite_won"].mean() if not subset.empty else pd.NA
                ),
            }
        )
    return pd.DataFrame(rows)


def save_outputs(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    model_comparison: pd.DataFrame,
    favorite_calibration: pd.DataFrame,
    draw_calibration: pd.DataFrame,
    strength_diff: pd.DataFrame,
    model_config: dict | None = None,
    predictions_output_path: Path = PREDICTIONS_OUTPUT_PATH,
    summary_output_path: Path = SUMMARY_OUTPUT_PATH,
    model_comparison_output_path: Path = MODEL_COMPARISON_OUTPUT_PATH,
    favorite_calibration_output_path: Path = FAVORITE_CALIBRATION_OUTPUT_PATH,
    draw_calibration_output_path: Path = DRAW_CALIBRATION_OUTPUT_PATH,
    strength_diff_output_path: Path = STRENGTH_DIFF_OUTPUT_PATH,
    calibration_output_path: Path = CALIBRATION_OUTPUT_PATH,
) -> None:
    model_config = load_model_config() if model_config is None else model_config
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    prediction_columns = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "p_home_win",
        "p_draw",
        "p_away_win",
        "predicted_outcome",
        "actual_outcome",
        "correct",
        "brier_score",
        "log_loss",
        "lambda_home",
        "lambda_away",
        "model_version",
        "model_status",
        "parameter_config_path",
        "rating_col",
        "bracket_source",
        "uses_random_pairing",
        "random_seed_used_for",
    ]
    final_predictions = predictions.loc[predictions["model_version"] == FINAL_MODEL_NAME]
    for key, value in metadata_columns(model_config).items():
        if key not in final_predictions.columns:
            final_predictions = final_predictions.assign(**{key: value})
    final_predictions[prediction_columns].to_csv(predictions_output_path, index=False)
    summary.to_csv(summary_output_path, index=False)
    model_comparison.to_csv(model_comparison_output_path, index=False)
    favorite_calibration.to_csv(favorite_calibration_output_path, index=False)
    favorite_calibration.to_csv(calibration_output_path, index=False)
    draw_calibration.to_csv(draw_calibration_output_path, index=False)
    strength_diff.to_csv(strength_diff_output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CupCast single-match predictions.")
    parser.add_argument(
        "--mode",
        default="default",
        choices=["default", "experimental", "test"],
        help="Model parameter mode.",
    )
    parser.add_argument("--train-start", default="2014-01-01")
    parser.add_argument("--test-start", default="2022-01-01")
    parser.add_argument("--test-end", default=None)
    args = parser.parse_args()
    model_config = load_model_config(args.mode)
    predictions_output = output_path("evaluation_predictions", model_config)
    summary_output = output_path("evaluation_summary", model_config)
    model_comparison_output = output_path("evaluation_model_comparison", model_config)
    favorite_calibration_output = output_path("evaluation_favorite_calibration", model_config)
    draw_calibration_output = output_path("evaluation_draw_calibration", model_config)
    strength_diff_output = output_path("evaluation_by_strength_diff", model_config)
    calibration_output = output_path("calibration_table", model_config)

    matches_path = resolve_matches_path()
    matches = load_clean_matches(matches_path)
    (
        predictions,
        summary,
        model_comparison,
        favorite_calibration,
        draw_calibration,
        strength_diff,
    ) = evaluate_predictions(
        matches=matches,
        train_start=args.train_start,
        test_start=args.test_start,
        test_end=args.test_end,
        model_config=model_config,
    )
    save_outputs(
        predictions,
        summary,
        model_comparison,
        favorite_calibration,
        draw_calibration,
        strength_diff,
        model_config=model_config,
        predictions_output_path=predictions_output,
        summary_output_path=summary_output,
        model_comparison_output_path=model_comparison_output,
        favorite_calibration_output_path=favorite_calibration_output,
        draw_calibration_output_path=draw_calibration_output,
        strength_diff_output_path=strength_diff_output,
        calibration_output_path=calibration_output,
    )

    print(f"Loaded cleaned matches from {matches_path}")
    print(f"Mode: {args.mode}")
    print(f"Model version: {model_config['model_version']}")
    print(f"Parameter config: {model_config['parameter_config_path']}")
    print(f"Saved prediction rows to {predictions_output}")
    print(f"Saved summary to {summary_output}")
    print(f"Saved model comparison to {model_comparison_output}")
    print(f"Saved favorite calibration to {favorite_calibration_output}")
    print(f"Saved draw calibration to {draw_calibration_output}")
    print(f"Saved strength-difference diagnostics to {strength_diff_output}")
    print(model_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
