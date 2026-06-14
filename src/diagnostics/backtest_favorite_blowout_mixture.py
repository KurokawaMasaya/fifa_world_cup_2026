from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.model_config import (
    draw_calibration_kwargs,
    load_model_config,
    poisson_parameter_kwargs,
)
from src.evaluation.evaluate_match_predictions import (
    HOME_ADVANTAGE,
    INITIAL_RATING,
    actual_outcome,
    update_ratings_after_match,
)
from src.models.favorite_blowout_mixture_scoreline import (
    BlowoutMixtureParams,
    favorite_blowout_mixture_scoreline_grid,
    sorted_scorelines,
    tail_metrics,
    top_scorelines,
)
from src.models.negative_binomial_scoreline import (
    get_top_scorelines,
    negative_binomial_scoreline_grid,
)
from src.models.poisson_match_model import (
    apply_draw_boost_to_outcomes,
    expected_goals_from_strength,
    outcome_probabilities,
    scoreline_probabilities,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "games_result_clean.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "diagnostics"
    / "scoreline_research"
    / "v23_favorite_blowout_mixture"
)
EPSILON = 1e-15
FORBIDDEN_PREDICTION_PATTERNS = [
    "goal",
    "score",
    "result",
    "gd",
    "goal_diff",
    "is_draw",
    "shootout",
]


def parse_scoreline(scoreline: str) -> tuple[int, int]:
    goals_a, goals_b = scoreline.split("-")
    return int(goals_a), int(goals_b)


def implied_result(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "team_a_win"
    if goals_a < goals_b:
        return "team_b_win"
    return "draw"


def margin_bucket(goal_diff: int) -> str:
    margin = abs(int(goal_diff))
    if margin == 0:
        return "draw"
    if margin == 1:
        return "one_goal_win"
    if margin == 2:
        return "two_goal_win"
    if margin == 3:
        return "three_goal_win"
    return "four_plus_goal_win"


def load_matches(path: Path, start_date: str) -> pd.DataFrame:
    matches = pd.read_csv(path)
    required = {"date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches["home_score"] = pd.to_numeric(matches["home_score"], errors="coerce")
    matches["away_score"] = pd.to_numeric(matches["away_score"], errors="coerce")
    matches = matches.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    matches = matches.loc[matches["date"] >= pd.Timestamp(start_date)].copy()
    matches["home_score"] = matches["home_score"].astype(int)
    matches["away_score"] = matches["away_score"].astype(int)
    matches["neutral"] = matches["neutral"].astype(bool)
    return matches.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


def split_prediction_and_evaluation_frames(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual_cols = [
        col
        for col in matches.columns
        if any(pattern in col.lower() for pattern in FORBIDDEN_PREDICTION_PATTERNS)
    ]
    prediction_df = matches.drop(columns=actual_cols).copy()
    evaluation_df = matches[
        [
            "date",
            "home_team",
            "away_team",
            "tournament",
            "home_score",
            "away_score",
        ]
    ].copy()
    forbidden_remaining = [
        col
        for col in prediction_df.columns
        if any(pattern in col.lower() for pattern in FORBIDDEN_PREDICTION_PATTERNS)
    ]
    if forbidden_remaining:
        raise ValueError(
            "prediction_df still contains post-match target columns: "
            f"{forbidden_remaining}"
        )
    print(f"Removed target columns before prediction: {actual_cols}")
    return prediction_df, evaluation_df


def probability_for_actual(grid: dict[str, float], actual_scoreline: str) -> float:
    return float(grid.get(actual_scoreline, 0.0))


def actual_rank(grid: dict[str, float], actual_scoreline: str) -> int | None:
    for index, (scoreline, _) in enumerate(sorted_scorelines(grid), start=1):
        if scoreline == actual_scoreline:
            return index
    return None


def build_match_type(favorite_win_prob: float) -> str:
    if favorite_win_prob < 0.45:
        return "balanced"
    if favorite_win_prob < 0.60:
        return "slight_favorite"
    if favorite_win_prob < 0.75:
        return "clear_favorite"
    return "heavy_favorite"


def scoreline_metrics_for_grid(
    grid: dict[str, float],
    actual_scoreline: str,
    actual_goals_a: int,
    actual_goals_b: int,
    favorite_side: str,
    prefix: str,
) -> dict[str, object]:
    top5 = top_scorelines(grid, top_n=5)
    top3_scorelines = [str(row["scoreline"]) for row in top5[:3]]
    top5_scorelines = [str(row["scoreline"]) for row in top5]
    selected = top5[0]
    predicted_goals_a, predicted_goals_b = parse_scoreline(str(selected["scoreline"]))
    actual_result = implied_result(actual_goals_a, actual_goals_b)
    predicted_result = implied_result(predicted_goals_a, predicted_goals_b)
    actual_goal_diff = actual_goals_a - actual_goals_b
    predicted_goal_diff = predicted_goals_a - predicted_goals_b
    fav_actual_goals = actual_goals_a if favorite_side == "team_a" else actual_goals_b
    dog_actual_goals = actual_goals_b if favorite_side == "team_a" else actual_goals_a
    fav_actual_margin = fav_actual_goals - dog_actual_goals
    tail = tail_metrics(grid, favorite_side=favorite_side)
    actual_probability = max(EPSILON, probability_for_actual(grid, actual_scoreline))
    return {
        f"{prefix}_predicted_scoreline": selected["scoreline"],
        f"{prefix}_predicted_goals_a": predicted_goals_a,
        f"{prefix}_predicted_goals_b": predicted_goals_b,
        f"{prefix}_predicted_scoreline_probability": selected["probability"],
        f"{prefix}_top_3_scorelines": json.dumps(top3_scorelines),
        f"{prefix}_top_5_scorelines": json.dumps(top5_scorelines),
        f"{prefix}_top_3_scoreline_probs": json.dumps([row["probability"] for row in top5[:3]]),
        f"{prefix}_top_5_scoreline_probs": json.dumps([row["probability"] for row in top5]),
        f"{prefix}_predicted_result_from_scoreline": predicted_result,
        f"{prefix}_exact_scoreline_hit": selected["scoreline"] == actual_scoreline,
        f"{prefix}_top3_scoreline_hit": actual_scoreline in top3_scorelines,
        f"{prefix}_top5_scoreline_hit": actual_scoreline in top5_scorelines,
        f"{prefix}_actual_scoreline_rank": actual_rank(grid, actual_scoreline),
        f"{prefix}_actual_scoreline_log_probability": math.log(actual_probability),
        f"{prefix}_winner_direction_correct": predicted_result == actual_result,
        f"{prefix}_margin_bucket_predicted": margin_bucket(predicted_goal_diff),
        f"{prefix}_margin_bucket_correct": margin_bucket(predicted_goal_diff) == margin_bucket(actual_goal_diff),
        f"{prefix}_predicted_total_goals": predicted_goals_a + predicted_goals_b,
        f"{prefix}_total_goals_error": (predicted_goals_a + predicted_goals_b) - (actual_goals_a + actual_goals_b),
        f"{prefix}_predicted_goal_diff": predicted_goal_diff,
        f"{prefix}_goal_diff_error": predicted_goal_diff - actual_goal_diff,
        f"{prefix}_abs_goal_diff_error": abs(predicted_goal_diff - actual_goal_diff),
        f"{prefix}_predicted_clean_sheet": predicted_goals_a == 0 or predicted_goals_b == 0,
        f"{prefix}_predicted_btts": predicted_goals_a > 0 and predicted_goals_b > 0,
        f"{prefix}_predicted_over_2_5": predicted_goals_a + predicted_goals_b >= 3,
        f"{prefix}_predicted_over_3_5": predicted_goals_a + predicted_goals_b >= 4,
        f"{prefix}_predicted_two_plus_win": abs(predicted_goal_diff) >= 2,
        f"{prefix}_predicted_favorite_scores_4_plus": fav_actual_goals >= 4,
        f"{prefix}_p_favorite_scores_4_plus": tail["p_favorite_scores_4_plus"],
        f"{prefix}_p_favorite_scores_5_plus": tail["p_favorite_scores_5_plus"],
        f"{prefix}_p_total_goals_5_plus": tail["p_total_goals_5_plus"],
        f"{prefix}_p_margin_4_plus": tail["p_margin_4_plus"],
        f"{prefix}_p_margin_5_plus": tail["p_margin_5_plus"],
        f"{prefix}_tail_risk_index": tail["tail_risk_index"],
        f"{prefix}_actual_favorite_scores_4_plus": fav_actual_goals >= 4,
        f"{prefix}_actual_favorite_scores_5_plus": fav_actual_goals >= 5,
        f"{prefix}_actual_margin_4_plus": fav_actual_margin >= 4,
        f"{prefix}_actual_margin_5_plus": fav_actual_margin >= 5,
    }


def build_predictions(
    matches: pd.DataFrame,
    model_config: dict,
    max_goals: int,
    dispersion_k: float,
    mixture_params: BlowoutMixtureParams | None = None,
) -> pd.DataFrame:
    mixture_params = mixture_params or BlowoutMixtureParams(normal_k=dispersion_k)
    ratings: defaultdict[str, float] = defaultdict(lambda: INITIAL_RATING)
    rows: list[dict[str, object]] = []
    for match_id, row in matches.iterrows():
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        home_strength = ratings[home_team]
        away_strength = ratings[away_team]
        advantage_home = 0.0 if bool(row["neutral"]) else HOME_ADVANTAGE
        strength_diff = home_strength + advantage_home - away_strength
        expected_goals = expected_goals_from_strength(
            strength_a=home_strength,
            strength_b=away_strength,
            advantage_a=advantage_home,
            **poisson_parameter_kwargs(model_config),
        )
        poisson_grid = scoreline_probabilities(
            expected_goals["lambda_a"],
            expected_goals["lambda_b"],
            max_goals=max_goals,
        )
        outcome_probs = outcome_probabilities(poisson_grid)
        outcome_probs = apply_draw_boost_to_outcomes(
            outcome_probs,
            strength_diff=expected_goals["strength_diff"],
            **draw_calibration_kwargs(model_config),
        )

        v20_grid = negative_binomial_scoreline_grid(
            lambda_a=expected_goals["lambda_a"],
            lambda_b=expected_goals["lambda_b"],
            dispersion_k=dispersion_k,
            max_goals=max_goals,
            aggressiveness=0.0,
        )
        v23_grid, metadata = favorite_blowout_mixture_scoreline_grid(
            lambda_a=expected_goals["lambda_a"],
            lambda_b=expected_goals["lambda_b"],
            p_team_a_win=outcome_probs["team_a_win"],
            p_draw=outcome_probs["draw"],
            p_team_b_win=outcome_probs["team_b_win"],
            rating_gap=strength_diff,
            max_goals=max_goals,
            params=mixture_params,
        )

        actual_goals_a = int(row["home_score"])
        actual_goals_b = int(row["away_score"])
        actual_score = f"{actual_goals_a}-{actual_goals_b}"
        favorite_side = str(metadata["favorite_side"])
        favorite_team = home_team if favorite_side == "team_a" else away_team
        actual_result = implied_result(actual_goals_a, actual_goals_b)
        base = {
            "match_id": match_id + 1,
            "date": row["date"].strftime("%Y-%m-%d"),
            "tournament": row["tournament"],
            "team_a": home_team,
            "team_b": away_team,
            "favorite_team": favorite_team,
            "favorite_win_prob": metadata["favorite_win_prob"],
            "favorite_bucket": metadata["favorite_bucket"],
            "match_type": build_match_type(float(metadata["favorite_win_prob"])),
            "strength_diff": strength_diff,
            "lambda_a": expected_goals["lambda_a"],
            "lambda_b": expected_goals["lambda_b"],
            "p_team_a_win": outcome_probs["team_a_win"],
            "p_draw": outcome_probs["draw"],
            "p_team_b_win": outcome_probs["team_b_win"],
            "base_lambda_fav": metadata["base_lambda_fav"],
            "base_lambda_dog": metadata["base_lambda_dog"],
            "p_blowout": metadata["p_blowout"],
            "blowout_lambda_fav": metadata["blowout_lambda_fav"],
            "blowout_lambda_dog": metadata["blowout_lambda_dog"],
            "normal_top_scoreline": get_top_scorelines(v20_grid, top_n=1, mode="mode")[0]["scoreline"],
            "mixture_top_scoreline": top_scorelines(v23_grid, top_n=1)[0]["scoreline"],
            "top_5_scorelines": json.dumps([row["scoreline"] for row in top_scorelines(v23_grid, top_n=5)]),
            "top_5_probs": json.dumps([row["probability"] for row in top_scorelines(v23_grid, top_n=5)]),
            "actual_goals_a": actual_goals_a,
            "actual_goals_b": actual_goals_b,
            "actual_scoreline": actual_score,
            "actual_result": actual_result,
            "margin_bucket_actual": margin_bucket(actual_goals_a - actual_goals_b),
            "actual_total_goals": actual_goals_a + actual_goals_b,
            "actual_goal_diff": actual_goals_a - actual_goals_b,
            "actual_draw": actual_result == "draw",
            "actual_clean_sheet": actual_goals_a == 0 or actual_goals_b == 0,
            "actual_btts": actual_goals_a > 0 and actual_goals_b > 0,
            "actual_over_2_5": actual_goals_a + actual_goals_b >= 3,
            "actual_over_3_5": actual_goals_a + actual_goals_b >= 4,
            "actual_two_plus_win": abs(actual_goals_a - actual_goals_b) >= 2,
        }
        base.update(
            scoreline_metrics_for_grid(
                v20_grid,
                actual_scoreline=actual_score,
                actual_goals_a=actual_goals_a,
                actual_goals_b=actual_goals_b,
                favorite_side=favorite_side,
                prefix="v20",
            )
        )
        base.update(
            scoreline_metrics_for_grid(
                v23_grid,
                actual_scoreline=actual_score,
                actual_goals_a=actual_goals_a,
                actual_goals_b=actual_goals_b,
                favorite_side=favorite_side,
                prefix="v23",
            )
        )
        rows.append(base)
        update_ratings_after_match(ratings, row)
    return pd.DataFrame(rows)


def summarize(match_level: pd.DataFrame, prefix: str, label: str) -> dict[str, object]:
    return {
        "model": label,
        "sample_size": len(match_level),
        "exact_top1_accuracy": match_level[f"{prefix}_exact_scoreline_hit"].mean(),
        "top3_accuracy": match_level[f"{prefix}_top3_scoreline_hit"].mean(),
        "top5_accuracy": match_level[f"{prefix}_top5_scoreline_hit"].mean(),
        "winner_direction_accuracy_from_scoreline": match_level[f"{prefix}_winner_direction_correct"].mean(),
        "margin_bucket_accuracy": match_level[f"{prefix}_margin_bucket_correct"].mean(),
        "mean_actual_scoreline_log_probability": match_level[f"{prefix}_actual_scoreline_log_probability"].mean(),
        "mean_total_goals_actual": match_level["actual_total_goals"].mean(),
        "mean_total_goals_predicted": match_level[f"{prefix}_predicted_total_goals"].mean(),
        "mean_total_goals_error": match_level[f"{prefix}_total_goals_error"].mean(),
        "mean_abs_total_goals_error": match_level[f"{prefix}_total_goals_error"].abs().mean(),
        "mean_goal_diff_error": match_level[f"{prefix}_goal_diff_error"].mean(),
        "mean_abs_goal_diff_error": match_level[f"{prefix}_abs_goal_diff_error"].mean(),
        "predicted_draw_rate_from_scoreline": match_level[f"{prefix}_predicted_result_from_scoreline"].eq("draw").mean(),
        "actual_draw_rate": match_level["actual_draw"].mean(),
        "predicted_one_goal_win_rate": match_level[f"{prefix}_margin_bucket_predicted"].eq("one_goal_win").mean(),
        "actual_one_goal_win_rate": match_level["margin_bucket_actual"].eq("one_goal_win").mean(),
        "predicted_two_plus_goal_win_rate": match_level[f"{prefix}_predicted_two_plus_win"].mean(),
        "actual_two_plus_goal_win_rate": match_level["actual_two_plus_win"].mean(),
        "predicted_clean_sheet_rate": match_level[f"{prefix}_predicted_clean_sheet"].mean(),
        "actual_clean_sheet_rate": match_level["actual_clean_sheet"].mean(),
        "predicted_both_teams_score_rate": match_level[f"{prefix}_predicted_btts"].mean(),
        "actual_both_teams_score_rate": match_level["actual_btts"].mean(),
        "predicted_over_2_5_rate": match_level[f"{prefix}_predicted_over_2_5"].mean(),
        "actual_over_2_5_rate": match_level["actual_over_2_5"].mean(),
        "predicted_over_3_5_rate": match_level[f"{prefix}_predicted_over_3_5"].mean(),
        "actual_over_3_5_rate": match_level["actual_over_3_5"].mean(),
        "mean_p_favorite_scores_4_plus": match_level[f"{prefix}_p_favorite_scores_4_plus"].mean(),
        "actual_favorite_scores_4_plus_rate": match_level[f"{prefix}_actual_favorite_scores_4_plus"].mean(),
        "mean_p_favorite_scores_5_plus": match_level[f"{prefix}_p_favorite_scores_5_plus"].mean(),
        "actual_favorite_scores_5_plus_rate": match_level[f"{prefix}_actual_favorite_scores_5_plus"].mean(),
        "mean_p_margin_4_plus": match_level[f"{prefix}_p_margin_4_plus"].mean(),
        "actual_margin_4_plus_rate": match_level[f"{prefix}_actual_margin_4_plus"].mean(),
        "mean_p_margin_5_plus": match_level[f"{prefix}_p_margin_5_plus"].mean(),
        "actual_margin_5_plus_rate": match_level[f"{prefix}_actual_margin_5_plus"].mean(),
        "mean_tail_risk_index": match_level[f"{prefix}_tail_risk_index"].mean(),
    }


def summarize_by_bucket(match_level: pd.DataFrame, bucket_col: str = "favorite_bucket") -> pd.DataFrame:
    rows = []
    for bucket, subset in match_level.groupby(bucket_col, sort=False):
        row = {"favorite_bucket": bucket, "sample_size": len(subset)}
        for prefix, label in [("v20", "v20_stable"), ("v23", "v23_favorite_blowout_mixture")]:
            summary = summarize(subset, prefix=prefix, label=label)
            for key, value in summary.items():
                if key not in {"model", "sample_size"}:
                    row[f"{label}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def build_tail_analysis(match_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket, subset in match_level.groupby("favorite_bucket", sort=False):
        rows.append(
            {
                "favorite_bucket": bucket,
                "sample_size": len(subset),
                "mean_p_blowout": subset["p_blowout"].mean(),
                "v20_p_favorite_scores_4_plus": subset["v20_p_favorite_scores_4_plus"].mean(),
                "v23_p_favorite_scores_4_plus": subset["v23_p_favorite_scores_4_plus"].mean(),
                "actual_favorite_scores_4_plus": subset["v23_actual_favorite_scores_4_plus"].mean(),
                "v20_p_favorite_scores_5_plus": subset["v20_p_favorite_scores_5_plus"].mean(),
                "v23_p_favorite_scores_5_plus": subset["v23_p_favorite_scores_5_plus"].mean(),
                "actual_favorite_scores_5_plus": subset["v23_actual_favorite_scores_5_plus"].mean(),
                "v20_p_total_goals_5_plus": subset["v20_p_total_goals_5_plus"].mean(),
                "v23_p_total_goals_5_plus": subset["v23_p_total_goals_5_plus"].mean(),
                "actual_total_goals_5_plus": (subset["actual_total_goals"] >= 5).mean(),
                "v20_p_margin_4_plus": subset["v20_p_margin_4_plus"].mean(),
                "v23_p_margin_4_plus": subset["v23_p_margin_4_plus"].mean(),
                "actual_margin_4_plus": subset["v23_actual_margin_4_plus"].mean(),
                "v20_tail_risk_index": subset["v20_tail_risk_index"].mean(),
                "v23_tail_risk_index": subset["v23_tail_risk_index"].mean(),
            }
        )
    return pd.DataFrame(rows)


def save_outputs(match_level: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [
            summarize(match_level, prefix="v20", label="v20_stable"),
            summarize(match_level, prefix="v23", label="v23_favorite_blowout_mixture"),
        ]
    )
    by_bucket = summarize_by_bucket(match_level)
    tail_analysis = build_tail_analysis(match_level)
    comparison = summary.copy()
    comparison["research_status"] = comparison["model"].map(
        {
            "v20_stable": "production_scoreline_baseline",
            "v23_favorite_blowout_mixture": "research_only_not_deployed",
        }
    )
    paths = {
        "match_level": output_dir / "match_level.csv",
        "summary": output_dir / "summary.csv",
        "by_favorite_bucket": output_dir / "by_favorite_bucket.csv",
        "blowout_tail_analysis": output_dir / "blowout_tail_analysis.csv",
        "comparison_vs_v20": output_dir / "comparison_vs_v20.csv",
    }
    match_level.to_csv(paths["match_level"], index=False)
    summary.to_csv(paths["summary"], index=False)
    by_bucket.to_csv(paths["by_favorite_bucket"], index=False)
    tail_analysis.to_csv(paths["blowout_tail_analysis"], index=False)
    comparison.to_csv(paths["comparison_vs_v20"], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest research-only V2.3 favorite blowout mixture scoreline model."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default="2014-01-01")
    parser.add_argument("--max-goals", type=int, default=10)
    parser.add_argument("--dispersion-k", type=float, default=12.0)
    args = parser.parse_args()

    model_config = load_model_config("default")
    matches = load_matches(args.input, start_date=args.start_date)
    prediction_df, evaluation_df = split_prediction_and_evaluation_frames(matches)
    if len(prediction_df) != len(evaluation_df):
        raise ValueError("Prediction/evaluation frame length mismatch")
    match_level = build_predictions(
        matches=matches,
        model_config=model_config,
        max_goals=args.max_goals,
        dispersion_k=args.dispersion_k,
        mixture_params=BlowoutMixtureParams(normal_k=args.dispersion_k),
    )
    if (pd.to_datetime(match_level["date"]) < pd.Timestamp(args.start_date)).any():
        raise ValueError("Backtest output contains rows before start date")
    paths = save_outputs(match_level, output_dir=args.output_dir)
    summary = pd.read_csv(paths["summary"])
    tail = pd.read_csv(paths["blowout_tail_analysis"])
    print(f"Backtested rows: {len(match_level)}")
    print(f"Output directory: {args.output_dir}")
    print("Summary:")
    print(summary.to_string(index=False))
    print("Tail analysis by favorite bucket:")
    print(tail.to_string(index=False))
    print("Files:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
