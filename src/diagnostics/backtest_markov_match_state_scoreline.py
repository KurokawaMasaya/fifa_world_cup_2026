from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.model_config import load_model_config
from src.diagnostics.backtest_favorite_blowout_mixture import (
    DEFAULT_INPUT,
    build_predictions,
    load_matches,
    margin_bucket,
    split_prediction_and_evaluation_frames,
)
from src.models.favorite_blowout_mixture_scoreline import BlowoutMixtureParams
from src.models.markov_match_state_scoreline import (
    MatchStateConfig,
    simulate_markov_scoreline_distribution,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "diagnostics"
    / "scoreline_research"
    / "v3_markov_match_state"
)
EPSILON = 1e-15


def parse_scoreline(scoreline: str) -> tuple[int, int]:
    goals_a, goals_b = scoreline.split("-")
    return int(goals_a), int(goals_b)


def implied_result(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "team_a_win"
    if goals_a < goals_b:
        return "team_b_win"
    return "draw"


def actual_scoreline_rank(grid: dict[str, float], actual_scoreline: str) -> int | None:
    for index, (scoreline, _) in enumerate(
        sorted(grid.items(), key=lambda item: (-item[1], item[0])),
        start=1,
    ):
        if scoreline == actual_scoreline:
            return index
    return None


def score_markov_prediction(row: pd.Series, markov: dict[str, object]) -> dict[str, object]:
    grid = markov["scoreline_probabilities"]
    actual_scoreline = row["actual_scoreline"]
    top5 = markov["top_5_scorelines"]
    top3_scorelines = [item["scoreline"] for item in top5[:3]]
    top5_scorelines = [item["scoreline"] for item in top5]
    pred_goals_a, pred_goals_b = parse_scoreline(str(markov["top_1_scoreline"]))
    actual_goals_a = int(row["actual_goals_a"])
    actual_goals_b = int(row["actual_goals_b"])
    actual_result = implied_result(actual_goals_a, actual_goals_b)
    predicted_result = implied_result(pred_goals_a, pred_goals_b)
    actual_goal_diff = actual_goals_a - actual_goals_b
    predicted_goal_diff = pred_goals_a - pred_goals_b
    favorite_side = str(markov["favorite_side"])
    fav_goals_actual = actual_goals_a if favorite_side == "team_a" else actual_goals_b
    dog_goals_actual = actual_goals_b if favorite_side == "team_a" else actual_goals_a
    fav_margin_actual = fav_goals_actual - dog_goals_actual
    actual_probability = max(EPSILON, float(grid.get(actual_scoreline, 0.0)))
    state_rates = markov["state_visit_rates"]
    return {
        "markov_top_scoreline": markov["top_1_scoreline"],
        "markov_top_3_scorelines": json.dumps(top3_scorelines),
        "markov_top_5_scorelines": json.dumps(top5_scorelines),
        "markov_top_5_probs": json.dumps([item["probability"] for item in top5]),
        "markov_exact_scoreline_hit": markov["top_1_scoreline"] == actual_scoreline,
        "markov_top3_scoreline_hit": actual_scoreline in top3_scorelines,
        "markov_top5_scoreline_hit": actual_scoreline in top5_scorelines,
        "markov_winner_direction_correct": predicted_result == actual_result,
        "markov_predicted_result_from_scoreline": predicted_result,
        "markov_predicted_total_goals": pred_goals_a + pred_goals_b,
        "markov_predicted_goal_diff": predicted_goal_diff,
        "markov_margin_bucket_predicted": margin_bucket(predicted_goal_diff),
        "markov_margin_bucket_correct": margin_bucket(predicted_goal_diff) == margin_bucket(actual_goal_diff),
        "markov_actual_scoreline_rank": actual_scoreline_rank(grid, actual_scoreline),
        "markov_actual_scoreline_log_probability": math.log(actual_probability),
        "markov_over_2_5_probability": markov["over_2_5_probability"],
        "markov_over_3_5_probability": markov["over_3_5_probability"],
        "markov_btts_probability": markov["btts_probability"],
        "markov_favorite_scores_4_plus_probability": markov[
            "favorite_scores_4_plus_probability"
        ],
        "markov_favorite_scores_5_plus_probability": markov[
            "favorite_scores_5_plus_probability"
        ],
        "markov_margin_4_plus_probability": markov["margin_4_plus_probability"],
        "markov_total_goals_5_plus_probability": markov[
            "total_goals_5_plus_probability"
        ],
        "markov_blowout_path_probability": markov["blowout_path_probability"],
        "markov_normal_visit_rate": state_rates["normal"],
        "markov_open_game_visit_rate": state_rates["open_game"],
        "markov_favorite_pressure_visit_rate": state_rates["favorite_pressure"],
        "markov_blowout_visit_rate": state_rates["blowout"],
        "actual_favorite_scores_4_plus": fav_goals_actual >= 4,
        "actual_favorite_scores_5_plus": fav_goals_actual >= 5,
        "actual_margin_4_plus": fav_margin_actual >= 4,
        "actual_total_goals_5_plus": actual_goals_a + actual_goals_b >= 5,
    }


def add_markov_predictions(
    base: pd.DataFrame,
    n_sims: int,
    step_minutes: int,
    random_seed: int,
    config: MatchStateConfig,
) -> pd.DataFrame:
    rows = []
    for _, row in base.iterrows():
        markov = simulate_markov_scoreline_distribution(
            lambda_a=float(row["lambda_a"]),
            lambda_b=float(row["lambda_b"]),
            team_a_win_pct=float(row["p_team_a_win"]),
            draw_pct=float(row["p_draw"]),
            team_b_win_pct=float(row["p_team_b_win"]),
            n_sims=n_sims,
            step_minutes=step_minutes,
            random_seed=random_seed + int(row["match_id"]),
            config=config,
        )
        updated = row.to_dict()
        updated.update(score_markov_prediction(row, markov))
        rows.append(updated)
    return pd.DataFrame(rows)


def summarize_model(match_level: pd.DataFrame, prefix: str, label: str) -> dict[str, object]:
    if prefix == "markov":
        return {
            "model": label,
            "sample_size": len(match_level),
            "exact_top1_accuracy": match_level["markov_exact_scoreline_hit"].mean(),
            "top3_accuracy": match_level["markov_top3_scoreline_hit"].mean(),
            "top5_accuracy": match_level["markov_top5_scoreline_hit"].mean(),
            "winner_direction_accuracy": match_level["markov_winner_direction_correct"].mean(),
            "margin_bucket_accuracy": match_level["markov_margin_bucket_correct"].mean(),
            "mean_predicted_goals": match_level["markov_predicted_total_goals"].mean(),
            "actual_mean_goals": match_level["actual_total_goals"].mean(),
            "predicted_over_2_5": match_level["markov_over_2_5_probability"].mean(),
            "actual_over_2_5": match_level["actual_over_2_5"].mean(),
            "predicted_over_3_5": match_level["markov_over_3_5_probability"].mean(),
            "actual_over_3_5": match_level["actual_over_3_5"].mean(),
            "predicted_btts": match_level["markov_btts_probability"].mean(),
            "actual_btts": match_level["actual_btts"].mean(),
            "predicted_favorite_scores_4_plus": match_level["markov_favorite_scores_4_plus_probability"].mean(),
            "actual_favorite_scores_4_plus": match_level["actual_favorite_scores_4_plus"].mean(),
            "predicted_favorite_scores_5_plus": match_level["markov_favorite_scores_5_plus_probability"].mean(),
            "actual_favorite_scores_5_plus": match_level["actual_favorite_scores_5_plus"].mean(),
            "predicted_margin_4_plus": match_level["markov_margin_4_plus_probability"].mean(),
            "actual_margin_4_plus": match_level["actual_margin_4_plus"].mean(),
            "predicted_total_goals_5_plus": match_level["markov_total_goals_5_plus_probability"].mean(),
            "actual_total_goals_5_plus": match_level["actual_total_goals_5_plus"].mean(),
            "mean_actual_scoreline_log_probability": match_level[
                "markov_actual_scoreline_log_probability"
            ].mean(),
        }
    return {
        "model": label,
        "sample_size": len(match_level),
        "exact_top1_accuracy": match_level[f"{prefix}_exact_scoreline_hit"].mean(),
        "top3_accuracy": match_level[f"{prefix}_top3_scoreline_hit"].mean(),
        "top5_accuracy": match_level[f"{prefix}_top5_scoreline_hit"].mean(),
        "winner_direction_accuracy": match_level[f"{prefix}_winner_direction_correct"].mean(),
        "margin_bucket_accuracy": match_level[f"{prefix}_margin_bucket_correct"].mean(),
        "mean_predicted_goals": match_level[f"{prefix}_predicted_total_goals"].mean(),
        "actual_mean_goals": match_level["actual_total_goals"].mean(),
        "predicted_over_2_5": match_level[f"{prefix}_predicted_over_2_5"].mean(),
        "actual_over_2_5": match_level["actual_over_2_5"].mean(),
        "predicted_over_3_5": match_level[f"{prefix}_predicted_over_3_5"].mean(),
        "actual_over_3_5": match_level["actual_over_3_5"].mean(),
        "predicted_btts": match_level[f"{prefix}_predicted_btts"].mean(),
        "actual_btts": match_level["actual_btts"].mean(),
        "predicted_favorite_scores_4_plus": match_level[f"{prefix}_p_favorite_scores_4_plus"].mean(),
        "actual_favorite_scores_4_plus": match_level[f"{prefix}_actual_favorite_scores_4_plus"].mean(),
        "predicted_favorite_scores_5_plus": match_level[f"{prefix}_p_favorite_scores_5_plus"].mean(),
        "actual_favorite_scores_5_plus": match_level[f"{prefix}_actual_favorite_scores_5_plus"].mean(),
        "predicted_margin_4_plus": match_level[f"{prefix}_p_margin_4_plus"].mean(),
        "actual_margin_4_plus": match_level[f"{prefix}_actual_margin_4_plus"].mean(),
        "predicted_total_goals_5_plus": match_level[f"{prefix}_p_total_goals_5_plus"].mean(),
        "actual_total_goals_5_plus": (match_level["actual_total_goals"] >= 5).mean(),
        "mean_actual_scoreline_log_probability": match_level[
            f"{prefix}_actual_scoreline_log_probability"
        ].mean(),
    }


def build_summary(match_level: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            summarize_model(match_level, "v20", "v20_stable_nb"),
            summarize_model(match_level, "v23", "v23_base_blowout_mixture"),
            summarize_model(match_level, "markov", "v3_markov_match_state"),
        ]
    )


def build_by_bucket(match_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket, subset in match_level.groupby("favorite_bucket", sort=False):
        for prefix, label in [
            ("v20", "v20_stable_nb"),
            ("v23", "v23_base_blowout_mixture"),
            ("markov", "v3_markov_match_state"),
        ]:
            row = summarize_model(subset, prefix, label)
            row["favorite_bucket"] = bucket
            rows.append(row)
    return pd.DataFrame(rows)


def build_tail_calibration(match_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket, subset in match_level.groupby("favorite_bucket", sort=False):
        rows.append(
            {
                "favorite_bucket": bucket,
                "sample_size": len(subset),
                "v20_favorite_scores_5_plus": subset["v20_p_favorite_scores_5_plus"].mean(),
                "v23_favorite_scores_5_plus": subset["v23_p_favorite_scores_5_plus"].mean(),
                "markov_favorite_scores_5_plus": subset[
                    "markov_favorite_scores_5_plus_probability"
                ].mean(),
                "actual_favorite_scores_5_plus": subset["actual_favorite_scores_5_plus"].mean(),
                "v20_margin_4_plus": subset["v20_p_margin_4_plus"].mean(),
                "v23_margin_4_plus": subset["v23_p_margin_4_plus"].mean(),
                "markov_margin_4_plus": subset["markov_margin_4_plus_probability"].mean(),
                "actual_margin_4_plus": subset["actual_margin_4_plus"].mean(),
                "v20_total_goals_5_plus": subset["v20_p_total_goals_5_plus"].mean(),
                "v23_total_goals_5_plus": subset["v23_p_total_goals_5_plus"].mean(),
                "markov_total_goals_5_plus": subset[
                    "markov_total_goals_5_plus_probability"
                ].mean(),
                "actual_total_goals_5_plus": subset["actual_total_goals_5_plus"].mean(),
            }
        )
    return pd.DataFrame(rows)


def build_state_diagnostics(match_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket, subset in match_level.groupby("favorite_bucket", sort=False):
        tail_event = (
            subset["actual_favorite_scores_4_plus"]
            | subset["actual_favorite_scores_5_plus"]
            | subset["actual_margin_4_plus"]
            | subset["actual_total_goals_5_plus"]
        )
        corr = (
            subset["markov_blowout_path_probability"].corr(tail_event.astype(float))
            if tail_event.nunique() > 1
            else pd.NA
        )
        rows.append(
            {
                "favorite_bucket": bucket,
                "sample_size": len(subset),
                "average_normal_visit_rate": subset["markov_normal_visit_rate"].mean(),
                "average_open_game_visit_rate": subset["markov_open_game_visit_rate"].mean(),
                "average_favorite_pressure_visit_rate": subset[
                    "markov_favorite_pressure_visit_rate"
                ].mean(),
                "average_blowout_visit_rate": subset["markov_blowout_visit_rate"].mean(),
                "average_blowout_path_probability": subset[
                    "markov_blowout_path_probability"
                ].mean(),
                "tail_event_rate": tail_event.mean(),
                "corr_blowout_path_probability_actual_tail_event": corr,
                "avg_blowout_path_if_tail_event": subset.loc[
                    tail_event, "markov_blowout_path_probability"
                ].mean(),
                "avg_blowout_path_if_no_tail_event": subset.loc[
                    ~tail_event, "markov_blowout_path_probability"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def save_outputs(match_level: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "match_level": output_dir / "markov_match_state__match_level.csv",
        "summary": output_dir / "markov_match_state__summary.csv",
        "by_favorite_bucket": output_dir / "markov_match_state__by_favorite_bucket.csv",
        "tail_calibration": output_dir / "markov_match_state__tail_calibration.csv",
        "state_diagnostics": output_dir / "markov_match_state__state_diagnostics.csv",
        "comparison": output_dir / "markov_match_state__comparison_vs_v20_v23.csv",
    }
    summary = build_summary(match_level)
    by_bucket = build_by_bucket(match_level)
    tail = build_tail_calibration(match_level)
    state = build_state_diagnostics(match_level)
    match_level.to_csv(outputs["match_level"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    by_bucket.to_csv(outputs["by_favorite_bucket"], index=False)
    tail.to_csv(outputs["tail_calibration"], index=False)
    state.to_csv(outputs["state_diagnostics"], index=False)
    summary.to_csv(outputs["comparison"], index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest research-only V3 Markov match-state scoreline simulator."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default="2014-01-01")
    parser.add_argument("--n-sims", type=int, default=2000)
    parser.add_argument("--step-minutes", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-goals", type=int, default=10)
    args = parser.parse_args()

    matches = load_matches(args.input, start_date=args.start_date)
    prediction_df, evaluation_df = split_prediction_and_evaluation_frames(matches)
    if len(prediction_df) != len(evaluation_df):
        raise ValueError("Prediction/evaluation frame length mismatch")

    model_config = load_model_config("default")
    base = build_predictions(
        matches=matches,
        model_config=model_config,
        max_goals=args.max_goals,
        dispersion_k=12.0,
        mixture_params=BlowoutMixtureParams(
            normal_k=12.0,
            blowout_k=6.0,
            blowout_lambda_multiplier=1.90,
            blowout_lambda_add=0.90,
            max_blowout_lambda_fav=6.2,
            underdog_blowout_lambda_multiplier=0.80,
            p_favorite_weight=0.28,
            p_imbalance_weight=0.10,
            p_rating_weight=0.04,
            p_multiplier=1.15,
        ),
    )
    match_level = add_markov_predictions(
        base=base,
        n_sims=args.n_sims,
        step_minutes=args.step_minutes,
        random_seed=args.random_seed,
        config=MatchStateConfig(max_goals=args.max_goals),
    )
    outputs = save_outputs(match_level, args.output_dir)
    summary = pd.read_csv(outputs["summary"])
    tail = pd.read_csv(outputs["tail_calibration"])
    state = pd.read_csv(outputs["state_diagnostics"])
    print(f"Backtested rows: {len(match_level)}")
    print(f"Markov simulations per match: {args.n_sims}")
    print("Summary:")
    print(summary.to_string(index=False))
    print("Tail calibration:")
    print(tail.to_string(index=False))
    print("State diagnostics:")
    print(state.to_string(index=False))
    print("Files:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
