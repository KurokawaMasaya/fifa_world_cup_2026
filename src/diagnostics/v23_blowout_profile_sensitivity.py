from __future__ import annotations

import argparse
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
    split_prediction_and_evaluation_frames,
)
from src.models.favorite_blowout_mixture_scoreline import BlowoutMixtureParams


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "diagnostics"
    / "scoreline_research"
    / "v23_favorite_blowout_mixture"
    / "profile_sensitivity"
)


PROFILES: dict[str, BlowoutMixtureParams] = {
    "v23_light": BlowoutMixtureParams(
        normal_k=12.0,
        blowout_k=7.5,
        blowout_lambda_multiplier=1.60,
        blowout_lambda_add=0.55,
        max_blowout_lambda_fav=5.6,
        underdog_blowout_lambda_multiplier=0.88,
        p_favorite_weight=0.18,
        p_imbalance_weight=0.06,
        p_rating_weight=0.02,
        p_multiplier=0.85,
    ),
    "v23_base": BlowoutMixtureParams(
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
    "v23_aggressive": BlowoutMixtureParams(
        normal_k=12.0,
        blowout_k=5.0,
        blowout_lambda_multiplier=2.20,
        blowout_lambda_add=1.10,
        max_blowout_lambda_fav=6.8,
        underdog_blowout_lambda_multiplier=0.74,
        p_favorite_weight=0.36,
        p_imbalance_weight=0.14,
        p_rating_weight=0.05,
        p_multiplier=1.35,
    ),
}


def summarize_profile(match_level: pd.DataFrame, profile: str, prefix: str = "v23") -> dict[str, object]:
    return {
        "profile": profile,
        "sample_size": len(match_level),
        "exact_top1": match_level[f"{prefix}_exact_scoreline_hit"].mean(),
        "top3": match_level[f"{prefix}_top3_scoreline_hit"].mean(),
        "top5": match_level[f"{prefix}_top5_scoreline_hit"].mean(),
        "winner_direction": match_level[f"{prefix}_winner_direction_correct"].mean(),
        "mean_predicted_goals": match_level[f"{prefix}_predicted_total_goals"].mean(),
        "pred_p_favorite_scores_4_plus": match_level[f"{prefix}_p_favorite_scores_4_plus"].mean(),
        "actual_favorite_scores_4_plus_rate": match_level[
            f"{prefix}_actual_favorite_scores_4_plus"
        ].mean(),
        "pred_p_favorite_scores_5_plus": match_level[f"{prefix}_p_favorite_scores_5_plus"].mean(),
        "actual_favorite_scores_5_plus_rate": match_level[
            f"{prefix}_actual_favorite_scores_5_plus"
        ].mean(),
        "pred_p_margin_4_plus": match_level[f"{prefix}_p_margin_4_plus"].mean(),
        "actual_margin_4_plus_rate": match_level[f"{prefix}_actual_margin_4_plus"].mean(),
        "pred_p_total_goals_5_plus": match_level[f"{prefix}_p_total_goals_5_plus"].mean(),
        "actual_total_goals_5_plus_rate": (match_level["actual_total_goals"] >= 5).mean(),
        "mean_p_blowout": 0.0 if prefix == "v20" else match_level["p_blowout"].mean(),
        "mean_tail_risk_index": match_level[f"{prefix}_tail_risk_index"].mean(),
    }


def summarize_profile_by_bucket(
    match_level: pd.DataFrame,
    profile: str,
    prefix: str = "v23",
) -> pd.DataFrame:
    rows = []
    for bucket, subset in match_level.groupby("favorite_bucket", sort=False):
        row = summarize_profile(subset, profile=profile, prefix=prefix)
        row["favorite_bucket"] = bucket
        rows.append(row)
    columns = ["profile", "favorite_bucket"] + [
        col for col in rows[0].keys() if col not in {"profile", "favorite_bucket"}
    ]
    return pd.DataFrame(rows)[columns]


def build_tail_examples(match_level: pd.DataFrame, profile: str, limit: int = 40) -> pd.DataFrame:
    examples = match_level.copy()
    examples["profile"] = profile
    examples["tail_delta_fav_5_plus_vs_v20"] = (
        examples["v23_p_favorite_scores_5_plus"] - examples["v20_p_favorite_scores_5_plus"]
    )
    examples["tail_delta_margin_4_plus_vs_v20"] = (
        examples["v23_p_margin_4_plus"] - examples["v20_p_margin_4_plus"]
    )
    cols = [
        "profile",
        "date",
        "team_a",
        "team_b",
        "favorite_team",
        "favorite_win_prob",
        "favorite_bucket",
        "p_blowout",
        "lambda_a",
        "lambda_b",
        "base_lambda_fav",
        "base_lambda_dog",
        "blowout_lambda_fav",
        "blowout_lambda_dog",
        "normal_top_scoreline",
        "mixture_top_scoreline",
        "top_5_scorelines",
        "top_5_probs",
        "v20_p_favorite_scores_5_plus",
        "v23_p_favorite_scores_5_plus",
        "tail_delta_fav_5_plus_vs_v20",
        "v20_p_margin_4_plus",
        "v23_p_margin_4_plus",
        "tail_delta_margin_4_plus_vs_v20",
        "actual_scoreline",
    ]
    return examples.sort_values(
        ["favorite_bucket", "tail_delta_fav_5_plus_vs_v20"],
        ascending=[True, False],
    )[cols].head(limit)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run V2.3 favorite blowout mixture profile sensitivity backtest."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default="2014-01-01")
    parser.add_argument("--max-goals", type=int, default=10)
    args = parser.parse_args()

    matches = load_matches(args.input, start_date=args.start_date)
    prediction_df, evaluation_df = split_prediction_and_evaluation_frames(matches)
    if len(prediction_df) != len(evaluation_df):
        raise ValueError("Prediction/evaluation frame length mismatch")

    model_config = load_model_config("default")
    summary_rows: list[dict[str, object]] = []
    by_bucket_frames: list[pd.DataFrame] = []
    example_frames: list[pd.DataFrame] = []

    baseline_written = False
    for profile, params in PROFILES.items():
        print(f"Running profile: {profile}")
        match_level = build_predictions(
            matches=matches,
            model_config=model_config,
            max_goals=args.max_goals,
            dispersion_k=params.normal_k,
            mixture_params=params,
        )
        if not baseline_written:
            summary_rows.append(summarize_profile(match_level, profile="v20_original", prefix="v20"))
            by_bucket_frames.append(
                summarize_profile_by_bucket(
                    match_level,
                    profile="v20_original",
                    prefix="v20",
                )
            )
            baseline_written = True
        summary_rows.append(summarize_profile(match_level, profile=profile))
        by_bucket_frames.append(summarize_profile_by_bucket(match_level, profile=profile))
        example_frames.append(build_tail_examples(match_level, profile=profile))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    by_bucket = pd.concat(by_bucket_frames, ignore_index=True)
    tail_calibration = by_bucket[
        [
            "profile",
            "favorite_bucket",
            "sample_size",
            "mean_p_blowout",
            "pred_p_favorite_scores_4_plus",
            "actual_favorite_scores_4_plus_rate",
            "pred_p_favorite_scores_5_plus",
            "actual_favorite_scores_5_plus_rate",
            "pred_p_margin_4_plus",
            "actual_margin_4_plus_rate",
            "pred_p_total_goals_5_plus",
            "actual_total_goals_5_plus_rate",
            "mean_tail_risk_index",
        ]
    ].copy()
    examples = pd.concat(example_frames, ignore_index=True)

    paths = {
        "summary": output_dir / "v23_profile_comparison_summary.csv",
        "by_bucket": output_dir / "v23_profile_comparison_by_bucket.csv",
        "tail_calibration": output_dir / "v23_tail_calibration_summary.csv",
        "examples": output_dir / "v23_tail_examples.csv",
    }
    summary.to_csv(paths["summary"], index=False)
    by_bucket.to_csv(paths["by_bucket"], index=False)
    tail_calibration.to_csv(paths["tail_calibration"], index=False)
    examples.to_csv(paths["examples"], index=False)

    print("Profile comparison summary:")
    print(summary.to_string(index=False))
    print("Tail calibration by bucket:")
    print(tail_calibration.to_string(index=False))
    print("Files:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
