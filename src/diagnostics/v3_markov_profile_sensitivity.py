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
from src.diagnostics.backtest_markov_match_state_scoreline import (
    add_markov_predictions,
    summarize_model,
)
from src.models.favorite_blowout_mixture_scoreline import BlowoutMixtureParams
from src.models.markov_match_state_scoreline import MatchStateConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "diagnostics"
    / "scoreline_research"
    / "v3_markov_match_state"
    / "profile_sensitivity"
)


PROFILES: dict[str, MatchStateConfig] = {
    "markov_base_current": MatchStateConfig(),
    "markov_light": MatchStateConfig(
        favorite_goal_multipliers={
            "normal": 1.00,
            "open_game": 1.08,
            "favorite_pressure": 1.22,
            "blowout": 1.65,
        },
        underdog_goal_multipliers={
            "normal": 1.00,
            "open_game": 1.08,
            "favorite_pressure": 0.94,
            "blowout": 0.78,
        },
        transition_open_scale=0.70,
        transition_pressure_scale=0.85,
        transition_blowout_scale=0.65,
        late_control_strength=0.35,
        underdog_resistance_strength=0.25,
    ),
    "markov_conservative": MatchStateConfig(
        favorite_goal_multipliers={
            "normal": 1.00,
            "open_game": 1.05,
            "favorite_pressure": 1.15,
            "blowout": 1.45,
        },
        underdog_goal_multipliers={
            "normal": 1.00,
            "open_game": 1.05,
            "favorite_pressure": 0.95,
            "blowout": 0.80,
        },
        initial_pressure_by_bucket={
            "balanced": 0.00,
            "slight_favorite": 0.005,
            "clear_favorite": 0.015,
            "heavy_favorite": 0.04,
            "extreme_mismatch": 0.08,
        },
        initial_open_by_bucket={
            "balanced": 0.025,
            "slight_favorite": 0.030,
            "clear_favorite": 0.035,
            "heavy_favorite": 0.040,
            "extreme_mismatch": 0.040,
        },
        transition_open_scale=0.45,
        transition_pressure_scale=0.65,
        transition_blowout_scale=0.35,
        late_control_strength=0.55,
        underdog_resistance_strength=0.45,
    ),
    "markov_tail_only": MatchStateConfig(
        favorite_goal_multipliers={
            "normal": 1.00,
            "open_game": 1.01,
            "favorite_pressure": 1.08,
            "blowout": 1.80,
        },
        underdog_goal_multipliers={
            "normal": 1.00,
            "open_game": 1.01,
            "favorite_pressure": 0.97,
            "blowout": 0.76,
        },
        initial_pressure_by_bucket={
            "balanced": 0.00,
            "slight_favorite": 0.00,
            "clear_favorite": 0.01,
            "heavy_favorite": 0.035,
            "extreme_mismatch": 0.075,
        },
        initial_open_by_bucket={
            "balanced": 0.015,
            "slight_favorite": 0.018,
            "clear_favorite": 0.020,
            "heavy_favorite": 0.020,
            "extreme_mismatch": 0.020,
        },
        transition_open_scale=0.20,
        transition_pressure_scale=0.45,
        transition_blowout_scale=0.55,
        late_control_strength=0.70,
        underdog_resistance_strength=0.60,
    ),
}


def summary_row(match_level: pd.DataFrame, profile: str) -> dict[str, object]:
    row = summarize_model(match_level, "markov", profile)
    row["profile"] = profile
    return row


def by_bucket_rows(match_level: pd.DataFrame, profile: str) -> list[dict[str, object]]:
    rows = []
    for bucket, subset in match_level.groupby("favorite_bucket", sort=False):
        row = summary_row(subset, profile)
        row["favorite_bucket"] = bucket
        rows.append(row)
    return rows


def state_rows(match_level: pd.DataFrame, profile: str) -> list[dict[str, object]]:
    rows = []
    for bucket, subset in match_level.groupby("favorite_bucket", sort=False):
        rows.append(
            {
                "profile": profile,
                "favorite_bucket": bucket,
                "sample_size": len(subset),
                "normal_visit_rate": subset["markov_normal_visit_rate"].mean(),
                "open_game_visit_rate": subset["markov_open_game_visit_rate"].mean(),
                "favorite_pressure_visit_rate": subset[
                    "markov_favorite_pressure_visit_rate"
                ].mean(),
                "blowout_visit_rate": subset["markov_blowout_visit_rate"].mean(),
                "blowout_path_probability": subset[
                    "markov_blowout_path_probability"
                ].mean(),
            }
        )
    return rows


def tail_rows(match_level: pd.DataFrame, profile: str) -> list[dict[str, object]]:
    rows = []
    for bucket, subset in match_level.groupby("favorite_bucket", sort=False):
        rows.append(
            {
                "profile": profile,
                "favorite_bucket": bucket,
                "sample_size": len(subset),
                "favorite_scores_5_plus_pred": subset[
                    "markov_favorite_scores_5_plus_probability"
                ].mean(),
                "favorite_scores_5_plus_actual": subset[
                    "actual_favorite_scores_5_plus"
                ].mean(),
                "margin_4_plus_pred": subset["markov_margin_4_plus_probability"].mean(),
                "margin_4_plus_actual": subset["actual_margin_4_plus"].mean(),
                "total_goals_5_plus_pred": subset[
                    "markov_total_goals_5_plus_probability"
                ].mean(),
                "total_goals_5_plus_actual": subset["actual_total_goals_5_plus"].mean(),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run conservative V3 Markov profile sensitivity diagnostics."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default="2014-01-01")
    parser.add_argument("--n-sims", type=int, default=1500)
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

    summary = [
        {**summarize_model(base, "v20", "v20_stable_nb"), "profile": "v20_stable_nb"},
        {
            **summarize_model(base, "v23", "v23_base_blowout_mixture"),
            "profile": "v23_base_blowout_mixture",
        },
    ]
    by_bucket = []
    state = []
    tail = []

    for profile, config in PROFILES.items():
        print(f"Running {profile}")
        markov = add_markov_predictions(
            base=base,
            n_sims=args.n_sims,
            step_minutes=args.step_minutes,
            random_seed=args.random_seed,
            config=config,
        )
        summary.append(summary_row(markov, profile))
        by_bucket.extend(by_bucket_rows(markov, profile))
        state.extend(state_rows(markov, profile))
        tail.extend(tail_rows(markov, profile))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary)
    by_bucket_df = pd.DataFrame(by_bucket)
    state_df = pd.DataFrame(state)
    tail_df = pd.DataFrame(tail)
    paths = {
        "summary": output_dir / "markov_profile_comparison_summary.csv",
        "by_bucket": output_dir / "markov_profile_comparison_by_bucket.csv",
        "state": output_dir / "markov_state_visit_rates.csv",
        "tail": output_dir / "markov_tail_calibration.csv",
    }
    summary_df.to_csv(paths["summary"], index=False)
    by_bucket_df.to_csv(paths["by_bucket"], index=False)
    state_df.to_csv(paths["state"], index=False)
    tail_df.to_csv(paths["tail"], index=False)
    print("Profile summary:")
    print(summary_df.to_string(index=False))
    print("Files:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
