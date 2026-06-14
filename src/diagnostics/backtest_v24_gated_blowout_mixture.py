from __future__ import annotations

import argparse
import json
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
from src.models.favorite_blowout_mixture_scoreline import (
    BlowoutMixtureParams,
    GatedBlowoutParams,
    gated_favorite_blowout_mixture_scoreline_grid,
    tail_metrics,
    top_scorelines,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "diagnostics"
    / "scoreline_research"
    / "v24_gated_blowout_mixture"
)


def base_params(**kwargs: object) -> GatedBlowoutParams:
    values = {
        "normal_k": 12.0,
        "blowout_k": 6.0,
        "blowout_lambda_multiplier": 1.90,
        "blowout_lambda_add": 0.90,
        "max_blowout_lambda_fav": 6.2,
        "underdog_blowout_lambda_multiplier": 0.80,
        "p_favorite_weight": 0.28,
        "p_imbalance_weight": 0.10,
        "p_rating_weight": 0.04,
        "p_multiplier": 1.15,
    }
    values.update(kwargs)
    return GatedBlowoutParams(**values)


PROFILES: dict[str, GatedBlowoutParams] = {
    "v24_gate_A_favorite_dominance_only": base_params(
        use_lambda_imbalance_gate=False,
        use_favorite_scoring_capacity_gate=False,
        use_underdog_suppression_gate=False,
        use_motivation_gate=False,
    ),
    "v24_gate_B_lambda_imbalance_only": base_params(
        use_favorite_dominance_gate=False,
        use_favorite_scoring_capacity_gate=False,
        use_underdog_suppression_gate=False,
        use_motivation_gate=False,
    ),
    "v24_gate_C_favorite_scoring_capacity_only": base_params(
        use_favorite_dominance_gate=False,
        use_lambda_imbalance_gate=False,
        use_underdog_suppression_gate=False,
        use_motivation_gate=False,
    ),
    "v24_gate_D_underdog_suppression_only": base_params(
        use_favorite_dominance_gate=False,
        use_lambda_imbalance_gate=False,
        use_favorite_scoring_capacity_gate=False,
        use_motivation_gate=False,
    ),
    "v24_gate_E_motivation_only": base_params(
        use_favorite_dominance_gate=False,
        use_lambda_imbalance_gate=False,
        use_favorite_scoring_capacity_gate=False,
        use_underdog_suppression_gate=False,
    ),
    "v24_gate_no_E_ABCD": base_params(use_motivation_gate=False),
    "v24_gate_full_ABCDE": base_params(),
    "v24_gate_light": base_params(blowout_k_factor=0.75),
    "v24_gate_base": base_params(blowout_k_factor=1.00),
    "v24_gate_aggressive": base_params(blowout_k_factor=1.25),
}


def parse_scoreline(scoreline: str) -> tuple[int, int]:
    goals_a, goals_b = scoreline.split("-")
    return int(goals_a), int(goals_b)


def implied_result(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "team_a_win"
    if goals_a < goals_b:
        return "team_b_win"
    return "draw"


def summarize_grid_for_match(row: pd.Series, profile: str, params: GatedBlowoutParams) -> dict[str, object]:
    grid, metadata = gated_favorite_blowout_mixture_scoreline_grid(
        lambda_a=float(row["lambda_a"]),
        lambda_b=float(row["lambda_b"]),
        p_team_a_win=float(row["p_team_a_win"]),
        p_draw=float(row["p_draw"]),
        p_team_b_win=float(row["p_team_b_win"]),
        rating_gap=float(row["strength_diff"]),
        params=params,
        max_goals=10,
    )
    top5 = top_scorelines(grid, top_n=5)
    top_scoreline = str(top5[0]["scoreline"])
    pred_a, pred_b = parse_scoreline(top_scoreline)
    actual_a = int(row["actual_goals_a"])
    actual_b = int(row["actual_goals_b"])
    actual_scoreline = str(row["actual_scoreline"])
    actual_result = implied_result(actual_a, actual_b)
    predicted_result = implied_result(pred_a, pred_b)
    favorite_side = str(metadata["favorite_side"])
    tail = tail_metrics(grid, favorite_side=favorite_side)
    fav_actual_goals = actual_a if favorite_side == "team_a" else actual_b
    dog_actual_goals = actual_b if favorite_side == "team_a" else actual_a
    fav_margin_actual = fav_actual_goals - dog_actual_goals
    pred_goal_diff = pred_a - pred_b
    actual_goal_diff = actual_a - actual_b
    return {
        "profile": profile,
        "match_id": row["match_id"],
        "date": row["date"],
        "team_a": row["team_a"],
        "team_b": row["team_b"],
        "favorite_team": row["favorite_team"],
        "favorite_bucket": metadata["favorite_bucket"],
        "favorite_win_prob": metadata["favorite_win_prob"],
        "base_lambda_fav": metadata["base_lambda_fav"],
        "base_lambda_dog": metadata["base_lambda_dog"],
        "p_blowout_raw": metadata["p_blowout_raw"],
        "blowout_gate": metadata["blowout_gate"],
        "blowout_k": metadata["blowout_k"],
        "p_blowout_final": metadata["p_blowout_final"],
        "bucket_cap": metadata["bucket_cap"],
        "favorite_dominance_gate": metadata["favorite_dominance_gate"],
        "lambda_imbalance_gate": metadata["lambda_imbalance_gate"],
        "favorite_scoring_capacity_gate": metadata["favorite_scoring_capacity_gate"],
        "underdog_suppression_gate": metadata["underdog_suppression_gate"],
        "motivation_gate": metadata["motivation_gate"],
        "gate_suppression_reason": metadata["gate_suppression_reason"],
        "predicted_scoreline": top_scoreline,
        "top_3_scorelines": json.dumps([item["scoreline"] for item in top5[:3]]),
        "top_5_scorelines": json.dumps([item["scoreline"] for item in top5]),
        "top_5_probs": json.dumps([item["probability"] for item in top5]),
        "exact_top1_hit": top_scoreline == actual_scoreline,
        "top3_hit": actual_scoreline in [item["scoreline"] for item in top5[:3]],
        "top5_hit": actual_scoreline in [item["scoreline"] for item in top5],
        "winner_direction_correct": predicted_result == actual_result,
        "margin_bucket_accuracy": row["margin_bucket_actual"] == (
            "draw"
            if pred_goal_diff == 0
            else "one_goal_win"
            if abs(pred_goal_diff) == 1
            else "two_goal_win"
            if abs(pred_goal_diff) == 2
            else "three_goal_win"
            if abs(pred_goal_diff) == 3
            else "four_plus_goal_win"
        ),
        "mean_goals_pred": pred_a + pred_b,
        "mean_goals_actual": actual_a + actual_b,
        "over_2_5_pred": pred_a + pred_b >= 3,
        "over_2_5_actual": actual_a + actual_b >= 3,
        "over_3_5_pred": pred_a + pred_b >= 4,
        "over_3_5_actual": actual_a + actual_b >= 4,
        "total_goals_5_plus_pred": tail["p_total_goals_5_plus"],
        "total_goals_5_plus_actual": actual_a + actual_b >= 5,
        "favorite_scores_4_plus_pred": tail["p_favorite_scores_4_plus"],
        "favorite_scores_4_plus_actual": fav_actual_goals >= 4,
        "favorite_scores_5_plus_pred": tail["p_favorite_scores_5_plus"],
        "favorite_scores_5_plus_actual": fav_actual_goals >= 5,
        "margin_4_plus_pred": tail["p_margin_4_plus"],
        "margin_4_plus_actual": fav_margin_actual >= 4,
        "margin_5_plus_pred": tail["p_margin_5_plus"],
        "margin_5_plus_actual": fav_margin_actual >= 5,
    }


def summarize_profile(df: pd.DataFrame, profile: str) -> dict[str, object]:
    fav5_err = abs(df["favorite_scores_5_plus_pred"].mean() - df["favorite_scores_5_plus_actual"].mean())
    margin4_err = abs(df["margin_4_plus_pred"].mean() - df["margin_4_plus_actual"].mean())
    total5_err = abs(df["total_goals_5_plus_pred"].mean() - df["total_goals_5_plus_actual"].mean())
    over25_err = abs(df["over_2_5_pred"].mean() - df["over_2_5_actual"].mean())
    combined_tail = (fav5_err + margin4_err + total5_err) / 3.0
    return {
        "profile": profile,
        "sample_size": len(df),
        "exact_top1": df["exact_top1_hit"].mean(),
        "top3": df["top3_hit"].mean(),
        "top5": df["top5_hit"].mean(),
        "winner_direction": df["winner_direction_correct"].mean(),
        "margin_bucket_accuracy": df["margin_bucket_accuracy"].mean(),
        "mean_goals_pred": df["mean_goals_pred"].mean(),
        "mean_goals_actual": df["mean_goals_actual"].mean(),
        "over_2_5_pred": df["over_2_5_pred"].mean(),
        "over_2_5_actual": df["over_2_5_actual"].mean(),
        "over_3_5_pred": df["over_3_5_pred"].mean(),
        "over_3_5_actual": df["over_3_5_actual"].mean(),
        "total_goals_5_plus_pred": df["total_goals_5_plus_pred"].mean(),
        "total_goals_5_plus_actual": df["total_goals_5_plus_actual"].mean(),
        "favorite_scores_4_plus_pred": df["favorite_scores_4_plus_pred"].mean(),
        "favorite_scores_4_plus_actual": df["favorite_scores_4_plus_actual"].mean(),
        "favorite_scores_5_plus_pred": df["favorite_scores_5_plus_pred"].mean(),
        "favorite_scores_5_plus_actual": df["favorite_scores_5_plus_actual"].mean(),
        "margin_4_plus_pred": df["margin_4_plus_pred"].mean(),
        "margin_4_plus_actual": df["margin_4_plus_actual"].mean(),
        "margin_5_plus_pred": df["margin_5_plus_pred"].mean(),
        "margin_5_plus_actual": df["margin_5_plus_actual"].mean(),
        "fav5_abs_error": fav5_err,
        "margin4_abs_error": margin4_err,
        "total5_abs_error": total5_err,
        "over25_abs_error": over25_err,
        "combined_tail_abs_error": combined_tail,
        "avg_p_blowout_raw": df["p_blowout_raw"].mean(),
        "avg_blowout_gate": df["blowout_gate"].mean(),
        "avg_p_blowout_final": df["p_blowout_final"].mean(),
        "pct_p_blowout_final_gt_0_01": (df["p_blowout_final"] > 0.01).mean(),
        "pct_p_blowout_final_gt_0_05": (df["p_blowout_final"] > 0.05).mean(),
        "pct_p_blowout_final_gt_0_10": (df["p_blowout_final"] > 0.10).mean(),
    }


def summarize_baseline(base: pd.DataFrame, prefix: str, profile: str) -> dict[str, object]:
    def actual_col(name: str) -> str:
        return f"{prefix}_actual_{name}" if f"{prefix}_actual_{name}" in base.columns else f"{name}_actual"

    fav5_err = abs(base[f"{prefix}_p_favorite_scores_5_plus"].mean() - base[f"{prefix}_actual_favorite_scores_5_plus"].mean())
    margin4_err = abs(base[f"{prefix}_p_margin_4_plus"].mean() - base[f"{prefix}_actual_margin_4_plus"].mean())
    total5_err = abs(base[f"{prefix}_p_total_goals_5_plus"].mean() - (base["actual_total_goals"] >= 5).mean())
    over25_err = abs(base[f"{prefix}_predicted_over_2_5"].mean() - base["actual_over_2_5"].mean())
    return {
        "profile": profile,
        "sample_size": len(base),
        "exact_top1": base[f"{prefix}_exact_scoreline_hit"].mean(),
        "top3": base[f"{prefix}_top3_scoreline_hit"].mean(),
        "top5": base[f"{prefix}_top5_scoreline_hit"].mean(),
        "winner_direction": base[f"{prefix}_winner_direction_correct"].mean(),
        "margin_bucket_accuracy": base[f"{prefix}_margin_bucket_correct"].mean(),
        "mean_goals_pred": base[f"{prefix}_predicted_total_goals"].mean(),
        "mean_goals_actual": base["actual_total_goals"].mean(),
        "over_2_5_pred": base[f"{prefix}_predicted_over_2_5"].mean(),
        "over_2_5_actual": base["actual_over_2_5"].mean(),
        "over_3_5_pred": base[f"{prefix}_predicted_over_3_5"].mean(),
        "over_3_5_actual": base["actual_over_3_5"].mean(),
        "total_goals_5_plus_pred": base[f"{prefix}_p_total_goals_5_plus"].mean(),
        "total_goals_5_plus_actual": (base["actual_total_goals"] >= 5).mean(),
        "favorite_scores_4_plus_pred": base[f"{prefix}_p_favorite_scores_4_plus"].mean(),
        "favorite_scores_4_plus_actual": base[f"{prefix}_actual_favorite_scores_4_plus"].mean(),
        "favorite_scores_5_plus_pred": base[f"{prefix}_p_favorite_scores_5_plus"].mean(),
        "favorite_scores_5_plus_actual": base[f"{prefix}_actual_favorite_scores_5_plus"].mean(),
        "margin_4_plus_pred": base[f"{prefix}_p_margin_4_plus"].mean(),
        "margin_4_plus_actual": base[f"{prefix}_actual_margin_4_plus"].mean(),
        "margin_5_plus_pred": base[f"{prefix}_p_margin_5_plus"].mean(),
        "margin_5_plus_actual": base[f"{prefix}_actual_margin_5_plus"].mean(),
        "fav5_abs_error": fav5_err,
        "margin4_abs_error": margin4_err,
        "total5_abs_error": total5_err,
        "over25_abs_error": over25_err,
        "combined_tail_abs_error": (fav5_err + margin4_err + total5_err) / 3.0,
        "avg_p_blowout_raw": 0.0 if prefix == "v20" else base["p_blowout"].mean(),
        "avg_blowout_gate": 1.0 if prefix == "v23" else 0.0,
        "avg_p_blowout_final": 0.0 if prefix == "v20" else base["p_blowout"].mean(),
        "pct_p_blowout_final_gt_0_01": 0.0 if prefix == "v20" else (base["p_blowout"] > 0.01).mean(),
        "pct_p_blowout_final_gt_0_05": 0.0 if prefix == "v20" else (base["p_blowout"] > 0.05).mean(),
        "pct_p_blowout_final_gt_0_10": 0.0 if prefix == "v20" else (base["p_blowout"] > 0.10).mean(),
    }


def add_recommendations(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    recs = []
    for _, row in out.iterrows():
        if row["profile"] in {"v20_stable", "v23_base"}:
            recs.append("baseline")
        elif row["top5"] < 0.510 or row["winner_direction"] < 0.495:
            recs.append("reject")
        elif row["combined_tail_abs_error"] <= 0.025 and row["over25_abs_error"] < 0.48:
            recs.append("shadow_candidate")
        elif row["combined_tail_abs_error"] <= 0.04:
            recs.append("research_only")
        else:
            recs.append("reject")
    out["recommendation"] = recs
    if "v24_gate_E_motivation_only" in set(out["profile"]):
        out.loc[
            out["profile"].eq("v24_gate_E_motivation_only"),
            "recommendation",
        ] = "reject_non_informative_without_real_motivation_data"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest V2.4 gated blowout mixture profiles.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default="2014-01-01")
    args = parser.parse_args()

    matches = load_matches(args.input, start_date=args.start_date)
    prediction_df, evaluation_df = split_prediction_and_evaluation_frames(matches)
    if len(prediction_df) != len(evaluation_df):
        raise ValueError("Prediction/evaluation frame length mismatch")
    model_config = load_model_config("default")
    base = build_predictions(
        matches=matches,
        model_config=model_config,
        max_goals=10,
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

    profile_frames = []
    for profile, params in PROFILES.items():
        print(f"Running {profile}")
        rows = [summarize_grid_for_match(row, profile, params) for _, row in base.iterrows()]
        profile_frames.append(pd.DataFrame(rows))
    match_level = pd.concat(profile_frames, ignore_index=True)

    summary_rows = [
        summarize_baseline(base, "v20", "v20_stable"),
        summarize_baseline(base, "v23", "v23_base"),
    ]
    for profile, subset in match_level.groupby("profile", sort=False):
        summary_rows.append(summarize_profile(subset, profile))
    summary = add_recommendations(pd.DataFrame(summary_rows))

    by_bucket_rows = []
    tail_rows = []
    gate_rows = []
    for profile, subset in match_level.groupby("profile", sort=False):
        for bucket, bucket_df in subset.groupby("favorite_bucket", sort=False):
            row = summarize_profile(bucket_df, profile)
            row["favorite_bucket"] = bucket
            by_bucket_rows.append(row)
            tail_rows.append(
                {
                    "profile": profile,
                    "favorite_bucket": bucket,
                    "sample_size": len(bucket_df),
                    "favorite_scores_5_plus_pred": bucket_df["favorite_scores_5_plus_pred"].mean(),
                    "favorite_scores_5_plus_actual": bucket_df["favorite_scores_5_plus_actual"].mean(),
                    "margin_4_plus_pred": bucket_df["margin_4_plus_pred"].mean(),
                    "margin_4_plus_actual": bucket_df["margin_4_plus_actual"].mean(),
                    "total_goals_5_plus_pred": bucket_df["total_goals_5_plus_pred"].mean(),
                    "total_goals_5_plus_actual": bucket_df["total_goals_5_plus_actual"].mean(),
                }
            )
        gate_rows.append(
            {
                "profile": profile,
                "avg_p_blowout_raw": subset["p_blowout_raw"].mean(),
                "avg_blowout_gate": subset["blowout_gate"].mean(),
                "avg_p_blowout_final": subset["p_blowout_final"].mean(),
                "pct_p_blowout_final_gt_0_01": (subset["p_blowout_final"] > 0.01).mean(),
                "pct_p_blowout_final_gt_0_05": (subset["p_blowout_final"] > 0.05).mean(),
                "pct_p_blowout_final_gt_0_10": (subset["p_blowout_final"] > 0.10).mean(),
                "motivation_note": "motivation_gate is neutral; no real motivation data available",
            }
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "v24_profile_comparison_summary.csv",
        "by_bucket": output_dir / "v24_profile_comparison_by_bucket.csv",
        "match_level": output_dir / "v24_gate_diagnostics_match_level.csv",
        "tail": output_dir / "v24_tail_calibration.csv",
        "examples": output_dir / "v24_gate_examples.csv",
        "ablation_summary": output_dir / "v24_ablation_comparison_summary.csv",
        "ablation_by_bucket": output_dir / "v24_ablation_comparison_by_bucket.csv",
        "ablation_tail": output_dir / "v24_ablation_tail_calibration.csv",
        "ablation_gate": output_dir / "v24_ablation_gate_diagnostics.csv",
    }
    by_bucket = pd.DataFrame(by_bucket_rows)
    tail = pd.DataFrame(tail_rows)
    gates = pd.DataFrame(gate_rows)
    examples = match_level.sort_values("p_blowout_final", ascending=False).head(80)
    summary.to_csv(paths["summary"], index=False)
    by_bucket.to_csv(paths["by_bucket"], index=False)
    match_level.to_csv(paths["match_level"], index=False)
    tail.to_csv(paths["tail"], index=False)
    examples.to_csv(paths["examples"], index=False)
    summary.to_csv(paths["ablation_summary"], index=False)
    by_bucket.to_csv(paths["ablation_by_bucket"], index=False)
    tail.to_csv(paths["ablation_tail"], index=False)
    gates.to_csv(paths["ablation_gate"], index=False)

    print("V2.4 summary:")
    print(summary.to_string(index=False))
    print("Files:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
