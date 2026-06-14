from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

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
from src.diagnostics.backtest_v24_gated_blowout_mixture import (
    PROFILES as V24_PROFILES,
    summarize_baseline,
    summarize_grid_for_match,
    summarize_profile,
)
from src.models.favorite_blowout_mixture_scoreline import (
    BlowoutMixtureParams,
    tail_metrics,
    top_scorelines,
)
from src.models.learned_gated_blowout_mixture import (
    FEATURE_COLUMNS,
    build_composite_tail_target,
    build_learned_gate_features,
    fit_logistic_gate,
    gate_auc,
    learned_bucket_capped_probability,
    learned_gated_scoreline_grid,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "diagnostics"
    / "scoreline_research"
    / "v25_learned_gated_blowout_mixture"
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


def actual_rank(grid: dict[str, float], actual_scoreline: str) -> int | None:
    for idx, (scoreline, _) in enumerate(
        sorted(grid.items(), key=lambda item: (-item[1], item[0])),
        start=1,
    ):
        if scoreline == actual_scoreline:
            return idx
    return None


def summarize_v25_match(row: pd.Series, profile: str, p_final: float, raw: float, logit: float, global_k: float) -> dict[str, object]:
    grid, metadata = learned_gated_scoreline_grid(row, p_blowout_final=p_final, max_goals=10)
    top5 = top_scorelines(grid, top_n=5)
    pred_scoreline = str(top5[0]["scoreline"])
    pred_a, pred_b = parse_scoreline(pred_scoreline)
    actual_a = int(row["actual_goals_a"])
    actual_b = int(row["actual_goals_b"])
    actual_scoreline = str(row["actual_scoreline"])
    favorite_side = str(metadata["favorite_side"])
    fav_actual = actual_a if favorite_side == "team_a" else actual_b
    dog_actual = actual_b if favorite_side == "team_a" else actual_a
    fav_margin_actual = fav_actual - dog_actual
    pred_result = implied_result(pred_a, pred_b)
    actual_result = implied_result(actual_a, actual_b)
    tail = tail_metrics(grid, favorite_side=favorite_side)
    p_actual = max(EPSILON, float(grid.get(actual_scoreline, 0.0)))
    return {
        "profile": profile,
        "match_id": row["match_id"],
        "date": row["date"],
        "team_a": row["team_a"],
        "team_b": row["team_b"],
        "favorite_team": row["favorite_team"],
        "underdog_team": row["team_b"] if row["favorite_team"] == row["team_a"] else row["team_a"],
        "favorite_win_prob": row["favorite_win_prob"],
        "lambda_favorite": row["base_lambda_fav"],
        "lambda_underdog": row["base_lambda_dog"],
        "lambda_imbalance": abs(row["lambda_a"] - row["lambda_b"]) / max(row["lambda_a"] + row["lambda_b"], 1e-9),
        "favorite_bucket": metadata["favorite_bucket"],
        "p_blowout_v23": row["p_blowout"],
        "p_blowout_v24_ABCD": row.get("p_blowout_v24_ABCD", pd.NA),
        "p_blowout_v25_raw": raw,
        "p_blowout_v25_final": p_final,
        "bucket_cap": metadata["bucket_cap"],
        "global_k": global_k,
        "learned_gate_logit": logit,
        "top_5_scorelines": json.dumps([item["scoreline"] for item in top5]),
        "top_5_probs": json.dumps([item["probability"] for item in top5]),
        "predicted_scoreline": pred_scoreline,
        "actual_scoreline": actual_scoreline,
        "actual_scoreline_rank": actual_rank(grid, actual_scoreline),
        "log_probability_of_actual_scoreline": math.log(p_actual),
        "actual_tail_event": bool(row["composite_blowout_tail_actual"]),
        "composite_blowout_tail_actual": bool(row["composite_blowout_tail_actual"]),
        "composite_blowout_tail_pred": p_final,
        "favorite_scores_5_plus_actual": fav_actual >= 5,
        "margin_4_plus_actual": fav_margin_actual >= 4,
        "margin_5_plus_actual": fav_margin_actual >= 5,
        "total_goals_5_plus_actual": actual_a + actual_b >= 5,
        "exact_top1_hit": pred_scoreline == actual_scoreline,
        "top3_hit": actual_scoreline in [item["scoreline"] for item in top5[:3]],
        "top5_hit": actual_scoreline in [item["scoreline"] for item in top5],
        "winner_direction": pred_result == actual_result,
        "margin_bucket_accuracy": margin_bucket(pred_a - pred_b) == row["margin_bucket_actual"],
        "mean_goals_pred": pred_a + pred_b,
        "mean_goals_actual": actual_a + actual_b,
        "over_2_5_pred": pred_a + pred_b >= 3,
        "over_2_5_actual": actual_a + actual_b >= 3,
        "over_3_5_pred": pred_a + pred_b >= 4,
        "over_3_5_actual": actual_a + actual_b >= 4,
        "total_goals_5_plus_pred": tail["p_total_goals_5_plus"],
        "favorite_scores_4_plus_pred": tail["p_favorite_scores_4_plus"],
        "favorite_scores_4_plus_actual": fav_actual >= 4,
        "favorite_scores_5_plus_pred": tail["p_favorite_scores_5_plus"],
        "margin_4_plus_pred": tail["p_margin_4_plus"],
        "margin_5_plus_pred": tail["p_margin_5_plus"],
    }


def summarize_v25(df: pd.DataFrame, profile: str) -> dict[str, object]:
    fav5_err = abs(df["favorite_scores_5_plus_pred"].mean() - df["favorite_scores_5_plus_actual"].mean())
    margin4_err = abs(df["margin_4_plus_pred"].mean() - df["margin_4_plus_actual"].mean())
    total5_err = abs(df["total_goals_5_plus_pred"].mean() - df["total_goals_5_plus_actual"].mean())
    over25_err = abs(df["over_2_5_pred"].mean() - df["over_2_5_actual"].mean())
    y = df["composite_blowout_tail_actual"].astype(int)
    p = df["p_blowout_v25_final"].astype(float)
    auc = roc_auc_score(y, p) if y.nunique() > 1 else pd.NA
    return {
        "profile": profile,
        "sample_size": len(df),
        "exact_top1": df["exact_top1_hit"].mean(),
        "top3": df["top3_hit"].mean(),
        "top5": df["top5_hit"].mean(),
        "winner_direction": df["winner_direction"].mean(),
        "margin_bucket_accuracy": df["margin_bucket_accuracy"].mean(),
        "actual_scoreline_rank_mean": df["actual_scoreline_rank"].mean(),
        "log_probability_of_actual_scoreline": df["log_probability_of_actual_scoreline"].mean(),
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
        "composite_blowout_tail_pred": df["composite_blowout_tail_pred"].mean(),
        "composite_blowout_tail_actual": y.mean(),
        "gate_auc_composite": auc,
        "gate_brier_composite": brier_score_loss(y, p),
        "fav5_abs_error": fav5_err,
        "margin4_abs_error": margin4_err,
        "total5_abs_error": total5_err,
        "over25_abs_error": over25_err,
        "combined_tail_abs_error": (fav5_err + margin4_err + total5_err) / 3.0,
        "avg_p_blowout_final": p.mean(),
        "pct_p_blowout_final_gt_0_01": (p > 0.01).mean(),
        "pct_p_blowout_final_gt_0_05": (p > 0.05).mean(),
        "pct_p_blowout_final_gt_0_10": (p > 0.10).mean(),
    }


def add_recommendation(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    recs = []
    for _, row in out.iterrows():
        profile = row["profile"]
        if profile in {"v20_stable", "v23_base", "v24_gate_A_favorite_dominance_only", "v24_gate_no_E_ABCD"}:
            recs.append("baseline")
        elif row["top5"] >= 0.515 and row["combined_tail_abs_error"] <= 0.025 and row["gate_auc_composite"] >= 0.55:
            recs.append("shadow_candidate")
        elif row["top5"] >= 0.510 and row["gate_auc_composite"] >= 0.52:
            recs.append("research_only")
        else:
            recs.append("reject")
    out["recommendation"] = recs
    return out


def calibration_deciles(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for profile, subset in df.groupby("profile", sort=False):
        ranked = subset.copy()
        ranked["decile"] = pd.qcut(
            ranked["p_blowout_v25_final"].rank(method="first"),
            10,
            labels=False,
        ) + 1
        for decile, decile_df in ranked.groupby("decile"):
            rows.append(
                {
                    "profile": profile,
                    "decile": int(decile),
                    "n_matches": len(decile_df),
                    "avg_p_blowout_final": decile_df["p_blowout_v25_final"].mean(),
                    "actual_composite_tail_rate": decile_df["composite_blowout_tail_actual"].mean(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest V2.5 learned gated blowout mixture.")
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
    base["date"] = pd.to_datetime(base["date"])
    base["composite_blowout_tail_actual"] = build_composite_tail_target(base)
    features = build_learned_gate_features(base)
    train_mask = base["date"] < pd.Timestamp("2022-01-01")
    val_mask = (base["date"] >= pd.Timestamp("2022-01-01")) & (base["date"] <= pd.Timestamp("2023-12-31"))
    test_mask = base["date"] >= pd.Timestamp("2024-01-01")
    if not test_mask.any():
        raise ValueError("No 2024+ test rows available")

    gate = fit_logistic_gate(
        features.loc[train_mask],
        base.loc[train_mask, "composite_blowout_tail_actual"],
        global_k=1.0,
    )

    test = base.loc[test_mask].copy()
    test_features = features.loc[test_mask].copy()
    profile_frames = []
    for profile, k in [
        ("v25_logit_gate_k075", 0.75),
        ("v25_logit_gate_k100", 1.00),
        ("v25_logit_gate_k125", 1.25),
        ("v25_logit_gate_simple", 1.00),
    ]:
        active_gate = gate if profile != "v25_logit_gate_simple" else fit_logistic_gate(
            features.loc[train_mask],
            base.loc[train_mask, "composite_blowout_tail_actual"],
            feature_columns=["favorite_win_prob_feature", "lambda_imbalance_feature", "favorite_scoring_capacity_feature"],
            global_k=1.0,
        )
        probs = learned_bucket_capped_probability(
            active_gate,
            test_features,
            test["favorite_bucket"],
        )
        probs["p_blowout_v25_final"] = (
            probs["p_blowout_v25_final"] * k / active_gate.global_k
        ).clip(upper=probs["bucket_cap"])
        rows = []
        for idx, row in test.iterrows():
            rows.append(
                summarize_v25_match(
                    row,
                    profile=profile,
                    p_final=float(probs.loc[idx, "p_blowout_v25_final"]),
                    raw=float(probs.loc[idx, "p_blowout_v25_raw"]),
                    logit=float(probs.loc[idx, "learned_gate_logit"]),
                    global_k=k,
                )
            )
        profile_frames.append(pd.DataFrame(rows))
    match_level = pd.concat(profile_frames, ignore_index=True)

    # Baselines on the same 2024+ test slice.
    v24_a = pd.DataFrame([summarize_grid_for_match(row, "v24_gate_A_favorite_dominance_only", __import__("src.diagnostics.backtest_v24_gated_blowout_mixture", fromlist=["PROFILES"]).PROFILES["v24_gate_A_favorite_dominance_only"]) for _, row in test.iterrows()])
    v24_abcd = pd.DataFrame([summarize_grid_for_match(row, "v24_gate_no_E_ABCD", __import__("src.diagnostics.backtest_v24_gated_blowout_mixture", fromlist=["PROFILES"]).PROFILES["v24_gate_no_E_ABCD"]) for _, row in test.iterrows()])
    summary_rows = [
        summarize_baseline(test, "v20", "v20_stable"),
        summarize_baseline(test, "v23", "v23_base"),
        summarize_profile(v24_a, "v24_gate_A_favorite_dominance_only"),
        summarize_profile(v24_abcd, "v24_gate_no_E_ABCD"),
    ]
    for profile, subset in match_level.groupby("profile", sort=False):
        summary_rows.append(summarize_v25(subset, profile))
    summary = add_recommendation(pd.DataFrame(summary_rows))

    by_bucket_rows = []
    tail_rows = []
    for profile, subset in match_level.groupby("profile", sort=False):
        for bucket, bucket_df in subset.groupby("favorite_bucket", sort=False):
            row = summarize_v25(bucket_df, profile)
            row["favorite_bucket"] = bucket
            by_bucket_rows.append(row)
            tail_rows.append(
                {
                    "profile": profile,
                    "favorite_bucket": bucket,
                    "favorite_scores_5_plus_pred": bucket_df["favorite_scores_5_plus_pred"].mean(),
                    "favorite_scores_5_plus_actual": bucket_df["favorite_scores_5_plus_actual"].mean(),
                    "margin_4_plus_pred": bucket_df["margin_4_plus_pred"].mean(),
                    "margin_4_plus_actual": bucket_df["margin_4_plus_actual"].mean(),
                    "total_goals_5_plus_pred": bucket_df["total_goals_5_plus_pred"].mean(),
                    "total_goals_5_plus_actual": bucket_df["total_goals_5_plus_actual"].mean(),
                    "avg_p_blowout_final": bucket_df["p_blowout_v25_final"].mean(),
                }
            )

    coef_model = gate.pipeline.named_steps["logit"]
    coefs = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "coefficient_standardized": coef_model.coef_[0],
        }
    )
    coefs.loc[len(coefs)] = {"feature": "intercept", "coefficient_standardized": coef_model.intercept_[0]}
    contrib = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "mean_abs_standardized_contribution": abs(coef_model.coef_[0]),
        }
    ).sort_values("mean_abs_standardized_contribution", ascending=False)
    gate_cal = calibration_deciles(match_level)
    examples = match_level.sort_values("p_blowout_v25_final", ascending=False).head(80)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "match_level": output_dir / "v25_match_level.csv",
        "summary": output_dir / "v25_profile_comparison_summary.csv",
        "by_bucket": output_dir / "v25_profile_comparison_by_bucket.csv",
        "tail": output_dir / "v25_tail_calibration.csv",
        "gate_cal": output_dir / "v25_gate_calibration_deciles.csv",
        "coefs": output_dir / "v25_gate_coefficients.csv",
        "contrib": output_dir / "v25_gate_feature_contributions.csv",
        "examples": output_dir / "v25_tail_examples.csv",
        "readiness": output_dir / "v25_production_readiness.md",
    }
    match_level.to_csv(paths["match_level"], index=False)
    summary.to_csv(paths["summary"], index=False)
    pd.DataFrame(by_bucket_rows).to_csv(paths["by_bucket"], index=False)
    pd.DataFrame(tail_rows).to_csv(paths["tail"], index=False)
    gate_cal.to_csv(paths["gate_cal"], index=False)
    coefs.to_csv(paths["coefs"], index=False)
    contrib.to_csv(paths["contrib"], index=False)
    examples.to_csv(paths["examples"], index=False)
    best = summary.loc[summary["profile"].astype(str).str.startswith("v25")].sort_values(
        ["recommendation", "combined_tail_abs_error", "top5"],
        ascending=[True, True, False],
    ).head(1)
    paths["readiness"].write_text(
        "# V2.5 Learned Gate Production Readiness\n\n"
        "Status: research only. V2.0 remains production and V2.4 ABCD remains the stronger shadow baseline unless V2.5 is manually promoted after review.\n\n"
        f"Train: 2014-2021. Validation: 2022-2023. Test: 2024+.\n\n"
        "W/D/L probabilities were not modified.\n\n"
        f"Best V2.5 row by diagnostic sort:\n\n{best.to_string(index=False)}\n"
    )
    print("V2.5 summary:")
    print(summary.to_string(index=False))
    print("Coefficients:")
    print(coefs.to_string(index=False))
    print("Gate calibration:")
    print(gate_cal.to_string(index=False))
    print("Files:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
