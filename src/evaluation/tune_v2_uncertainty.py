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

from src.config.model_config import load_model_config  # noqa: E402
from src.evaluation.evaluate_match_predictions import (  # noqa: E402
    FINAL_MODEL_NAME,
    evaluate_predictions,
    load_clean_matches,
    resolve_matches_path,
)
from src.evaluation.evaluate_v2_uncertainty import (  # noqa: E402
    actual_result,
    brier,
    log_loss,
    pick,
    summarize,
)
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTICS_DIR = PROJECT_ROOT / "output" / "diagnostics"
TUNING_RESULTS_PATH = DIAGNOSTICS_DIR / "v2_uncertainty_tuning_results.csv"
BEST_CONFIG_PATH = PROJECT_ROOT / "config" / "model_params_v2_uncertainty_tuned.json"
V1_CONFIG_PATH = PROJECT_ROOT / "config" / "model_params_v1.json"

MAX_SHIFT_GRID = [0.015, 0.025, 0.035, 0.045]
VOLATILITY_WEIGHT_GRID = [0.004, 0.008, 0.010, 0.012]
STABILITY_EDGE_WEIGHT_GRID = [0.004, 0.006, 0.008, 0.010]
MAX_STAR_SHIFT_GRID = [0.015, 0.025, 0.035, 0.040]
STAR_WEIGHT_GRID = [0.006, 0.010, 0.012, 0.016]
DRAW_RATE_WARNING_TOLERANCE = 0.02


def build_v1_match_level(
    train_start: str,
    test_start: str,
    test_end: str | None,
) -> tuple[pd.DataFrame, dict]:
    v1_config = (
        load_model_config(config_path=V1_CONFIG_PATH)
        if V1_CONFIG_PATH.exists()
        else load_model_config("default")
    )
    matches = load_clean_matches(resolve_matches_path())
    predictions, *_ = evaluate_predictions(
        matches=matches,
        train_start=train_start,
        test_start=test_start,
        test_end=test_end,
        model_config=v1_config,
    )
    final_predictions = (
        predictions.loc[predictions["model_version"] == FINAL_MODEL_NAME]
        .copy()
        .reset_index(drop=True)
    )
    return final_predictions, v1_config


def evaluate_parameter_set(
    v1_predictions: pd.DataFrame,
    squad_features: pd.DataFrame,
    superstar_features: pd.DataFrame,
    max_shift: float,
    volatility_weight: float,
    stability_edge_weight: float,
    max_star_shift: float,
    star_weight: float,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    for index, row in v1_predictions.iterrows():
        actual = actual_result(row)
        v1_probs = {
            "team_a_win": float(row["p_home_win"]),
            "draw": float(row["p_draw"]),
            "team_b_win": float(row["p_away_win"]),
        }
        adjustment = adjust_probabilities_for_uncertainty(
            p_team_a_win=v1_probs["team_a_win"],
            p_draw=v1_probs["draw"],
            p_team_b_win=v1_probs["team_b_win"],
            team_a=row["home_team"],
            team_b=row["away_team"],
            squad_features_df=squad_features,
            max_shift=max_shift,
            volatility_weight=volatility_weight,
            stability_edge_weight=stability_edge_weight,
        )
        v2_probs = {
            "team_a_win": adjustment["adjusted_p_team_a_win"],
            "draw": adjustment["adjusted_p_draw"],
            "team_b_win": adjustment["adjusted_p_team_b_win"],
        }
        superstar_adjustment = adjust_probabilities_for_superstar_impact(
            p_team_a_win=v2_probs["team_a_win"],
            p_draw=v2_probs["draw"],
            p_team_b_win=v2_probs["team_b_win"],
            team_a=row["home_team"],
            team_b=row["away_team"],
            superstar_features_df=superstar_features,
            max_star_shift=max_star_shift,
            star_weight=star_weight,
        )
        v2_probs = {
            "team_a_win": superstar_adjustment["star_adjusted_p_team_a_win"],
            "draw": superstar_adjustment["star_adjusted_p_draw"],
            "team_b_win": superstar_adjustment["star_adjusted_p_team_b_win"],
        }
        v2_pick = pick(v2_probs)
        rows.append(
            {
                "match_id": index + 1,
                "date": row["date"],
                "team_a": row["home_team"],
                "team_b": row["away_team"],
                "actual_result": actual,
                "v2_p_team_a_win": v2_probs["team_a_win"],
                "v2_p_draw": v2_probs["draw"],
                "v2_p_team_b_win": v2_probs["team_b_win"],
                "v2_predicted_result": v2_pick,
                "v2_correct": v2_pick == actual,
                "v2_brier_score": brier(v2_probs, actual),
                "v2_log_loss": log_loss(v2_probs, actual),
                "v2_actual_outcome_probability": v2_probs[actual],
                "uncertainty_shift": adjustment["uncertainty_shift"],
                "adjustment_reason": adjustment["adjustment_reason"],
                "superstar_shift": superstar_adjustment["superstar_shift"],
                "superstar_adjustment_reason": superstar_adjustment[
                    "superstar_adjustment_reason"
                ],
            }
        )
    match_level = pd.DataFrame(rows)
    metrics = summarize(match_level, prefix="v2", model_version="v2_uncertainty")
    metrics.update(
        {
            "max_shift": max_shift,
            "volatility_weight": volatility_weight,
            "stability_edge_weight": stability_edge_weight,
            "max_star_shift": max_star_shift,
            "star_weight": star_weight,
        }
    )
    return metrics, match_level


def baseline_metrics(v1_predictions: pd.DataFrame) -> dict:
    rows = []
    for index, row in v1_predictions.iterrows():
        actual = actual_result(row)
        v1_probs = {
            "team_a_win": float(row["p_home_win"]),
            "draw": float(row["p_draw"]),
            "team_b_win": float(row["p_away_win"]),
        }
        v1_pick = pick(v1_probs)
        rows.append(
            {
                "match_id": index + 1,
                "actual_result": actual,
                "v1_p_team_a_win": v1_probs["team_a_win"],
                "v1_p_draw": v1_probs["draw"],
                "v1_p_team_b_win": v1_probs["team_b_win"],
                "v1_predicted_result": v1_pick,
                "v1_correct": v1_pick == actual,
                "v1_brier_score": brier(v1_probs, actual),
                "v1_log_loss": log_loss(v1_probs, actual),
                "v1_actual_outcome_probability": v1_probs[actual],
            }
        )
    return summarize(pd.DataFrame(rows), prefix="v1", model_version="v1_default")


def save_best_config(best_row: pd.Series, v1_config: dict, notes: str) -> dict:
    config = {
        "model_version": "v2_uncertainty_tuned",
        "model_status": "experimental",
        "max_shift": float(best_row["max_shift"]),
        "volatility_weight": float(best_row["volatility_weight"]),
        "stability_edge_weight": float(best_row["stability_edge_weight"]),
        "max_star_shift": float(best_row["max_star_shift"]),
        "star_weight": float(best_row["star_weight"]),
        "selection_metric": "mean_log_loss",
        "rating_file": v1_config.get("rating_source_path", "data/processed/team_ratings_world_cup_elo.csv"),
        "rating_col": v1_config.get("rating_col", "anchored_final_strength"),
        "squad_values_file": "data/raw/squad_values.csv",
        "superstar_features_file": "data/processed/superstar_features.csv",
        "tuning_results_file": "output/diagnostics/v2_uncertainty_tuning_results.csv",
        "notes": notes,
    }
    BEST_CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")
    return config


def tune_v2_uncertainty(
    train_start: str = "2014-01-01",
    test_start: str = "2022-01-01",
    test_end: str | None = None,
    squad_values_path: Path = DEFAULT_SQUAD_VALUES_PATH,
    superstar_features_path: Path = DEFAULT_SUPERSTAR_FEATURES_PATH,
) -> tuple[pd.DataFrame, dict, dict]:
    v1_predictions, v1_config = build_v1_match_level(
        train_start=train_start,
        test_start=test_start,
        test_end=test_end,
    )
    squad_features = load_squad_uncertainty_features(squad_values_path)
    superstar_features = load_superstar_features(superstar_features_path)
    squad_lookup = dict(
        zip(squad_features["team_name"].astype(str), squad_features["volatility_score"].astype(float))
    )
    superstar_lookup = {
        str(row["team"]): {
            "superstar_score": float(row["superstar_score"]),
            "top_player_name": "" if pd.isna(row["top_player_name"]) else str(row["top_player_name"]),
        }
        for _, row in superstar_features.iterrows()
    }
    v1_metrics = baseline_metrics(v1_predictions)
    v1_draw_error = float(v1_metrics["draw_calibration_error"])

    rows = []
    for (
        max_shift,
        volatility_weight,
        stability_edge_weight,
        max_star_shift,
        star_weight,
    ) in itertools.product(
        MAX_SHIFT_GRID,
        VOLATILITY_WEIGHT_GRID,
        STABILITY_EDGE_WEIGHT_GRID,
        MAX_STAR_SHIFT_GRID,
        STAR_WEIGHT_GRID,
    ):
        try:
            metrics, _ = evaluate_parameter_set(
                v1_predictions=v1_predictions,
                squad_features=squad_lookup,
                superstar_features=superstar_lookup,
                max_shift=max_shift,
                volatility_weight=volatility_weight,
                stability_edge_weight=stability_edge_weight,
                max_star_shift=max_star_shift,
                star_weight=star_weight,
            )
            metrics["draw_rate_warning"] = (
                float(metrics["draw_calibration_error"])
                > v1_draw_error + DRAW_RATE_WARNING_TOLERANCE
            )
            metrics["error"] = ""
            rows.append(metrics)
        except Exception as exc:  # keep the full grid auditable.
            rows.append(
                {
                    "model_version": "v2_uncertainty",
                    "n_matches": len(v1_predictions),
                    "accuracy": math.nan,
                    "mean_brier_score": math.nan,
                    "mean_log_loss": math.nan,
                    "mean_actual_outcome_probability": math.nan,
                    "predicted_draw_rate": math.nan,
                    "predicted_pick_draw_rate": math.nan,
                    "actual_draw_rate": v1_metrics["actual_draw_rate"],
                    "draw_calibration_error": math.nan,
                    "mean_p_draw": math.nan,
                    "max_shift": max_shift,
                    "volatility_weight": volatility_weight,
                    "stability_edge_weight": stability_edge_weight,
                    "max_star_shift": max_star_shift,
                    "star_weight": star_weight,
                    "draw_rate_warning": True,
                    "error": str(exc),
                }
            )

    tuning = pd.DataFrame(rows)
    tuning = tuning.sort_values(
        ["mean_log_loss", "mean_brier_score"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(TUNING_RESULTS_PATH, index=False)

    best = tuning.loc[tuning["error"].fillna("").eq("")].iloc[0]
    improves_log_loss = float(best["mean_log_loss"]) < float(v1_metrics["mean_log_loss"])
    improves_brier = float(best["mean_brier_score"]) < float(v1_metrics["mean_brier_score"])
    notes = (
        "V2 uncertainty remains experimental; do not replace V1 unless explicitly promoted. "
        f"Improves log loss: {improves_log_loss}. Improves Brier score: {improves_brier}. "
        f"Draw rate warning: {bool(best['draw_rate_warning'])}."
    )
    best_config = save_best_config(best, v1_config=v1_config, notes=notes)
    return tuning, v1_metrics, best_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune V2 uncertainty adjustment parameters.")
    parser.add_argument("--train-start", default="2014-01-01")
    parser.add_argument("--test-start", default="2022-01-01")
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--squad-values", type=Path, default=DEFAULT_SQUAD_VALUES_PATH)
    parser.add_argument("--superstar-features", type=Path, default=DEFAULT_SUPERSTAR_FEATURES_PATH)
    args = parser.parse_args()

    tuning, v1_metrics, best_config = tune_v2_uncertainty(
        train_start=args.train_start,
        test_start=args.test_start,
        test_end=args.test_end,
        squad_values_path=args.squad_values,
        superstar_features_path=args.superstar_features,
    )
    best = tuning.iloc[0]
    improves_log_loss = float(best["mean_log_loss"]) < float(v1_metrics["mean_log_loss"])
    improves_brier = float(best["mean_brier_score"]) < float(v1_metrics["mean_brier_score"])
    should_promote = improves_log_loss and improves_brier and not bool(best["draw_rate_warning"])

    print(f"Saved tuning results to {TUNING_RESULTS_PATH}")
    print(f"Saved best config to {BEST_CONFIG_PATH}")
    print("\nBest parameter combination")
    print(
        best[
            [
                "max_shift",
                "volatility_weight",
                "stability_edge_weight",
                "max_star_shift",
                "star_weight",
                "mean_log_loss",
                "mean_brier_score",
                "accuracy",
                "draw_rate_warning",
            ]
        ].to_string()
    )
    print("\nV1 baseline metrics")
    print(pd.Series(v1_metrics).to_string())
    print("\nBest V2 metrics")
    print(best.to_string())
    print(f"\nV2 improves log loss: {improves_log_loss}")
    print(f"V2 improves Brier score: {improves_brier}")
    print(f"Recommendation: {'promote candidate' if should_promote else 'remain experimental'}")
    print("\nBest config")
    print(json.dumps(best_config, indent=2))


if __name__ == "__main__":
    main()
