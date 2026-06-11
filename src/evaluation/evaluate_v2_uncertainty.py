from __future__ import annotations

import argparse
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTICS_DIR = PROJECT_ROOT / "output" / "diagnostics"
SUMMARY_OUTPUT_PATH = DIAGNOSTICS_DIR / "v1_vs_v2_layer_ablation.csv"
SUPERSTAR_SUMMARY_OUTPUT_PATH = DIAGNOSTICS_DIR / "v1_vs_v2_superstar_evaluation.csv"
MATCH_LEVEL_OUTPUT_PATH = DIAGNOSTICS_DIR / "v2_uncertainty_match_level_evaluation.csv"
TUNED_V2_CONFIG_PATH = PROJECT_ROOT / "config" / "model_params_v2_uncertainty_tuned.json"
EPSILON = 1e-15
OUTCOMES = ["team_a_win", "draw", "team_b_win"]


def actual_result(row: pd.Series) -> str:
    if int(row["home_score"]) > int(row["away_score"]):
        return "team_a_win"
    if int(row["home_score"]) < int(row["away_score"]):
        return "team_b_win"
    return "draw"


def pick(probabilities: dict[str, float]) -> str:
    return max(probabilities, key=probabilities.get)


def brier(probabilities: dict[str, float], actual: str) -> float:
    return sum(
        (probabilities[outcome] - (1.0 if outcome == actual else 0.0)) ** 2
        for outcome in OUTCOMES
    )


def log_loss(probabilities: dict[str, float], actual: str) -> float:
    probability = max(EPSILON, min(1.0, probabilities[actual]))
    return -math.log(probability)


def summarize(match_level: pd.DataFrame, prefix: str, model_version: str) -> dict:
    predicted_col = f"{prefix}_predicted_result"
    correct_col = f"{prefix}_correct"
    actual_probability_col = f"{prefix}_actual_outcome_probability"
    brier_col = f"{prefix}_brier_score"
    log_loss_col = f"{prefix}_log_loss"
    draw_probability_col = f"{prefix}_p_draw"

    predicted_draw_rate = match_level[draw_probability_col].mean()
    predicted_pick_draw_rate = match_level[predicted_col].eq("draw").mean()
    actual_draw_rate = match_level["actual_result"].eq("draw").mean()
    validate_draw_probability_summary(
        model_version=model_version,
        p_draw=match_level[draw_probability_col],
        predicted_draw_rate=predicted_draw_rate,
        verbose=False,
    )
    return {
        "model_version": model_version,
        "n_matches": len(match_level),
        "accuracy": match_level[correct_col].mean(),
        "mean_brier_score": match_level[brier_col].mean(),
        "mean_log_loss": match_level[log_loss_col].mean(),
        "mean_actual_outcome_probability": match_level[actual_probability_col].mean(),
        "predicted_draw_rate": predicted_draw_rate,
        "predicted_pick_draw_rate": predicted_pick_draw_rate,
        "actual_draw_rate": actual_draw_rate,
        "draw_calibration_error": abs(predicted_draw_rate - actual_draw_rate),
        "mean_p_draw": match_level[draw_probability_col].mean(),
    }


def validate_draw_probability_summary(
    model_version: str,
    p_draw: pd.Series,
    predicted_draw_rate: float,
    verbose: bool = True,
) -> None:
    positive_draw_values = pd.to_numeric(p_draw, errors="coerce").fillna(0.0).gt(0).any()
    if predicted_draw_rate == 0 and positive_draw_values:
        raise ValueError(
            f"{model_version} predicted_draw_rate is 0 but positive p_draw values exist"
        )
    if verbose:
        print(
            f"{model_version} p_draw min/mean/max: "
            f"{p_draw.min():.4f} / {p_draw.mean():.4f} / {p_draw.max():.4f}"
        )
    if predicted_draw_rate < 0.10 or predicted_draw_rate > 0.35:
        print(
            f"WARNING: {model_version} mean p_draw {predicted_draw_rate:.4f} "
            "is outside the expected 0.10-0.35 range"
        )


def evaluate_v2_uncertainty(
    train_start: str = "2014-01-01",
    test_start: str = "2022-01-01",
    test_end: str | None = None,
    squad_values_path: Path = DEFAULT_SQUAD_VALUES_PATH,
    superstar_features_path: Path = DEFAULT_SUPERSTAR_FEATURES_PATH,
    club_form_features_path: Path = DEFAULT_CLUB_FORM_FEATURES_PATH,
    config_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    v1_config = load_model_config("default")
    if config_path is None and TUNED_V2_CONFIG_PATH.exists():
        config_path = TUNED_V2_CONFIG_PATH
    v2_config = load_model_config(config_path=config_path) if config_path else load_model_config("v2")
    matches = load_clean_matches(resolve_matches_path())
    predictions, *_ = evaluate_predictions(
        matches=matches,
        train_start=train_start,
        test_start=test_start,
        test_end=test_end,
        model_config=v1_config,
    )
    v1_predictions = (
        predictions.loc[predictions["model_version"] == FINAL_MODEL_NAME]
        .copy()
        .reset_index(drop=True)
    )
    squad_features = load_squad_uncertainty_features(squad_values_path)
    superstar_features = load_superstar_features(superstar_features_path)
    club_form_features = load_team_club_form_features(club_form_features_path)
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
    club_form_lookup = {
        str(row["team"]): {
            "club_form_signal": float(row["club_form_signal"]),
            "club_form_data_coverage": float(row["club_form_data_coverage"]),
        }
        for _, row in club_form_features.iterrows()
    }
    max_star_shift = float(v2_config.get("max_star_shift", 0.040))
    star_weight = float(v2_config.get("star_weight", 0.012))
    max_club_form_shift = float(v2_config.get("max_club_form_shift", 0.025))
    club_form_weight = float(v2_config.get("club_form_weight", 0.008))

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
            squad_features_df=squad_lookup,
            max_shift=float(v2_config.get("max_shift", v2_config.get("v2_uncertainty_max_shift", 0.035))),
            volatility_weight=float(v2_config.get("volatility_weight", v2_config.get("v2_volatility_weight", 0.010))),
            stability_edge_weight=float(v2_config.get("stability_edge_weight", v2_config.get("v2_stability_edge_weight", 0.008))),
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
            superstar_features_df=superstar_lookup,
            max_star_shift=max_star_shift,
            star_weight=star_weight,
        )
        v2_star_probs = {
            "team_a_win": superstar_adjustment["star_adjusted_p_team_a_win"],
            "draw": superstar_adjustment["star_adjusted_p_draw"],
            "team_b_win": superstar_adjustment["star_adjusted_p_team_b_win"],
        }
        club_form_adjustment = adjust_probabilities_for_club_form(
            p_team_a_win=v2_star_probs["team_a_win"],
            p_draw=v2_star_probs["draw"],
            p_team_b_win=v2_star_probs["team_b_win"],
            team_a=row["home_team"],
            team_b=row["away_team"],
            team_club_form_features_df=club_form_lookup,
            max_club_form_shift=max_club_form_shift,
            club_form_weight=club_form_weight,
        )
        v2_full_probs = {
            "team_a_win": club_form_adjustment["club_form_adjusted_p_team_a_win"],
            "draw": club_form_adjustment["club_form_adjusted_p_draw"],
            "team_b_win": club_form_adjustment["club_form_adjusted_p_team_b_win"],
        }
        v1_pick = pick(v1_probs)
        v2_pick = pick(v2_probs)
        v2_star_pick = pick(v2_star_probs)
        v2_full_pick = pick(v2_full_probs)
        rows.append(
            {
                "match_id": index + 1,
                "date": row["date"],
                "team_a": row["home_team"],
                "team_b": row["away_team"],
                "actual_result": actual,
                "v1_p_team_a_win": v1_probs["team_a_win"],
                "v1_p_draw": v1_probs["draw"],
                "v1_p_team_b_win": v1_probs["team_b_win"],
                "v2_p_team_a_win": v2_probs["team_a_win"],
                "v2_p_draw": v2_probs["draw"],
                "v2_p_team_b_win": v2_probs["team_b_win"],
                "v2_star_p_team_a_win": v2_star_probs["team_a_win"],
                "v2_star_p_draw": v2_star_probs["draw"],
                "v2_star_p_team_b_win": v2_star_probs["team_b_win"],
                "v2_full_p_team_a_win": v2_full_probs["team_a_win"],
                "v2_full_p_draw": v2_full_probs["draw"],
                "v2_full_p_team_b_win": v2_full_probs["team_b_win"],
                "v1_predicted_result": v1_pick,
                "v2_predicted_result": v2_pick,
                "v2_star_predicted_result": v2_star_pick,
                "v2_full_predicted_result": v2_full_pick,
                "v1_correct": v1_pick == actual,
                "v2_correct": v2_pick == actual,
                "v2_star_correct": v2_star_pick == actual,
                "v2_full_correct": v2_full_pick == actual,
                "v1_brier_score": brier(v1_probs, actual),
                "v2_brier_score": brier(v2_probs, actual),
                "v2_star_brier_score": brier(v2_star_probs, actual),
                "v2_full_brier_score": brier(v2_full_probs, actual),
                "v1_log_loss": log_loss(v1_probs, actual),
                "v2_log_loss": log_loss(v2_probs, actual),
                "v2_star_log_loss": log_loss(v2_star_probs, actual),
                "v2_full_log_loss": log_loss(v2_full_probs, actual),
                "v1_actual_outcome_probability": v1_probs[actual],
                "v2_actual_outcome_probability": v2_probs[actual],
                "v2_star_actual_outcome_probability": v2_star_probs[actual],
                "v2_full_actual_outcome_probability": v2_full_probs[actual],
                "uncertainty_shift": adjustment["uncertainty_shift"],
                "adjustment_reason": adjustment["adjustment_reason"],
                "superstar_shift": superstar_adjustment["superstar_shift"],
                "superstar_adjustment_reason": superstar_adjustment[
                    "superstar_adjustment_reason"
                ],
                "missing_squad_feature_flag": adjustment["missing_squad_feature_flag"],
                "missing_superstar_feature_flag": superstar_adjustment[
                    "missing_superstar_feature_flag"
                ],
                "club_form_shift": club_form_adjustment["club_form_shift"],
                "club_form_edge": club_form_adjustment["club_form_edge"],
                "club_form_adjustment_reason": club_form_adjustment[
                    "club_form_adjustment_reason"
                ],
                "missing_club_form_feature_flag": club_form_adjustment[
                    "missing_club_form_feature_flag"
                ],
            }
        )

    match_level = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            summarize(match_level, prefix="v1", model_version="v1_default"),
            summarize(match_level, prefix="v2", model_version="v2_uncertainty"),
            summarize(match_level, prefix="v2_star", model_version="v2_uncertainty_superstar"),
            summarize(
                match_level,
                prefix="v2_full",
                model_version="v2_uncertainty_superstar_club_form",
            ),
        ]
    )
    v1 = summary.loc[summary["model_version"] == "v1_default"].iloc[0]
    v2 = summary.loc[
        summary["model_version"] == "v2_uncertainty_superstar_club_form"
    ].iloc[0]
    improves = (
        v2["mean_log_loss"] < v1["mean_log_loss"]
        or v2["mean_brier_score"] < v1["mean_brier_score"]
    )
    summary["model_status"] = summary["model_version"].map(
        {
            "v1_default": "default",
            "v2_uncertainty": "experimental" if not improves else "candidate",
            "v2_uncertainty_superstar": "experimental" if not improves else "candidate",
            "v2_uncertainty_superstar_club_form": (
                "experimental" if not improves else "candidate"
            ),
        }
    )
    summary["note"] = summary["model_version"].map(
        {
            "v1_default": "Default model remains the production baseline.",
            "v2_uncertainty": (
                "Experimental; keep V1 default unless this improves log loss or Brier score."
                if not improves
                else "Candidate; V2 uncertainty improved at least one probability metric."
            ),
            "v2_uncertainty_superstar": (
                "Experimental; keep V1 default unless superstar layer improves log loss or Brier score."
                if not improves
                else "Candidate; V2 uncertainty plus superstar improved at least one probability metric."
            ),
            "v2_uncertainty_superstar_club_form": (
                "Experimental; keep club form diagnostic unless it improves log loss or Brier score."
                if not improves
                else "Candidate; full V2 stack improved at least one probability metric."
            ),
        }
    )
    return summary, match_level


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V2 uncertainty adjustment.")
    parser.add_argument("--train-start", default="2014-01-01")
    parser.add_argument("--test-start", default="2022-01-01")
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--squad-values", type=Path, default=DEFAULT_SQUAD_VALUES_PATH)
    parser.add_argument("--superstar-features", type=Path, default=DEFAULT_SUPERSTAR_FEATURES_PATH)
    parser.add_argument("--club-form-features", type=Path, default=DEFAULT_CLUB_FORM_FEATURES_PATH)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_OUTPUT_PATH)
    parser.add_argument("--superstar-summary-output", type=Path, default=SUPERSTAR_SUMMARY_OUTPUT_PATH)
    parser.add_argument("--match-output", type=Path, default=MATCH_LEVEL_OUTPUT_PATH)
    args = parser.parse_args()

    summary, match_level = evaluate_v2_uncertainty(
        train_start=args.train_start,
        test_start=args.test_start,
        test_end=args.test_end,
        squad_values_path=args.squad_values,
        superstar_features_path=args.superstar_features,
        club_form_features_path=args.club_form_features,
        config_path=args.config,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.match_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    summary.to_csv(args.superstar_summary_output, index=False)
    match_level.to_csv(args.match_output, index=False)

    print(f"Saved V1 vs V2 uncertainty summary to {args.summary_output}")
    print(f"Saved V1 vs V2 superstar summary to {args.superstar_summary_output}")
    print(f"Saved match-level V2 uncertainty evaluation to {args.match_output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
