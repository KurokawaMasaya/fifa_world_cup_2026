from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = PROJECT_ROOT / "output" / "live"
LOCKED_PREDICTIONS_PATH = LIVE_DIR / "worldcup_group_stage_locked_predictions.csv"
ACTUAL_RESULTS_PATH = LIVE_DIR / "fixtures_results.csv"
EVALUATION_OUTPUT_PATH = LIVE_DIR / "worldcup_group_stage_live_evaluation.csv"
SUMMARY_OUTPUT_PATH = LIVE_DIR / "worldcup_group_stage_live_summary.csv"
CALIBRATION_OUTPUT_PATH = LIVE_DIR / "worldcup_group_stage_live_calibration.csv"
EPSILON = 1e-15

OUTCOME_COLUMNS = {
    "team_a_win": "p_team_a_win",
    "draw": "p_draw",
    "team_b_win": "p_team_b_win",
}
CALIBRATION_BINS = [
    (0.35, 0.45),
    (0.45, 0.55),
    (0.55, 0.65),
    (0.65, 0.75),
    (0.75, 0.85),
    (0.85, 1.00),
]


def normalize_result(value: object) -> str | pd.NA:
    if pd.isna(value) or str(value).strip() == "":
        return pd.NA
    result = str(value).strip()
    if result not in OUTCOME_COLUMNS:
        raise ValueError(
            f"Invalid actual_result '{result}'. Expected one of {sorted(OUTCOME_COLUMNS)}"
        )
    return result


def actual_result_from_goals(row: pd.Series) -> str | pd.NA:
    if "status" in row and str(row.get("status", "")).strip().lower() != "final":
        return pd.NA
    explicit_result = normalize_result(row.get("actual_result"))
    if not pd.isna(explicit_result):
        return explicit_result
    goals_a = pd.to_numeric(row.get("goals_a"), errors="coerce")
    goals_b = pd.to_numeric(row.get("goals_b"), errors="coerce")
    if pd.isna(goals_a) or pd.isna(goals_b):
        return pd.NA
    if goals_a > goals_b:
        return "team_a_win"
    if goals_a < goals_b:
        return "team_b_win"
    return "draw"


def pick_from_probabilities(row: pd.Series) -> tuple[str, str, float]:
    probabilities = {
        "team_a_win": float(row["p_team_a_win"]),
        "draw": float(row["p_draw"]),
        "team_b_win": float(row["p_team_b_win"]),
    }
    pick = max(probabilities, key=probabilities.get)
    if pick == "team_a_win":
        pick_label = f"{row['team_a']} win"
    elif pick == "team_b_win":
        pick_label = f"{row['team_b']} win"
    else:
        pick_label = "Draw"
    return pick, pick_label, probabilities[pick]


def risk_level_from_pick_probability(probability: float) -> str:
    if probability >= 0.65:
        return "low"
    if probability >= 0.50:
        return "medium"
    return "high"


def draw_risk_level_from_probability(draw_probability: float) -> str:
    if draw_probability >= 0.28:
        return "high"
    if draw_probability >= 0.22:
        return "medium"
    return "low"


def validate_locked_predictions(predictions: pd.DataFrame) -> None:
    required = {
        "match_id",
        "group",
        "team_a",
        "team_b",
        "p_team_a_win",
        "p_draw",
        "p_team_b_win",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Locked prediction file is missing columns: {sorted(missing)}")
    probability_sum = (
        predictions["p_team_a_win"] + predictions["p_draw"] + predictions["p_team_b_win"]
    )
    if not probability_sum.between(0.999, 1.001).all():
        raise ValueError("Locked probabilities must sum to 1 before live evaluation")


def load_completed_results(path: Path) -> pd.DataFrame:
    actual = pd.read_csv(path)
    required = {"match_id", "team_a", "team_b", "goals_a", "goals_b"}
    missing = required - set(actual.columns)
    if missing:
        raise ValueError(f"Actual-results file is missing columns: {sorted(missing)}")
    actual = actual.copy()
    # When a status column is present, restrict to completed (final) matches.
    # When it is absent, treat every supplied row as a completed result and let
    # actual_result_from_goals derive the outcome from goals_a/goals_b.
    if "status" in actual.columns:
        actual["status"] = actual["status"].astype(str).str.strip().str.lower()
        actual = actual.loc[actual["status"].eq("final")].copy()
    if "actual_result" not in actual.columns:
        actual["actual_result"] = pd.NA
    actual["actual_result"] = actual.apply(actual_result_from_goals, axis=1)
    return actual.loc[actual["actual_result"].notna()].copy()


def brier_score(row: pd.Series) -> float:
    return sum(
        (float(row[column]) - (1.0 if result == row["actual_result"] else 0.0)) ** 2
        for result, column in OUTCOME_COLUMNS.items()
    )


def log_loss(row: pd.Series) -> float:
    probability = max(EPSILON, min(1.0, float(row[OUTCOME_COLUMNS[row["actual_result"]]])))
    return -math.log(probability)


def build_live_evaluation(locked: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    validate_locked_predictions(locked)
    actual_columns = ["match_id", "team_a", "team_b", "goals_a", "goals_b", "actual_result"]
    actual = actual[actual_columns].copy()
    locked_ids = set(locked["match_id"].astype(str))
    actual_ids = set(actual["match_id"].astype(str))
    if locked_ids & actual_ids:
        merged = locked.merge(
            actual[["match_id", "goals_a", "goals_b", "actual_result"]],
            on="match_id",
            how="inner",
        )
    else:
        merged = locked.merge(
            actual[["team_a", "team_b", "goals_a", "goals_b", "actual_result"]],
            on=["team_a", "team_b"],
            how="inner",
        )
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "match_id",
                "group",
                "team_a",
                "team_b",
                "p_team_a_win",
                "p_draw",
                "p_team_b_win",
                "pick_cn",
                "pick_probability",
                "actual_result",
                "correct",
                "actual_outcome_probability",
                "brier_score",
                "log_loss",
                "risk_level",
                "draw_risk_level",
                "model_version",
            ]
        )

    picks = merged.apply(pick_from_probabilities, axis=1, result_type="expand")
    merged["predicted_result"] = picks[0]
    merged["pick_cn"] = picks[1]
    merged["pick_probability"] = picks[2].astype(float)
    merged["correct"] = merged["predicted_result"] == merged["actual_result"]
    merged["actual_outcome_probability"] = merged.apply(
        lambda row: row[OUTCOME_COLUMNS[row["actual_result"]]], axis=1
    )
    merged["brier_score"] = merged.apply(brier_score, axis=1)
    merged["log_loss"] = merged.apply(log_loss, axis=1)
    merged["risk_level"] = merged["pick_probability"].map(risk_level_from_pick_probability)
    merged["draw_risk_level"] = merged["p_draw"].map(draw_risk_level_from_probability)
    if "model_version" not in merged.columns:
        merged["model_version"] = pd.NA

    columns = [
        "match_id",
        "group",
        "team_a",
        "team_b",
        "p_team_a_win",
        "p_draw",
        "p_team_b_win",
        "pick_cn",
        "pick_probability",
        "actual_result",
        "correct",
        "actual_outcome_probability",
        "brier_score",
        "log_loss",
        "risk_level",
        "draw_risk_level",
        "model_version",
    ]
    return merged[columns].sort_values("match_id").reset_index(drop=True)


def build_live_summary(evaluation: pd.DataFrame) -> pd.DataFrame:
    n_completed = len(evaluation)
    if n_completed == 0:
        return pd.DataFrame(
            [
                {
                    "completed_matches": 0,
                    "status": "waiting_for_matches",
                    "n_completed_matches": 0,
                    "accuracy": pd.NA,
                    "mean_brier_score": pd.NA,
                    "mean_log_loss": pd.NA,
                    "mean_actual_outcome_probability": pd.NA,
                    "predicted_draw_rate": pd.NA,
                    "predicted_pick_draw_rate": pd.NA,
                    "actual_draw_rate": pd.NA,
                    "draw_calibration_error": pd.NA,
                }
            ]
        )

    predicted_draw_rate = evaluation["p_draw"].mean()
    predicted_pick_draw_rate = (evaluation["pick_cn"] == "Draw").mean()
    actual_draw_rate = evaluation["actual_result"].eq("draw").mean()
    return pd.DataFrame(
        [
            {
                "completed_matches": n_completed,
                "status": "evaluated",
                "n_completed_matches": n_completed,
                "accuracy": evaluation["correct"].mean(),
                "mean_brier_score": evaluation["brier_score"].mean(),
                "mean_log_loss": evaluation["log_loss"].mean(),
                "mean_actual_outcome_probability": evaluation[
                    "actual_outcome_probability"
                ].mean(),
                "predicted_draw_rate": predicted_draw_rate,
                "predicted_pick_draw_rate": predicted_pick_draw_rate,
                "actual_draw_rate": actual_draw_rate,
                "draw_calibration_error": abs(predicted_draw_rate - actual_draw_rate),
            }
        ]
    )


def build_live_calibration(evaluation: pd.DataFrame) -> pd.DataFrame:
    if evaluation.empty:
        return pd.DataFrame(
            [
                {
                    "pick_probability_bin": f"{lower:.2f}-{upper:.2f}",
                    "n_matches": 0,
                    "avg_pick_probability": pd.NA,
                    "actual_pick_success_rate": pd.NA,
                }
                for lower, upper in CALIBRATION_BINS
            ]
        )
    rows = []
    for lower, upper in CALIBRATION_BINS:
        if upper == 1.00:
            in_bin = evaluation["pick_probability"].between(lower, upper, inclusive="both")
        else:
            in_bin = (evaluation["pick_probability"] >= lower) & (
                evaluation["pick_probability"] < upper
            )
        subset = evaluation.loc[in_bin]
        rows.append(
            {
                "pick_probability_bin": f"{lower:.2f}-{upper:.2f}",
                "n_matches": len(subset),
                "avg_pick_probability": (
                    subset["pick_probability"].mean() if not subset.empty else pd.NA
                ),
                "actual_pick_success_rate": (
                    subset["correct"].mean() if not subset.empty else pd.NA
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_live_group_stage(
    locked_predictions_path: Path = LOCKED_PREDICTIONS_PATH,
    actual_results_path: Path = ACTUAL_RESULTS_PATH,
    evaluation_output_path: Path = EVALUATION_OUTPUT_PATH,
    summary_output_path: Path = SUMMARY_OUTPUT_PATH,
    calibration_output_path: Path = CALIBRATION_OUTPUT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    locked = pd.read_csv(locked_predictions_path)
    actual = load_completed_results(actual_results_path)
    evaluation = build_live_evaluation(locked, actual)
    summary = build_live_summary(evaluation)
    calibration = build_live_calibration(evaluation)

    for path in [evaluation_output_path, summary_output_path, calibration_output_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(evaluation_output_path, index=False)
    summary.to_csv(summary_output_path, index=False)
    calibration.to_csv(calibration_output_path, index=False)
    return evaluation, summary, calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate locked World Cup group-stage predictions.")
    parser.add_argument("--locked-predictions", type=Path, default=LOCKED_PREDICTIONS_PATH)
    parser.add_argument("--actual-results", type=Path, default=ACTUAL_RESULTS_PATH)
    parser.add_argument("--evaluation-output", type=Path, default=EVALUATION_OUTPUT_PATH)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_OUTPUT_PATH)
    parser.add_argument("--calibration-output", type=Path, default=CALIBRATION_OUTPUT_PATH)
    args = parser.parse_args()

    evaluation, summary, calibration = evaluate_live_group_stage(
        locked_predictions_path=args.locked_predictions,
        actual_results_path=args.actual_results,
        evaluation_output_path=args.evaluation_output,
        summary_output_path=args.summary_output,
        calibration_output_path=args.calibration_output,
    )
    print(f"Completed matches evaluated: {len(evaluation)}")
    if len(evaluation) == 0:
        print(
            "No completed matches yet. Live evaluation will start after final results are available."
        )
    print(f"Saved live evaluation to {args.evaluation_output}")
    print(f"Saved live summary to {args.summary_output}")
    print(f"Saved live calibration to {args.calibration_output}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
