from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.model_config import load_model_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = PROJECT_ROOT / "output" / "live"
REQUESTED_INPUT_PATH = PROJECT_ROOT / "output" / "predictions" / "group_stage_prediction_confidence.csv"
FALLBACK_INPUT_PATH = PROJECT_ROOT / "output" / "predictions" / "group_stage_predictions_v2_uncertainty.csv"
LOCKED_OUTPUT_PATH = LIVE_DIR / "worldcup_group_stage_locked_predictions.csv"
ACTUAL_RESULTS_TEMPLATE_PATH = LIVE_DIR / "worldcup_group_stage_actual_results.csv"
EXPECTED_MATCHES = 72


def resolve_input_path(path: Path | None) -> Path:
    if path is not None:
        resolved = path if path.is_absolute() else PROJECT_ROOT / path
        if not resolved.exists():
            raise FileNotFoundError(f"Prediction input file not found: {resolved}")
        return resolved
    if REQUESTED_INPUT_PATH.exists():
        return REQUESTED_INPUT_PATH
    if FALLBACK_INPUT_PATH.exists():
        return FALLBACK_INPUT_PATH
    raise FileNotFoundError(
        f"Could not find {REQUESTED_INPUT_PATH} or fallback {FALLBACK_INPUT_PATH}"
    )


def validate_predictions(predictions: pd.DataFrame) -> None:
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
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    if len(predictions) != EXPECTED_MATCHES:
        raise ValueError(f"Expected {EXPECTED_MATCHES} group-stage predictions, found {len(predictions)}")
    probability_sum = (
        predictions["p_team_a_win"] + predictions["p_draw"] + predictions["p_team_b_win"]
    )
    if not probability_sum.between(0.999, 1.001).all():
        bad = predictions.loc[
            ~probability_sum.between(0.999, 1.001), ["match_id", "team_a", "team_b"]
        ]
        raise ValueError(f"Probabilities do not sum to 1 for: {bad.to_dict(orient='records')}")


def add_metadata(predictions: pd.DataFrame, mode: str) -> pd.DataFrame:
    output = predictions.copy()
    config = load_model_config(mode)
    if "main_pick" in output.columns and "predicted_result" not in output.columns:
        output["predicted_result"] = output["main_pick"]
    if "main_pick_probability" in output.columns and "pick_probability" not in output.columns:
        output["pick_probability"] = output["main_pick_probability"]
    if "pick_cn" not in output.columns and "main_pick" in output.columns:
        output["pick_cn"] = output.apply(
            lambda row: (
                f"{row['team_a']} win"
                if row["main_pick"] == "team_a_win"
                else f"{row['team_b']} win"
                if row["main_pick"] == "team_b_win"
                else "Draw"
            ),
            axis=1,
        )
    if "prediction_timestamp" not in output.columns:
        output["prediction_timestamp"] = datetime.now(timezone.utc).isoformat()
    if "model_version" not in output.columns:
        output["model_version"] = config.get("model_version")
    if "rating_col" not in output.columns:
        output["rating_col"] = config.get("rating_col")
    return output


def create_actual_results_template(locked_predictions: pd.DataFrame) -> pd.DataFrame:
    template = locked_predictions[["match_id", "team_a", "team_b"]].copy()
    template["goals_a"] = pd.NA
    template["goals_b"] = pd.NA
    template["actual_result"] = pd.NA
    return template


def lock_predictions(
    input_path: Path | None = None,
    output_path: Path = LOCKED_OUTPUT_PATH,
    actual_template_path: Path = ACTUAL_RESULTS_TEMPLATE_PATH,
    mode: str = "default",
    overwrite: bool = False,
) -> pd.DataFrame:
    resolved_input = resolve_input_path(input_path)
    resolved_output = output_path if output_path.is_absolute() else PROJECT_ROOT / output_path
    resolved_template = (
        actual_template_path if actual_template_path.is_absolute() else PROJECT_ROOT / actual_template_path
    )
    if resolved_output.exists() and not overwrite:
        raise FileExistsError(
            f"Locked prediction file already exists: {resolved_output}. "
            "Pass --overwrite only if you intentionally want to replace it."
        )

    predictions = pd.read_csv(resolved_input)
    validate_predictions(predictions)
    locked = add_metadata(predictions, mode=mode)

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    locked.to_csv(resolved_output, index=False)

    if overwrite or not resolved_template.exists():
        create_actual_results_template(locked).to_csv(resolved_template, index=False)
    return locked


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock pre-match World Cup group-stage forecasts.")
    parser.add_argument("--input", type=Path, default=None, help="Prediction confidence CSV.")
    parser.add_argument("--output", type=Path, default=LOCKED_OUTPUT_PATH)
    parser.add_argument("--actual-template-output", type=Path, default=ACTUAL_RESULTS_TEMPLATE_PATH)
    parser.add_argument("--mode", default="default", choices=["default", "v2", "experimental", "test"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    locked = lock_predictions(
        input_path=args.input,
        output_path=args.output,
        actual_template_path=args.actual_template_output,
        mode=args.mode,
        overwrite=args.overwrite,
    )
    print(f"Locked {len(locked)} group-stage predictions to {args.output}")
    print(f"Actual-results template: {args.actual_template_output}")


if __name__ == "__main__":
    main()
