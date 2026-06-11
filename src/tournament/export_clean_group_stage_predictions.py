from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "predictions"
    / "group_stage_predictions_v2_uncertainty_tuned.csv"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "output" / "predictions" / "group_stage_predictions_clean.csv"
)
CLEAN_COLUMNS = [
    "match_id",
    "group",
    "team_a",
    "team_b",
    "team_a_win_pct",
    "draw_pct",
    "team_b_win_pct",
    "predicted_scoreline",
    "scoreline_probability_pct",
]


def export_clean_predictions(input_path: Path, output_path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(input_path)
    required = {
        "match_id",
        "group",
        "team_a",
        "team_b",
        "v2_p_team_a_win",
        "v2_p_draw",
        "v2_p_team_b_win",
        "nb_top_scoreline_1",
        "nb_top_scoreline_1_probability",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

    probability_sum = (
        predictions["v2_p_team_a_win"]
        + predictions["v2_p_draw"]
        + predictions["v2_p_team_b_win"]
    )
    if (probability_sum - 1.0).abs().max() > 1e-9:
        raise ValueError("Official V2 probabilities must sum to 1 before clean export")

    clean = pd.DataFrame(
        {
            "match_id": predictions["match_id"],
            "group": predictions["group"],
            "team_a": predictions["team_a"],
            "team_b": predictions["team_b"],
            "team_a_win_pct": (predictions["v2_p_team_a_win"] * 100).round().astype(int),
            "draw_pct": (predictions["v2_p_draw"] * 100).round().astype(int),
            "team_b_win_pct": (predictions["v2_p_team_b_win"] * 100).round().astype(int),
            "predicted_scoreline": predictions["nb_top_scoreline_1"],
            "scoreline_probability_pct": (
                predictions["nb_top_scoreline_1_probability"] * 100
            )
            .round()
            .astype(int),
        }
    )[CLEAN_COLUMNS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False)
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export presentation-ready group-stage predictions."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    clean = export_clean_predictions(input_path=args.input, output_path=args.output)
    print(f"Saved clean group-stage predictions to {args.output}")
    print(f"Rows: {len(clean)}")
    print(f"Columns: {', '.join(clean.columns)}")
    print(clean.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
