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
    "top_5_scorelines",
    "top_5_scoreline_probability_pct",
]


def export_clean_predictions(input_path: Path, output_path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(input_path)
    required = {"match_id", "group", "team_a", "team_b"}
    probability_columns = {"v2_p_team_a_win", "v2_p_draw", "v2_p_team_b_win"}
    percentage_columns = {"team_a_win_pct", "draw_pct", "team_b_win_pct"}
    if probability_columns.issubset(predictions.columns):
        required |= probability_columns
    else:
        required |= percentage_columns

    if "top_5_scorelines" in predictions.columns and "top_5_scoreline_probs" in predictions.columns:
        top5_source = "v24_top5"
    elif all(
        col in predictions.columns
        for col in [
            "nb_top_scoreline_1",
            "nb_top_scoreline_1_probability",
            "nb_top_scoreline_2",
            "nb_top_scoreline_2_probability",
            "nb_top_scoreline_3",
            "nb_top_scoreline_3_probability",
        ]
    ):
        top5_source = "nb_top3"
    else:
        top5_source = "single_scoreline"
        required |= {"predicted_scoreline", "scoreline_probability_pct"}

    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

    if probability_columns.issubset(predictions.columns):
        probability_sum = (
            predictions["v2_p_team_a_win"]
            + predictions["v2_p_draw"]
            + predictions["v2_p_team_b_win"]
        )
        if (probability_sum - 1.0).abs().max() > 1e-9:
            raise ValueError("Official V2 probabilities must sum to 1 before clean export")
        team_a_win_pct = (predictions["v2_p_team_a_win"] * 100).round().astype(int)
        draw_pct = (predictions["v2_p_draw"] * 100).round().astype(int)
        team_b_win_pct = (predictions["v2_p_team_b_win"] * 100).round().astype(int)
    else:
        team_a_win_pct = predictions["team_a_win_pct"].round().astype(int)
        draw_pct = predictions["draw_pct"].round().astype(int)
        team_b_win_pct = predictions["team_b_win_pct"].round().astype(int)

    if top5_source == "v24_top5":
        top_5_scorelines = predictions["top_5_scorelines"]
        top_5_probs = predictions["top_5_scoreline_probs"]
    elif top5_source == "nb_top3":
        top_5_scorelines = predictions.apply(
            lambda row: str(
                [
                    row["nb_top_scoreline_1"],
                    row["nb_top_scoreline_2"],
                    row["nb_top_scoreline_3"],
                ]
            ),
            axis=1,
        )
        top_5_probs = predictions.apply(
            lambda row: str(
                [
                    float(row["nb_top_scoreline_1_probability"]),
                    float(row["nb_top_scoreline_2_probability"]),
                    float(row["nb_top_scoreline_3_probability"]),
                ]
            ),
            axis=1,
        )
    else:
        top_5_scorelines = predictions["predicted_scoreline"].map(lambda value: str([str(value)]))
        top_5_probs = predictions["scoreline_probability_pct"].map(lambda value: str([float(value) / 100]))

    clean = pd.DataFrame(
        {
            "match_id": predictions["match_id"],
            "group": predictions["group"],
            "team_a": predictions["team_a"],
            "team_b": predictions["team_b"],
            "team_a_win_pct": team_a_win_pct,
            "draw_pct": draw_pct,
            "team_b_win_pct": team_b_win_pct,
            "top_5_scorelines": top_5_scorelines,
            "top_5_scoreline_probability_pct": top_5_probs,
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
