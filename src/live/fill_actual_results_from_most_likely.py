from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.poisson_match_model import (  # noqa: E402
    most_likely_scoreline,
    scoreline_probabilities,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = PROJECT_ROOT / "output" / "live"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "output" / "predictions" / "group_stage_prediction_confidence.csv"
DEFAULT_ACTUAL_RESULTS_PATH = LIVE_DIR / "worldcup_group_stage_actual_results.csv"


def actual_result_from_goals(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "team_a_win"
    if goals_a < goals_b:
        return "team_b_win"
    return "draw"


def fill_actual_results_from_most_likely(
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    output_path: Path = DEFAULT_ACTUAL_RESULTS_PATH,
    max_goals: int = 8,
) -> pd.DataFrame:
    """Fill the live results template with deterministic most-likely scorelines.

    This is only for model sanity checks before real World Cup results are
    entered. It does not sample outcomes and must not be treated as live truth.
    The actual_result column is intentionally left empty for manual entry.
    """
    predictions = pd.read_csv(predictions_path)
    required = {"match_id", "team_a", "team_b", "lambda_a", "lambda_b"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")

    rows = []
    for _, row in predictions.sort_values("match_id").iterrows():
        score_probs = scoreline_probabilities(
            lambda_a=float(row["lambda_a"]),
            lambda_b=float(row["lambda_b"]),
            max_goals=max_goals,
        )
        goals_a, goals_b = most_likely_scoreline(score_probs)[0]
        rows.append(
            {
                "match_id": int(row["match_id"]),
                "team_a": row["team_a"],
                "team_b": row["team_b"],
                "goals_a": goals_a,
                "goals_b": goals_b,
                "actual_result": "",
            }
        )

    output = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill live group-stage actual results with model-most-likely scorelines."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_ACTUAL_RESULTS_PATH)
    parser.add_argument("--max-goals", type=int, default=8)
    args = parser.parse_args()

    output = fill_actual_results_from_most_likely(
        predictions_path=args.predictions,
        output_path=args.output,
        max_goals=args.max_goals,
    )
    print(f"Filled goals for {len(output)} rows from most-likely scorelines.")
    print(f"Saved to {args.output}")
    print("actual_result was left blank for manual entry.")


if __name__ == "__main__":
    main()
