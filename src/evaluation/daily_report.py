from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.evaluate_live_prediction_vs_actual import (  # noqa: E402
    DEFAULT_ACTUALS_PATH,
    DEFAULT_CLOSE_WIN_RATE_THRESHOLD,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PREDICTIONS_PATH,
    evaluate,
)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Daily CupCast report: compare completed match results with W/D/L "
            "and top-1/top-3/top-5 scoreline predictions."
        )
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--actuals", type=Path, default=DEFAULT_ACTUALS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--close-win-rate-threshold",
        type=float,
        default=DEFAULT_CLOSE_WIN_RATE_THRESHOLD,
        help=(
            "Daily-report W/D/L credit threshold. If team win rates are within "
            "this many percentage points, exact draw top-1 hits and higher-win-side "
            "results receive W/D/L credit."
        ),
    )
    args = parser.parse_args()

    match_level, summary = evaluate(
        args.predictions,
        args.actuals,
        args.output_dir,
        close_win_rate_threshold=args.close_win_rate_threshold,
    )

    print("\nDaily Report Summary")
    print("====================")
    if summary.empty or int(summary.iloc[0]["matches_evaluated"]) == 0:
        print("No completed matches available for evaluation.")
        return

    row = summary.iloc[0]
    print(f"Matches evaluated: {int(row['matches_evaluated'])}")
    print(f"W/D/L correct probability: {pct(float(row['wdl_correct_rate']))}")
    print(
        "Mean probability assigned to actual W/D/L outcome: "
        f"{float(row['mean_wdl_actual_outcome_probability_pct']):.2f}%"
    )
    print(f"Scoreline exact top-1 probability: {pct(float(row['scoreline_exact_top1_rate']))}")
    print(f"Scoreline top-3 probability: {pct(float(row['scoreline_top3_rate']))}")
    print(f"Scoreline top-5 probability: {pct(float(row['scoreline_top5_rate']))}")
    print(f"Exact-draw W/D/L credit count: {int(row['wdl_exact_draw_credit_count'])}")

    display_cols = [
        "match_id",
        "team_a",
        "team_b",
        "actual_scoreline",
        "team_a_win_pct",
        "draw_pct",
        "team_b_win_pct",
        "wdl_argmax_outcome",
        "top1_scoreline",
        "wdl_direction_correct",
        "exact_top1_correct",
        "top3_correct",
        "top5_correct",
    ]
    print("\nCompleted Match Detail")
    print(match_level[display_cols].to_string(index=False))
    print(f"\nSaved match-level CSV: {args.output_dir / 'live_prediction_vs_actual_match_level.csv'}")
    print(f"Saved summary CSV: {args.output_dir / 'live_prediction_vs_actual_summary.csv'}")


if __name__ == "__main__":
    main()
