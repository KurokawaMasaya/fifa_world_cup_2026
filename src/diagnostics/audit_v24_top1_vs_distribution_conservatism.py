from __future__ import annotations

import argparse
import json
import math
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
    favorite_blowout_mixture_scoreline_grid,
    gated_favorite_blowout_mixture_scoreline_grid,
    top_scorelines,
)
from src.models.negative_binomial_scoreline import negative_binomial_scoreline_grid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "diagnostics"
    / "scoreline_research"
    / "v24_top1_vs_distribution_audit"
)
EPSILON = 1e-15


def parse_scoreline(scoreline: str) -> tuple[int, int]:
    a, b = scoreline.split("-")
    return int(a), int(b)


def outcome(a: int, b: int) -> str:
    if a > b:
        return "team_a_win"
    if a < b:
        return "team_b_win"
    return "draw"


def top_scoreline_rank(grid: dict[str, float], actual_scoreline: str) -> int | None:
    for rank, (scoreline, _) in enumerate(
        sorted(grid.items(), key=lambda item: (-item[1], item[0])),
        start=1,
    ):
        if scoreline == actual_scoreline:
            return rank
    return None


def grid_metrics(grid: dict[str, float], favorite_is_a: bool) -> dict[str, float]:
    expected_total = expected_fav = expected_dog = 0.0
    over15 = over25 = over35 = total5 = low = btts = 0.0
    fav3 = fav4 = fav5 = margin2 = margin3 = margin4 = 0.0
    for scoreline, probability in grid.items():
        a, b = parse_scoreline(scoreline)
        total = a + b
        fav = a if favorite_is_a else b
        dog = b if favorite_is_a else a
        margin = fav - dog
        expected_total += probability * total
        expected_fav += probability * fav
        expected_dog += probability * dog
        over15 += probability if total >= 2 else 0.0
        over25 += probability if total >= 3 else 0.0
        over35 += probability if total >= 4 else 0.0
        total5 += probability if total >= 5 else 0.0
        low += probability if total <= 2 else 0.0
        btts += probability if a > 0 and b > 0 else 0.0
        fav3 += probability if fav >= 3 else 0.0
        fav4 += probability if fav >= 4 else 0.0
        fav5 += probability if fav >= 5 else 0.0
        margin2 += probability if margin >= 2 else 0.0
        margin3 += probability if margin >= 3 else 0.0
        margin4 += probability if margin >= 4 else 0.0
    return {
        "distribution_expected_total_goals": expected_total,
        "distribution_expected_favorite_goals": expected_fav,
        "distribution_expected_underdog_goals": expected_dog,
        "distribution_P_over_1_5": over15,
        "distribution_P_over_2_5": over25,
        "distribution_P_over_3_5": over35,
        "distribution_P_total_goals_5_plus": total5,
        "distribution_P_low_score": low,
        "distribution_P_favorite_scores_3_plus": fav3,
        "distribution_P_favorite_scores_4_plus": fav4,
        "distribution_P_favorite_scores_5_plus": fav5,
        "distribution_P_margin_2_plus": margin2,
        "distribution_P_margin_3_plus": margin3,
        "distribution_P_margin_4_plus": margin4,
        "distribution_P_BTTS": btts,
    }


def actual_metrics(row: pd.Series, favorite_is_a: bool) -> dict[str, object]:
    a = int(row["actual_goals_a"])
    b = int(row["actual_goals_b"])
    fav = a if favorite_is_a else b
    dog = b if favorite_is_a else a
    margin = fav - dog
    total = a + b
    return {
        "actual_mean_total_goals": total,
        "actual_over_1_5": total >= 2,
        "actual_over_2_5": total >= 3,
        "actual_over_3_5": total >= 4,
        "actual_low_score_rate": total <= 2,
        "actual_favorite_3_plus": fav >= 3,
        "actual_favorite_4_plus": fav >= 4,
        "actual_favorite_5_plus": fav >= 5,
        "actual_margin_2_plus": margin >= 2,
        "actual_margin_3_plus": margin >= 3,
        "actual_margin_4_plus": margin >= 4,
        "actual_BTTS": a > 0 and b > 0,
    }


def displayed_metrics(top5: list[dict[str, object]], row: pd.Series, favorite_is_a: bool) -> dict[str, object]:
    top1 = str(top5[0]["scoreline"])
    a, b = parse_scoreline(top1)
    aa = int(row["actual_goals_a"])
    ab = int(row["actual_goals_b"])
    fav = a if favorite_is_a else b
    dog = b if favorite_is_a else a
    margin = fav - dog
    total = a + b
    actual = str(row["actual_scoreline"])
    return {
        "displayed_scoreline": top1,
        "displayed_scoreline_prob": float(top5[0]["probability"]),
        "displayed_exact_top1": top1 == actual,
        "displayed_top3": actual in [str(x["scoreline"]) for x in top5[:3]],
        "displayed_top5": actual in [str(x["scoreline"]) for x in top5],
        "displayed_winner_direction": outcome(a, b) == outcome(aa, ab),
        "displayed_mean_total_goals": total,
        "displayed_over_1_5_rate": total >= 2,
        "displayed_over_2_5_rate": total >= 3,
        "displayed_over_3_5_rate": total >= 4,
        "displayed_low_score_rate": total <= 2,
        "displayed_favorite_3_plus_rate": fav >= 3,
        "displayed_favorite_4_plus_rate": fav >= 4,
        "displayed_favorite_5_plus_rate": fav >= 5,
        "displayed_margin_2_plus_rate": margin >= 2,
        "displayed_margin_3_plus_rate": margin >= 3,
        "displayed_margin_4_plus_rate": margin >= 4,
    }


def build_model_grid(row: pd.Series, model: str) -> tuple[dict[str, float], dict[str, object]]:
    if model == "v20_stable":
        grid = negative_binomial_scoreline_grid(
            lambda_a=float(row["lambda_a"]),
            lambda_b=float(row["lambda_b"]),
            dispersion_k=12.0,
            max_goals=10,
            aggressiveness=0.0,
        )
        return dict(grid), {}
    if model == "v23_base":
        grid, meta = favorite_blowout_mixture_scoreline_grid(
            lambda_a=float(row["lambda_a"]),
            lambda_b=float(row["lambda_b"]),
            p_team_a_win=float(row["p_team_a_win"]),
            p_draw=float(row["p_draw"]),
            p_team_b_win=float(row["p_team_b_win"]),
            rating_gap=float(row["strength_diff"]),
            max_goals=10,
            params=BlowoutMixtureParams(
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
        return dict(grid), meta
    if model == "v24_abcd_no_E":
        grid, meta = gated_favorite_blowout_mixture_scoreline_grid(
            lambda_a=float(row["lambda_a"]),
            lambda_b=float(row["lambda_b"]),
            p_team_a_win=float(row["p_team_a_win"]),
            p_draw=float(row["p_draw"]),
            p_team_b_win=float(row["p_team_b_win"]),
            rating_gap=float(row["strength_diff"]),
            max_goals=10,
            params=GatedBlowoutParams(
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
                use_motivation_gate=False,
            ),
        )
        return dict(grid), meta
    raise ValueError(f"Unknown model: {model}")


def build_match_level(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in base.iterrows():
        favorite_is_a = row["favorite_team"] == row["team_a"]
        for model in ["v20_stable", "v23_base", "v24_abcd_no_E"]:
            grid, meta = build_model_grid(row, model)
            top10 = top_scorelines(grid, top_n=10)
            top5 = top10[:5]
            actual_scoreline = str(row["actual_scoreline"])
            p_actual = max(EPSILON, float(grid.get(actual_scoreline, 0.0)))
            output = {
                "model": model,
                "match_id": row["match_id"],
                "date": row["date"],
                "team_a": row["team_a"],
                "team_b": row["team_b"],
                "favorite": row["favorite_team"],
                "favorite_bucket": row["favorite_bucket"],
                "actual_scoreline": actual_scoreline,
                "top_5_scorelines": json.dumps([x["scoreline"] for x in top5]),
                "top_5_probs": json.dumps([x["probability"] for x in top5]),
                "top_10_scorelines": json.dumps([x["scoreline"] for x in top10]),
                "top_10_probs": json.dumps([x["probability"] for x in top10]),
                "actual_scoreline_rank": top_scoreline_rank(grid, actual_scoreline),
                "probability_assigned_to_actual_scoreline": p_actual,
                "log_probability_of_actual_scoreline": math.log(p_actual),
                "p_blowout": meta.get("p_blowout", meta.get("p_blowout_final", 0.0)),
            }
            output.update(displayed_metrics(top5, row, favorite_is_a))
            output.update(grid_metrics(grid, favorite_is_a))
            output.update(actual_metrics(row, favorite_is_a))
            rows.append(output)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict[str, object]:
    bool_cols = [
        "displayed_exact_top1",
        "displayed_top3",
        "displayed_top5",
        "displayed_winner_direction",
        "displayed_over_1_5_rate",
        "displayed_over_2_5_rate",
        "displayed_over_3_5_rate",
        "displayed_low_score_rate",
        "displayed_favorite_3_plus_rate",
        "displayed_favorite_4_plus_rate",
        "displayed_favorite_5_plus_rate",
        "displayed_margin_2_plus_rate",
        "displayed_margin_3_plus_rate",
        "displayed_margin_4_plus_rate",
        "actual_over_1_5",
        "actual_over_2_5",
        "actual_over_3_5",
        "actual_low_score_rate",
        "actual_favorite_3_plus",
        "actual_favorite_4_plus",
        "actual_favorite_5_plus",
        "actual_margin_2_plus",
        "actual_margin_3_plus",
        "actual_margin_4_plus",
        "actual_BTTS",
    ]
    row = {"model": df["model"].iloc[0], "sample_size": len(df)}
    for col in bool_cols:
        row[col] = df[col].mean()
    numeric_cols = [
        "displayed_mean_total_goals",
        "distribution_expected_total_goals",
        "distribution_expected_favorite_goals",
        "distribution_expected_underdog_goals",
        "distribution_P_over_1_5",
        "distribution_P_over_2_5",
        "distribution_P_over_3_5",
        "distribution_P_total_goals_5_plus",
        "distribution_P_low_score",
        "distribution_P_favorite_scores_3_plus",
        "distribution_P_favorite_scores_4_plus",
        "distribution_P_favorite_scores_5_plus",
        "distribution_P_margin_2_plus",
        "distribution_P_margin_3_plus",
        "distribution_P_margin_4_plus",
        "distribution_P_BTTS",
        "log_probability_of_actual_scoreline",
        "probability_assigned_to_actual_scoreline",
        "actual_scoreline_rank",
        "actual_mean_total_goals",
    ]
    for col in numeric_cols:
        row[col] = df[col].mean()
    return row


def selector_failures(match_level: pd.DataFrame) -> pd.DataFrame:
    v24 = match_level.loc[match_level["model"].eq("v24_abcd_no_E")].copy()
    v24["actual_total"] = v24["actual_scoreline"].map(lambda s: sum(parse_scoreline(s)))
    v24["displayed_total"] = v24["displayed_scoreline"].map(lambda s: sum(parse_scoreline(s)))
    def high_in_candidates(row: pd.Series) -> bool:
        scores = json.loads(row["top_10_scorelines"])
        return any(sum(parse_scoreline(score)) >= 3 for score in scores)
    failures = v24.loc[
        (v24["actual_total"] >= 3)
        & (v24["displayed_total"] <= 2)
        & v24.apply(high_in_candidates, axis=1)
    ].copy()
    return failures[
        [
            "model",
            "match_id",
            "team_a",
            "team_b",
            "favorite",
            "actual_scoreline",
            "displayed_scoreline",
            "displayed_scoreline_prob",
            "top_5_scorelines",
            "top_5_probs",
            "actual_scoreline_rank",
            "distribution_P_over_2_5",
            "distribution_P_margin_3_plus",
            "favorite_bucket",
        ]
    ]


def write_report(summary: pd.DataFrame, by_bucket: pd.DataFrame, output_dir: Path) -> None:
    v24 = summary.loc[summary["model"].eq("v24_abcd_no_E")].iloc[0]
    lines = [
        "# V2.4 Top-1 vs Distribution Conservatism Audit",
        "",
        "This diagnostic compares displayed top-1 scorelines against the full scoreline probability distribution.",
        "",
        "## Main Finding",
        "",
        (
            "V2.4 ABCD is much less conservative in the full distribution than the displayed top-1 "
            "scoreline suggests. The top-1 selector is the main reason displayed over-2.5 and "
            "favorite 3+ rates look extremely low."
        ),
        "",
        "## V2.4 Key Metrics",
        "",
        f"- Displayed over-2.5: {v24['displayed_over_2_5_rate']:.3f}",
        f"- Distribution P(over-2.5): {v24['distribution_P_over_2_5']:.3f}",
        f"- Actual over-2.5: {v24['actual_over_2_5']:.3f}",
        f"- Displayed low-score rate: {v24['displayed_low_score_rate']:.3f}",
        f"- Distribution P(low-score): {v24['distribution_P_low_score']:.3f}",
        f"- Actual low-score rate: {v24['actual_low_score_rate']:.3f}",
        f"- Displayed favorite 3+: {v24['displayed_favorite_3_plus_rate']:.3f}",
        f"- Distribution P(favorite 3+): {v24['distribution_P_favorite_scores_3_plus']:.3f}",
        f"- Actual favorite 3+: {v24['actual_favorite_3_plus']:.3f}",
        "",
        "## Recommendation",
        "",
        (
            "Do not tune distribution parameters yet. The next improvement should target output "
            "presentation or a calibrated selector that surfaces top-5/tail-risk information, "
            "because forcing top-1 upward has already damaged exact accuracy in micro-tuning."
        ),
        "",
        "Suggested next directions:",
        "",
        "- A. Distribution calibration: not first priority; distribution is less conservative than top-1.",
        "- B. Scoreline selector: promising but must be learned/validated, not rule-forced.",
        "- C. Output presentation: strongest immediate option; show top-5 and tail-risk.",
        "- D. Richer input features: useful later for selecting among close-probability scorelines.",
    ]
    (output_dir / "v24_top1_vs_distribution_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V2.4 top-1 vs full-distribution conservatism.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--start-date", default="2014-01-01")
    args = parser.parse_args()

    matches = load_matches(args.input, start_date=args.start_date)
    split_prediction_and_evaluation_frames(matches)
    base = build_predictions(
        matches=matches,
        model_config=load_model_config("default"),
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
    match_level = build_match_level(base)
    summary = pd.DataFrame(
        [summarize(df) for _, df in match_level.groupby("model", sort=False)]
    )
    by_bucket = pd.DataFrame(
        [
            {**summarize(df), "favorite_bucket": bucket}
            for (_, bucket), df in match_level.groupby(["model", "favorite_bucket"], sort=False)
        ]
    )
    examples = selector_failures(match_level)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "v24_top1_vs_distribution_summary.csv", index=False)
    by_bucket.to_csv(OUTPUT_DIR / "v24_top1_vs_distribution_by_bucket.csv", index=False)
    match_level.to_csv(OUTPUT_DIR / "v24_top1_vs_distribution_match_level.csv", index=False)
    examples.to_csv(OUTPUT_DIR / "v24_selector_failure_examples.csv", index=False)
    write_report(summary, by_bucket, OUTPUT_DIR)
    print("Summary:")
    print(summary.to_string(index=False))
    print("Selector failures:", len(examples))
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
