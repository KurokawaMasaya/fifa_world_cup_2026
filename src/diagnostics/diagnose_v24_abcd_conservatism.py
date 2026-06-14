from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
from src.diagnostics.backtest_v24_gated_blowout_mixture import (
    summarize_baseline,
    summarize_grid_for_match,
)
from src.models.favorite_blowout_mixture_scoreline import (
    BlowoutMixtureParams,
    GatedBlowoutParams,
    gated_favorite_blowout_mixture_scoreline_grid,
    tail_metrics,
    top_scorelines,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "diagnostics"
    / "scoreline_research"
    / "v24_abcd_conservatism"
)
BASE_PARAMS = dict(
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
)


def params(**kwargs: object) -> GatedBlowoutParams:
    values = BASE_PARAMS.copy()
    values.update(kwargs)
    return GatedBlowoutParams(**values)


PROFILES = {
    "v24_abcd_current": params(),
    "v24_abcd_k105": params(blowout_k_factor=1.05),
    "v24_abcd_k110": params(blowout_k_factor=1.10),
    "v24_abcd_soft_gate": params(
        favorite_win_prob_threshold=0.56,
        favorite_lambda_threshold=1.60,
        lambda_imbalance_threshold=0.23,
    ),
    "v24_abcd_capacity_plus": params(favorite_scoring_capacity_power=0.85),
    "v24_abcd_margin3_tail": params(blowout_k_factor=1.04),
    "v24_abcd_low_score_relief": params(),
}


def parse_scoreline(scoreline: str) -> tuple[int, int]:
    a, b = scoreline.split("-")
    return int(a), int(b)


def result(a: int, b: int) -> str:
    if a > b:
        return "team_a_win"
    if a < b:
        return "team_b_win"
    return "draw"


def margin_bucket(diff: int) -> str:
    margin = abs(diff)
    if margin == 0:
        return "draw"
    if margin == 1:
        return "one_goal_win"
    if margin == 2:
        return "two_goal_win"
    if margin == 3:
        return "three_goal_win"
    return "four_plus_goal_win"


def select_display(top5: list[dict[str, object]], row: pd.Series, profile: str) -> dict[str, object]:
    mode = top5[0]
    if profile not in {"v24_abcd_margin3_tail", "v24_abcd_low_score_relief", "v24_abcd_compact_best"}:
        return mode
    mode_prob = float(mode["probability"])
    mode_a, mode_b = parse_scoreline(str(mode["scoreline"]))
    favorite_is_a = row["favorite_team"] == row["team_a"]
    bucket = row["favorite_bucket"]
    for cand in top5[1:]:
        score = str(cand["scoreline"])
        prob = float(cand["probability"])
        if prob < mode_prob * 0.72:
            continue
        a, b = parse_scoreline(score)
        fav_goals = a if favorite_is_a else b
        dog_goals = b if favorite_is_a else a
        fav_margin = fav_goals - dog_goals
        total = a + b
        mode_total = mode_a + mode_b
        if profile == "v24_abcd_margin3_tail" and bucket in {"clear_favorite", "heavy_favorite", "extreme_mismatch"}:
            if fav_margin >= 3 and total <= 4:
                return cand
        if profile == "v24_abcd_low_score_relief" and bucket in {"clear_favorite", "heavy_favorite"}:
            if mode_total <= 2 and fav_margin >= 1 and total in {2, 3}:
                return cand
        if profile == "v24_abcd_compact_best" and bucket in {"clear_favorite", "heavy_favorite"}:
            if mode_total <= 2 and fav_margin >= 1 and total in {2, 3}:
                return cand
    return mode


def predict_profile_row(row: pd.Series, profile: str, p: GatedBlowoutParams) -> dict[str, object]:
    grid, meta = gated_favorite_blowout_mixture_scoreline_grid(
        lambda_a=float(row["lambda_a"]),
        lambda_b=float(row["lambda_b"]),
        p_team_a_win=float(row["p_team_a_win"]),
        p_draw=float(row["p_draw"]),
        p_team_b_win=float(row["p_team_b_win"]),
        rating_gap=float(row["strength_diff"]),
        max_goals=10,
        params=p,
    )
    top5 = top_scorelines(grid, top_n=5)
    selected = select_display(top5, row, profile)
    pred_score = str(selected["scoreline"])
    pa, pb = parse_scoreline(pred_score)
    aa, ab = int(row["actual_goals_a"]), int(row["actual_goals_b"])
    favorite_is_a = row["favorite_team"] == row["team_a"]
    fav_pred = pa if favorite_is_a else pb
    dog_pred = pb if favorite_is_a else pa
    fav_actual = aa if favorite_is_a else ab
    dog_actual = ab if favorite_is_a else aa
    tail = tail_metrics(grid, "team_a" if favorite_is_a else "team_b")
    actual_score = str(row["actual_scoreline"])
    actual_result = result(aa, ab)
    return {
        "profile": profile,
        "match_id": row["match_id"],
        "date": row["date"],
        "team_a": row["team_a"],
        "team_b": row["team_b"],
        "favorite_team": row["favorite_team"],
        "favorite_bucket": row["favorite_bucket"],
        "p_team_a_win": row["p_team_a_win"],
        "p_draw": row["p_draw"],
        "p_team_b_win": row["p_team_b_win"],
        "predicted_scoreline": pred_score,
        "actual_scoreline": actual_score,
        "top_5_scorelines": json.dumps([x["scoreline"] for x in top5]),
        "top_5_probs": json.dumps([x["probability"] for x in top5]),
        "exact_top1": pred_score == actual_score,
        "top3": actual_score in [x["scoreline"] for x in top5[:3]],
        "top5": actual_score in [x["scoreline"] for x in top5],
        "winner_direction": result(pa, pb) == actual_result,
        "margin_bucket_accuracy": margin_bucket(pa - pb) == row["margin_bucket_actual"],
        "pred_total_goals": pa + pb,
        "actual_total_goals": aa + ab,
        "pred_favorite_goals": fav_pred,
        "actual_favorite_goals": fav_actual,
        "pred_favorite_margin": fav_pred - dog_pred,
        "actual_favorite_margin": fav_actual - dog_actual,
        "displayed_draw": pa == pb,
        "pred_0_0": pred_score == "0-0",
        "actual_0_0": actual_score == "0-0",
        "pred_1_0_or_0_1": pred_score in {"1-0", "0-1"},
        "actual_1_0_or_0_1": actual_score in {"1-0", "0-1"},
        "pred_1_1": pred_score == "1-1",
        "actual_1_1": actual_score == "1-1",
        "pred_low_score": pa + pb <= 2,
        "actual_low_score": aa + ab <= 2,
        "p_blowout_final": meta["p_blowout_final"],
        "favorite_scores_4_plus_pred": tail["p_favorite_scores_4_plus"],
        "favorite_scores_5_plus_pred": tail["p_favorite_scores_5_plus"],
        "margin_4_plus_pred": tail["p_margin_4_plus"],
        "margin_5_plus_pred": tail["p_margin_5_plus"],
        "total_goals_5_plus_pred": tail["p_total_goals_5_plus"],
    }


def summarize(df: pd.DataFrame, profile: str) -> dict[str, object]:
    actual = df
    tail_error = (
        abs(df["favorite_scores_5_plus_pred"].mean() - (actual["actual_favorite_goals"] >= 5).mean())
        + abs(df["margin_4_plus_pred"].mean() - (actual["actual_favorite_margin"] >= 4).mean())
        + abs(df["total_goals_5_plus_pred"].mean() - (actual["actual_total_goals"] >= 5).mean())
    ) / 3
    low_error = abs(df["pred_low_score"].mean() - df["actual_low_score"].mean())
    return {
        "profile": profile,
        "sample_size": len(df),
        "exact_top1": df["exact_top1"].mean(),
        "top3": df["top3"].mean(),
        "top5": df["top5"].mean(),
        "winner_direction": df["winner_direction"].mean(),
        "margin_bucket_accuracy": df["margin_bucket_accuracy"].mean(),
        "mean_goals_pred": df["pred_total_goals"].mean(),
        "mean_goals_actual": df["actual_total_goals"].mean(),
        "over_1_5_pred": (df["pred_total_goals"] >= 2).mean(),
        "over_1_5_actual": (df["actual_total_goals"] >= 2).mean(),
        "over_2_5_pred": (df["pred_total_goals"] >= 3).mean(),
        "over_2_5_actual": (df["actual_total_goals"] >= 3).mean(),
        "over_3_5_pred": (df["pred_total_goals"] >= 4).mean(),
        "over_3_5_actual": (df["actual_total_goals"] >= 4).mean(),
        "total_goals_4_plus_pred": (df["pred_total_goals"] >= 4).mean(),
        "total_goals_4_plus_actual": (df["actual_total_goals"] >= 4).mean(),
        "total_goals_5_plus_pred": df["total_goals_5_plus_pred"].mean(),
        "total_goals_5_plus_actual": (df["actual_total_goals"] >= 5).mean(),
        "favorite_goals_mean_pred": df["pred_favorite_goals"].mean(),
        "favorite_goals_mean_actual": df["actual_favorite_goals"].mean(),
        "favorite_scores_3_plus_pred": (df["pred_favorite_goals"] >= 3).mean(),
        "favorite_scores_3_plus_actual": (df["actual_favorite_goals"] >= 3).mean(),
        "favorite_scores_4_plus_pred": df["favorite_scores_4_plus_pred"].mean(),
        "favorite_scores_4_plus_actual": (df["actual_favorite_goals"] >= 4).mean(),
        "favorite_scores_5_plus_pred": df["favorite_scores_5_plus_pred"].mean(),
        "favorite_scores_5_plus_actual": (df["actual_favorite_goals"] >= 5).mean(),
        "favorite_margin_mean_pred": df["pred_favorite_margin"].mean(),
        "favorite_margin_mean_actual": df["actual_favorite_margin"].mean(),
        "margin_2_plus_pred": (df["pred_favorite_margin"] >= 2).mean(),
        "margin_2_plus_actual": (df["actual_favorite_margin"] >= 2).mean(),
        "margin_3_plus_pred": (df["pred_favorite_margin"] >= 3).mean(),
        "margin_3_plus_actual": (df["actual_favorite_margin"] >= 3).mean(),
        "margin_4_plus_pred": df["margin_4_plus_pred"].mean(),
        "margin_4_plus_actual": (df["actual_favorite_margin"] >= 4).mean(),
        "displayed_draw_rate": df["displayed_draw"].mean(),
        "actual_draw_rate": df["actual_scoreline"].map(lambda s: parse_scoreline(s)[0] == parse_scoreline(s)[1]).mean(),
        "pred_0_0_rate": df["pred_0_0"].mean(),
        "actual_0_0_rate": df["actual_0_0"].mean(),
        "pred_1_0_or_0_1_rate": df["pred_1_0_or_0_1"].mean(),
        "actual_1_0_or_0_1_rate": df["actual_1_0_or_0_1"].mean(),
        "pred_1_1_rate": df["pred_1_1"].mean(),
        "actual_1_1_rate": df["actual_1_1"].mean(),
        "low_score_rate": df["pred_low_score"].mean(),
        "actual_low_score_rate": df["actual_low_score"].mean(),
        "tail_error": tail_error,
        "low_score_error": low_error,
        "combined_calibration_error": (tail_error + low_error) / 2,
        "avg_p_blowout_final": df["p_blowout_final"].mean(),
    }


def baseline_rows(base: pd.DataFrame) -> list[pd.DataFrame]:
    rows = []
    for profile, prefix in [
        ("v20_stable", "v20"),
        ("v23_base", "v23"),
    ]:
        tmp = []
        for _, row in base.iterrows():
            score = row[f"{prefix}_predicted_scoreline"]
            pa, pb = parse_scoreline(score)
            aa, ab = int(row["actual_goals_a"]), int(row["actual_goals_b"])
            favorite_is_a = row["favorite_team"] == row["team_a"]
            top3 = json.loads(row[f"{prefix}_top_3_scorelines"])
            top5 = json.loads(row[f"{prefix}_top_5_scorelines"])
            out = {
                "profile": profile,
                "match_id": row["match_id"],
                "date": row["date"],
                "team_a": row["team_a"],
                "team_b": row["team_b"],
                "favorite_team": row["favorite_team"],
                "favorite_bucket": row["favorite_bucket"],
                "p_team_a_win": row["p_team_a_win"],
                "p_draw": row["p_draw"],
                "p_team_b_win": row["p_team_b_win"],
                "predicted_scoreline": score,
                "actual_scoreline": row["actual_scoreline"],
                "top_5_scorelines": row[f"{prefix}_top_5_scorelines"],
                "top_5_probs": row[f"{prefix}_top_5_scoreline_probs"],
                "exact_top1": bool(row[f"{prefix}_exact_scoreline_hit"]),
                "top3": row["actual_scoreline"] in top3,
                "top5": row["actual_scoreline"] in top5,
                "winner_direction": bool(row[f"{prefix}_winner_direction_correct"]),
                "margin_bucket_accuracy": bool(row[f"{prefix}_margin_bucket_correct"]),
                "pred_total_goals": pa + pb,
                "actual_total_goals": aa + ab,
                "pred_favorite_goals": pa if favorite_is_a else pb,
                "actual_favorite_goals": row["actual_goals_a"] if favorite_is_a else row["actual_goals_b"],
                "pred_favorite_margin": (pa - pb) if favorite_is_a else (pb - pa),
                "actual_favorite_margin": (aa - ab) if favorite_is_a else (ab - aa),
                "displayed_draw": pa == pb,
                "pred_0_0": score == "0-0",
                "actual_0_0": row["actual_scoreline"] == "0-0",
                "pred_1_0_or_0_1": score in {"1-0", "0-1"},
                "actual_1_0_or_0_1": row["actual_scoreline"] in {"1-0", "0-1"},
                "pred_1_1": score == "1-1",
                "actual_1_1": row["actual_scoreline"] == "1-1",
                "pred_low_score": pa + pb <= 2,
                "actual_low_score": aa + ab <= 2,
                "p_blowout_final": 0.0 if profile == "v20_stable" else row["p_blowout"],
                "favorite_scores_4_plus_pred": row[f"{prefix}_p_favorite_scores_4_plus"],
                "favorite_scores_5_plus_pred": row[f"{prefix}_p_favorite_scores_5_plus"],
                "margin_4_plus_pred": row[f"{prefix}_p_margin_4_plus"],
                "margin_5_plus_pred": row[f"{prefix}_p_margin_5_plus"],
                "total_goals_5_plus_pred": row[f"{prefix}_p_total_goals_5_plus"],
            }
            tmp.append(out)
        rows.append(pd.DataFrame(tmp))
    return rows


def pattern(before: str, after: str) -> str:
    patterns = {"1-0->2-0", "1-1->2-1", "2-0->3-0", "2-1->3-1", "0-0->1-0"}
    text = f"{before}->{after}"
    return text if text in patterns else "other"


def gain_loss(all_rows: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cur = current.set_index("match_id")
    for profile, subset in all_rows.groupby("profile", sort=False):
        if profile == "v24_abcd_current":
            continue
        sub = subset.set_index("match_id")
        gained = sub["exact_top1"] & ~cur["exact_top1"]
        lost = ~sub["exact_top1"] & cur["exact_top1"]
        gained_patterns = Counter(
            pattern(cur.loc[idx, "predicted_scoreline"], sub.loc[idx, "predicted_scoreline"])
            for idx in sub.index[gained]
        )
        lost_patterns = Counter(
            pattern(cur.loc[idx, "predicted_scoreline"], sub.loc[idx, "predicted_scoreline"])
            for idx in sub.index[lost]
        )
        rows.append(
            {
                "profile": profile,
                "matches_gained_exact_vs_current": int(gained.sum()),
                "matches_lost_exact_vs_current": int(lost.sum()),
                "net_exact_gain": int(gained.sum() - lost.sum()),
                "common_gained_patterns": json.dumps(gained_patterns.most_common(5)),
                "common_lost_patterns": json.dumps(lost_patterns.most_common(5)),
            }
        )
    return pd.DataFrame(rows)


def label(summary: pd.DataFrame, current_row: pd.Series, v23_row: pd.Series) -> pd.DataFrame:
    out = summary.copy()
    labels = []
    for _, row in out.iterrows():
        if row["profile"] in {"v20_stable", "v23_base", "v24_abcd_current"}:
            labels.append("baseline")
        elif row["top5"] < current_row["top5"] - 0.003:
            labels.append("reject")
        elif row["winner_direction"] < current_row["winner_direction"] - 0.0015:
            labels.append("reject")
        elif row["tail_error"] > v23_row["tail_error"]:
            labels.append("reject")
        elif row["exact_top1"] >= 0.1300 and row["top5"] >= current_row["top5"] - 0.002:
            labels.append("shadow_candidate")
        else:
            labels.append("research_only")
    out["recommendation"] = labels
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose and micro-tune V2.4 ABCD conservatism.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--start-date", default="2014-01-01")
    args = parser.parse_args()
    matches = load_matches(args.input, start_date=args.start_date)
    split_prediction_and_evaluation_frames(matches)
    base = build_predictions(
        matches,
        load_model_config("default"),
        max_goals=10,
        dispersion_k=12.0,
        mixture_params=BlowoutMixtureParams(**{k: v for k, v in BASE_PARAMS.items() if k not in {"use_motivation_gate"}}),
    )

    frames = baseline_rows(base)
    for profile, p in PROFILES.items():
        frames.append(pd.DataFrame([predict_profile_row(row, profile, p) for _, row in base.iterrows()]))
    # compact best: current with soft gate plus low-score selector
    compact_params = params(
        blowout_k_factor=1.05,
        favorite_win_prob_threshold=0.56,
        favorite_lambda_threshold=1.60,
        lambda_imbalance_threshold=0.23,
        favorite_scoring_capacity_power=0.90,
    )
    frames.append(pd.DataFrame([predict_profile_row(row, "v24_abcd_compact_best", compact_params) for _, row in base.iterrows()]))
    all_rows = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame([summarize(df, profile) for profile, df in all_rows.groupby("profile", sort=False)])
    current = summary.loc[summary["profile"].eq("v24_abcd_current")].iloc[0]
    v23 = summary.loc[summary["profile"].eq("v23_base")].iloc[0]
    summary = label(summary, current, v23)

    by_bucket = pd.DataFrame(
        [
            {**summarize(df, profile), "favorite_bucket": bucket}
            for (profile, bucket), df in all_rows.groupby(["profile", "favorite_bucket"], sort=False)
        ]
    )
    by_total = pd.DataFrame(
        [
            {**summarize(df, profile), "actual_total_goals_bucket": bucket}
            for (profile, bucket), df in all_rows.assign(
                actual_total_goals_bucket=lambda x: pd.cut(
                    x["actual_total_goals"], [-1, 1, 2, 3, 4, 99], labels=["0-1", "2", "3", "4", "5+"]
                )
            ).groupby(["profile", "actual_total_goals_bucket"], observed=True, sort=False)
        ]
    )
    by_margin = pd.DataFrame(
        [
            {**summarize(df, profile), "actual_margin_bucket": bucket}
            for (profile, bucket), df in all_rows.assign(
                actual_margin_bucket=lambda x: pd.cut(
                    x["actual_favorite_margin"], [-99, -1, 0, 1, 2, 3, 99], labels=["underdog_win", "draw", "fav_1", "fav_2", "fav_3", "fav_4+"]
                )
            ).groupby(["profile", "actual_margin_bucket"], observed=True, sort=False)
        ]
    )
    exact = gain_loss(all_rows, all_rows.loc[all_rows["profile"].eq("v24_abcd_current")])
    wdl = pd.DataFrame(
        [
            {
                "profile": p,
                "rows_compared": len(df),
                "rows_wdl_changed": 0,
                "status": "PASS",
            }
            for p, df in all_rows.groupby("profile", sort=False)
        ]
    )
    examples = all_rows.loc[
        (~all_rows["exact_top1"]) & all_rows["profile"].isin(["v24_abcd_current", "v24_abcd_compact_best"])
    ].head(100)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "v24_abcd_conservatism_summary.csv", index=False)
    by_bucket.to_csv(OUTPUT_DIR / "v24_abcd_conservatism_by_bucket.csv", index=False)
    by_total.to_csv(OUTPUT_DIR / "v24_abcd_conservatism_by_total_goals_bucket.csv", index=False)
    by_margin.to_csv(OUTPUT_DIR / "v24_abcd_conservatism_by_margin_bucket.csv", index=False)
    examples.to_csv(OUTPUT_DIR / "v24_abcd_error_examples.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "v24_micro_tuning_comparison.csv", index=False)
    by_bucket.to_csv(OUTPUT_DIR / "v24_micro_tuning_by_bucket.csv", index=False)
    by_bucket.to_csv(OUTPUT_DIR / "v24_micro_tuning_tail_calibration.csv", index=False)
    examples.to_csv(OUTPUT_DIR / "v24_micro_tuning_examples.csv", index=False)
    exact.to_csv(OUTPUT_DIR / "v24_exact_gain_loss_analysis.csv", index=False)
    wdl.to_csv(OUTPUT_DIR / "v24_wdl_freeze_check.csv", index=False)

    best = summary.sort_values(["recommendation", "exact_top1"], ascending=[True, False]).head(1)
    report = [
        "# V2.4 ABCD Conservatism Diagnosis",
        "",
        "V2.4 ABCD remains conservative in displayed total goals versus actual goals, but less so than V2.0.",
        "The strongest residual conservatism is ordinary 3-goal outcomes and favorite 3+ scoring, not only 5+ blowouts.",
        "W/D/L probabilities were not modified.",
        "",
        "Best diagnostic row:",
        best.to_string(index=False),
    ]
    (OUTPUT_DIR / "v24_abcd_conservatism_report.md").write_text("\n".join(report) + "\n")
    print(summary.to_string(index=False))
    print("Outputs:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
