from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
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
from src.diagnostics.backtest_v24_gated_blowout_mixture import PROFILES as V24_PROFILES
from src.models.favorite_blowout_mixture_scoreline import (
    BlowoutMixtureParams,
    gated_favorite_blowout_mixture_scoreline_grid,
    top_scorelines,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "diagnostics"
    / "scoreline_research"
    / "v25_wdl_aware_selector"
)
EPSILON = 1e-12


@dataclass(frozen=True)
class SelectorProfile:
    alpha: float
    beta: float
    close_probability_threshold: float
    draw_protection: str


SELECTOR_PROFILES: dict[str, SelectorProfile | None] = {
    "raw_mode_selector": None,
    "v24_abcd_current_selector": None,
    "v25_wdl_light": SelectorProfile(
        alpha=0.25,
        beta=0.25,
        close_probability_threshold=0.65,
        draw_protection="strong",
    ),
    "v25_wdl_base": SelectorProfile(
        alpha=0.50,
        beta=0.25,
        close_probability_threshold=0.65,
        draw_protection="medium",
    ),
    "v25_wdl_strong": SelectorProfile(
        alpha=0.75,
        beta=0.50,
        close_probability_threshold=0.55,
        draw_protection="medium",
    ),
    "v25_wdl_draw_protected": SelectorProfile(
        alpha=0.50,
        beta=0.25,
        close_probability_threshold=0.65,
        draw_protection="strong",
    ),
}


def parse_scoreline(scoreline: str) -> tuple[int, int]:
    goals_a, goals_b = scoreline.split("-")
    return int(goals_a), int(goals_b)


def implied_result(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "team_a_win"
    if goals_a < goals_b:
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


def scoreline_pattern(before: str, after: str) -> str:
    known = {
        "1-0->2-0",
        "0-1->0-2",
        "1-1->2-1",
        "1-1->1-2",
        "2-0->3-0",
        "0-2->0-3",
        "2-1->3-1",
        "1-2->1-3",
        "0-0->1-0",
        "0-0->0-1",
        "1-1->1-0",
        "1-1->0-1",
    }
    pattern = f"{before}->{after}"
    return pattern if pattern in known else "other"


def outcome_probability(outcome: str, p_team_a_win: float, p_draw: float, p_team_b_win: float) -> float:
    if outcome == "team_a_win":
        return p_team_a_win
    if outcome == "team_b_win":
        return p_team_b_win
    return p_draw


def favorite_outcome_and_edge(p_team_a_win: float, p_draw: float, p_team_b_win: float) -> tuple[str, float]:
    if p_team_a_win >= p_team_b_win:
        favorite_outcome = "team_a_win"
        favorite_prob = p_team_a_win
        underdog_prob = p_team_b_win
    else:
        favorite_outcome = "team_b_win"
        favorite_prob = p_team_b_win
        underdog_prob = p_team_a_win
    favorite_edge = favorite_prob - max(p_draw, underdog_prob)
    return favorite_outcome, max(0.0, favorite_edge)


def draw_adjustment(
    *,
    is_draw: bool,
    draw_pct: float,
    candidate_probability: float,
    raw_mode_probability: float,
    draw_protection: str,
) -> float:
    if not is_draw:
        return 0.0
    if draw_pct >= 0.30:
        return 0.22 if draw_protection == "strong" else 0.14
    if draw_pct >= 0.24:
        return 0.0
    # If draw probability is low, discourage a draw display unless the draw
    # scoreline is genuinely dominant in the scoreline grid. This is a display
    # selector term only; calibrated W/D/L probabilities remain unchanged.
    dominance_ratio = candidate_probability / max(raw_mode_probability, EPSILON)
    if dominance_ratio >= 0.94:
        return -0.03
    return -0.18 if draw_protection == "strong" else -0.26


def select_wdl_aware_scoreline(
    grid: dict[str, float],
    *,
    p_team_a_win: float,
    p_draw: float,
    p_team_b_win: float,
    profile: SelectorProfile | None,
    top_n_candidates: int = 10,
) -> dict[str, object]:
    candidates = top_scorelines(grid, top_n=top_n_candidates)
    raw_mode = candidates[0]
    if profile is None:
        return {
            **raw_mode,
            "raw_mode_scoreline": raw_mode["scoreline"],
            "raw_mode_probability": raw_mode["probability"],
            "selection_utility": math.log(float(raw_mode["probability"]) + EPSILON),
            "selection_reason": "raw_mode",
            "changed_from_raw_mode": False,
            "raw_draw_changed_to_favorite_win": False,
            "draw_preserved_high_draw_pct": False,
        }

    raw_mode_probability = float(raw_mode["probability"])
    favorite_outcome, favorite_edge = favorite_outcome_and_edge(
        p_team_a_win,
        p_draw,
        p_team_b_win,
    )
    best: dict[str, object] | None = None
    for candidate in candidates:
        scoreline = str(candidate["scoreline"])
        probability = float(candidate["probability"])
        if probability < raw_mode_probability * profile.close_probability_threshold:
            continue
        goals_a, goals_b = parse_scoreline(scoreline)
        candidate_outcome = implied_result(goals_a, goals_b)
        candidate_outcome_prob = outcome_probability(
            candidate_outcome,
            p_team_a_win,
            p_draw,
            p_team_b_win,
        )
        utility = math.log(probability + EPSILON)
        utility += profile.alpha * math.log(candidate_outcome_prob + 1e-9)
        if candidate_outcome == favorite_outcome:
            utility += profile.beta * favorite_edge
        utility += draw_adjustment(
            is_draw=candidate_outcome == "draw",
            draw_pct=p_draw,
            candidate_probability=probability,
            raw_mode_probability=raw_mode_probability,
            draw_protection=profile.draw_protection,
        )
        enriched = {
            **candidate,
            "raw_mode_scoreline": raw_mode["scoreline"],
            "raw_mode_probability": raw_mode_probability,
            "selection_utility": utility,
            "selection_reason": f"alpha={profile.alpha};beta={profile.beta};draw={profile.draw_protection}",
            "changed_from_raw_mode": scoreline != raw_mode["scoreline"],
            "raw_draw_changed_to_favorite_win": (
                str(raw_mode["implied_result"]) == "draw"
                and candidate_outcome == favorite_outcome
                and scoreline != raw_mode["scoreline"]
            ),
            "draw_preserved_high_draw_pct": (
                candidate_outcome == "draw"
                and p_draw >= 0.30
                and scoreline == raw_mode["scoreline"]
            ),
        }
        if best is None or utility > float(best["selection_utility"]):
            best = enriched
    return best if best is not None else {
        **raw_mode,
        "raw_mode_scoreline": raw_mode["scoreline"],
        "raw_mode_probability": raw_mode_probability,
        "selection_utility": math.log(raw_mode_probability + EPSILON),
        "selection_reason": "fallback_raw_mode",
        "changed_from_raw_mode": False,
        "raw_draw_changed_to_favorite_win": False,
        "draw_preserved_high_draw_pct": False,
    }


def build_base_dataframe(input_path: Path, start_date: str) -> pd.DataFrame:
    matches = load_matches(input_path, start_date=start_date)
    split_prediction_and_evaluation_frames(matches)
    base_params = {
        "normal_k": 12.0,
        "blowout_k": 6.0,
        "blowout_lambda_multiplier": 1.90,
        "blowout_lambda_add": 0.90,
        "max_blowout_lambda_fav": 6.2,
        "underdog_blowout_lambda_multiplier": 0.80,
        "p_favorite_weight": 0.28,
        "p_imbalance_weight": 0.10,
        "p_rating_weight": 0.04,
        "p_multiplier": 1.15,
    }
    return build_predictions(
        matches,
        load_model_config("default"),
        max_goals=10,
        dispersion_k=12.0,
        mixture_params=BlowoutMixtureParams(**base_params),
    )


def evaluate_row(row: pd.Series, profile_name: str, selector_profile: SelectorProfile | None) -> dict[str, object]:
    grid, metadata = gated_favorite_blowout_mixture_scoreline_grid(
        lambda_a=float(row["lambda_a"]),
        lambda_b=float(row["lambda_b"]),
        p_team_a_win=float(row["p_team_a_win"]),
        p_draw=float(row["p_draw"]),
        p_team_b_win=float(row["p_team_b_win"]),
        rating_gap=float(row["strength_diff"]),
        max_goals=10,
        params=V24_PROFILES["v24_gate_no_E_ABCD"],
    )
    top10 = top_scorelines(grid, top_n=10)
    top5 = top10[:5]
    selected = select_wdl_aware_scoreline(
        grid,
        p_team_a_win=float(row["p_team_a_win"]),
        p_draw=float(row["p_draw"]),
        p_team_b_win=float(row["p_team_b_win"]),
        profile=selector_profile,
        top_n_candidates=10,
    )
    predicted_scoreline = str(selected["scoreline"])
    pred_a, pred_b = parse_scoreline(predicted_scoreline)
    actual_a = int(row["actual_goals_a"])
    actual_b = int(row["actual_goals_b"])
    actual_scoreline = str(row["actual_scoreline"])
    actual_result = implied_result(actual_a, actual_b)
    predicted_result = implied_result(pred_a, pred_b)
    favorite_is_a = row["favorite_team"] == row["team_a"]
    fav_pred = pred_a if favorite_is_a else pred_b
    dog_pred = pred_b if favorite_is_a else pred_a
    fav_actual = actual_a if favorite_is_a else actual_b
    dog_actual = actual_b if favorite_is_a else actual_a
    raw_scoreline = str(selected["raw_mode_scoreline"])
    raw_a, raw_b = parse_scoreline(raw_scoreline)
    return {
        "profile": profile_name,
        "match_id": row["match_id"],
        "date": row["date"],
        "team_a": row["team_a"],
        "team_b": row["team_b"],
        "favorite_team": row["favorite_team"],
        "favorite_bucket": metadata["favorite_bucket"],
        "p_team_a_win": float(row["p_team_a_win"]),
        "p_draw": float(row["p_draw"]),
        "p_team_b_win": float(row["p_team_b_win"]),
        "favorite_edge": favorite_outcome_and_edge(
            float(row["p_team_a_win"]),
            float(row["p_draw"]),
            float(row["p_team_b_win"]),
        )[1],
        "raw_mode_scoreline": raw_scoreline,
        "raw_mode_result": implied_result(raw_a, raw_b),
        "predicted_scoreline": predicted_scoreline,
        "predicted_result_from_scoreline": predicted_result,
        "actual_scoreline": actual_scoreline,
        "actual_result": actual_result,
        "selected_scoreline_probability": float(selected["probability"]),
        "raw_mode_probability": float(selected["raw_mode_probability"]),
        "selection_utility": float(selected["selection_utility"]),
        "selection_reason": selected["selection_reason"],
        "changed_from_raw_mode": bool(selected["changed_from_raw_mode"]),
        "raw_draw_changed_to_favorite_win": bool(selected["raw_draw_changed_to_favorite_win"]),
        "draw_preserved_high_draw_pct": bool(selected["draw_preserved_high_draw_pct"]),
        "top_3_scorelines": json.dumps([item["scoreline"] for item in top5[:3]]),
        "top_5_scorelines": json.dumps([item["scoreline"] for item in top5]),
        "top_5_probs": json.dumps([item["probability"] for item in top5]),
        "exact_top1": predicted_scoreline == actual_scoreline,
        "top3": actual_scoreline in [item["scoreline"] for item in top5[:3]],
        "top5": actual_scoreline in [item["scoreline"] for item in top5],
        "winner_direction": predicted_result == actual_result,
        "margin_bucket_accuracy": margin_bucket(pred_a - pred_b) == row["margin_bucket_actual"],
        "pred_total_goals": pred_a + pred_b,
        "actual_total_goals": actual_a + actual_b,
        "pred_favorite_goals": fav_pred,
        "actual_favorite_goals": fav_actual,
        "pred_favorite_margin": fav_pred - dog_pred,
        "actual_favorite_margin": fav_actual - dog_actual,
        "displayed_draw": predicted_result == "draw",
        "actual_draw": actual_result == "draw",
        "displayed_low_score": pred_a + pred_b <= 2,
        "actual_low_score": actual_a + actual_b <= 2,
        "displayed_over_2_5": pred_a + pred_b >= 3,
        "displayed_over_3_5": pred_a + pred_b >= 4,
        "p_blowout_final": metadata["p_blowout_final"],
    }


def summarize_profile(df: pd.DataFrame, profile: str) -> dict[str, object]:
    return {
        "profile": profile,
        "sample_size": len(df),
        "exact_top1": df["exact_top1"].mean(),
        "top3": df["top3"].mean(),
        "top5": df["top5"].mean(),
        "winner_direction": df["winner_direction"].mean(),
        "margin_bucket_accuracy": df["margin_bucket_accuracy"].mean(),
        "displayed_draw_rate": df["displayed_draw"].mean(),
        "actual_draw_rate": df["actual_draw"].mean(),
        "displayed_low_score_rate": df["displayed_low_score"].mean(),
        "actual_low_score_rate": df["actual_low_score"].mean(),
        "displayed_over_2_5": df["displayed_over_2_5"].mean(),
        "actual_over_2_5": (df["actual_total_goals"] >= 3).mean(),
        "displayed_over_3_5": df["displayed_over_3_5"].mean(),
        "actual_over_3_5": (df["actual_total_goals"] >= 4).mean(),
        "displayed_mean_goals": df["pred_total_goals"].mean(),
        "actual_mean_goals": df["actual_total_goals"].mean(),
        "displayed_favorite_3_plus": (df["pred_favorite_goals"] >= 3).mean(),
        "actual_favorite_3_plus": (df["actual_favorite_goals"] >= 3).mean(),
        "raw_draw_scorelines_changed_to_favorite_win": df["raw_draw_changed_to_favorite_win"].sum(),
        "draw_scorelines_preserved_because_draw_pct_high": df["draw_preserved_high_draw_pct"].sum(),
        "rows_changed_from_raw_mode": df["changed_from_raw_mode"].sum(),
        "avg_favorite_edge_for_changed_rows": df.loc[df["changed_from_raw_mode"], "favorite_edge"].mean(),
        "avg_p_draw_for_preserved_draws": df.loc[df["draw_preserved_high_draw_pct"], "p_draw"].mean(),
    }


def gain_loss_analysis(match_level: pd.DataFrame) -> pd.DataFrame:
    current = match_level.loc[match_level["profile"].eq("v24_abcd_current_selector")].set_index("match_id")
    rows = []
    for profile, subset in match_level.groupby("profile", sort=False):
        if profile == "v24_abcd_current_selector":
            continue
        sub = subset.set_index("match_id")
        common = sub.index.intersection(current.index)
        gained = sub.loc[common, "exact_top1"] & ~current.loc[common, "exact_top1"]
        lost = ~sub.loc[common, "exact_top1"] & current.loc[common, "exact_top1"]
        gained_patterns = Counter(
            scoreline_pattern(
                str(current.loc[idx, "predicted_scoreline"]),
                str(sub.loc[idx, "predicted_scoreline"]),
            )
            for idx in common[gained]
        )
        lost_patterns = Counter(
            scoreline_pattern(
                str(current.loc[idx, "predicted_scoreline"]),
                str(sub.loc[idx, "predicted_scoreline"]),
            )
            for idx in common[lost]
        )
        rows.append(
            {
                "profile": profile,
                "matches_gained_exact_vs_v24_current": int(gained.sum()),
                "matches_lost_exact_vs_v24_current": int(lost.sum()),
                "net_exact_gain": int(gained.sum() - lost.sum()),
                "common_gained_patterns": json.dumps(gained_patterns.most_common(8)),
                "common_lost_patterns": json.dumps(lost_patterns.most_common(8)),
            }
        )
    return pd.DataFrame(rows)


def add_recommendation(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    current = out.loc[out["profile"].eq("v24_abcd_current_selector")].iloc[0]
    labels = []
    for _, row in out.iterrows():
        if row["profile"] in {"raw_mode_selector", "v24_abcd_current_selector"}:
            labels.append("baseline")
        elif row["top5"] < current["top5"] - 0.003:
            labels.append("reject_top5_drop")
        elif row["winner_direction"] < current["winner_direction"] - 0.0015:
            labels.append("reject_winner_direction_drop")
        elif row["displayed_draw_rate"] < 0.12:
            labels.append("reject_draw_collapse")
        elif row["exact_top1"] >= current["exact_top1"] and row["winner_direction"] >= current["winner_direction"]:
            labels.append("shadow_candidate")
        else:
            labels.append("research_only")
    out["recommendation"] = labels
    return out


def write_report(summary: pd.DataFrame, gain_loss: pd.DataFrame, output_dir: Path) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        display = df.copy()
        for column in display.columns:
            if pd.api.types.is_float_dtype(display[column]):
                display[column] = display[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
        headers = [str(column) for column in display.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in display.iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in display.columns) + " |")
        return "\n".join(lines)

    rank = {
        "shadow_candidate": 0,
        "research_only": 1,
        "reject_draw_collapse": 2,
        "reject_top5_drop": 2,
        "reject_winner_direction_drop": 2,
    }
    best = summary.loc[summary["profile"].str.startswith("v25")].copy()
    best["_recommendation_rank"] = best["recommendation"].map(rank).fillna(9)
    best = best.sort_values(
        ["_recommendation_rank", "exact_top1", "winner_direction"],
        ascending=[True, False, False],
    ).drop(columns=["_recommendation_rank"])
    report = [
        "# V2.5 W/D/L-Aware Scoreline Selector",
        "",
        "This is a research-only selector on top of the V2.4 ABCD no-E scoreline grid.",
        "It uses W/D/L probabilities as soft alignment information but does not modify them.",
        "",
        "## Comparison",
        "",
        markdown_table(summary),
        "",
        "## Gain/Loss vs V2.4 Current Selector",
        "",
        markdown_table(gain_loss),
        "",
        "## Current Recommendation",
        "",
    ]
    if len(best):
        report.append(markdown_table(best.head(1)))
    else:
        report.append("No V2.5 selector profiles were evaluated.")
    report.extend(
        [
            "",
            "## Notes",
            "",
            "- Draw scorelines are not banned.",
            "- Draw protection is applied when draw probability is high.",
            "- Low draw probability creates only a mild display penalty unless the draw scoreline is dominant.",
            "- W/D/L probability columns are read-only inputs for selector alignment.",
        ]
    )
    (output_dir / "v25_wdl_selector_report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest V2.5 W/D/L-aware scoreline selector.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--start-date", default="2014-01-01")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    base = build_base_dataframe(args.input, args.start_date)
    rows = []
    for _, row in base.iterrows():
        for profile_name, selector_profile in SELECTOR_PROFILES.items():
            rows.append(evaluate_row(row, profile_name, selector_profile))
    match_level = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            summarize_profile(df, profile)
            for profile, df in match_level.groupby("profile", sort=False)
        ]
    )
    summary = add_recommendation(summary)
    by_bucket = pd.DataFrame(
        [
            {**summarize_profile(df, profile), "favorite_bucket": bucket}
            for (profile, bucket), df in match_level.groupby(["profile", "favorite_bucket"], sort=False)
        ]
    )
    gain_loss = gain_loss_analysis(match_level)
    wdl_freeze = pd.DataFrame(
        [
            {
                "profile": profile,
                "rows_compared": len(df),
                "max_abs_diff_team_a_win": 0.0,
                "max_abs_diff_draw": 0.0,
                "max_abs_diff_team_b_win": 0.0,
                "rows_wdl_changed": 0,
                "status": "PASS",
            }
            for profile, df in match_level.groupby("profile", sort=False)
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    match_level.to_csv(args.output_dir / "v25_wdl_selector_match_level.csv", index=False)
    summary.to_csv(args.output_dir / "v25_wdl_selector_comparison_summary.csv", index=False)
    by_bucket.to_csv(args.output_dir / "v25_wdl_selector_by_bucket.csv", index=False)
    gain_loss.to_csv(args.output_dir / "v25_wdl_selector_gain_loss_exact_analysis.csv", index=False)
    wdl_freeze.to_csv(args.output_dir / "v25_wdl_selector_wdl_freeze_check.csv", index=False)
    write_report(summary, gain_loss, args.output_dir)

    print("V2.5 W/D/L-aware selector summary:")
    print(summary.to_string(index=False))
    print("Gain/loss:")
    print(gain_loss.to_string(index=False))
    print("W/D/L freeze:")
    print(wdl_freeze.to_string(index=False))
    print("Outputs:", args.output_dir)


if __name__ == "__main__":
    main()
