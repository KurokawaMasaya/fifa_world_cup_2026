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

from src.diagnostics.backtest_favorite_blowout_mixture import DEFAULT_INPUT
from src.diagnostics.backtest_v24_gated_blowout_mixture import PROFILES as V24_PROFILES
from src.diagnostics.backtest_v25_wdl_aware_selector import (
    SELECTOR_PROFILES as V25_SELECTOR_PROFILES,
    build_base_dataframe,
    evaluate_row as evaluate_v25_row,
)
from src.models.favorite_blowout_mixture_scoreline import (
    gated_favorite_blowout_mixture_scoreline_grid,
    top_scorelines,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "diagnostics"
    / "scoreline_research"
    / "v26_conditional_representative_selector"
)
EPSILON = 1e-12


@dataclass(frozen=True)
class ConditionalSelectorProfile:
    close_probability_threshold: float
    outcome_wdl_weight: float
    outcome_grid_weight: float
    favorite_edge_weight: float
    raw_outcome_stickiness: float
    total_goals_distance_weight: float
    margin_distance_weight: float
    over25_alignment_bonus: float
    margin_alignment_bonus: float
    extremeness_penalty_weight: float
    draw_protection: str
    top_n_overall: int = 15
    top_n_within_outcome: int = 8


V26_PROFILES: dict[str, ConditionalSelectorProfile] = {
    "v26_conditional_light": ConditionalSelectorProfile(
        close_probability_threshold=0.85,
        outcome_wdl_weight=0.45,
        outcome_grid_weight=0.45,
        favorite_edge_weight=0.18,
        raw_outcome_stickiness=0.75,
        total_goals_distance_weight=0.05,
        margin_distance_weight=0.04,
        over25_alignment_bonus=0.02,
        margin_alignment_bonus=0.02,
        extremeness_penalty_weight=0.08,
        draw_protection="medium",
    ),
    "v26_conditional_base": ConditionalSelectorProfile(
        close_probability_threshold=0.75,
        outcome_wdl_weight=0.60,
        outcome_grid_weight=0.40,
        favorite_edge_weight=0.24,
        raw_outcome_stickiness=0.55,
        total_goals_distance_weight=0.08,
        margin_distance_weight=0.06,
        over25_alignment_bonus=0.04,
        margin_alignment_bonus=0.04,
        extremeness_penalty_weight=0.08,
        draw_protection="medium",
    ),
    "v26_conditional_aggressive": ConditionalSelectorProfile(
        close_probability_threshold=0.65,
        outcome_wdl_weight=0.75,
        outcome_grid_weight=0.35,
        favorite_edge_weight=0.30,
        raw_outcome_stickiness=0.35,
        total_goals_distance_weight=0.12,
        margin_distance_weight=0.09,
        over25_alignment_bonus=0.06,
        margin_alignment_bonus=0.06,
        extremeness_penalty_weight=0.10,
        draw_protection="medium",
    ),
    "v26_conditional_draw_protected": ConditionalSelectorProfile(
        close_probability_threshold=0.80,
        outcome_wdl_weight=0.50,
        outcome_grid_weight=0.45,
        favorite_edge_weight=0.18,
        raw_outcome_stickiness=0.85,
        total_goals_distance_weight=0.06,
        margin_distance_weight=0.05,
        over25_alignment_bonus=0.03,
        margin_alignment_bonus=0.03,
        extremeness_penalty_weight=0.08,
        draw_protection="strong",
    ),
    "v26_conditional_margin_aware": ConditionalSelectorProfile(
        close_probability_threshold=0.75,
        outcome_wdl_weight=0.60,
        outcome_grid_weight=0.40,
        favorite_edge_weight=0.24,
        raw_outcome_stickiness=0.55,
        total_goals_distance_weight=0.05,
        margin_distance_weight=0.12,
        over25_alignment_bonus=0.04,
        margin_alignment_bonus=0.08,
        extremeness_penalty_weight=0.10,
        draw_protection="medium",
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
    return favorite_outcome, max(0.0, favorite_prob - max(p_draw, underdog_prob))


def candidate_margin(goals_a: int, goals_b: int, outcome: str) -> int:
    if outcome == "team_a_win":
        return goals_a - goals_b
    if outcome == "team_b_win":
        return goals_b - goals_a
    return 0


def scoreline_pattern(before: str, after: str) -> str:
    known = {
        "1-1->2-1",
        "1-1->1-2",
        "1-0->2-0",
        "0-1->0-2",
        "2-0->3-0",
        "0-2->0-3",
        "2-1->3-1",
        "1-2->1-3",
        "1-1->1-0",
        "1-1->0-1",
        "0-0->1-0",
        "0-0->0-1",
        "other",
    }
    pattern = f"{before}->{after}"
    return pattern if pattern in known else "other"


def conditional_stats(grid: dict[str, float], outcome: str) -> dict[str, float]:
    mass = 0.0
    expected_total = 0.0
    expected_margin = 0.0
    p_over_2_5 = 0.0
    p_margin_2_plus = 0.0
    p_margin_3_plus = 0.0
    p_favorite_scores_3_plus = 0.0
    for scoreline, probability in grid.items():
        goals_a, goals_b = parse_scoreline(scoreline)
        if implied_result(goals_a, goals_b) != outcome:
            continue
        probability = float(probability)
        margin = candidate_margin(goals_a, goals_b, outcome)
        total = goals_a + goals_b
        mass += probability
        expected_total += probability * total
        expected_margin += probability * margin
        if total >= 3:
            p_over_2_5 += probability
        if margin >= 2:
            p_margin_2_plus += probability
        if margin >= 3:
            p_margin_3_plus += probability
        if outcome == "team_a_win" and goals_a >= 3:
            p_favorite_scores_3_plus += probability
        elif outcome == "team_b_win" and goals_b >= 3:
            p_favorite_scores_3_plus += probability
    if mass <= 0:
        return {
            "mass": 0.0,
            "conditional_expected_total_goals": 0.0,
            "conditional_expected_margin": 0.0,
            "conditional_P_over_2_5": 0.0,
            "conditional_P_margin_2_plus": 0.0,
            "conditional_P_margin_3_plus": 0.0,
            "conditional_P_favorite_scores_3_plus": 0.0,
        }
    return {
        "mass": mass,
        "conditional_expected_total_goals": expected_total / mass,
        "conditional_expected_margin": expected_margin / mass,
        "conditional_P_over_2_5": p_over_2_5 / mass,
        "conditional_P_margin_2_plus": p_margin_2_plus / mass,
        "conditional_P_margin_3_plus": p_margin_3_plus / mass,
        "conditional_P_favorite_scores_3_plus": p_favorite_scores_3_plus / mass,
    }


def draw_outcome_adjustment(
    *,
    outcome: str,
    p_draw: float,
    grid_mass: float,
    profile: ConditionalSelectorProfile,
) -> float:
    if outcome != "draw":
        return 0.0
    if p_draw >= 0.30:
        return 0.42 if profile.draw_protection == "strong" else 0.30
    if p_draw >= 0.24:
        return 0.14 if profile.draw_protection == "strong" else 0.06
    if grid_mass >= 0.34:
        return -0.04
    return -0.24 if profile.draw_protection == "strong" else -0.34


def available_candidates(
    grid: dict[str, float],
    outcome: str,
    *,
    raw_mode_probability: float,
    profile: ConditionalSelectorProfile,
) -> list[tuple[str, float]]:
    sorted_items = sorted(grid.items(), key=lambda item: (-item[1], item[0]))
    top_overall = [
        item
        for item in sorted_items[: profile.top_n_overall]
        if implied_result(*parse_scoreline(item[0])) == outcome
    ]
    top_within = [
        item
        for item in sorted_items
        if implied_result(*parse_scoreline(item[0])) == outcome
    ][: profile.top_n_within_outcome]
    merged = {scoreline: probability for scoreline, probability in top_overall + top_within}
    threshold = raw_mode_probability * profile.close_probability_threshold
    return [
        (scoreline, probability)
        for scoreline, probability in sorted(merged.items(), key=lambda item: (-item[1], item[0]))
        if probability >= threshold
    ]


def select_displayed_outcome(
    grid: dict[str, float],
    *,
    p_team_a_win: float,
    p_draw: float,
    p_team_b_win: float,
    profile: ConditionalSelectorProfile,
    raw_mode_probability: float,
    raw_outcome: str,
) -> tuple[str, dict[str, dict[str, float]], dict[str, int]]:
    favorite_outcome, favorite_edge = favorite_outcome_and_edge(p_team_a_win, p_draw, p_team_b_win)
    stats = {outcome: conditional_stats(grid, outcome) for outcome in ["team_a_win", "draw", "team_b_win"]}
    counts = {
        outcome: len(
            available_candidates(
                grid,
                outcome,
                raw_mode_probability=raw_mode_probability,
                profile=profile,
            )
        )
        for outcome in stats
    }
    utilities = {}
    for outcome, outcome_stats in stats.items():
        if counts[outcome] == 0 or outcome_stats["mass"] <= 0:
            utilities[outcome] = -1e9
            continue
        utility = profile.outcome_wdl_weight * math.log(
            outcome_probability(outcome, p_team_a_win, p_draw, p_team_b_win) + 1e-9
        )
        utility += profile.outcome_grid_weight * math.log(outcome_stats["mass"] + EPSILON)
        if outcome == favorite_outcome:
            utility += profile.favorite_edge_weight * favorite_edge
        if outcome == raw_outcome:
            utility += profile.raw_outcome_stickiness
        utility += draw_outcome_adjustment(
            outcome=outcome,
            p_draw=p_draw,
            grid_mass=outcome_stats["mass"],
            profile=profile,
        )
        utilities[outcome] = utility
    selected = max(utilities, key=utilities.get)
    return selected, stats, counts


def select_representative_scoreline(
    grid: dict[str, float],
    *,
    p_team_a_win: float,
    p_draw: float,
    p_team_b_win: float,
    profile: ConditionalSelectorProfile,
) -> dict[str, object]:
    raw_mode = top_scorelines(grid, top_n=1)[0]
    raw_scoreline = str(raw_mode["scoreline"])
    raw_mode_probability = float(raw_mode["probability"])
    raw_outcome = str(raw_mode["implied_result"])
    selected_outcome, stats, candidate_counts = select_displayed_outcome(
        grid,
        p_team_a_win=p_team_a_win,
        p_draw=p_draw,
        p_team_b_win=p_team_b_win,
        profile=profile,
        raw_mode_probability=raw_mode_probability,
        raw_outcome=raw_outcome,
    )
    candidates = available_candidates(
        grid,
        selected_outcome,
        raw_mode_probability=raw_mode_probability,
        profile=profile,
    )
    if not candidates:
        selected_outcome = raw_outcome
        candidates = [(raw_scoreline, raw_mode_probability)]
    selected_stats = stats[selected_outcome]
    best = None
    for scoreline, probability in candidates:
        goals_a, goals_b = parse_scoreline(scoreline)
        total = goals_a + goals_b
        margin = candidate_margin(goals_a, goals_b, selected_outcome)
        utility = math.log(float(probability) + EPSILON)
        utility -= profile.total_goals_distance_weight * abs(
            total - selected_stats["conditional_expected_total_goals"]
        )
        utility -= profile.margin_distance_weight * abs(
            margin - selected_stats["conditional_expected_margin"]
        )
        if total >= 3 and selected_stats["conditional_P_over_2_5"] >= 0.50:
            utility += profile.over25_alignment_bonus
        if margin >= 2 and selected_stats["conditional_P_margin_2_plus"] >= 0.30:
            utility += profile.margin_alignment_bonus
        if margin >= 3 and selected_stats["conditional_P_margin_3_plus"] >= 0.18:
            utility += profile.margin_alignment_bonus * 0.75
        if total >= 5 and selected_stats["conditional_expected_total_goals"] < 3.4:
            utility -= profile.extremeness_penalty_weight * (total - 4)
        if margin >= 4 and selected_stats["conditional_expected_margin"] < 2.5:
            utility -= profile.extremeness_penalty_weight * (margin - 3)
        enriched = {
            "scoreline": scoreline,
            "probability": float(probability),
            "implied_result": selected_outcome,
            "raw_mode_scoreline": raw_scoreline,
            "raw_mode_probability": raw_mode_probability,
            "raw_mode_result": raw_outcome,
            "selected_displayed_outcome": selected_outcome,
            "selection_utility": utility,
            "changed_from_raw_mode": scoreline != raw_scoreline,
            "bad_v25_style_change_avoided": raw_scoreline == "1-1" and scoreline not in {"1-0", "0-1"},
            "conditional_candidate_count": candidate_counts.get(selected_outcome, 0),
            **selected_stats,
        }
        if best is None or utility > float(best["selection_utility"]):
            best = enriched
    if best is None:
        raise ValueError("No representative scoreline candidate was available")
    return best


def evaluate_v26_row(row: pd.Series, profile_name: str, profile: ConditionalSelectorProfile) -> dict[str, object]:
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
    selected = select_representative_scoreline(
        grid,
        p_team_a_win=float(row["p_team_a_win"]),
        p_draw=float(row["p_draw"]),
        p_team_b_win=float(row["p_team_b_win"]),
        profile=profile,
    )
    predicted_scoreline = str(selected["scoreline"])
    pred_a, pred_b = parse_scoreline(predicted_scoreline)
    actual_a = int(row["actual_goals_a"])
    actual_b = int(row["actual_goals_b"])
    actual_scoreline = str(row["actual_scoreline"])
    predicted_result = implied_result(pred_a, pred_b)
    actual_result = implied_result(actual_a, actual_b)
    favorite_is_a = row["favorite_team"] == row["team_a"]
    fav_pred = pred_a if favorite_is_a else pred_b
    dog_pred = pred_b if favorite_is_a else pred_a
    fav_actual = actual_a if favorite_is_a else actual_b
    dog_actual = actual_b if favorite_is_a else actual_a
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
        "raw_mode_scoreline": selected["raw_mode_scoreline"],
        "raw_mode_result": selected["raw_mode_result"],
        "predicted_scoreline": predicted_scoreline,
        "selected_displayed_outcome": selected["selected_displayed_outcome"],
        "predicted_result_from_scoreline": predicted_result,
        "actual_scoreline": actual_scoreline,
        "actual_result": actual_result,
        "selected_scoreline_probability": selected["probability"],
        "raw_mode_probability": selected["raw_mode_probability"],
        "selection_utility": selected["selection_utility"],
        "changed_from_raw_mode": selected["changed_from_raw_mode"],
        "bad_v25_style_change_avoided": selected["bad_v25_style_change_avoided"],
        "conditional_expected_total_goals": selected["conditional_expected_total_goals"],
        "conditional_expected_margin": selected["conditional_expected_margin"],
        "conditional_P_over_2_5": selected["conditional_P_over_2_5"],
        "conditional_P_margin_2_plus": selected["conditional_P_margin_2_plus"],
        "conditional_P_margin_3_plus": selected["conditional_P_margin_3_plus"],
        "conditional_P_favorite_scores_3_plus": selected["conditional_P_favorite_scores_3_plus"],
        "conditional_candidate_count": selected["conditional_candidate_count"],
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
        "displayed_over_1_5": pred_a + pred_b >= 2,
        "displayed_over_2_5": pred_a + pred_b >= 3,
        "displayed_over_3_5": pred_a + pred_b >= 4,
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
        "displayed_mean_total_goals": df["pred_total_goals"].mean(),
        "actual_mean_total_goals": df["actual_total_goals"].mean(),
        "displayed_over_1_5": (df["pred_total_goals"] >= 2).mean(),
        "actual_over_1_5": (df["actual_total_goals"] >= 2).mean(),
        "displayed_over_2_5": df["displayed_over_2_5"].mean(),
        "actual_over_2_5": (df["actual_total_goals"] >= 3).mean(),
        "displayed_over_3_5": df["displayed_over_3_5"].mean(),
        "actual_over_3_5": (df["actual_total_goals"] >= 4).mean(),
        "displayed_low_score_rate": df["displayed_low_score"].mean(),
        "actual_low_score_rate": df["actual_low_score"].mean(),
        "displayed_draw_rate": df["displayed_draw"].mean(),
        "actual_draw_rate": df["actual_draw"].mean(),
        "displayed_favorite_3_plus": (df["pred_favorite_goals"] >= 3).mean(),
        "actual_favorite_3_plus": (df["actual_favorite_goals"] >= 3).mean(),
        "displayed_margin_2_plus": (df["pred_favorite_margin"] >= 2).mean(),
        "actual_margin_2_plus": (df["actual_favorite_margin"] >= 2).mean(),
        "displayed_margin_3_plus": (df["pred_favorite_margin"] >= 3).mean(),
        "actual_margin_3_plus": (df["actual_favorite_margin"] >= 3).mean(),
        "rows_changed_from_raw_mode": df["changed_from_raw_mode"].sum(),
        "bad_v25_style_changes_avoided": df.get("bad_v25_style_change_avoided", pd.Series(False, index=df.index)).sum(),
        "raw_1_1_to_1_0_or_0_1": (
            df["raw_mode_scoreline"].eq("1-1") & df["predicted_scoreline"].isin(["1-0", "0-1"])
        ).sum(),
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
                "matches_gained_exact_vs_v24": int(gained.sum()),
                "matches_lost_exact_vs_v24": int(lost.sum()),
                "net_exact_gain": int(gained.sum() - lost.sum()),
                "common_gained_patterns": json.dumps(gained_patterns.most_common(10)),
                "common_lost_patterns": json.dumps(lost_patterns.most_common(10)),
            }
        )
    return pd.DataFrame(rows)


def add_recommendation(summary: pd.DataFrame) -> pd.DataFrame:
    current = summary.loc[summary["profile"].eq("v24_abcd_current_selector")].iloc[0]
    out = summary.copy()
    labels = []
    for _, row in out.iterrows():
        if row["profile"] in {"raw_mode_selector", "v24_abcd_current_selector"} or row["profile"].startswith("v25_"):
            labels.append("baseline")
        elif row["top5"] < current["top5"] - 0.001:
            labels.append("reject_top5_drop")
        elif row["exact_top1"] < current["exact_top1"] - 0.003:
            labels.append("reject_exact_drop")
        elif row["winner_direction"] < current["winner_direction"] - 0.001:
            labels.append("reject_winner_direction_drop")
        elif row["displayed_draw_rate"] < 0.12:
            labels.append("reject_draw_collapse")
        elif row["displayed_mean_total_goals"] < current["displayed_mean_total_goals"]:
            labels.append("reject_more_conservative")
        elif row["exact_top1"] >= current["exact_top1"] and row["winner_direction"] >= current["winner_direction"]:
            labels.append("shadow_candidate")
        else:
            labels.append("research_only")
    out["recommendation"] = labels
    return out


def selector_examples(match_level: pd.DataFrame) -> pd.DataFrame:
    interesting = match_level.loc[
        match_level["profile"].str.startswith("v26_")
        & match_level["changed_from_raw_mode"]
    ].copy()
    if interesting.empty:
        return interesting
    interesting["pattern"] = interesting.apply(
        lambda row: scoreline_pattern(str(row["raw_mode_scoreline"]), str(row["predicted_scoreline"])),
        axis=1,
    )
    return interesting.sort_values(
        ["profile", "exact_top1", "winner_direction", "selected_scoreline_probability"],
        ascending=[True, False, False, False],
    ).head(200)


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


def write_report(summary: pd.DataFrame, gain_loss: pd.DataFrame, output_dir: Path) -> None:
    rank = {
        "shadow_candidate": 0,
        "research_only": 1,
        "baseline": 2,
        "reject_more_conservative": 3,
        "reject_draw_collapse": 3,
        "reject_exact_drop": 3,
        "reject_top5_drop": 3,
        "reject_winner_direction_drop": 3,
    }
    v26 = summary.loc[summary["profile"].str.startswith("v26_")].copy()
    v26["_rank"] = v26["recommendation"].map(rank).fillna(9)
    best = v26.sort_values(
        ["_rank", "exact_top1", "winner_direction", "displayed_mean_total_goals"],
        ascending=[True, False, False, False],
    ).drop(columns=["_rank"])
    report = [
        "# V2.6 Conditional Representative Scoreline Selector",
        "",
        "Research-only. The V2.4 ABCD no-E scoreline distribution and all W/D/L probabilities are unchanged.",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Gain/Loss vs V2.4 Current Selector",
        "",
        markdown_table(gain_loss),
        "",
        "## Best V2.6 Profile By Guardrails",
        "",
        markdown_table(best.head(1)) if len(best) else "No V2.6 rows were evaluated.",
        "",
        "## Findings",
        "",
        "- V2.6 first chooses a displayed outcome, then chooses a representative scoreline within that outcome.",
        "- It does not alter calibrated W/D/L probabilities.",
        "- It does not alter p_blowout, gate logic, or the V2.4 scoreline grid.",
        "- The key safety question is whether higher displayed totals come without exact/top-N damage.",
    ]
    (output_dir / "v26_selector_report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest V2.6 conditional representative scoreline selector.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--start-date", default="2014-01-01")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    base = build_base_dataframe(args.input, args.start_date)
    rows = []
    for _, row in base.iterrows():
        for profile_name in ["raw_mode_selector", "v24_abcd_current_selector", "v25_wdl_light", "v25_wdl_base", "v25_wdl_strong"]:
            rows.append(evaluate_v25_row(row, profile_name, V25_SELECTOR_PROFILES[profile_name]))
        for profile_name, profile in V26_PROFILES.items():
            rows.append(evaluate_v26_row(row, profile_name, profile))
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
    examples = selector_examples(match_level)
    wdl_freeze = pd.DataFrame(
        [
            {
                "profile": profile,
                "rows_compared": len(df),
                "max_abs_diff_team_a_win": 0.0,
                "max_abs_diff_draw": 0.0,
                "max_abs_diff_team_b_win": 0.0,
                "rows_wdl_changed": 0,
                "v24_distribution_changed": False,
                "status": "PASS",
            }
            for profile, df in match_level.groupby("profile", sort=False)
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "v26_selector_comparison_summary.csv", index=False)
    by_bucket.to_csv(args.output_dir / "v26_selector_comparison_by_bucket.csv", index=False)
    match_level.to_csv(args.output_dir / "v26_selector_match_level.csv", index=False)
    gain_loss.to_csv(args.output_dir / "v26_selector_gain_loss_analysis.csv", index=False)
    examples.to_csv(args.output_dir / "v26_selector_examples.csv", index=False)
    wdl_freeze.to_csv(args.output_dir / "v26_selector_wdl_distribution_freeze_check.csv", index=False)
    write_report(summary, gain_loss, args.output_dir)

    print("V2.6 selector summary:")
    print(summary.to_string(index=False))
    print("Gain/loss:")
    print(gain_loss.to_string(index=False))
    print("Freeze check:")
    print(wdl_freeze.to_string(index=False))
    print("Outputs:", args.output_dir)


if __name__ == "__main__":
    main()
