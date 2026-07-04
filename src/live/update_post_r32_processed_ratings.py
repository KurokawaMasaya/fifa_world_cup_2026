from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.model_config import (
    draw_calibration_kwargs,
    load_model_config,
    metadata_columns,
    poisson_parameter_kwargs,
)
from src.live.update_group_stage_processed_ratings import (
    MAX_ATTACK_DELTA,
    MAX_DEFENSE_DELTA,
    MAX_HEALTH_RATING_DELTA,
    V24_MAX_GOALS,
    V24_SCORELINE_VERSION,
    match_key,
    normalize_team,
    points_for,
    team_health_deltas,
)
from src.models.favorite_blowout_mixture_scoreline import (
    canonical_v24_abcd_no_e_params,
    gated_favorite_blowout_mixture_scoreline_grid,
    top_scorelines as v24_top_scorelines,
)
from src.models.v2_probability_stack import (
    is_v2_probability_stack_enabled,
    load_v2_feature_context,
)
from src.simulation.group_stage_simulator import (
    TEAMS_PATH,
    load_base_total_goals,
)
from src.simulation.knockout_resolution import predicted_knockout_resolution
from src.simulation.tournament_simulator import (
    RANDOM_SEED_USED_FOR,
    precompute_knockout_predictions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_DIR = PROJECT_ROOT / "output" / "predictions"
LIVE_DIR = PROJECT_ROOT / "output" / "live"

RATINGS_PATH = PROCESSED_DIR / "team_ratings_world_cup_elo.csv"
BACKUP_PATH = PROCESSED_DIR / "team_ratings_world_cup_elo_pre_r16_update_backup.csv"
UPDATED_COPY_PATH = PROCESSED_DIR / "team_ratings_world_cup_elo_after_r32.csv"
UPDATE_LOG_PATH = PROCESSED_DIR / "r32_strength_update_log.csv"
PERFORMANCE_SUMMARY_PATH = PROCESSED_DIR / "r32_team_performance_summary.csv"
LIVE_FIXTURES_PATH = LIVE_DIR / "fixtures_results.csv"
R32_PREDICTIONS_PATH = PREDICTIONS_DIR / "round_of_32_predictions_after_group_stage.csv"
R32_CLEAN_PATH = PREDICTIONS_DIR / "round_of_32_predictions_after_group_stage_clean.csv"

R16_PREDICTIONS_PATH = PREDICTIONS_DIR / "round_of_16_predictions_after_r32.csv"
R16_CLEAN_PATH = PREDICTIONS_DIR / "round_of_16_predictions_after_r32_clean.csv"
R16_METADATA_PATH = PREDICTIONS_DIR / "round_of_16_predictions_after_r32_clean__metadata.json"
R16_BRACKET_PATH = PREDICTIONS_DIR / "round_of_16_bracket_after_r32.csv"

POINTS_WEIGHT = 3.0
GD_WEIGHT = 1.5
ADVANCEMENT_WEIGHT = 1.5
UPSET_ADVANCEMENT_WEIGHT = 2.0
ATTACK_WEIGHT = 0.04
DEFENSE_WEIGHT = 0.04
MAX_R32_RATING_DELTA = 12.0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical_team_lookup(ratings: pd.DataFrame) -> dict[str, str]:
    return {normalize_team(team): team for team in ratings["team_name"].astype(str)}


def canonical_team(value: object, lookup: dict[str, str]) -> str:
    key = normalize_team(value)
    if key not in lookup:
        raise ValueError(f"Team not found in rating CSV after normalization: {value!r} -> {key!r}")
    return lookup[key]


def load_r32_prediction_inputs(rating_lookup: dict[str, str]) -> pd.DataFrame:
    detailed = pd.read_csv(R32_PREDICTIONS_PATH)
    clean = pd.read_csv(R32_CLEAN_PATH)
    detailed["_match_key"] = detailed.apply(lambda row: match_key(row["team_a"], row["team_b"]), axis=1)
    clean["_match_key"] = clean.apply(lambda row: match_key(row["team_a"], row["team_b"]), axis=1)
    clean_ids = clean[["_match_key", "match_id"]].rename(columns={"match_id": "espn_event_id"})
    merged = detailed.merge(clean_ids, on="_match_key", how="left", validate="one_to_one")
    if merged["espn_event_id"].isna().any():
        missing = merged.loc[merged["espn_event_id"].isna(), ["team_a", "team_b"]]
        raise ValueError("Could not map R32 predictions to ESPN ids:\n" + missing.to_string(index=False))
    for column in ["team_a", "team_b"]:
        merged[column] = merged[column].map(lambda value: canonical_team(value, rating_lookup))
    return merged


def load_completed_r32_actuals(rating_lookup: dict[str, str]) -> pd.DataFrame:
    fixtures = pd.read_csv(LIVE_FIXTURES_PATH)
    fixtures["match_id_numeric"] = pd.to_numeric(fixtures["match_id"], errors="coerce")
    r32 = fixtures.loc[
        fixtures["match_id_numeric"].between(760486, 760501)
        & fixtures["status"].astype(str).str.lower().eq("final")
    ].copy()
    if len(r32) != 16:
        raise ValueError(f"Expected 16 completed R32 matches, found {len(r32)}")
    for column in ["team_a", "team_b"]:
        r32[column] = r32[column].map(lambda value: canonical_team(value, rating_lookup))
    r32["_match_key"] = r32.apply(lambda row: match_key(row["team_a"], row["team_b"]), axis=1)
    return r32


def load_r16_fixtures(rating_lookup: dict[str, str]) -> pd.DataFrame:
    fixtures = pd.read_csv(LIVE_FIXTURES_PATH)
    fixtures["match_id_numeric"] = pd.to_numeric(fixtures["match_id"], errors="coerce")
    r16 = fixtures.loc[fixtures["match_id_numeric"].between(760502, 760509)].copy()
    if len(r16) != 8:
        raise ValueError(f"Expected 8 R16 fixtures from ESPN, found {len(r16)}")
    for column in ["team_a", "team_b"]:
        r16[column] = r16[column].map(lambda value: canonical_team(value, rating_lookup))
    return r16.sort_values(["match_id_numeric"]).reset_index(drop=True)


def advancing_teams_from_r16(r16: pd.DataFrame) -> set[str]:
    return set(r16["team_a"]).union(set(r16["team_b"]))


def numeric(value: object, default: float = 0.0) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(parsed) else float(parsed)


def build_match_update_rows(predictions: pd.DataFrame, actuals: pd.DataFrame, r16: pd.DataFrame) -> pd.DataFrame:
    actuals_by_id = actuals.set_index(actuals["match_id"].astype(str))
    advancers = advancing_teams_from_r16(r16)
    rows: list[dict[str, Any]] = []
    for _, pred in predictions.iterrows():
        event_id = str(int(pred["espn_event_id"]))
        if event_id not in actuals_by_id.index:
            raise ValueError(f"Missing completed actual for R32 ESPN event {event_id}")
        actual = actuals_by_id.loc[event_id]
        same_orientation = normalize_team(actual["team_a"]) == normalize_team(pred["team_a"])
        goals_a = int(actual["goals_a"] if same_orientation else actual["goals_b"])
        goals_b = int(actual["goals_b"] if same_orientation else actual["goals_a"])
        lambda_a = numeric(pred.get("lambda_a"), math.nan)
        lambda_b = numeric(pred.get("lambda_b"), math.nan)
        expected_margin_a = 0.0 if math.isnan(lambda_a) or math.isnan(lambda_b) else lambda_a - lambda_b
        p_a = numeric(pred["team_a_win_pct"]) / 100.0
        p_d = numeric(pred["draw_pct"]) / 100.0
        p_b = numeric(pred["team_b_win_pct"]) / 100.0
        teams = [
            (pred["team_a"], pred["team_b"], goals_a, goals_b, 3 * p_a + p_d, lambda_a, lambda_b, p_a, p_b),
            (pred["team_b"], pred["team_a"], goals_b, goals_a, 3 * p_b + p_d, lambda_b, lambda_a, p_b, p_a),
        ]
        for team, opponent, gf, ga, expected_points, xgf, xga, own_win_prob, opp_win_prob in teams:
            actual_points = points_for(gf, ga)
            actual_margin = gf - ga
            expected_margin = expected_margin_a if team == pred["team_a"] else -expected_margin_a
            margin_surprise = actual_margin - expected_margin
            attack_surprise = 0.0 if math.isnan(xgf) else gf - xgf
            defense_surprise = 0.0 if math.isnan(xga) else xga - ga
            advanced = team in advancers
            eliminated = not advanced
            was_underdog = own_win_prob < opp_win_prob
            advancement_delta = ADVANCEMENT_WEIGHT if advanced else -ADVANCEMENT_WEIGHT
            upset_advancement_delta = 0.0
            if advanced and was_underdog:
                upset_advancement_delta += UPSET_ADVANCEMENT_WEIGHT
            if eliminated and not was_underdog:
                upset_advancement_delta -= UPSET_ADVANCEMENT_WEIGHT
            rows.append(
                {
                    "match_id": int(event_id),
                    "date": actual.get("date"),
                    "team": team,
                    "opponent": opponent,
                    "goals_for": gf,
                    "goals_against": ga,
                    "advanced_to_r16": bool(advanced),
                    "actual_points": actual_points,
                    "expected_points": expected_points,
                    "points_surprise": actual_points - expected_points,
                    "expected_goals_for": xgf,
                    "expected_goals_against": xga,
                    "attack_surprise": attack_surprise,
                    "defense_surprise": defense_surprise,
                    "expected_margin": expected_margin,
                    "actual_margin": actual_margin,
                    "margin_surprise": margin_surprise,
                    "advancement_delta": advancement_delta,
                    "upset_advancement_delta": upset_advancement_delta,
                    "own_win_prob": own_win_prob,
                    "opponent_win_prob": opp_win_prob,
                }
            )
    return pd.DataFrame(rows)


def build_team_performance(match_updates: pd.DataFrame) -> pd.DataFrame:
    outcomes = match_updates.assign(
        win=(match_updates["goals_for"] > match_updates["goals_against"]).astype(int),
        draw=(match_updates["goals_for"] == match_updates["goals_against"]).astype(int),
        loss=(match_updates["goals_for"] < match_updates["goals_against"]).astype(int),
    )
    perf = (
        outcomes.groupby("team", as_index=False)
        .agg(
            matches_played=("match_id", "count"),
            wins=("win", "sum"),
            draws=("draw", "sum"),
            losses=("loss", "sum"),
            advanced_to_r16=("advanced_to_r16", "max"),
            points=("actual_points", "sum"),
            expected_points=("expected_points", "sum"),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
            actual_goal_difference=("actual_margin", "sum"),
            expected_goal_difference=("expected_margin", "sum"),
            attack_surprise=("attack_surprise", "sum"),
            defense_surprise=("defense_surprise", "sum"),
            advancement_delta=("advancement_delta", "sum"),
            upset_advancement_delta=("upset_advancement_delta", "sum"),
        )
    )
    perf["goal_difference"] = perf["actual_goal_difference"]
    perf["points_surprise"] = perf["points"] - perf["expected_points"]
    perf["goal_difference_surprise"] = perf["actual_goal_difference"] - perf["expected_goal_difference"]
    perf["r32_delta_raw"] = (
        POINTS_WEIGHT * perf["points_surprise"]
        + GD_WEIGHT * perf["goal_difference_surprise"]
        + perf["advancement_delta"]
        + perf["upset_advancement_delta"]
    )
    perf["r32_performance_delta"] = perf["r32_delta_raw"].clip(-MAX_R32_RATING_DELTA, MAX_R32_RATING_DELTA)
    perf["r32_attack_delta"] = (ATTACK_WEIGHT * perf["attack_surprise"]).clip(-MAX_ATTACK_DELTA, MAX_ATTACK_DELTA)
    perf["r32_defense_delta"] = (DEFENSE_WEIGHT * perf["defense_surprise"]).clip(-MAX_DEFENSE_DELTA, MAX_DEFENSE_DELTA)
    return perf.sort_values(["advanced_to_r16", "r32_performance_delta"], ascending=[False, False])


def apply_rating_update(ratings: pd.DataFrame, performance: pd.DataFrame, rating_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    health = team_health_deltas()
    perf = performance.copy()
    perf["_team_key"] = perf["team"].map(normalize_team)
    health["_team_key"] = health["team"].map(normalize_team) if not health.empty else pd.Series(dtype=object)
    output = ratings.copy()
    output["_team_key"] = output["team_name"].map(normalize_team)
    output = output.merge(
        perf[["_team_key", "r32_performance_delta", "r32_attack_delta", "r32_defense_delta"]],
        on="_team_key",
        how="left",
    )
    if health.empty:
        for column in [
            "post_group_health_rating_delta",
            "post_group_health_attack_delta",
            "post_group_health_defense_delta",
        ]:
            output[column] = 0.0
    else:
        output = output.merge(
            health[["_team_key", "post_group_health_rating_delta", "post_group_health_attack_delta", "post_group_health_defense_delta"]],
            on="_team_key",
            how="left",
        )
    for column in [
        "r32_performance_delta",
        "r32_attack_delta",
        "r32_defense_delta",
        "post_group_health_rating_delta",
        "post_group_health_attack_delta",
        "post_group_health_defense_delta",
    ]:
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
    output["rating_before_r16_update"] = pd.to_numeric(output[rating_col], errors="coerce")
    output["r32_health_delta"] = output["post_group_health_rating_delta"].clip(-MAX_HEALTH_RATING_DELTA, MAX_HEALTH_RATING_DELTA)
    output["r32_total_delta"] = (output["r32_performance_delta"] + output["r32_health_delta"]).clip(
        -MAX_R32_RATING_DELTA,
        MAX_R32_RATING_DELTA,
    )
    output["rating_after_r32"] = output["rating_before_r16_update"] + output["r32_total_delta"]
    output[rating_col] = output["rating_after_r32"]
    output["effective_rating"] = output["rating_after_r32"]
    output["effective_attack"] = pd.to_numeric(output.get("effective_attack", 0.0), errors="coerce").fillna(0.0) + output[
        "r32_attack_delta"
    ] + output["post_group_health_attack_delta"]
    output["effective_defense"] = pd.to_numeric(output.get("effective_defense", 0.0), errors="coerce").fillna(0.0) + output[
        "r32_defense_delta"
    ] + output["post_group_health_defense_delta"]
    log = output.merge(
        perf.drop(columns=["_team_key", "r32_performance_delta", "r32_attack_delta", "r32_defense_delta"], errors="ignore"),
        left_on="team_name",
        right_on="team",
        how="left",
    )
    log = log[
        [
            "team_name",
            "rating_before_r16_update",
            "r32_performance_delta",
            "r32_health_delta",
            "r32_total_delta",
            "rating_after_r32",
            "r32_attack_delta",
            "effective_attack",
            "r32_defense_delta",
            "effective_defense",
            "matches_played",
            "advanced_to_r16",
            "points",
            "goals_for",
            "goals_against",
            "goal_difference",
            "expected_points",
            "points_surprise",
            "goal_difference_surprise",
            "advancement_delta",
            "upset_advancement_delta",
        ]
    ].rename(columns={"team_name": "team", "rating_before_r16_update": "before_rating", "rating_after_r32": "after_rating"})
    log["reason"] = (
        "r32_results_delta="
        + log["r32_performance_delta"].round(3).astype(str)
        + "; health_delta="
        + log["r32_health_delta"].round(3).astype(str)
    )
    return output.drop(columns=["_team_key"], errors="ignore"), log


def write_r16_predictions(model_config: dict, ratings: pd.DataFrame, rating_col: str, r16: pd.DataFrame) -> pd.DataFrame:
    teams_df = pd.read_csv(TEAMS_PATH)
    base_total_goals = load_base_total_goals(model_config=model_config)
    model_kwargs = {**poisson_parameter_kwargs(model_config), **draw_calibration_kwargs(model_config)}
    v2_context = load_v2_feature_context(model_config) if is_v2_probability_stack_enabled(model_config) else None
    knockout_predictions, rating_lookup = precompute_knockout_predictions(
        teams_df=teams_df,
        ratings_df=ratings,
        rating_col=rating_col,
        base_total_goals=base_total_goals,
        model_kwargs=model_kwargs,
        model_config=model_config,
        v2_feature_context=v2_context,
    )
    metadata = metadata_columns(model_config)
    scoreline_params = canonical_v24_abcd_no_e_params()
    rows: list[dict[str, Any]] = []
    for index, fixture in r16.reset_index(drop=True).iterrows():
        team_a = fixture["team_a"]
        team_b = fixture["team_b"]
        prediction = knockout_predictions[(team_a, team_b)]
        rating_gap = rating_lookup[team_a] - rating_lookup[team_b]
        v24_grid, v24_meta = gated_favorite_blowout_mixture_scoreline_grid(
            lambda_a=float(prediction["lambda_a"]),
            lambda_b=float(prediction["lambda_b"]),
            p_team_a_win=float(prediction["p_team_a_win"]),
            p_draw=float(prediction["p_draw"]),
            p_team_b_win=float(prediction["p_team_b_win"]),
            rating_gap=rating_gap,
            max_goals=V24_MAX_GOALS,
            params=scoreline_params,
        )
        v24_prediction = {**prediction, "scoreline_probabilities": v24_grid}
        top5_items = v24_top_scorelines(v24_grid, top_n=5)
        top5 = [str(item["scoreline"]) for item in top5_items]
        top5_probs = [round(float(item["probability"]) * 100) for item in top5_items]
        resolution = predicted_knockout_resolution(
            team_a=team_a,
            team_b=team_b,
            prediction=v24_prediction,
            rating_a=rating_lookup[team_a],
            rating_b=rating_lookup[team_b],
        )
        favorite_win_pct = max(prediction["p_team_a_win"], prediction["p_team_b_win"]) * 100
        favorite_fail = 100 - favorite_win_pct
        rows.append(
            {
                **metadata,
                "match_id": int(fixture["match_id"]),
                "round": "R16",
                "match_number": 89 + index,
                "kickoff_time_utc": fixture.get("kickoff_time_utc", ""),
                "date": fixture.get("date", ""),
                "team_a": team_a,
                "team_b": team_b,
                "scoreline_version": V24_SCORELINE_VERSION,
                "scoreline_model": "gated_favorite_blowout_mixture_scoreline_grid",
                "scoreline_params_profile": "v24_gate_no_E_ABCD",
                "p_blowout_final": float(v24_meta["p_blowout_final"]),
                "blowout_gate": float(v24_meta["blowout_gate"]),
                "favorite_bucket": v24_meta["favorite_bucket"],
                "predicted_90min_score": resolution["predicted_90min_score"],
                "predicted_90min_result": resolution["predicted_90min_result"],
                "team_a_win_pct": round(prediction["p_team_a_win"] * 100, 1),
                "draw_pct": round(prediction["p_draw"] * 100, 1),
                "team_b_win_pct": round(prediction["p_team_b_win"] * 100, 1),
                "team_a_90min_win_pct": round(prediction["p_team_a_win"] * 100, 1),
                "draw_90min_pct": round(prediction["p_draw"] * 100, 1),
                "team_b_90min_win_pct": round(prediction["p_team_b_win"] * 100, 1),
                "predicted_120min_score": resolution["predicted_120min_score"],
                "predicted_120min_result": resolution["predicted_120min_result"],
                "extra_time_goal_increment": resolution["extra_time_goal_increment"],
                "penalty_winner": resolution["penalty_winner"],
                "penalty_win_prob_team_a": resolution["penalty_win_prob_team_a"],
                "penalty_win_prob_team_b": resolution["penalty_win_prob_team_b"],
                "final_winner": resolution["final_winner"],
                "match_resolution": resolution["match_resolution"],
                "final_advancing_team": resolution["final_advancing_team"],
                "lambda_a": prediction["lambda_a"],
                "lambda_b": prediction["lambda_b"],
                "top_5_scorelines": str(top5),
                "top_5_scoreline_probability_pct": str(top5_probs),
                "risk": round(min(prediction["p_team_a_win"], prediction["p_team_b_win"], prediction["p_draw"]) * 100),
                "favourite_fail_to_win_risk": round(favorite_fail),
                "bracket_source": "output/live/fixtures_results.csv",
                "uses_random_pairing": False,
                "official_bracket": True,
                "random_seed_used_for": str(RANDOM_SEED_USED_FOR),
            }
        )
    output = pd.DataFrame(rows).sort_values(["kickoff_time_utc", "match_id"]).reset_index(drop=True)
    output.to_csv(R16_PREDICTIONS_PATH, index=False)
    clean = output[
        [
            "match_id",
            "round",
            "kickoff_time_utc",
            "team_a",
            "team_b",
            "team_a_win_pct",
            "team_b_win_pct",
            "draw_pct",
            "top_5_scorelines",
            "top_5_scoreline_probability_pct",
            "predicted_90min_score",
            "predicted_90min_result",
            "predicted_120min_score",
            "predicted_120min_result",
            "extra_time_goal_increment",
            "penalty_winner",
            "penalty_win_prob_team_a",
            "penalty_win_prob_team_b",
            "final_winner",
            "match_resolution",
            "final_advancing_team",
            "risk",
            "favourite_fail_to_win_risk",
        ]
    ].rename(
        columns={
            "round": "Round",
            "top_5_scorelines": "top5 scoreline",
            "top_5_scoreline_probability_pct": "top5 scoreline_probability_pct",
        }
    )
    clean.to_csv(R16_CLEAN_PATH, index=False)
    r16[["match_id", "date", "kickoff_time_utc", "team_a", "team_b", "status"]].to_csv(R16_BRACKET_PATH, index=False)
    metadata = {
        "prediction_scope": "round_of_16_after_r32_strength_update",
        "source_fixtures_file": "output/live/fixtures_results.csv",
        "rating_source_path": "data/processed/team_ratings_world_cup_elo.csv",
        "rating_update_log": "data/processed/r32_strength_update_log.csv",
        "wdl_model_version": output["model_version"].iloc[0],
        "scoreline_version": V24_SCORELINE_VERSION,
        "scoreline_model": "gated_favorite_blowout_mixture_scoreline_grid",
        "row_count": int(len(output)),
        "generated_at": utc_now_iso(),
        "notes": "R16 predictions use ratings updated after completed R32 results. Core W/D/L and V2.4 scoreline model logic were not modified.",
    }
    R16_METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply post-R32 rating update and write R16 predictions.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "model_params_default.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = load_model_config(config_path=args.config)
    rating_col = str(model_config.get("rating_col", "anchored_final_strength"))
    ratings_current = pd.read_csv(RATINGS_PATH)
    lookup = canonical_team_lookup(ratings_current)
    r32_predictions = load_r32_prediction_inputs(lookup)
    r32_actuals = load_completed_r32_actuals(lookup)
    r16_fixtures = load_r16_fixtures(lookup)
    if not BACKUP_PATH.exists():
        shutil.copy2(RATINGS_PATH, BACKUP_PATH)
        print(f"Created backup: {BACKUP_PATH}")
    ratings = pd.read_csv(BACKUP_PATH)
    match_updates = build_match_update_rows(r32_predictions, r32_actuals, r16_fixtures)
    performance = build_team_performance(match_updates)
    updated_ratings, update_log = apply_rating_update(ratings, performance, rating_col)
    updated_ratings.to_csv(RATINGS_PATH, index=False)
    updated_ratings.to_csv(UPDATED_COPY_PATH, index=False)
    performance.to_csv(PERFORMANCE_SUMMARY_PATH, index=False)
    update_log.to_csv(UPDATE_LOG_PATH, index=False)
    model_config["rating_source_path"] = "data/processed/team_ratings_world_cup_elo.csv"
    model_config["rating_col"] = rating_col
    r16_predictions = write_r16_predictions(model_config, updated_ratings, rating_col, r16_fixtures)
    if len(r16_predictions) != 8:
        raise AssertionError(f"Expected 8 R16 predictions, found {len(r16_predictions)}")
    print(f"Updated ratings: {RATINGS_PATH}")
    print(f"Saved updated ratings copy: {UPDATED_COPY_PATH}")
    print(f"Saved R32 update log: {UPDATE_LOG_PATH}")
    print(f"Saved R32 team performance summary: {PERFORMANCE_SUMMARY_PATH}")
    print(f"Saved R16 predictions: {R16_PREDICTIONS_PATH}")
    print(f"Saved R16 clean predictions: {R16_CLEAN_PATH}")
    print(f"Saved R16 metadata: {R16_METADATA_PATH}")
    print("\nChampion simulation command, not run:")
    print(
        "python3 src/simulation/post_r32_simulator.py "
        "--n-simulations 100000 "
        "--output output/simulations/tournament_simulation_after_r32.csv "
        "--sample-bracket-output output/brackets/sample_knockout_bracket_after_r32.csv"
    )


if __name__ == "__main__":
    main()
