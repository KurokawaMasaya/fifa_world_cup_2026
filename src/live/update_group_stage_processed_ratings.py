from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import unicodedata
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
from src.models.v2_probability_stack import (
    is_v2_probability_stack_enabled,
    load_v2_feature_context,
)
from src.simulation.group_stage_simulator import (
    TEAMS_PATH,
    load_base_total_goals,
)
from src.simulation.tournament_simulator import (
    RANDOM_SEED_USED_FOR,
    precompute_knockout_predictions,
)
from src.models.favorite_blowout_mixture_scoreline import (
    canonical_v24_abcd_no_e_params,
    gated_favorite_blowout_mixture_scoreline_grid,
    top_scorelines as v24_top_scorelines,
)
from src.simulation.knockout_resolution import predicted_knockout_resolution
from src.simulation.r32_bracket_source import normalize_r32_bracket


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_DIR = PROJECT_ROOT / "output" / "predictions"
LIVE_DIR = PROJECT_ROOT / "output" / "live"

RATINGS_PATH = PROCESSED_DIR / "team_ratings_world_cup_elo.csv"
BACKUP_PATH = PROCESSED_DIR / "team_ratings_world_cup_elo_pre_group_stage_update_backup.csv"
UPDATED_COPY_PATH = PROCESSED_DIR / "team_ratings_world_cup_elo_after_group_stage.csv"
UPDATE_LOG_PATH = PROCESSED_DIR / "group_stage_strength_update_log.csv"
PERFORMANCE_SUMMARY_PATH = PROCESSED_DIR / "group_stage_team_performance_summary.csv"
HEALTH_STATUS_PATH = PROCESSED_DIR / "player_health_status.csv"
HEALTH_ADJUSTMENTS_PATH = PROCESSED_DIR / "player_health_adjustments.csv"
HEALTH_UPDATE_LOG_PATH = PROCESSED_DIR / "player_health_update_log.csv"
LIVE_FIXTURES_PATH = LIVE_DIR / "fixtures_results.csv"

R1_PREDICTIONS_PATH = PREDICTIONS_DIR / "live" / "group_stage_round1_predictions__v24_abcd_no_e.csv"
R2_PREDICTIONS_PATH = PREDICTIONS_DIR / "live" / "group_stage_round2_predictions_clean.csv"
R3_HEALTH_CLEAN_PATH = PROJECT_ROOT / "output" / "prediction" / "group_stage_predictions_health_adjusted_clean.csv"
R3_HEALTH_DETAIL_PATH = PROJECT_ROOT / "output" / "prediction" / "group_stage_predictions_health_adjusted.csv"
R2_DETAIL_PATH = PREDICTIONS_DIR / "live" / "group_stage_round2_predictions.csv"

R32_PREDICTIONS_PATH = PREDICTIONS_DIR / "round_of_32_predictions_after_group_stage.csv"
R32_PREDICTIONS_WITH_ET_PATH = (
    PREDICTIONS_DIR / "round_of_32_predictions_after_group_stage_with_et_penalties.csv"
)
KNOCKOUT_PREDICTIONS_PATH = PREDICTIONS_DIR / "knockout_predictions_after_group_stage.csv"
R32_BRACKET_PATH = PREDICTIONS_DIR / "round_of_32_bracket_after_group_stage.csv"

POINTS_WEIGHT = 3.0
GD_WEIGHT = 1.5
ATTACK_WEIGHT = 0.04
DEFENSE_WEIGHT = 0.04
MAX_GROUP_STAGE_RATING_DELTA = 20.0
MAX_ATTACK_DELTA = 0.25
MAX_DEFENSE_DELTA = 0.25
MAX_HEALTH_RATING_DELTA = 15.0
V24_SCORELINE_VERSION = "v24_abcd_no_e"
V24_MAX_GOALS = 10


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_team(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text.lower().replace("&", "and"))
    text = " ".join(text.split())
    aliases = {
        "bosnia herzegovina": "bosnia and herzegovina",
        "usa": "united states",
        "us": "united states",
        "ir iran": "iran",
        "cape verde": "cabo verde",
        "ivory coast": "cote d ivoire",
        "cote d ivoire": "cote d ivoire",
        "turkiye": "turkey",
        "morroco": "morocco",
        "congo dr": "dr congo",
        "democratic republic of congo": "dr congo",
    }
    return aliases.get(text, text)


def match_key(team_a: object, team_b: object) -> str:
    return "::".join(sorted([normalize_team(team_a), normalize_team(team_b)]))


def clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def points_for(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def load_predictions() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for round_label, path in [("R1", R1_PREDICTIONS_PATH), ("R2", R2_PREDICTIONS_PATH)]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {round_label} predictions: {path}")
        frame = pd.read_csv(path)
        frame["prediction_round"] = round_label
        frames.append(frame)
    if not R3_HEALTH_CLEAN_PATH.exists():
        raise FileNotFoundError(f"Missing health-adjusted R3 predictions: {R3_HEALTH_CLEAN_PATH}")
    r3 = pd.read_csv(R3_HEALTH_CLEAN_PATH)
    r3 = r3.loc[pd.to_numeric(r3["match_id"], errors="coerce").between(49, 72)].copy()
    r3["prediction_round"] = "R3_health_adjusted"
    frames.append(r3)
    predictions = pd.concat(frames, ignore_index=True).drop_duplicates("match_id", keep="last")
    predictions["_match_key"] = predictions.apply(lambda row: match_key(row["team_a"], row["team_b"]), axis=1)
    return predictions


def load_prediction_lambdas() -> pd.DataFrame:
    frames = []
    for path in [R2_DETAIL_PATH, R3_HEALTH_DETAIL_PATH]:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if {"match_id", "lambda_a", "lambda_b"}.issubset(frame.columns):
            frames.append(frame[["match_id", "lambda_a", "lambda_b"]].copy())
    if not frames:
        return pd.DataFrame(columns=["match_id", "lambda_a", "lambda_b"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("match_id", keep="last")


def load_completed_group_actuals() -> pd.DataFrame:
    actuals = pd.read_csv(LIVE_FIXTURES_PATH)
    finals = actuals.loc[actuals["status"].astype(str).str.lower().eq("final")].copy()
    finals = finals.dropna(subset=["goals_a", "goals_b"])
    finals["_match_key"] = finals.apply(lambda row: match_key(row["team_a"], row["team_b"]), axis=1)
    return finals


def build_match_update_rows(predictions: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    lambdas = load_prediction_lambdas()
    predictions = predictions.merge(lambdas, on="match_id", how="left")
    rows: list[dict[str, Any]] = []
    for _, pred in predictions.iterrows():
        actual_matches = actuals.loc[actuals["_match_key"].eq(pred["_match_key"])]
        if actual_matches.empty:
            continue
        actual = actual_matches.iloc[0]
        same_orientation = normalize_team(actual["team_a"]) == normalize_team(pred["team_a"])
        goals_a = int(actual["goals_a"] if same_orientation else actual["goals_b"])
        goals_b = int(actual["goals_b"] if same_orientation else actual["goals_a"])
        lambda_a = pd.to_numeric(pred.get("lambda_a"), errors="coerce")
        lambda_b = pd.to_numeric(pred.get("lambda_b"), errors="coerce")
        if pd.isna(lambda_a) or pd.isna(lambda_b):
            lambda_a = math.nan
            lambda_b = math.nan

        teams = [
            (
                pred["team_a"],
                pred["team_b"],
                goals_a,
                goals_b,
                3 * float(pred["team_a_win_pct"]) / 100.0 + float(pred["draw_pct"]) / 100.0,
                lambda_a,
                lambda_b,
            ),
            (
                pred["team_b"],
                pred["team_a"],
                goals_b,
                goals_a,
                3 * float(pred["team_b_win_pct"]) / 100.0 + float(pred["draw_pct"]) / 100.0,
                lambda_b,
                lambda_a,
            ),
        ]
        for team, opponent, gf, ga, expected_points, xgf, xga in teams:
            actual_points = points_for(gf, ga)
            points_surprise = actual_points - expected_points
            expected_margin = xgf - xga if not pd.isna(xgf) and not pd.isna(xga) else math.nan
            actual_margin = gf - ga
            margin_surprise = actual_margin - expected_margin if not pd.isna(expected_margin) else 0.0
            attack_surprise = gf - xgf if not pd.isna(xgf) else 0.0
            defense_surprise = xga - ga if not pd.isna(xga) else 0.0
            rows.append(
                {
                    "match_id": pred["match_id"],
                    "date": actual.get("date"),
                    "prediction_round": pred.get("prediction_round"),
                    "team": team,
                    "opponent": opponent,
                    "goals_for": gf,
                    "goals_against": ga,
                    "actual_points": actual_points,
                    "expected_points": expected_points,
                    "points_surprise": points_surprise,
                    "expected_goals_for": xgf,
                    "expected_goals_against": xga,
                    "attack_surprise": attack_surprise,
                    "defense_surprise": defense_surprise,
                    "expected_margin": expected_margin,
                    "actual_margin": actual_margin,
                    "margin_surprise": margin_surprise,
                }
            )
    return pd.DataFrame(rows)


def build_team_performance(match_updates: pd.DataFrame) -> pd.DataFrame:
    if match_updates.empty:
        raise ValueError("No completed group-stage matches matched predictions")
    perf = (
        match_updates.groupby("team", as_index=False)
        .agg(
            matches_played=("match_id", "count"),
            points=("actual_points", "sum"),
            expected_points=("expected_points", "sum"),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
            actual_goal_difference=("actual_margin", "sum"),
            expected_goal_difference=("expected_margin", "sum"),
            attack_surprise=("attack_surprise", "sum"),
            defense_surprise=("defense_surprise", "sum"),
        )
    )
    outcomes = match_updates.assign(
        win=lambda df: (df["goals_for"] > df["goals_against"]).astype(int),
        draw=lambda df: (df["goals_for"] == df["goals_against"]).astype(int),
        loss=lambda df: (df["goals_for"] < df["goals_against"]).astype(int),
    )
    record = outcomes.groupby("team", as_index=False).agg(
        wins=("win", "sum"), draws=("draw", "sum"), losses=("loss", "sum")
    )
    perf = perf.merge(record, on="team", how="left")
    perf["goals_for_per_match"] = perf["goals_for"] / perf["matches_played"]
    perf["goals_against_per_match"] = perf["goals_against"] / perf["matches_played"]
    perf["points_surprise"] = perf["points"] - perf["expected_points"]
    perf["goal_difference"] = perf["actual_goal_difference"]
    perf["goal_difference_surprise"] = (
        perf["actual_goal_difference"] - pd.to_numeric(perf["expected_goal_difference"], errors="coerce").fillna(0)
    )
    perf["performance_delta_raw"] = POINTS_WEIGHT * perf["points_surprise"] + GD_WEIGHT * perf["goal_difference_surprise"]
    perf["performance_delta"] = perf["performance_delta_raw"].clip(-MAX_GROUP_STAGE_RATING_DELTA, MAX_GROUP_STAGE_RATING_DELTA)
    perf["attack_delta"] = (ATTACK_WEIGHT * perf["attack_surprise"]).clip(-MAX_ATTACK_DELTA, MAX_ATTACK_DELTA)
    perf["defense_delta"] = (DEFENSE_WEIGHT * perf["defense_surprise"]).clip(-MAX_DEFENSE_DELTA, MAX_DEFENSE_DELTA)
    ordered_columns = [
        "team",
        "matches_played",
        "wins",
        "draws",
        "losses",
        "points",
        "goals_for",
        "goals_against",
        "goal_difference",
        "goals_for_per_match",
        "goals_against_per_match",
        "expected_points",
        "points_surprise",
        "expected_goal_difference",
        "actual_goal_difference",
        "goal_difference_surprise",
        "attack_surprise",
        "defense_surprise",
        "performance_delta_raw",
        "performance_delta",
        "attack_delta",
        "defense_delta",
    ]
    return perf[ordered_columns].sort_values(["points", "goal_difference", "goals_for"], ascending=False)


def write_health_status() -> pd.DataFrame:
    columns = [
        "team",
        "player",
        "status",
        "injury_or_suspension_type",
        "expected_return",
        "source",
        "source_url",
        "confidence",
        "role",
        "importance_score",
        "availability_pct",
        "rating_impact",
        "attack_impact",
        "defense_impact",
        "last_updated",
    ]
    if HEALTH_ADJUSTMENTS_PATH.exists():
        adjustments = pd.read_csv(HEALTH_ADJUSTMENTS_PATH)
    else:
        adjustments = pd.DataFrame()
    if adjustments.empty:
        status = pd.DataFrame(columns=columns)
    else:
        status = pd.DataFrame(
            {
                "team": adjustments.get("team"),
                "player": adjustments.get("player"),
                "status": adjustments.get("status"),
                "injury_or_suspension_type": "",
                "expected_return": "",
                "source": adjustments.get("source"),
                "source_url": "",
                "confidence": adjustments.get("confidence"),
                "role": adjustments.get("role"),
                "importance_score": adjustments.get("importance_score"),
                "availability_pct": adjustments.get("availability_pct"),
                "rating_impact": adjustments.get("rating_impact"),
                "attack_impact": adjustments.get("attack_impact"),
                "defense_impact": adjustments.get("defense_impact"),
                "last_updated": adjustments.get("last_updated"),
            }
        )
    status.to_csv(HEALTH_STATUS_PATH, index=False)
    return status


def team_health_deltas() -> pd.DataFrame:
    if not HEALTH_UPDATE_LOG_PATH.exists():
        return pd.DataFrame(columns=["team", "post_group_health_rating_delta", "post_group_health_attack_delta", "post_group_health_defense_delta"])
    health = pd.read_csv(HEALTH_UPDATE_LOG_PATH)
    if health.empty:
        return pd.DataFrame(columns=["team", "post_group_health_rating_delta", "post_group_health_attack_delta", "post_group_health_defense_delta"])
    for column in ["health_rating_delta", "health_attack_delta", "health_defense_delta"]:
        if column not in health.columns:
            health[column] = 0.0
    health = health[["team", "health_rating_delta", "health_attack_delta", "health_defense_delta"]].copy()
    health["health_rating_delta"] = pd.to_numeric(health["health_rating_delta"], errors="coerce").fillna(0).clip(
        -MAX_HEALTH_RATING_DELTA, MAX_HEALTH_RATING_DELTA
    )
    health["health_attack_delta"] = pd.to_numeric(health["health_attack_delta"], errors="coerce").fillna(0).clip(
        -MAX_ATTACK_DELTA, MAX_ATTACK_DELTA
    )
    health["health_defense_delta"] = pd.to_numeric(health["health_defense_delta"], errors="coerce").fillna(0).clip(
        -MAX_DEFENSE_DELTA, MAX_DEFENSE_DELTA
    )
    return health.rename(
        columns={
            "health_rating_delta": "post_group_health_rating_delta",
            "health_attack_delta": "post_group_health_attack_delta",
            "health_defense_delta": "post_group_health_defense_delta",
        }
    )


def apply_rating_update(ratings: pd.DataFrame, performance: pd.DataFrame, rating_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    health = team_health_deltas()
    perf = performance.copy()
    perf["_team_key"] = perf["team"].map(normalize_team)
    health["_team_key"] = health["team"].map(normalize_team) if not health.empty else pd.Series(dtype=object)
    output = ratings.copy()
    output["_team_key"] = output["team_name"].map(normalize_team)
    output = output.merge(
        perf[["_team_key", "performance_delta", "attack_delta", "defense_delta"]],
        on="_team_key",
        how="left",
    )
    output = output.merge(
        health[["_team_key", "post_group_health_rating_delta", "post_group_health_attack_delta", "post_group_health_defense_delta"]],
        on="_team_key",
        how="left",
    )
    for column in [
        "performance_delta",
        "attack_delta",
        "defense_delta",
        "post_group_health_rating_delta",
        "post_group_health_attack_delta",
        "post_group_health_defense_delta",
    ]:
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
    output["rating_before_group_stage_update"] = pd.to_numeric(output[rating_col], errors="coerce")
    output["group_stage_performance_delta"] = output["performance_delta"]
    output["group_stage_health_delta"] = output["post_group_health_rating_delta"]
    output["group_stage_total_delta"] = (
        output["group_stage_performance_delta"] + output["group_stage_health_delta"]
    ).clip(-MAX_GROUP_STAGE_RATING_DELTA, MAX_GROUP_STAGE_RATING_DELTA)
    output["rating_after_group_stage"] = output["rating_before_group_stage_update"] + output["group_stage_total_delta"]
    output[rating_col] = output["rating_after_group_stage"]
    output["effective_rating"] = output["rating_after_group_stage"]
    output["effective_attack"] = (output.get("effective_attack", 0.0).fillna(0.0) if "effective_attack" in output else 0.0) + output[
        "attack_delta"
    ] + output["post_group_health_attack_delta"]
    output["effective_defense"] = (output.get("effective_defense", 0.0).fillna(0.0) if "effective_defense" in output else 0.0) + output[
        "defense_delta"
    ] + output["post_group_health_defense_delta"]
    log = output.merge(
        perf.drop(columns=["_team_key", "performance_delta", "attack_delta", "defense_delta"], errors="ignore"),
        left_on="team_name",
        right_on="team",
        how="left",
    )
    log = log[
        [
            "team_name",
            "rating_before_group_stage_update",
            "group_stage_performance_delta",
            "group_stage_health_delta",
            "group_stage_total_delta",
            "rating_after_group_stage",
            "attack_delta",
            "post_group_health_attack_delta",
            "effective_attack",
            "defense_delta",
            "post_group_health_defense_delta",
            "effective_defense",
            "matches_played",
            "points",
            "goals_for",
            "goals_against",
            "goal_difference",
            "expected_points",
            "points_surprise",
            "goal_difference_surprise",
        ]
    ].rename(columns={"team_name": "team", "rating_before_group_stage_update": "before_rating", "rating_after_group_stage": "after_rating"})
    log["reason"] = (
        "group_stage_results_delta="
        + log["group_stage_performance_delta"].round(3).astype(str)
        + "; health_delta="
        + log["group_stage_health_delta"].round(3).astype(str)
    )
    output = output.drop(columns=["_team_key", "performance_delta", "attack_delta", "defense_delta"], errors="ignore")
    return output, log


def top_scorelines(scoreline_probabilities: dict[str, float], n: int = 5) -> tuple[list[str], list[int]]:
    ordered = sorted(scoreline_probabilities.items(), key=lambda item: item[1], reverse=True)[:n]
    return [score for score, _ in ordered], [round(prob * 100) for _, prob in ordered]


def write_knockout_predictions(model_config: dict, ratings: pd.DataFrame, rating_col: str) -> pd.DataFrame:
    teams_df = pd.read_csv(TEAMS_PATH)
    r32_bracket = normalize_r32_bracket(ratings)
    r32_bracket = r32_bracket.to_dict("records")
    pd.DataFrame(r32_bracket).to_csv(R32_BRACKET_PATH, index=False)

    base_total_goals = load_base_total_goals(model_config=model_config)
    model_kwargs = {
        **poisson_parameter_kwargs(model_config),
        **draw_calibration_kwargs(model_config),
    }
    v2_context = load_v2_feature_context(model_config) if is_v2_probability_stack_enabled(model_config) else None
    knockout_predictions, _ = precompute_knockout_predictions(
        teams_df=teams_df,
        ratings_df=ratings,
        rating_col=rating_col,
        base_total_goals=base_total_goals,
        model_kwargs=model_kwargs,
        model_config=model_config,
        v2_feature_context=v2_context,
    )
    metadata = metadata_columns(model_config)
    rating_lookup = ratings.set_index("team_name")[rating_col].astype(float).to_dict()
    scoreline_params = canonical_v24_abcd_no_e_params()
    rows = []
    for row in r32_bracket:
        prediction = knockout_predictions[(row["team_a"], row["team_b"])]
        rating_gap = rating_lookup[row["team_a"]] - rating_lookup[row["team_b"]]
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
            team_a=row["team_a"],
            team_b=row["team_b"],
            prediction=v24_prediction,
            rating_a=rating_lookup[row["team_a"]],
            rating_b=rating_lookup[row["team_b"]],
        )
        rows.append(
            {
                **metadata,
                "match_number": row["match_number"],
                "round": row["round"],
                "slot_a": row["slot_a"],
                "slot_b": row["slot_b"],
                "team_a": row["team_a"],
                "team_b": row["team_b"],
                "section": row.get("section"),
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
                "bracket_source": row["bracket_source"],
                "uses_random_pairing": row["uses_random_pairing"],
                "official_bracket": row["official_bracket"],
                "random_seed_used_for": str(RANDOM_SEED_USED_FOR),
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(R32_PREDICTIONS_PATH, index=False)
    output.to_csv(R32_PREDICTIONS_WITH_ET_PATH, index=False)
    output.to_csv(KNOCKOUT_PREDICTIONS_PATH, index=False)
    return output


def validate_outputs(ratings: pd.DataFrame, performance: pd.DataFrame, knockout: pd.DataFrame, rating_col: str) -> None:
    if not BACKUP_PATH.exists():
        raise AssertionError("Backup file missing")
    if ratings["team_name"].duplicated().any():
        raise AssertionError("Duplicate team rows in ratings")
    if ratings["team_name"].isna().any() or ratings[rating_col].isna().any():
        raise AssertionError("Missing team names or updated ratings")
    if len(performance) != 48:
        raise AssertionError(f"Expected 48 team performance rows, found {len(performance)}")
    if (performance["matches_played"] != 3).any():
        raise AssertionError("Every team should have 3 completed group-stage matches")
    if len(knockout) != 16:
        raise AssertionError(f"Expected 16 R32 prediction rows, found {len(knockout)}")
    knockout_teams = set(knockout["team_a"]).union(set(knockout["team_b"]))
    missing = knockout_teams - set(ratings["team_name"])
    if missing:
        raise AssertionError(f"Knockout teams missing updated ratings: {sorted(missing)}")
    if not ratings[rating_col].between(1200, 2100).all():
        raise AssertionError("Updated ratings outside reasonable range")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply formal post-group-stage strength update.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "model_params_default.json")
    parser.add_argument("--skip-predictions", action="store_true", help="Update data files but do not write R32 predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = load_model_config(config_path=args.config)
    rating_col = str(model_config.get("rating_col", "anchored_final_strength"))
    if not BACKUP_PATH.exists():
        shutil.copy2(RATINGS_PATH, BACKUP_PATH)
        print(f"Created backup: {BACKUP_PATH}")
    ratings = pd.read_csv(BACKUP_PATH)
    predictions = load_predictions()
    actuals = load_completed_group_actuals()
    if len(actuals) < 72:
        raise SystemExit(f"Group stage is not complete: found {len(actuals)} final matches")
    match_updates = build_match_update_rows(predictions, actuals)
    performance = build_team_performance(match_updates)
    write_health_status()
    updated_ratings, update_log = apply_rating_update(ratings, performance, rating_col)
    updated_ratings.to_csv(RATINGS_PATH, index=False)
    updated_ratings.to_csv(UPDATED_COPY_PATH, index=False)
    performance.to_csv(PERFORMANCE_SUMMARY_PATH, index=False)
    update_log.to_csv(UPDATE_LOG_PATH, index=False)
    model_config["rating_source_path"] = "data/processed/team_ratings_world_cup_elo.csv"
    model_config["rating_col"] = rating_col
    knockout = pd.DataFrame()
    if not args.skip_predictions:
        knockout = write_knockout_predictions(model_config, updated_ratings, rating_col)
    validate_outputs(updated_ratings, performance, knockout, rating_col)
    print(f"Updated ratings: {RATINGS_PATH}")
    print(f"Saved updated ratings copy: {UPDATED_COPY_PATH}")
    print(f"Saved strength update log: {UPDATE_LOG_PATH}")
    print(f"Saved team performance summary: {PERFORMANCE_SUMMARY_PATH}")
    print(f"Saved player health status: {HEALTH_STATUS_PATH}")
    print(f"Saved R32 predictions: {R32_PREDICTIONS_PATH}")
    print(f"Saved knockout predictions: {KNOCKOUT_PREDICTIONS_PATH}")
    print("\nSimulation commands:")
    print(
        "python3 src/simulation/post_group_stage_simulator.py "
        "--n-simulations 1000 "
        "--output output/simulations/tournament_simulation_after_group_stage_smoke.csv"
    )
    print(
        "python3 src/simulation/post_group_stage_simulator.py "
        "--n-simulations 100000 "
        "--output output/simulations/tournament_simulation_after_group_stage.csv"
    )


if __name__ == "__main__":
    main()
