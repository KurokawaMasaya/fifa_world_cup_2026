from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.model_config import draw_calibration_kwargs, load_model_config, poisson_parameter_kwargs
from src.simulation.group_stage_simulator import (
    TEAMS_PATH,
    _apply_result,
    _empty_standings,
    initialize_monte_carlo_counts,
    load_base_total_goals,
    load_default_ratings,
    load_group_stage_fixtures,
    sample_scoreline,
)
from src.simulation.tournament_simulator import (
    build_fixed_r32_bracket,
    initialize_tournament_counts,
    precompute_knockout_predictions,
    select_qualifier_records,
    simulate_knockout_bracket,
    update_group_counts_from_records,
    update_knockout_counts,
)
from src.models.poisson_match_model import scoreline_probabilities
from src.models.v2_probability_stack import (
    is_v2_probability_stack_enabled,
    load_v2_feature_context,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = PROJECT_ROOT / "output" / "live"
LIVE_FIXTURES_PATH = LIVE_DIR / "fixtures_results.csv"
PREDICTIONS_CLEAN_PATH = PROJECT_ROOT / "output" / "predictions" / "group_stage_predictions_clean.csv"
LIVE_TOURNAMENT_OUTPUT_PATH = LIVE_DIR / "live_tournament_simulation.csv"
LIVE_GROUP_OUTPUT_PATH = LIVE_DIR / "live_group_projection.csv"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_required_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return pd.read_csv(path)


def normalize_team_key(value: object) -> str:
    return str(value).strip().casefold()


def current_standings_from_finals(fixtures: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    """Build current group standings from final ESPN rows only."""
    standings_by_group = []
    for group_letter, group_teams in teams_df.groupby("group_letter"):
        standings = _empty_standings(group_teams)
        finals = fixtures.loc[
            fixtures["status"].astype(str).str.lower().eq("final")
            & fixtures["team_a"].isin(group_teams["team_name"])
            & fixtures["team_b"].isin(group_teams["team_name"])
        ].copy()
        for _, match in finals.iterrows():
            goals_a = int(match["goals_a"])
            goals_b = int(match["goals_b"])
            _apply_result(standings, match["team_a"], goals_a, goals_b)
            _apply_result(standings, match["team_b"], goals_b, goals_a)
        standings_by_group.extend(standings.values())
    return pd.DataFrame(standings_by_group)


def live_fixture_lookup(live_fixtures: pd.DataFrame) -> dict[tuple[str, str], dict]:
    """Map ESPN final rows by unordered team pair.

    This is a state-conditioning layer only. It locks completed scores into the
    simulation and does not change team ratings or future-match probabilities.
    """
    lookup = {}
    if live_fixtures.empty:
        return lookup
    finals = live_fixtures.loc[live_fixtures["status"].astype(str).str.lower().eq("final")]
    for _, row in finals.iterrows():
        key = tuple(sorted([normalize_team_key(row["team_a"]), normalize_team_key(row["team_b"])]))
        lookup[key] = row.to_dict()
    return lookup


def prediction_lookup(predictions: pd.DataFrame) -> dict[int, dict]:
    required = {
        "match_id",
        "team_a_win_pct",
        "draw_pct",
        "team_b_win_pct",
        "predicted_scoreline",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Clean predictions missing columns: {sorted(missing)}")
    return {int(row["match_id"]): row.to_dict() for _, row in predictions.iterrows()}


def scoreline_grid_from_clean_prediction(prediction: Mapping[str, object]) -> dict[str, float]:
    """Create a small outcome-consistent scoreline grid for unfinished group matches.

    The clean prediction file exposes W/D/L probabilities, not full scoreline
    distributions. We preserve those W/D/L probabilities exactly and use the
    displayed scoreline only as a plausible representative score within the
    sampled outcome.
    """
    p_home = float(prediction["team_a_win_pct"]) / 100.0
    p_draw = float(prediction["draw_pct"]) / 100.0
    p_away = float(prediction["team_b_win_pct"]) / 100.0
    total = p_home + p_draw + p_away
    if total <= 0:
        raise ValueError("Prediction probabilities must have positive mass")
    p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total

    goals_a, goals_b = [int(part) for part in str(prediction["predicted_scoreline"]).split("-")]
    if goals_a > goals_b:
        home_score = f"{goals_a}-{goals_b}"
    else:
        home_score = "1-0"
    if goals_a == goals_b:
        draw_score = f"{goals_a}-{goals_b}"
    else:
        draw_score = "1-1"
    if goals_b > goals_a:
        away_score = f"{goals_a}-{goals_b}"
    else:
        away_score = "0-1"
    return {home_score: p_home, draw_score: p_draw, away_score: p_away}


def simulate_live_group_records(
    group_fixtures: list[dict],
    group_teams: list[dict],
    clean_predictions: Mapping[int, Mapping[str, object]],
    final_lookup: Mapping[tuple[str, str], Mapping[str, object]],
    rng: random.Random,
) -> list[dict]:
    standings = _empty_standings(pd.DataFrame(group_teams))
    for fixture in group_fixtures:
        team_a = fixture["home_team"]
        team_b = fixture["away_team"]
        final_key = tuple(sorted([normalize_team_key(team_a), normalize_team_key(team_b)]))
        if final_key in final_lookup:
            final_row = final_lookup[final_key]
            if normalize_team_key(final_row["team_a"]) == normalize_team_key(team_a):
                goals_a = int(final_row["goals_a"])
                goals_b = int(final_row["goals_b"])
            else:
                goals_a = int(final_row["goals_b"])
                goals_b = int(final_row["goals_a"])
        else:
            grid = scoreline_grid_from_clean_prediction(clean_predictions[int(fixture["match_number"])])
            goals_a, goals_b = sample_scoreline(grid, rng=rng)
        _apply_result(standings, team_a, goals_a, goals_b)
        _apply_result(standings, team_b, goals_b, goals_a)

    ranked = sorted(
        standings.values(),
        key=lambda row: (
            -row["points"],
            -row["goal_difference"],
            -row["goals_for"],
            -row["wins"],
            row["team_name"],
        ),
    )
    for index, row in enumerate(ranked, start=1):
        row["group_rank"] = index
    return ranked


def build_live_results(
    counts: Mapping[str, Mapping[str, object]],
    current: pd.DataFrame,
    n_simulations: int,
    last_updated: str,
) -> pd.DataFrame:
    current_lookup = current.set_index("team_name").to_dict("index")
    rows = []
    for team_name, row in counts.items():
        simulations = int(row["simulations"])
        current_row = current_lookup.get(team_name, {})
        rows.append(
            {
                "team": team_name,
                "group": row["group_letter"],
                "current_points": int(current_row.get("points", 0)),
                "current_gd": int(current_row.get("goal_difference", 0)),
                "current_goals_for": int(current_row.get("goals_for", 0)),
                "current_goals_against": int(current_row.get("goals_against", 0)),
                "simulations": n_simulations,
                "advance_pct": 100.0 * row["qualified_count"] / simulations,
                "round_of_32_pct": 100.0 * row["r32_count"] / simulations,
                "round_of_16_pct": 100.0 * row["r16_count"] / simulations,
                "quarterfinal_pct": 100.0 * row["qf_count"] / simulations,
                "semifinal_pct": 100.0 * row["sf_count"] / simulations,
                "final_pct": 100.0 * row["final_count"] / simulations,
                "champion_pct": 100.0 * row["champion_count"] / simulations,
                "last_updated": last_updated,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["champion_pct", "final_pct", "advance_pct", "team"],
        ascending=[False, False, False, True],
    )


def run_live_tournament_simulation(
    n_simulations: int = 10000,
    seed: int = 2026,
    output_path: Path = LIVE_TOURNAMENT_OUTPUT_PATH,
    group_output_path: Path = LIVE_GROUP_OUTPUT_PATH,
    model_config: dict | None = None,
) -> pd.DataFrame:
    """Run state-conditioned tournament simulations from current live results.

    Completed ESPN matches are fixed as actual scores. Remaining matches use
    existing model probabilities. Ratings are not updated and no model is
    retrained.
    """
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive")
    model_config = load_model_config() if model_config is None else model_config
    fixtures = load_group_stage_fixtures()
    teams_df = pd.read_csv(TEAMS_PATH)
    live_fixtures = load_required_csv(LIVE_FIXTURES_PATH, "live ESPN fixtures/results")
    clean_predictions = prediction_lookup(
        load_required_csv(PREDICTIONS_CLEAN_PATH, "clean group-stage predictions")
    )
    current = current_standings_from_finals(live_fixtures, teams_df)

    ratings_df, rating_col = load_default_ratings(model_config)
    base_total_goals = load_base_total_goals(model_config=model_config)
    model_kwargs = {
        **poisson_parameter_kwargs(model_config),
        **draw_calibration_kwargs(model_config),
    }
    v2_feature_context = (
        load_v2_feature_context(model_config)
        if is_v2_probability_stack_enabled(model_config)
        else None
    )
    knockout_predictions, rating_lookup = precompute_knockout_predictions(
        teams_df=teams_df,
        ratings_df=ratings_df,
        rating_col=rating_col,
        base_total_goals=base_total_goals,
        model_kwargs=model_kwargs,
        model_config=model_config,
        v2_feature_context=v2_feature_context,
    )
    grouped_fixtures = {
        group_letter: group.sort_values("match_number").to_dict("records")
        for group_letter, group in fixtures.groupby("group_letter")
    }
    grouped_teams = {
        group_letter: group.to_dict("records")
        for group_letter, group in teams_df.groupby("group_letter")
    }
    final_lookup = live_fixture_lookup(live_fixtures)
    counts = initialize_tournament_counts(teams_df)
    rng = random.Random(seed)
    for _ in range(n_simulations):
        ranked_groups = [
            simulate_live_group_records(
                group_fixtures=grouped_fixtures[group_letter],
                group_teams=grouped_teams[group_letter],
                clean_predictions=clean_predictions,
                final_lookup=final_lookup,
                rng=rng,
            )
            for group_letter in sorted(grouped_fixtures)
        ]
        winners, runners_up, best_third = select_qualifier_records(ranked_groups)
        update_group_counts_from_records(counts, ranked_groups, best_third)
        bracket = build_fixed_r32_bracket(winners, runners_up, best_third)
        stages, _ = simulate_knockout_bracket(
            r32_bracket=bracket,
            knockout_predictions=knockout_predictions,
            rating_lookup=rating_lookup,
            rng=rng,
        )
        update_knockout_counts(counts, stages)

    last_updated = utc_now_iso()
    output = build_live_results(counts, current, n_simulations, last_updated=last_updated)
    group_projection = current.sort_values(["group_letter", "points", "goal_difference"], ascending=[True, False, False])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    group_projection.to_csv(group_output_path, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live state-conditioned tournament simulation.")
    parser.add_argument("--n-sims", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=LIVE_TOURNAMENT_OUTPUT_PATH)
    args = parser.parse_args()
    results = run_live_tournament_simulation(
        n_simulations=args.n_sims,
        seed=args.seed,
        output_path=args.output,
    )
    print(f"Saved live tournament simulation to {args.output}")
    print(f"Rows: {len(results)}")
    print("Top 10 champion probabilities:")
    print(results[["team", "champion_pct", "advance_pct"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
