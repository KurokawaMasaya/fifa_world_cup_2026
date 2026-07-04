from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Mapping

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.model_config import (
    draw_calibration_kwargs,
    load_model_config,
    metadata_columns,
    poisson_parameter_kwargs,
)
from src.live.update_group_stage_processed_ratings import normalize_team
from src.models.v2_probability_stack import is_v2_probability_stack_enabled, load_v2_feature_context
from src.simulation.group_stage_simulator import TEAMS_PATH, load_base_total_goals, load_default_ratings
from src.simulation.tournament_simulator import (
    RANDOM_SEED_USED_FOR,
    precompute_knockout_predictions,
    simulate_knockout_match_with_resolution,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_FIXTURES_PATH = PROJECT_ROOT / "output" / "live" / "fixtures_results.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "simulations" / "tournament_simulation_after_r32.csv"
DEFAULT_SAMPLE_BRACKET = PROJECT_ROOT / "output" / "brackets" / "sample_knockout_bracket_after_r32.csv"


def canonical_team_lookup(ratings: pd.DataFrame) -> dict[str, str]:
    return {normalize_team(team): team for team in ratings["team_name"].astype(str)}


def load_r16_pairings(ratings: pd.DataFrame) -> list[dict[str, object]]:
    fixtures = pd.read_csv(LIVE_FIXTURES_PATH)
    fixtures["match_id_numeric"] = pd.to_numeric(fixtures["match_id"], errors="coerce")
    r16 = fixtures.loc[fixtures["match_id_numeric"].between(760502, 760509)].copy()
    if len(r16) != 8:
        raise ValueError(f"Expected 8 R16 fixtures, found {len(r16)}")
    lookup = canonical_team_lookup(ratings)
    rows = []
    for _, row in r16.sort_values("match_id_numeric").iterrows():
        team_a_key = normalize_team(row["team_a"])
        team_b_key = normalize_team(row["team_b"])
        if team_a_key not in lookup or team_b_key not in lookup:
            raise ValueError(f"Unresolved R16 fixture team: {row['team_a']} vs {row['team_b']}")
        rows.append(
            {
                "match_id": int(row["match_id_numeric"]),
                "round": "R16",
                "team_a": lookup[team_a_key],
                "team_b": lookup[team_b_key],
                "kickoff_time_utc": row.get("kickoff_time_utc", ""),
                "bracket_source": "output/live/fixtures_results.csv",
            }
        )
    return rows


def initialize_counts(teams_df: pd.DataFrame, r16_teams: set[str]) -> dict[str, dict]:
    counts = {}
    for _, row in teams_df.iterrows():
        team = row["team_name"]
        counts[team] = {
            "team_name": team,
            "fifa_code": row.get("fifa_code"),
            "group_letter": row.get("group_letter"),
            "r16_count": 1 if team in r16_teams else 0,
            "qf_count": 0,
            "sf_count": 0,
            "final_count": 0,
            "champion_count": 0,
        }
    return counts


def pair_adjacent(teams: list[str]) -> list[tuple[str, str]]:
    if len(teams) % 2:
        raise ValueError(f"Cannot pair odd team count: {len(teams)}")
    return [(teams[index], teams[index + 1]) for index in range(0, len(teams), 2)]


def simulate_round(
    round_name: str,
    pairings: list[tuple[str, str]],
    predictions: Mapping[tuple[str, str], Mapping[str, object]],
    rating_lookup: Mapping[str, float],
    rng: random.Random,
    match_number_start: int,
) -> tuple[list[str], list[dict[str, object]]]:
    winners = []
    rows = []
    for offset, (team_a, team_b) in enumerate(pairings):
        resolution = simulate_knockout_match_with_resolution(
            team_a=team_a,
            team_b=team_b,
            knockout_predictions=predictions,
            rating_lookup=rating_lookup,
            rng=rng,
        )
        winners.append(str(resolution["winner"]))
        rows.append(
            {
                "match_number": match_number_start + offset,
                "round": round_name,
                "team_a": team_a,
                "team_b": team_b,
                "winner": resolution["winner"],
                "loser": resolution["loser"],
                "match_resolution": resolution["match_resolution"],
                "predicted_90min_score": resolution["predicted_90min_score"],
                "predicted_120min_score": resolution["predicted_120min_score"],
                "extra_time_goal_increment": resolution["extra_time_goal_increment"],
                "penalty_winner": resolution["penalty_winner"],
                "penalty_win_prob_team_a": resolution["penalty_win_prob_team_a"],
                "penalty_win_prob_team_b": resolution["penalty_win_prob_team_b"],
            }
        )
    return winners, rows


def build_results(counts: dict[str, dict], n_simulations: int, model_config: dict) -> pd.DataFrame:
    rows = []
    for row in counts.values():
        output = {
            "team_name": row["team_name"],
            "fifa_code": row["fifa_code"],
            "group_letter": row["group_letter"],
            "simulations": n_simulations,
            "r16_probability": row["r16_count"] / n_simulations,
            "qf_probability": row["qf_count"] / n_simulations,
            "sf_probability": row["sf_count"] / n_simulations,
            "final_probability": row["final_count"] / n_simulations,
            "champion_probability": row["champion_count"] / n_simulations,
            "r16_count": row["r16_count"],
            "qf_count": row["qf_count"],
            "sf_count": row["sf_count"],
            "final_count": row["final_count"],
            "champion_count": row["champion_count"],
            "bracket_source": "output/live/fixtures_results.csv",
            "uses_random_pairing": False,
            "official_bracket": True,
            "random_seed_used_for": str(RANDOM_SEED_USED_FOR),
        }
        output.update(metadata_columns(model_config))
        rows.append(output)
    return pd.DataFrame(rows).sort_values(
        ["champion_probability", "final_probability", "team_name"],
        ascending=[False, False, True],
    )


def run_post_r32_simulation(
    n_simulations: int,
    seed: int,
    output_path: Path,
    sample_bracket_path: Path,
    model_config: dict,
) -> pd.DataFrame:
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive")
    teams_df = pd.read_csv(TEAMS_PATH)
    ratings_df, rating_col = load_default_ratings(model_config)
    r16_rows = load_r16_pairings(ratings_df)
    r16_pairings = [(row["team_a"], row["team_b"]) for row in r16_rows]
    r16_teams = {team for pairing in r16_pairings for team in pairing}
    base_total_goals = load_base_total_goals(model_config=model_config)
    model_kwargs = {**poisson_parameter_kwargs(model_config), **draw_calibration_kwargs(model_config)}
    v2_context = load_v2_feature_context(model_config) if is_v2_probability_stack_enabled(model_config) else None
    predictions, rating_lookup = precompute_knockout_predictions(
        teams_df=teams_df,
        ratings_df=ratings_df,
        rating_col=rating_col,
        base_total_goals=base_total_goals,
        model_kwargs=model_kwargs,
        model_config=model_config,
        v2_feature_context=v2_context,
    )
    counts = initialize_counts(teams_df, r16_teams)
    rng = random.Random(seed)
    sample_rows = None
    for _ in range(n_simulations):
        r16_winners, r16_match_rows = simulate_round("R16", r16_pairings, predictions, rating_lookup, rng, 89)
        qf_winners, qf_match_rows = simulate_round("QF", pair_adjacent(r16_winners), predictions, rating_lookup, rng, 97)
        sf_winners, sf_match_rows = simulate_round("SF", pair_adjacent(qf_winners), predictions, rating_lookup, rng, 101)
        champion, final_match_rows = simulate_round("Final", pair_adjacent(sf_winners), predictions, rating_lookup, rng, 104)
        for team in r16_winners:
            counts[team]["qf_count"] += 1
        for team in qf_winners:
            counts[team]["sf_count"] += 1
        for team in sf_winners:
            counts[team]["final_count"] += 1
        counts[champion[0]]["champion_count"] += 1
        if sample_rows is None:
            sample_rows = r16_match_rows + qf_match_rows + sf_match_rows + final_match_rows
    results = build_results(counts, n_simulations, model_config)
    results["n_simulations"] = n_simulations
    results["random_seed"] = seed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    if sample_rows is not None:
        sample_bracket_path.parent.mkdir(parents=True, exist_ok=True)
        sample = pd.DataFrame(sample_rows)
        for column, value in metadata_columns(model_config).items():
            sample[column] = value
        sample.to_csv(sample_bracket_path, index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tournament simulation from fixed R16 fixtures after R32.")
    parser.add_argument("--n-simulations", "--num_simulations", dest="n_simulations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-bracket-output", type=Path, default=DEFAULT_SAMPLE_BRACKET)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "model_params_default.json")
    args = parser.parse_args()
    config = load_model_config(config_path=args.config)
    results = run_post_r32_simulation(
        n_simulations=args.n_simulations,
        seed=args.seed,
        output_path=args.output,
        sample_bracket_path=args.sample_bracket_output,
        model_config=config,
    )
    print(f"Ran {args.n_simulations} post-R32 simulations.")
    print(f"Saved simulation output to {args.output}")
    print("Top 3 champion probabilities")
    print(results[["team_name", "champion_probability"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
