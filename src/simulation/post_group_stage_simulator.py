from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.model_config import (
    draw_calibration_kwargs,
    load_model_config,
    metadata_columns,
    poisson_parameter_kwargs,
)
from src.models.v2_probability_stack import is_v2_probability_stack_enabled, load_v2_feature_context
from src.simulation.group_stage_simulator import (
    TEAMS_PATH,
    load_base_total_goals,
    load_default_ratings,
)
from src.simulation.r32_bracket_source import normalize_r32_bracket
from src.simulation.tournament_simulator import (
    RANDOM_SEED_USED_FOR,
    initialize_path_difficulty_counts,
    precompute_knockout_predictions,
    save_sample_knockout_bracket,
    simulate_knockout_bracket,
    update_knockout_counts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_FIXTURES_PATH = PROJECT_ROOT / "output" / "live" / "fixtures_results.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "simulations" / "tournament_simulation_after_group_stage.csv"
DEFAULT_SAMPLE_BRACKET = PROJECT_ROOT / "output" / "brackets" / "sample_knockout_bracket_after_group_stage.csv"


def initialize_post_group_counts(teams_df: pd.DataFrame, r32_teams: set[str]) -> dict[str, dict]:
    counts = {}
    for _, row in teams_df.iterrows():
        team_name = row["team_name"]
        counts[team_name] = {
            "team_name": team_name,
            "fifa_code": row.get("fifa_code"),
            "group_letter": row.get("group_letter"),
            "simulations": 0,
            "r32_count": 0,
            "qualified_after_group_stage": team_name in r32_teams,
            "r16_count": 0,
            "qf_count": 0,
            "sf_count": 0,
            "final_count": 0,
            "champion_count": 0,
        }
    return counts


def build_results(counts: dict[str, dict], n_simulations: int, model_config: dict) -> pd.DataFrame:
    rows = []
    for row in counts.values():
        output = {
            "team_name": row["team_name"],
            "fifa_code": row["fifa_code"],
            "group_letter": row["group_letter"],
            "simulations": n_simulations,
            "r32_probability": row["r32_count"] / n_simulations,
            "r16_probability": row["r16_count"] / n_simulations,
            "qf_probability": row["qf_count"] / n_simulations,
            "sf_probability": row["sf_count"] / n_simulations,
            "final_probability": row["final_count"] / n_simulations,
            "champion_probability": row["champion_count"] / n_simulations,
            "r32_count": row["r32_count"],
            "qualified_after_group_stage": row["qualified_after_group_stage"],
            "r16_count": row["r16_count"],
            "qf_count": row["qf_count"],
            "sf_count": row["sf_count"],
            "final_count": row["final_count"],
            "champion_count": row["champion_count"],
            "bracket_source": "data/processed/R32.rtf",
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


def run_post_group_stage_simulation(
    n_simulations: int,
    seed: int,
    output_path: Path,
    sample_bracket_path: Path,
    model_config: dict | None = None,
) -> pd.DataFrame:
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive")
    model_config = load_model_config() if model_config is None else model_config
    teams_df = pd.read_csv(TEAMS_PATH)
    live_fixtures = pd.read_csv(LIVE_FIXTURES_PATH)
    finals = live_fixtures.loc[live_fixtures["status"].astype(str).str.lower().eq("final")]
    if len(finals) < 72:
        raise ValueError(f"Group stage is not complete: found {len(finals)} final matches")

    ratings_df, rating_col = load_default_ratings(model_config)
    r32_bracket_df = normalize_r32_bracket(ratings_df)
    r32_bracket = r32_bracket_df.to_dict("records")
    r32_teams = {team for row in r32_bracket for team in [row["team_a"], row["team_b"]]}

    base_total_goals = load_base_total_goals(model_config=model_config)
    model_kwargs = {**poisson_parameter_kwargs(model_config), **draw_calibration_kwargs(model_config)}
    v2_context = load_v2_feature_context(model_config) if is_v2_probability_stack_enabled(model_config) else None
    knockout_predictions, rating_lookup = precompute_knockout_predictions(
        teams_df=teams_df,
        ratings_df=ratings_df,
        rating_col=rating_col,
        base_total_goals=base_total_goals,
        model_kwargs=model_kwargs,
        model_config=model_config,
        v2_feature_context=v2_context,
    )

    counts = initialize_post_group_counts(teams_df, r32_teams)
    path_counts = initialize_path_difficulty_counts(teams_df)
    sample_rows = None
    rng = random.Random(seed)
    for _ in range(n_simulations):
        for row in counts.values():
            row["simulations"] += 1
        stages, bracket_rows = simulate_knockout_bracket(
            r32_bracket=r32_bracket,
            knockout_predictions=knockout_predictions,
            rating_lookup=rating_lookup,
            rng=rng,
            path_counts=path_counts,
        )
        if sample_rows is None:
            sample_rows = bracket_rows
        update_knockout_counts(counts, stages)

    results = build_results(counts, n_simulations=n_simulations, model_config=model_config)
    results["n_simulations"] = n_simulations
    results["random_seed"] = seed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    if sample_rows is not None:
        save_sample_knockout_bracket(sample_rows, output_path=sample_bracket_path, model_config=model_config)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run knockout-only simulation after completed group stage.")
    parser.add_argument("--n-simulations", "--num_simulations", dest="n_simulations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-bracket-output", type=Path, default=DEFAULT_SAMPLE_BRACKET)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "model_params_default.json")
    args = parser.parse_args()
    config = load_model_config(config_path=args.config)
    results = run_post_group_stage_simulation(
        n_simulations=args.n_simulations,
        seed=args.seed,
        output_path=args.output,
        sample_bracket_path=args.sample_bracket_output,
        model_config=config,
    )
    print(f"Ran {args.n_simulations} post-group-stage knockout simulations.")
    print(f"Saved simulation output to {args.output}")
    print("Top 10 champion probabilities")
    print(results[["team_name", "champion_probability"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
