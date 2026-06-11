from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Mapping

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.poisson_match_model import (
    CALIBRATED_RATINGS_PATH,
    DEFAULT_MATCHES_PATH,
    DEFAULT_RATING_COL,
    DEFAULT_RATINGS_PATH,
    FALLBACK_RATING_COL,
    estimate_base_goals,
    predict_from_ratings,
)
from src.config.model_config import (
    SIMULATIONS_DIR,
    draw_calibration_kwargs,
    load_model_config,
    metadata_columns,
    poisson_parameter_kwargs,
    simulation_output_path,
)
from src.models.v2_probability_stack import (
    apply_v2_probability_stack,
    is_v2_probability_stack_enabled,
    load_v2_feature_context,
    rescale_scoreline_probabilities_to_outcomes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MATCHES_PATH = RAW_DATA_DIR / "matches.csv"
TEAMS_PATH = RAW_DATA_DIR / "teams.csv"
STAGES_PATH = RAW_DATA_DIR / "tournament_stages.csv"
SAMPLE_OUTPUT_PATH = SIMULATIONS_DIR / "group_stage_sample_simulation_default.csv"
MONTE_CARLO_OUTPUT_PATH = SIMULATIONS_DIR / "group_stage_monte_results_default.csv"

EXPECTED_GROUP_COUNT = 12
EXPECTED_GROUP_MATCH_COUNT = 72
EXPECTED_DIRECT_QUALIFIERS = 24
EXPECTED_THIRD_PLACE_QUALIFIERS = 8
EXPECTED_TOTAL_QUALIFIERS = 32
DEFAULT_MONTE_CARLO_SIMULATIONS = 10000


def load_group_stage_fixtures(
    matches_path: Path = MATCHES_PATH,
    teams_path: Path = TEAMS_PATH,
    stages_path: Path = STAGES_PATH,
) -> pd.DataFrame:
    """Load World Cup fixtures and return group-stage rows with team names.

    The raw fixture table stores team IDs, so this function joins through
    teams.csv before simulation. Keeping that join in one place makes the
    72-match group-stage sanity check explicit and avoids hard-coded fixtures.
    """
    matches = pd.read_csv(matches_path)
    teams = pd.read_csv(teams_path)
    stages = pd.read_csv(stages_path)

    fixtures = matches.merge(
        stages[["id", "stage_name"]],
        left_on="stage_id",
        right_on="id",
        how="left",
        suffixes=("", "_stage"),
    )
    fixtures = fixtures.loc[fixtures["stage_name"] == "Group Stage"].copy()

    home_teams = teams[["id", "team_name", "fifa_code", "group_letter"]].rename(
        columns={
            "id": "home_team_id",
            "team_name": "home_team",
            "fifa_code": "home_fifa_code",
            "group_letter": "home_group",
        }
    )
    away_teams = teams[["id", "team_name", "fifa_code", "group_letter"]].rename(
        columns={
            "id": "away_team_id",
            "team_name": "away_team",
            "fifa_code": "away_fifa_code",
            "group_letter": "away_group",
        }
    )
    fixtures = fixtures.merge(home_teams, on="home_team_id", how="left")
    fixtures = fixtures.merge(away_teams, on="away_team_id", how="left")
    fixtures["group_letter"] = fixtures["match_label"].str.extract(r"Group\s+([A-L])")

    missing = fixtures[
        fixtures[["home_team", "away_team", "group_letter"]].isna().any(axis=1)
    ]
    if not missing.empty:
        raise ValueError(
            "Some group-stage fixtures are missing team names or group labels: "
            f"{missing['match_number'].tolist()}"
        )

    if not (fixtures["home_group"] == fixtures["away_group"]).all():
        raise ValueError("Found group-stage fixtures with teams from different groups")

    columns = [
        "id",
        "match_number",
        "kickoff_at",
        "match_label",
        "group_letter",
        "home_team",
        "away_team",
        "home_fifa_code",
        "away_fifa_code",
    ]
    return fixtures[columns].sort_values("match_number").reset_index(drop=True)


def load_default_ratings(model_config: dict | None = None) -> tuple[pd.DataFrame, str]:
    """Load anchored final strength when available, otherwise legacy ratings."""
    model_config = load_model_config() if model_config is None else model_config
    configured_ratings_path = model_config.get("rating_source_path")
    ratings_path = (
        PROJECT_ROOT / configured_ratings_path
        if configured_ratings_path
        else DEFAULT_RATINGS_PATH
    )
    if not ratings_path.exists() and not DEFAULT_RATINGS_PATH.exists() and not CALIBRATED_RATINGS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find ratings file: {ratings_path}. "
            "Run src/ratings/build_elo_ratings.py or src/ratings/build_player_impacted_strength.py first."
        )

    if not ratings_path.exists():
        ratings_path = DEFAULT_RATINGS_PATH
    ratings = pd.read_csv(ratings_path)
    if DEFAULT_RATING_COL not in ratings.columns and CALIBRATED_RATINGS_PATH.exists():
        calibrated = pd.read_csv(CALIBRATED_RATINGS_PATH)
        if DEFAULT_RATING_COL in calibrated.columns:
            ratings_path = CALIBRATED_RATINGS_PATH
            ratings = calibrated

    configured_rating_col = model_config.get("rating_col", DEFAULT_RATING_COL)
    rating_col = configured_rating_col if configured_rating_col in ratings.columns else FALLBACK_RATING_COL
    if rating_col not in ratings.columns:
        raise ValueError(f"{ratings_path} does not contain a usable rating column")
    return ratings, rating_col


def load_base_total_goals(
    matches_path: Path = DEFAULT_MATCHES_PATH,
    model_config: dict | None = None,
) -> float:
    """Use the existing Poisson-model base-goals estimator for tournament scoring."""
    if model_config is not None and "base_total_goals" in model_config:
        return float(model_config["base_total_goals"])
    if not matches_path.exists():
        return 2.65
    return 2.0 * estimate_base_goals(pd.read_csv(matches_path))


def sample_scoreline(
    scoreline_probabilities: Mapping[str, float],
    rng: random.Random,
) -> tuple[int, int]:
    """Sample one scoreline from the normalized Poisson scoreline distribution."""
    threshold = rng.random()
    cumulative_probability = 0.0
    last_scoreline = "0-0"

    for scoreline, probability in scoreline_probabilities.items():
        cumulative_probability += probability
        last_scoreline = scoreline
        if threshold <= cumulative_probability:
            goals_a, goals_b = scoreline.split("-")
            return int(goals_a), int(goals_b)

    goals_a, goals_b = last_scoreline.split("-")
    return int(goals_a), int(goals_b)


def simulate_match(
    fixture: pd.Series,
    ratings_df: pd.DataFrame,
    rating_col: str,
    base_total_goals: float,
    rng: random.Random,
    model_kwargs: Mapping[str, object] | None = None,
) -> dict:
    """Predict and sample a single group-stage match using the Poisson model."""
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    model_kwargs.setdefault("base_total_goals", base_total_goals)
    prediction = predict_from_ratings(
        team_a=fixture["home_team"],
        team_b=fixture["away_team"],
        ratings_df=ratings_df,
        rating_col=rating_col,
        **model_kwargs,
    )
    home_goals, away_goals = sample_scoreline(
        prediction["scoreline_probabilities"],
        rng=rng,
    )

    return {
        "match_number": fixture["match_number"],
        "kickoff_at": fixture["kickoff_at"],
        "group_letter": fixture["group_letter"],
        "home_team": fixture["home_team"],
        "away_team": fixture["away_team"],
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_xg": prediction["lambda_a"],
        "away_xg": prediction["lambda_b"],
    }


def precompute_fixture_predictions(
    fixtures: pd.DataFrame,
    ratings_df: pd.DataFrame,
    rating_col: str,
    base_total_goals: float,
    model_kwargs: Mapping[str, object] | None = None,
    model_config: Mapping[str, object] | None = None,
    v2_feature_context: Mapping[str, object] | None = None,
) -> dict[int, dict]:
    """Precompute Poisson probabilities once so Monte Carlo runs stay fast."""
    predictions = {}
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    model_kwargs.setdefault("base_total_goals", base_total_goals)
    use_v2_stack = model_config is not None and is_v2_probability_stack_enabled(model_config)
    if use_v2_stack and v2_feature_context is None:
        v2_feature_context = load_v2_feature_context(model_config)

    for _, fixture in fixtures.iterrows():
        prediction = predict_from_ratings(
            team_a=fixture["home_team"],
            team_b=fixture["away_team"],
            ratings_df=ratings_df,
            rating_col=rating_col,
            **model_kwargs,
        )
        scoreline_probabilities = prediction["scoreline_probabilities"]
        final_probabilities = {
            "team_a_win": prediction["p_team_a_win"],
            "draw": prediction["p_draw"],
            "team_b_win": prediction["p_team_b_win"],
        }
        if use_v2_stack:
            v2_adjustment = apply_v2_probability_stack(
                team_a=fixture["home_team"],
                team_b=fixture["away_team"],
                p_team_a_win=prediction["p_team_a_win"],
                p_draw=prediction["p_draw"],
                p_team_b_win=prediction["p_team_b_win"],
                config=model_config,
                feature_context=v2_feature_context,
            )
            final_probabilities = {
                "team_a_win": v2_adjustment["v2_p_team_a_win"],
                "draw": v2_adjustment["v2_p_draw"],
                "team_b_win": v2_adjustment["v2_p_team_b_win"],
            }
            scoreline_probabilities = rescale_scoreline_probabilities_to_outcomes(
                scoreline_probabilities=scoreline_probabilities,
                target_outcomes=final_probabilities,
            )
        predictions[int(fixture["match_number"])] = {
            "home_xg": prediction["lambda_a"],
            "away_xg": prediction["lambda_b"],
            "scoreline_probabilities": scoreline_probabilities,
            "p_team_a_win": final_probabilities["team_a_win"],
            "p_draw": final_probabilities["draw"],
            "p_team_b_win": final_probabilities["team_b_win"],
            "model_probability_stack": "v2" if use_v2_stack else "v1",
        }
    return predictions


def simulate_match_from_prediction(
    fixture: pd.Series,
    fixture_prediction: Mapping[str, object],
    rng: random.Random,
) -> dict:
    """Sample a fixture from precomputed Poisson probabilities."""
    home_goals, away_goals = sample_scoreline(
        fixture_prediction["scoreline_probabilities"],
        rng=rng,
    )
    return {
        "match_number": fixture["match_number"],
        "kickoff_at": fixture["kickoff_at"],
        "group_letter": fixture["group_letter"],
        "home_team": fixture["home_team"],
        "away_team": fixture["away_team"],
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_xg": fixture_prediction["home_xg"],
        "away_xg": fixture_prediction["away_xg"],
    }


def _empty_standings(group_teams: pd.DataFrame) -> dict[str, dict]:
    standings = {}
    for _, team in group_teams.iterrows():
        standings[team["team_name"]] = {
            "group_letter": team["group_letter"],
            "team_name": team["team_name"],
            "fifa_code": team["fifa_code"],
            "matches_played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_difference": 0,
            "points": 0,
        }
    return standings


def _apply_result(standings: dict[str, dict], team: str, goals_for: int, goals_against: int) -> None:
    row = standings[team]
    row["matches_played"] += 1
    row["goals_for"] += goals_for
    row["goals_against"] += goals_against
    row["goal_difference"] = row["goals_for"] - row["goals_against"]

    if goals_for > goals_against:
        row["wins"] += 1
        row["points"] += 3
    elif goals_for < goals_against:
        row["losses"] += 1
    else:
        row["draws"] += 1
        row["points"] += 1


def rank_group(standings: Mapping[str, dict]) -> pd.DataFrame:
    """Rank a group by common table tiebreakers.

    This first-stage simulator uses points, goal difference, goals for, wins,
    and team name as a deterministic final fallback. Head-to-head tiebreakers
    can be layered in later when the knockout simulator is added.
    """
    table = pd.DataFrame(standings.values())
    table = table.sort_values(
        ["points", "goal_difference", "goals_for", "wins", "team_name"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    table["group_rank"] = range(1, len(table) + 1)
    return table


def simulate_one_group(
    group_letter: str,
    fixtures: pd.DataFrame,
    teams_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    rating_col: str,
    base_total_goals: float,
    rng: random.Random,
    model_kwargs: Mapping[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate every fixture in one group and return standings plus matches."""
    group_fixtures = fixtures.loc[fixtures["group_letter"] == group_letter].copy()
    group_teams = teams_df.loc[teams_df["group_letter"] == group_letter].copy()

    if len(group_teams) != 4:
        raise ValueError(f"Group {group_letter} should contain 4 teams, found {len(group_teams)}")
    if len(group_fixtures) != 6:
        raise ValueError(f"Group {group_letter} should contain 6 matches, found {len(group_fixtures)}")

    standings = _empty_standings(group_teams)
    match_rows = []
    for _, fixture in group_fixtures.sort_values("match_number").iterrows():
        result = simulate_match(
            fixture=fixture,
            ratings_df=ratings_df,
            rating_col=rating_col,
            base_total_goals=base_total_goals,
            rng=rng,
            model_kwargs=model_kwargs,
        )
        match_rows.append(result)
        _apply_result(
            standings,
            result["home_team"],
            result["home_goals"],
            result["away_goals"],
        )
        _apply_result(
            standings,
            result["away_team"],
            result["away_goals"],
            result["home_goals"],
        )

    return rank_group(standings), pd.DataFrame(match_rows)


def simulate_one_group_from_predictions(
    group_letter: str,
    fixtures: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixture_predictions: Mapping[int, Mapping[str, object]],
    rng: random.Random,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate one group using precomputed match-level probability grids."""
    group_fixtures = fixtures.loc[fixtures["group_letter"] == group_letter].copy()
    group_teams = teams_df.loc[teams_df["group_letter"] == group_letter].copy()

    if len(group_teams) != 4:
        raise ValueError(f"Group {group_letter} should contain 4 teams, found {len(group_teams)}")
    if len(group_fixtures) != 6:
        raise ValueError(f"Group {group_letter} should contain 6 matches, found {len(group_fixtures)}")

    standings = _empty_standings(group_teams)
    match_rows = []
    for _, fixture in group_fixtures.sort_values("match_number").iterrows():
        result = simulate_match_from_prediction(
            fixture=fixture,
            fixture_prediction=fixture_predictions[int(fixture["match_number"])],
            rng=rng,
        )
        match_rows.append(result)
        _apply_result(
            standings,
            result["home_team"],
            result["home_goals"],
            result["away_goals"],
        )
        _apply_result(
            standings,
            result["away_team"],
            result["away_goals"],
            result["home_goals"],
        )

    return rank_group(standings), pd.DataFrame(match_rows)


def select_group_stage_qualifiers(standings: pd.DataFrame) -> pd.DataFrame:
    """Select top two in each group plus the best eight third-place teams."""
    direct = standings.loc[standings["group_rank"] <= 2].copy()
    direct["qualification_status"] = "qualified_top_2"

    third_place = standings.loc[standings["group_rank"] == 3].copy()
    third_place = third_place.sort_values(
        ["points", "goal_difference", "goals_for", "wins", "team_name"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    best_third = third_place.head(EXPECTED_THIRD_PLACE_QUALIFIERS).copy()
    best_third["qualification_status"] = "qualified_best_third"

    qualifiers = pd.concat([direct, best_third], ignore_index=True)
    qualifiers["qualified"] = True
    return qualifiers


def simulate_all_groups(
    fixtures: pd.DataFrame | None = None,
    teams_df: pd.DataFrame | None = None,
    ratings_df: pd.DataFrame | None = None,
    rating_col: str | None = None,
    base_total_goals: float | None = None,
    seed: int = 2026,
    model_config: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate the full group stage and return standings, matches, qualifiers."""
    model_config = load_model_config() if model_config is None else model_config
    fixtures = load_group_stage_fixtures() if fixtures is None else fixtures.copy()
    teams_df = pd.read_csv(TEAMS_PATH) if teams_df is None else teams_df.copy()
    if ratings_df is None or rating_col is None:
        ratings_df, rating_col = load_default_ratings(model_config)
    if base_total_goals is None:
        base_total_goals = load_base_total_goals(model_config=model_config)
    model_kwargs = {
        **poisson_parameter_kwargs(model_config),
        **draw_calibration_kwargs(model_config),
    }
    fixture_predictions = precompute_fixture_predictions(
        fixtures=fixtures,
        ratings_df=ratings_df,
        rating_col=rating_col,
        base_total_goals=base_total_goals,
        model_kwargs=model_kwargs,
        model_config=model_config,
    )

    run_sanity_checks(fixtures=fixtures, phase="fixtures")

    rng = random.Random(seed)
    standings_tables = []
    simulated_matches = []
    for group_letter in sorted(fixtures["group_letter"].unique()):
        standings, matches = simulate_one_group_from_predictions(
            group_letter=group_letter,
            fixtures=fixtures,
            teams_df=teams_df,
            fixture_predictions=fixture_predictions,
            rng=rng,
        )
        standings_tables.append(standings)
        simulated_matches.append(matches)

    all_standings = pd.concat(standings_tables, ignore_index=True)
    all_matches = pd.concat(simulated_matches, ignore_index=True)
    qualifiers = select_group_stage_qualifiers(all_standings)
    run_sanity_checks(
        fixtures=fixtures,
        standings=all_standings,
        qualifiers=qualifiers,
        phase="complete",
    )
    return all_standings, all_matches, qualifiers


def simulate_all_groups_from_predictions(
    fixtures: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixture_predictions: Mapping[int, Mapping[str, object]],
    rng: random.Random,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate the full group stage using precomputed match probabilities."""
    run_sanity_checks(fixtures=fixtures, phase="fixtures")

    standings_tables = []
    simulated_matches = []
    for group_letter in sorted(fixtures["group_letter"].unique()):
        standings, matches = simulate_one_group_from_predictions(
            group_letter=group_letter,
            fixtures=fixtures,
            teams_df=teams_df,
            fixture_predictions=fixture_predictions,
            rng=rng,
        )
        standings_tables.append(standings)
        simulated_matches.append(matches)

    all_standings = pd.concat(standings_tables, ignore_index=True)
    all_matches = pd.concat(simulated_matches, ignore_index=True)
    qualifiers = select_group_stage_qualifiers(all_standings)
    run_sanity_checks(
        fixtures=fixtures,
        standings=all_standings,
        qualifiers=qualifiers,
        phase="complete",
    )
    return all_standings, all_matches, qualifiers


def run_sanity_checks(
    fixtures: pd.DataFrame,
    standings: pd.DataFrame | None = None,
    qualifiers: pd.DataFrame | None = None,
    phase: str = "complete",
) -> None:
    """Assert the group-stage structure and qualification counts."""
    group_count = fixtures["group_letter"].nunique()
    match_count = len(fixtures)
    if group_count != EXPECTED_GROUP_COUNT:
        raise AssertionError(f"Expected 12 groups, found {group_count}")
    if match_count != EXPECTED_GROUP_MATCH_COUNT:
        raise AssertionError(f"Expected 72 group-stage matches, found {match_count}")
    if phase == "fixtures":
        return

    if standings is None or qualifiers is None:
        raise AssertionError("Standings and qualifiers are required for complete checks")

    direct_count = int((qualifiers["qualification_status"] == "qualified_top_2").sum())
    third_count = int((qualifiers["qualification_status"] == "qualified_best_third").sum())
    total_count = len(qualifiers)

    if direct_count != EXPECTED_DIRECT_QUALIFIERS:
        raise AssertionError(f"Expected 24 direct qualifiers, found {direct_count}")
    if third_count != EXPECTED_THIRD_PLACE_QUALIFIERS:
        raise AssertionError(f"Expected 8 best third-place qualifiers, found {third_count}")
    if total_count != EXPECTED_TOTAL_QUALIFIERS:
        raise AssertionError(f"Expected 32 total qualified teams, found {total_count}")
    if standings["team_name"].nunique() != 48:
        raise AssertionError("Expected standings to contain all 48 teams exactly once")


def run_monte_carlo_sanity_checks(results: pd.DataFrame, n_simulations: int) -> None:
    """Validate aggregate team counts from a Monte Carlo group-stage run."""
    if len(results) != 48:
        raise AssertionError(f"Expected Monte Carlo output for 48 teams, found {len(results)}")

    rank_columns = {
        1: "group_rank_1_count",
        2: "group_rank_2_count",
        3: "group_rank_3_count",
        4: "group_rank_4_count",
    }
    for rank, column in rank_columns.items():
        expected = EXPECTED_GROUP_COUNT * n_simulations
        observed = int(results[column].sum())
        if observed != expected:
            raise AssertionError(
                f"Expected {expected} total rank-{rank} finishes, found {observed}"
            )

    expected_direct = EXPECTED_DIRECT_QUALIFIERS * n_simulations
    observed_direct = int(results["qualified_top_2_count"].sum())
    if observed_direct != expected_direct:
        raise AssertionError(
            f"Expected {expected_direct} direct qualifier finishes, found {observed_direct}"
        )

    expected_third = EXPECTED_THIRD_PLACE_QUALIFIERS * n_simulations
    observed_third = int(results["qualified_best_third_count"].sum())
    if observed_third != expected_third:
        raise AssertionError(
            f"Expected {expected_third} best-third qualifier finishes, found {observed_third}"
        )

    expected_total = EXPECTED_TOTAL_QUALIFIERS * n_simulations
    observed_total = int(results["qualified_count"].sum())
    if observed_total != expected_total:
        raise AssertionError(
            f"Expected {expected_total} total qualifier finishes, found {observed_total}"
        )

    rank_total = results[
        [
            "group_rank_1_count",
            "group_rank_2_count",
            "group_rank_3_count",
            "group_rank_4_count",
        ]
    ].sum(axis=1)
    if not (rank_total == n_simulations).all():
        raise AssertionError("Every team should finish exactly once in each simulation")


def initialize_monte_carlo_counts(teams_df: pd.DataFrame) -> dict[str, dict]:
    """Create per-team counters for rank and qualification probabilities."""
    counts = {}
    for _, team in teams_df.sort_values(["group_letter", "team_name"]).iterrows():
        counts[team["team_name"]] = {
            "team_name": team["team_name"],
            "fifa_code": team["fifa_code"],
            "group_letter": team["group_letter"],
            "simulations": 0,
            "group_rank_1_count": 0,
            "group_rank_2_count": 0,
            "group_rank_3_count": 0,
            "group_rank_4_count": 0,
            "qualified_top_2_count": 0,
            "qualified_best_third_count": 0,
            "qualified_count": 0,
            "points_total": 0,
            "goal_difference_total": 0,
        }
    return counts


def update_monte_carlo_counts(
    counts: dict[str, dict],
    standings: pd.DataFrame,
    qualifiers: pd.DataFrame,
) -> None:
    """Add one simulation's standings and qualifier outcomes to team counters."""
    qualifier_labels = qualifiers.set_index("team_name")["qualification_status"].to_dict()
    for _, row in standings.iterrows():
        team_name = row["team_name"]
        team_counts = counts[team_name]
        rank = int(row["group_rank"])
        team_counts["simulations"] += 1
        team_counts[f"group_rank_{rank}_count"] += 1
        team_counts["points_total"] += int(row["points"])
        team_counts["goal_difference_total"] += int(row["goal_difference"])

        qualification_status = qualifier_labels.get(team_name)
        if qualification_status == "qualified_top_2":
            team_counts["qualified_top_2_count"] += 1
            team_counts["qualified_count"] += 1
        elif qualification_status == "qualified_best_third":
            team_counts["qualified_best_third_count"] += 1
            team_counts["qualified_count"] += 1


def build_monte_carlo_results(counts: Mapping[str, Mapping[str, object]]) -> pd.DataFrame:
    """Convert Monte Carlo counts into team-level probabilities."""
    rows = []
    for team_counts in counts.values():
        simulations = int(team_counts["simulations"])
        row = dict(team_counts)
        row["group_rank_1_probability"] = row["group_rank_1_count"] / simulations
        row["group_rank_2_probability"] = row["group_rank_2_count"] / simulations
        row["group_rank_3_probability"] = row["group_rank_3_count"] / simulations
        row["group_rank_4_probability"] = row["group_rank_4_count"] / simulations
        row["qualified_top_2_probability"] = row["qualified_top_2_count"] / simulations
        row["qualified_best_third_probability"] = row["qualified_best_third_count"] / simulations
        row["qualification_probability"] = row["qualified_count"] / simulations
        row["average_points"] = row["points_total"] / simulations
        row["average_goal_difference"] = row["goal_difference_total"] / simulations
        rows.append(row)

    columns = [
        "team_name",
        "fifa_code",
        "group_letter",
        "simulations",
        "group_rank_1_probability",
        "group_rank_2_probability",
        "group_rank_3_probability",
        "group_rank_4_probability",
        "qualified_top_2_probability",
        "qualified_best_third_probability",
        "qualification_probability",
        "average_points",
        "average_goal_difference",
        "group_rank_1_count",
        "group_rank_2_count",
        "group_rank_3_count",
        "group_rank_4_count",
        "qualified_top_2_count",
        "qualified_best_third_count",
        "qualified_count",
    ]
    results = pd.DataFrame(rows)[columns]
    return results.sort_values(
        ["group_letter", "qualification_probability", "group_rank_1_probability", "team_name"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def _simulate_group_records(
    group_fixtures: list[dict],
    group_teams: list[dict],
    fixture_predictions: Mapping[int, Mapping[str, object]],
    rng: random.Random,
) -> list[dict]:
    """Fast group simulation for Monte Carlo runs without per-run DataFrames."""
    standings = _empty_standings(pd.DataFrame(group_teams))
    for fixture in group_fixtures:
        prediction = fixture_predictions[int(fixture["match_number"])]
        home_goals, away_goals = sample_scoreline(
            prediction["scoreline_probabilities"],
            rng=rng,
        )
        _apply_result(standings, fixture["home_team"], home_goals, away_goals)
        _apply_result(standings, fixture["away_team"], away_goals, home_goals)

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


def _update_monte_carlo_counts_from_records(
    counts: dict[str, dict],
    ranked_groups: list[list[dict]],
) -> None:
    """Update Monte Carlo counters directly from lightweight ranked group rows."""
    third_place_rows = []
    for ranked_group in ranked_groups:
        for row in ranked_group:
            team_counts = counts[row["team_name"]]
            rank = int(row["group_rank"])
            team_counts["simulations"] += 1
            team_counts[f"group_rank_{rank}_count"] += 1
            team_counts["points_total"] += int(row["points"])
            team_counts["goal_difference_total"] += int(row["goal_difference"])

            if rank <= 2:
                team_counts["qualified_top_2_count"] += 1
                team_counts["qualified_count"] += 1
            elif rank == 3:
                third_place_rows.append(row)

    best_third = sorted(
        third_place_rows,
        key=lambda row: (
            -row["points"],
            -row["goal_difference"],
            -row["goals_for"],
            -row["wins"],
            row["team_name"],
        ),
    )[:EXPECTED_THIRD_PLACE_QUALIFIERS]
    for row in best_third:
        team_counts = counts[row["team_name"]]
        team_counts["qualified_best_third_count"] += 1
        team_counts["qualified_count"] += 1


def run_group_stage_monte_carlo(
    n_simulations: int = DEFAULT_MONTE_CARLO_SIMULATIONS,
    seed: int = 2026,
    output_path: Path = MONTE_CARLO_OUTPUT_PATH,
    model_config: dict | None = None,
) -> pd.DataFrame:
    """Run repeated group-stage simulations and save team-level probabilities."""
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive")

    model_config = load_model_config() if model_config is None else model_config
    fixtures = load_group_stage_fixtures()
    teams_df = pd.read_csv(TEAMS_PATH)
    ratings_df, rating_col = load_default_ratings(model_config)
    base_total_goals = load_base_total_goals(model_config=model_config)
    model_kwargs = {
        **poisson_parameter_kwargs(model_config),
        **draw_calibration_kwargs(model_config),
    }
    fixture_predictions = precompute_fixture_predictions(
        fixtures=fixtures,
        ratings_df=ratings_df,
        rating_col=rating_col,
        base_total_goals=base_total_goals,
        model_kwargs=model_kwargs,
        model_config=model_config,
    )
    grouped_fixtures = {
        group_letter: group.sort_values("match_number").to_dict("records")
        for group_letter, group in fixtures.groupby("group_letter")
    }
    grouped_teams = {
        group_letter: group.to_dict("records")
        for group_letter, group in teams_df.groupby("group_letter")
    }

    counts = initialize_monte_carlo_counts(teams_df)
    rng = random.Random(seed)
    for _ in range(n_simulations):
        ranked_groups = [
            _simulate_group_records(
                group_fixtures=grouped_fixtures[group_letter],
                group_teams=grouped_teams[group_letter],
                fixture_predictions=fixture_predictions,
                rng=rng,
            )
            for group_letter in sorted(grouped_fixtures)
        ]
        _update_monte_carlo_counts_from_records(counts, ranked_groups=ranked_groups)

    results = build_monte_carlo_results(counts)
    for key, value in metadata_columns(model_config).items():
        results[key] = value
    run_monte_carlo_sanity_checks(results=results, n_simulations=n_simulations)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return results


def build_sample_output(
    standings: pd.DataFrame,
    qualifiers: pd.DataFrame,
) -> pd.DataFrame:
    """Create one CSV-friendly group standings output with qualification labels."""
    output = standings.copy()
    qualifier_labels = qualifiers.set_index("team_name")["qualification_status"].to_dict()
    output["qualified"] = output["team_name"].isin(qualifier_labels)
    output["qualification_status"] = output["team_name"].map(qualifier_labels).fillna("eliminated")
    return output.sort_values(["group_letter", "group_rank"]).reset_index(drop=True)


def save_sample_simulation(
    output_path: Path = SAMPLE_OUTPUT_PATH,
    seed: int = 2026,
    model_config: dict | None = None,
) -> pd.DataFrame:
    """Run one deterministic sample simulation and save group standings."""
    model_config = load_model_config() if model_config is None else model_config
    standings, _, qualifiers = simulate_all_groups(seed=seed, model_config=model_config)
    sample_output = build_sample_output(standings=standings, qualifiers=qualifiers)
    for key, value in metadata_columns(model_config).items():
        sample_output[key] = value
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_output.to_csv(output_path, index=False)
    return sample_output


def print_sanity_report(fixtures: pd.DataFrame, qualifiers: pd.DataFrame) -> None:
    """Print pass/fail-style counts for the group-stage simulation."""
    direct_count = int((qualifiers["qualification_status"] == "qualified_top_2").sum())
    third_count = int((qualifiers["qualification_status"] == "qualified_best_third").sum())
    checks = {
        "12 groups": fixtures["group_letter"].nunique() == EXPECTED_GROUP_COUNT,
        "72 group-stage matches": len(fixtures) == EXPECTED_GROUP_MATCH_COUNT,
        "24 direct qualifiers": direct_count == EXPECTED_DIRECT_QUALIFIERS,
        "8 best third-place qualifiers": third_count == EXPECTED_THIRD_PLACE_QUALIFIERS,
        "32 total qualified teams": len(qualifiers) == EXPECTED_TOTAL_QUALIFIERS,
    }

    print("\nSanity checks")
    for label, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} - {label}")


def print_monte_carlo_sanity_report(results: pd.DataFrame, n_simulations: int) -> None:
    """Print aggregate pass/fail checks for the Monte Carlo output."""
    checks = {
        "48 teams in Monte Carlo output": len(results) == 48,
        "rank 1 count per simulation": int(results["group_rank_1_count"].sum())
        == EXPECTED_GROUP_COUNT * n_simulations,
        "rank 2 count per simulation": int(results["group_rank_2_count"].sum())
        == EXPECTED_GROUP_COUNT * n_simulations,
        "rank 3 count per simulation": int(results["group_rank_3_count"].sum())
        == EXPECTED_GROUP_COUNT * n_simulations,
        "rank 4 count per simulation": int(results["group_rank_4_count"].sum())
        == EXPECTED_GROUP_COUNT * n_simulations,
        "24 direct qualifiers per simulation": int(results["qualified_top_2_count"].sum())
        == EXPECTED_DIRECT_QUALIFIERS * n_simulations,
        "8 best third-place qualifiers per simulation": int(results["qualified_best_third_count"].sum())
        == EXPECTED_THIRD_PLACE_QUALIFIERS * n_simulations,
        "32 total qualifiers per simulation": int(results["qualified_count"].sum())
        == EXPECTED_TOTAL_QUALIFIERS * n_simulations,
    }

    print("\nMonte Carlo sanity checks")
    for label, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} - {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate the 2026 World Cup group stage.")
    parser.add_argument(
        "--mode",
        default="default",
        choices=["default", "v2", "experimental", "test"],
        help="Model parameter mode.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=DEFAULT_MONTE_CARLO_SIMULATIONS,
        help="Number of Monte Carlo group-stage simulations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SAMPLE_OUTPUT_PATH,
        help="CSV path for the sample group-stage standings output.",
    )
    parser.add_argument(
        "--monte-carlo-output",
        type=Path,
        default=MONTE_CARLO_OUTPUT_PATH,
        help="CSV path for Monte Carlo team-level probability output.",
    )
    args = parser.parse_args()
    model_config = load_model_config(args.mode)
    if args.output == SAMPLE_OUTPUT_PATH:
        args.output = simulation_output_path("group_stage_sample_simulation", model_config)
    if args.monte_carlo_output == MONTE_CARLO_OUTPUT_PATH:
        args.monte_carlo_output = simulation_output_path("group_stage_monte_results", model_config)

    fixtures = load_group_stage_fixtures()
    standings, matches, qualifiers = simulate_all_groups(
        fixtures=fixtures,
        seed=args.seed,
        model_config=model_config,
    )
    sample_output = build_sample_output(standings=standings, qualifiers=qualifiers)
    for key, value in metadata_columns(model_config).items():
        sample_output[key] = value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample_output.to_csv(args.output, index=False)

    print(f"Simulated {len(matches)} group-stage matches with seed {args.seed}.")
    print(f"Saved sample output to {args.output}")
    print_sanity_report(fixtures=fixtures, qualifiers=qualifiers)

    monte_carlo_results = run_group_stage_monte_carlo(
        n_simulations=args.n_simulations,
        seed=args.seed,
        output_path=args.monte_carlo_output,
        model_config=model_config,
    )
    print(
        f"\nRan {args.n_simulations} Monte Carlo group-stage simulations "
        f"with seed {args.seed}."
    )
    print(f"Saved Monte Carlo output to {args.monte_carlo_output}")
    print_monte_carlo_sanity_report(
        results=monte_carlo_results,
        n_simulations=args.n_simulations,
    )


if __name__ == "__main__":
    main()
