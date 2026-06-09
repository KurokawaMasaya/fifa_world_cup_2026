from __future__ import annotations

import argparse
import inspect
import math
import random
import re
import sys
from pathlib import Path
from typing import Mapping

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.poisson_match_model import predict_from_strength
from src.config.model_config import (
    draw_calibration_kwargs,
    load_model_config,
    metadata_columns,
    output_path,
    poisson_parameter_kwargs,
)
from src.simulation.group_stage_simulator import (
    DEFAULT_MONTE_CARLO_SIMULATIONS,
    EXPECTED_DIRECT_QUALIFIERS,
    EXPECTED_GROUP_COUNT,
    EXPECTED_GROUP_MATCH_COUNT,
    EXPECTED_THIRD_PLACE_QUALIFIERS,
    EXPECTED_TOTAL_QUALIFIERS,
    PROCESSED_DATA_DIR,
    TEAMS_PATH,
    _simulate_group_records,
    build_monte_carlo_results,
    initialize_monte_carlo_counts,
    load_base_total_goals,
    load_default_ratings,
    load_group_stage_fixtures,
    precompute_fixture_predictions,
    run_monte_carlo_sanity_checks,
    run_sanity_checks,
    sample_scoreline,
)


TOURNAMENT_OUTPUT_PATH = PROCESSED_DATA_DIR / "tournament_simulation_results_default.csv"
TEAM_STRENGTH_DIAGNOSTICS_PATH = PROCESSED_DATA_DIR / "team_strength_default.csv"
HEAD_TO_HEAD_DIAGNOSTICS_PATH = PROCESSED_DATA_DIR / "diagnostics_head_to_head_default.csv"
PATH_DIFFICULTY_DIAGNOSTICS_PATH = PROCESSED_DATA_DIR / "diagnostics_path_difficulty_default.csv"
SAMPLE_KNOCKOUT_BRACKET_PATH = PROCESSED_DATA_DIR / "sample_knockout_bracket.csv"
WEIGHTED_ELO_PATH = PROCESSED_DATA_DIR / "team_ratings_weighted_elo.csv"
FIFA_ELO_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "qualifier_elo_rating.csv"
KNOCKOUT_SCHEDULE_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "matches.csv"
BRACKET_MODE = "fixed_approximation"
BRACKET_SOURCE = "matches.csv"
USES_RANDOM_PAIRING = False
OFFICIAL_BRACKET = False
RANDOM_SEED_USED_FOR = ["scoreline_sampling", "tiebreakers"]
BRACKET_MAPPING_NOTE = (
    "Uses fixed_approximation bracket slots from data/raw/matches.csv; "
    "best-third placement is approximated, not official FIFA bracket mapping."
)
DIAGNOSTIC_TEAMS = [
    "Spain",
    "Argentina",
    "France",
    "England",
    "Brazil",
    "Portugal",
    "Germany",
    "Netherlands",
    "Morocco",
    "Senegal",
    "Japan",
    "Curacao",
]
HEAD_TO_HEAD_MATCHUPS = [
    ("France", "Morocco"),
    ("France", "Brazil"),
    ("France", "Senegal"),
    ("Morocco", "Brazil"),
    ("Morocco", "Spain"),
    ("Brazil", "Spain"),
    ("Germany", "Curacao"),
]
KNOCKOUT_STAGE_COUNTS = {
    "r32_count": 32,
    "r16_count": 16,
    "qf_count": 8,
    "sf_count": 4,
    "final_count": 2,
    "champion_count": 1,
}
KNOCKOUT_ROUNDS = [
    ("R32", "r16_count"),
    ("R16", "qf_count"),
    ("QF", "sf_count"),
    ("SF", "final_count"),
    ("Final", "champion_count"),
]
ROUND_OPPONENT_COLUMNS = {
    "R32": "avg_r32_opponent_strength",
    "R16": "avg_r16_opponent_strength",
    "QF": "avg_qf_opponent_strength",
    "SF": "avg_sf_opponent_strength",
    "Final": "avg_final_opponent_strength",
}


def elo_tiebreak_probability(rating_a: float, rating_b: float) -> float:
    """Return team A's chance to advance when a knockout scoreline is drawn."""
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def precompute_knockout_predictions(
    teams_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    rating_col: str,
    base_total_goals: float,
    model_kwargs: Mapping[str, object] | None = None,
) -> tuple[dict[tuple[str, str], dict], dict[str, float]]:
    """Precompute neutral-site Poisson grids for every possible team pairing."""
    ratings = ratings_df.set_index("team_name")[rating_col].astype(float).to_dict()
    team_names = teams_df["team_name"].tolist()
    predictions = {}
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    model_kwargs.setdefault("base_total_goals", base_total_goals)

    for team_a in team_names:
        for team_b in team_names:
            if team_a == team_b:
                continue
            prediction = predict_from_strength(
                team_a=team_a,
                team_b=team_b,
                strength_a=ratings[team_a],
                strength_b=ratings[team_b],
                **model_kwargs,
            )
            predictions[(team_a, team_b)] = {
                "scoreline_probabilities": prediction["scoreline_probabilities"],
                "lambda_a": prediction["lambda_a"],
                "lambda_b": prediction["lambda_b"],
            }

    return predictions, ratings


def _third_place_sort_key(row: Mapping[str, object]) -> tuple:
    return (
        -int(row["points"]),
        -int(row["goal_difference"]),
        -int(row["goals_for"]),
        -int(row["wins"]),
        str(row["team_name"]),
    )


def select_qualifier_records(ranked_groups: list[list[dict]]) -> tuple[list[dict], list[dict], list[dict]]:
    """Select group winners, runners-up, and best third-place teams."""
    winners = []
    runners_up = []
    third_place_rows = []
    for ranked_group in ranked_groups:
        winners.extend(row for row in ranked_group if int(row["group_rank"]) == 1)
        runners_up.extend(row for row in ranked_group if int(row["group_rank"]) == 2)
        third_place_rows.extend(row for row in ranked_group if int(row["group_rank"]) == 3)

    winners = sorted(winners, key=lambda row: str(row["group_letter"]))
    runners_up = sorted(runners_up, key=lambda row: str(row["group_letter"]))
    best_third = sorted(third_place_rows, key=_third_place_sort_key)[
        :EXPECTED_THIRD_PLACE_QUALIFIERS
    ]
    return winners, runners_up, best_third


def _split_match_label(match_label: str) -> tuple[str, str]:
    left, right = re.split(r"\s+vs\s+", match_label.strip())
    return left, right


def load_knockout_schedule(schedule_path: Path = KNOCKOUT_SCHEDULE_PATH) -> pd.DataFrame:
    """Load fixed knockout bracket slots from data/raw/matches.csv.

    The raw schedule currently contains an impossible self-reference for match
    100: ``W95 vs W100``. The bracket graph cannot be simulated with a match
    depending on itself, so fixed_approximation repairs that source to
    ``W95 vs W96`` while preserving the original label for diagnostics.
    """
    schedule = pd.read_csv(schedule_path)
    schedule = schedule.loc[schedule["stage_id"].between(2, 7)].copy()
    schedule["original_match_label"] = schedule["match_label"]
    self_reference = schedule["match_label"].eq("W95 vs W100")
    schedule.loc[self_reference, "match_label"] = "W95 vs W96"
    return schedule[["match_number", "stage_id", "original_match_label", "match_label"]].sort_values(
        "match_number"
    )


def _finish_key(row: Mapping[str, object]) -> str:
    return f"{int(row['group_rank'])}{row['group_letter']}"


def build_finish_lookup(
    winners: list[dict],
    runners_up: list[dict],
    best_third: list[dict],
) -> dict[str, dict]:
    """Map fixed bracket finish slots such as 1A, 2B, and 3C to teams."""
    if len(winners) != EXPECTED_GROUP_COUNT:
        raise AssertionError(f"Expected 12 group winners, found {len(winners)}")
    if len(runners_up) != EXPECTED_DIRECT_QUALIFIERS - EXPECTED_GROUP_COUNT:
        raise AssertionError(f"Expected 12 runners-up, found {len(runners_up)}")
    if len(best_third) != EXPECTED_THIRD_PLACE_QUALIFIERS:
        raise AssertionError(f"Expected 8 best third-place teams, found {len(best_third)}")

    lookup = {}
    for row in winners + runners_up + best_third:
        lookup[_finish_key(row)] = row
    return lookup


def resolve_r32_slot(
    slot: str,
    finish_lookup: Mapping[str, dict],
    third_place_assignments: Mapping[str, str],
) -> dict:
    """Resolve one R32 slot from a group finish code.

    Slots like ``1A`` and ``2B`` are exact. Slots like ``3ABCDF`` are
    best-third placeholders; because the official 2026 third-place allocation
    table is not available in the project data, fixed_approximation chooses the
    first eligible best-third team by the slot's listed group order.
    """
    if re.fullmatch(r"[12][A-L]", slot):
        return finish_lookup[slot]

    match = re.fullmatch(r"3([A-L]+)", slot)
    if not match:
        raise ValueError(f"Unsupported R32 slot: {slot}")
    group_letter = third_place_assignments[slot]
    return finish_lookup[f"3{group_letter}"]


def assign_third_place_slots(third_place_slots: list[str], best_third: list[dict]) -> dict[str, str]:
    """Deterministically assign best-third teams to fixed R32 placeholder slots.

    This is an approximation because the official FIFA table that maps the
    exact combination of qualified third-place groups to bracket slots is not
    present in the data. The allocator still respects each slot's allowed group
    letters and uses every selected best-third group at most once.
    """
    selected_groups = {row["group_letter"] for row in best_third}
    allowed_by_slot = {
        slot: [group for group in re.fullmatch(r"3([A-L]+)", slot).group(1) if group in selected_groups]
        for slot in third_place_slots
    }
    if any(not groups for groups in allowed_by_slot.values()):
        raise NotImplementedError(
            "Third-place bracket assignment is not implemented for this combination"
        )

    ordered_slots = sorted(third_place_slots, key=lambda slot: (len(allowed_by_slot[slot]), slot))
    assignments: dict[str, str] = {}
    used_groups: set[str] = set()

    def backtrack(index: int) -> bool:
        if index == len(ordered_slots):
            return True
        slot = ordered_slots[index]
        for group in allowed_by_slot[slot]:
            if group in used_groups:
                continue
            assignments[slot] = group
            used_groups.add(group)
            if backtrack(index + 1):
                return True
            used_groups.remove(group)
            del assignments[slot]
        return False

    if not backtrack(0):
        raise NotImplementedError(
            "Third-place bracket assignment is not implemented for this combination"
        )
    return assignments


def build_fixed_r32_bracket(
    winners: list[dict],
    runners_up: list[dict],
    best_third: list[dict],
    knockout_schedule: pd.DataFrame | None = None,
    bracket_mode: str = BRACKET_MODE,
) -> list[dict]:
    """Build R32 matches from fixed schedule slots, never random ordering."""
    if bracket_mode != BRACKET_MODE:
        raise ValueError("Only bracket_mode='fixed_approximation' is implemented")

    schedule = load_knockout_schedule() if knockout_schedule is None else knockout_schedule
    r32_schedule = schedule.loc[schedule["stage_id"] == 2].copy()
    finish_lookup = build_finish_lookup(winners, runners_up, best_third)
    third_place_slots = []
    for label in r32_schedule["match_label"]:
        third_place_slots.extend(
            slot for slot in _split_match_label(label) if slot.startswith("3")
        )
    third_place_assignments = assign_third_place_slots(third_place_slots, best_third)
    bracket = []

    for _, match in r32_schedule.sort_values("match_number").iterrows():
        slot_a, slot_b = _split_match_label(match["match_label"])
        team_a = resolve_r32_slot(slot_a, finish_lookup, third_place_assignments)
        team_b = resolve_r32_slot(slot_b, finish_lookup, third_place_assignments)
        bracket.append(
            {
                "match_number": int(match["match_number"]),
                "stage_id": int(match["stage_id"]),
                "round": "R32",
                "slot_a": slot_a,
                "slot_b": slot_b,
                "team_a": team_a["team_name"],
                "team_b": team_b["team_name"],
                "team_a_group": team_a["group_letter"],
                "team_b_group": team_b["group_letter"],
                "team_a_group_rank": int(team_a["group_rank"]),
                "team_b_group_rank": int(team_b["group_rank"]),
                "original_match_label": match["original_match_label"],
                "resolved_match_label": match["match_label"],
                "bracket_mode": bracket_mode,
                "bracket_source": BRACKET_SOURCE,
                "uses_random_pairing": USES_RANDOM_PAIRING,
                "official_bracket": OFFICIAL_BRACKET,
                "random_seed_used_for": RANDOM_SEED_USED_FOR,
            }
        )

    validate_round_pairings("R32", [(row["team_a"], row["team_b"]) for row in bracket])
    return bracket


def simulate_knockout_match(
    team_a: str,
    team_b: str,
    knockout_predictions: Mapping[tuple[str, str], Mapping[str, object]],
    rating_lookup: Mapping[str, float],
    rng: random.Random,
) -> tuple[str, str]:
    """Sample a knockout match and return winner and loser."""
    prediction = knockout_predictions[(team_a, team_b)]
    goals_a, goals_b = sample_scoreline(prediction["scoreline_probabilities"], rng=rng)

    if goals_a > goals_b:
        return team_a, team_b
    if goals_b > goals_a:
        return team_b, team_a

    team_a_advances_probability = elo_tiebreak_probability(
        rating_lookup[team_a],
        rating_lookup[team_b],
    )
    if rng.random() < team_a_advances_probability:
        return team_a, team_b
    return team_b, team_a


def simulate_knockout_round(
    pairings: list[tuple[str, str]],
    knockout_predictions: Mapping[tuple[str, str], Mapping[str, object]],
    rating_lookup: Mapping[str, float],
    rng: random.Random,
) -> list[str]:
    """Simulate one knockout round and return winners in bracket order."""
    return [
        simulate_knockout_match(
            team_a=team_a,
            team_b=team_b,
            knockout_predictions=knockout_predictions,
            rating_lookup=rating_lookup,
            rng=rng,
        )[0]
        for team_a, team_b in pairings
    ]


def validate_round_pairings(round_name: str, pairings: list[tuple[str, str]]) -> None:
    """Validate match and team counts for one knockout round."""
    expected_matches = {
        "R32": 16,
        "R16": 8,
        "QF": 4,
        "SF": 2,
        "Final": 1,
    }
    expected_match_count = expected_matches[round_name]
    if len(pairings) != expected_match_count:
        raise AssertionError(
            f"{round_name} should have {expected_match_count} matches, found {len(pairings)}"
        )
    teams = [team for pairing in pairings for team in pairing]
    if len(teams) != expected_match_count * 2:
        raise AssertionError(f"{round_name} has an invalid team count")
    if len(set(teams)) != len(teams):
        raise AssertionError(f"A team appears twice in {round_name}")


def validate_knockout_stages(stages: Mapping[str, list[str]]) -> None:
    """Validate team counts after every knockout stage."""
    expected_counts = {
        "r32_count": 32,
        "r16_count": 16,
        "qf_count": 8,
        "sf_count": 4,
        "final_count": 2,
        "champion_count": 1,
    }
    for column, expected_count in expected_counts.items():
        observed = len(stages[column])
        if observed != expected_count:
            raise AssertionError(f"{column} should contain {expected_count} teams, found {observed}")
    if len(set(stages["r32_count"])) != 32:
        raise AssertionError("No team may appear twice in R32")


def _looks_like_placeholder(value: str) -> bool:
    return bool(re.fullmatch(r"[123][A-L]+|W\d+|RU\d+", value))


def resolve_knockout_placeholder(
    placeholder: str,
    winners: Mapping[int, str],
    losers: Mapping[int, str],
) -> str:
    """Resolve W/RU placeholders from completed knockout matches."""
    if placeholder.startswith("W"):
        match_number = int(placeholder.removeprefix("W"))
        if match_number not in winners:
            raise ValueError(
                f"{placeholder} was requested before Match {match_number} was simulated"
            )
        return winners[match_number]

    if placeholder.startswith("RU"):
        match_number = int(placeholder.removeprefix("RU"))
        if match_number not in losers:
            raise ValueError(
                f"{placeholder} was requested before Match {match_number} was simulated"
            )
        return losers[match_number]

    raise ValueError(f"Unsupported knockout placeholder: {placeholder}")


def resolve_knockout_match_slots(
    match_label: str,
    winners: Mapping[int, str],
    losers: Mapping[int, str],
) -> tuple[str, str, str, str]:
    """Resolve both team slots for a post-R32 knockout match."""
    slot_a, slot_b = _split_match_label(match_label)
    return (
        resolve_knockout_placeholder(slot_a, winners=winners, losers=losers),
        resolve_knockout_placeholder(slot_b, winners=winners, losers=losers),
        slot_a,
        slot_b,
    )


def update_path_difficulty_counts(
    path_counts: dict[str, dict],
    pairings: list[tuple[str, str]],
    round_name: str,
    rating_lookup: Mapping[str, float],
) -> None:
    """Record opponent strength faced by each team in a knockout round."""
    total_key = f"{round_name.lower()}_opponent_strength_total"
    count_key = f"{round_name.lower()}_opponent_count"
    for team_a, team_b in pairings:
        path_counts[team_a][total_key] += rating_lookup[team_b]
        path_counts[team_a][count_key] += 1
        path_counts[team_b][total_key] += rating_lookup[team_a]
        path_counts[team_b][count_key] += 1


def pair_adjacent(teams: list[str]) -> list[tuple[str, str]]:
    """Pair teams in bracket order for the next knockout round."""
    if len(teams) % 2 != 0:
        raise AssertionError(f"Cannot pair odd team count: {len(teams)}")
    return [(teams[index], teams[index + 1]) for index in range(0, len(teams), 2)]


def simulate_knockout_bracket(
    r32_bracket: list[dict],
    knockout_predictions: Mapping[tuple[str, str], Mapping[str, object]],
    rating_lookup: Mapping[str, float],
    rng: random.Random,
    path_counts: dict[str, dict] | None = None,
) -> tuple[dict[str, list[str]], list[dict]]:
    """Simulate R32 through Final using fixed match-number bracket sources."""
    schedule = load_knockout_schedule()
    match_winners: dict[int, str] = {}
    match_losers: dict[int, str] = {}
    match_rows = []
    r32_by_match = {row["match_number"]: row for row in r32_bracket}
    stage_reachers = {"r32_count": []}

    for _, schedule_match in schedule.sort_values("match_number").iterrows():
        stage_id = int(schedule_match["stage_id"])
        match_number = int(schedule_match["match_number"])
        round_name = {
            2: "R32",
            3: "R16",
            4: "QF",
            5: "SF",
            6: "Third Place",
            7: "Final",
        }[stage_id]

        if stage_id == 2:
            r32_row = r32_by_match[match_number]
            team_a = r32_row["team_a"]
            team_b = r32_row["team_b"]
            slot_a = r32_row["slot_a"]
            slot_b = r32_row["slot_b"]
            original_label = r32_row["original_match_label"]
            resolved_label = r32_row["resolved_match_label"]
        else:
            team_a, team_b, slot_a, slot_b = resolve_knockout_match_slots(
                schedule_match["match_label"],
                winners=match_winners,
                losers=match_losers,
            )
            original_label = schedule_match["original_match_label"]
            resolved_label = schedule_match["match_label"]

        winner, loser = simulate_knockout_match(
            team_a=team_a,
            team_b=team_b,
            knockout_predictions=knockout_predictions,
            rating_lookup=rating_lookup,
            rng=rng,
        )
        match_winners[match_number] = winner
        match_losers[match_number] = loser
        if stage_id == 2:
            stage_reachers["r32_count"].extend([team_a, team_b])
        if path_counts is not None and round_name in ROUND_OPPONENT_COLUMNS:
            update_path_difficulty_counts(
                path_counts,
                [(team_a, team_b)],
                round_name,
                rating_lookup,
            )
        match_rows.append(
            {
                "match_number": match_number,
                "round": round_name,
                "slot_a": slot_a,
                "slot_b": slot_b,
                "team_a": team_a,
                "team_b": team_b,
                "winner": winner,
                "original_match_label": original_label,
                "resolved_match_label": resolved_label,
                "bracket_mode": BRACKET_MODE,
                "bracket_source": BRACKET_SOURCE,
                "uses_random_pairing": USES_RANDOM_PAIRING,
                "official_bracket": OFFICIAL_BRACKET,
                "random_seed_used_for": RANDOM_SEED_USED_FOR,
            }
        )

    round_pairings = {
        "R32": [(row["team_a"], row["team_b"]) for row in match_rows if row["round"] == "R32"],
        "R16": [(row["team_a"], row["team_b"]) for row in match_rows if row["round"] == "R16"],
        "QF": [(row["team_a"], row["team_b"]) for row in match_rows if row["round"] == "QF"],
        "SF": [(row["team_a"], row["team_b"]) for row in match_rows if row["round"] == "SF"],
        "Final": [(row["team_a"], row["team_b"]) for row in match_rows if row["round"] == "Final"],
    }
    for round_name, pairings in round_pairings.items():
        validate_round_pairings(round_name, pairings)

    unresolved_values = [
        value
        for row in match_rows
        for value in [row["team_a"], row["team_b"]]
        if _looks_like_placeholder(str(value))
    ]
    if unresolved_values:
        raise AssertionError(f"Unresolved placeholders remain: {unresolved_values}")

    stages = {
        "r32_count": stage_reachers["r32_count"],
        "r16_count": [row["winner"] for row in match_rows if row["round"] == "R32"],
        "qf_count": [row["winner"] for row in match_rows if row["round"] == "R16"],
        "sf_count": [row["winner"] for row in match_rows if row["round"] == "QF"],
        "final_count": [row["winner"] for row in match_rows if row["round"] == "SF"],
        "champion_count": [row["winner"] for row in match_rows if row["round"] == "Final"],
    }
    validate_knockout_stages(stages)
    return stages, match_rows


def initialize_tournament_counts(teams_df: pd.DataFrame) -> dict[str, dict]:
    """Create counters for group ranks, qualification, and knockout stages."""
    counts = initialize_monte_carlo_counts(teams_df)
    for row in counts.values():
        for column in KNOCKOUT_STAGE_COUNTS:
            row[column] = 0
    return counts


def update_group_counts_from_records(
    counts: dict[str, dict],
    ranked_groups: list[list[dict]],
    best_third: list[dict],
) -> None:
    """Add group finish and group qualification outcomes to tournament counters."""
    best_third_names = {row["team_name"] for row in best_third}
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
            elif row["team_name"] in best_third_names:
                team_counts["qualified_best_third_count"] += 1
                team_counts["qualified_count"] += 1


def update_knockout_counts(counts: dict[str, dict], stages: Mapping[str, list[str]]) -> None:
    """Add one simulation's knockout stage reach counts."""
    for column, teams in stages.items():
        for team_name in teams:
            counts[team_name][column] += 1


def build_tournament_results(
    counts: Mapping[str, Mapping[str, object]],
    model_config: dict | None = None,
) -> pd.DataFrame:
    """Convert tournament counters into team-level probabilities."""
    model_config = load_model_config() if model_config is None else model_config
    group_results = build_monte_carlo_results(counts)
    extra_columns = []
    for count_column in KNOCKOUT_STAGE_COUNTS:
        probability_column = count_column.replace("_count", "_probability")
        group_results[count_column] = [
            counts[team_name][count_column] for team_name in group_results["team_name"]
        ]
        group_results[probability_column] = (
            group_results[count_column] / group_results["simulations"]
        )
        extra_columns.append(probability_column)

    columns = [
        "team_name",
        "fifa_code",
        "group_letter",
        "simulations",
        "group_rank_1_probability",
        "group_rank_2_probability",
        "group_rank_3_probability",
        "group_rank_4_probability",
        "qualification_probability",
        "r32_probability",
        "r16_probability",
        "qf_probability",
        "sf_probability",
        "final_probability",
        "champion_probability",
        "model_version",
        "model_status",
        "parameter_config_path",
        "rating_col",
        "bracket_mode",
        "bracket_source",
        "uses_random_pairing",
        "official_bracket",
        "random_seed_used_for",
        "bracket_mapping_note",
        "qualified_top_2_probability",
        "qualified_best_third_probability",
        "average_points",
        "average_goal_difference",
        "group_rank_1_count",
        "group_rank_2_count",
        "group_rank_3_count",
        "group_rank_4_count",
        "qualified_top_2_count",
        "qualified_best_third_count",
        "qualified_count",
        "r32_count",
        "r16_count",
        "qf_count",
        "sf_count",
        "final_count",
        "champion_count",
    ]
    group_results = group_results.rename(
        columns={
            "r32_count_probability": "r32_probability",
            "r16_count_probability": "r16_probability",
            "qf_count_probability": "qf_probability",
            "sf_count_probability": "sf_probability",
            "final_count_probability": "final_probability",
            "champion_count_probability": "champion_probability",
        }
    )
    group_results["bracket_mode"] = BRACKET_MODE
    group_results["bracket_source"] = BRACKET_SOURCE
    group_results["uses_random_pairing"] = USES_RANDOM_PAIRING
    group_results["official_bracket"] = OFFICIAL_BRACKET
    group_results["random_seed_used_for"] = str(RANDOM_SEED_USED_FOR)
    group_results["bracket_mapping_note"] = BRACKET_MAPPING_NOTE
    for key, value in metadata_columns(model_config).items():
        group_results[key] = value
    return group_results[columns].sort_values(
        ["champion_probability", "final_probability", "qualification_probability", "team_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def initialize_path_difficulty_counts(teams_df: pd.DataFrame) -> dict[str, dict]:
    """Create per-team counters for average knockout opponent strength."""
    counts = {}
    for team_name in teams_df["team_name"]:
        counts[team_name] = {}
        for round_name in ROUND_OPPONENT_COLUMNS:
            counts[team_name][f"{round_name.lower()}_opponent_strength_total"] = 0.0
            counts[team_name][f"{round_name.lower()}_opponent_count"] = 0
    return counts


def build_path_difficulty_diagnostics(
    teams_df: pd.DataFrame,
    path_counts: Mapping[str, Mapping[str, float | int]],
) -> pd.DataFrame:
    """Convert knockout opponent counters into average strength by round."""
    rows = []
    for _, team in teams_df.sort_values(["group_letter", "team_name"]).iterrows():
        row = {"team_name": team["team_name"]}
        team_counts = path_counts[team["team_name"]]
        for round_name, output_col in ROUND_OPPONENT_COLUMNS.items():
            total_key = f"{round_name.lower()}_opponent_strength_total"
            count_key = f"{round_name.lower()}_opponent_count"
            count = int(team_counts[count_key])
            row[output_col] = team_counts[total_key] / count if count else pd.NA
        rows.append(row)
    return pd.DataFrame(rows)


def save_path_difficulty_diagnostics(
    teams_df: pd.DataFrame,
    path_counts: Mapping[str, Mapping[str, float | int]],
    output_path: Path = PATH_DIFFICULTY_DIAGNOSTICS_PATH,
    model_config: dict | None = None,
) -> pd.DataFrame:
    model_config = load_model_config() if model_config is None else model_config
    diagnostics = build_path_difficulty_diagnostics(teams_df, path_counts)
    for key, value in metadata_columns(model_config).items():
        diagnostics[key] = value
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(output_path, index=False)
    return diagnostics


def save_team_strength_diagnostics(
    ratings_df: pd.DataFrame,
    rating_col: str,
    output_path: Path = TEAM_STRENGTH_DIAGNOSTICS_PATH,
    model_config: dict | None = None,
) -> pd.DataFrame:
    """Save a compact table showing rating inputs used by the Poisson model."""
    model_config = load_model_config() if model_config is None else model_config
    teams = pd.read_csv(TEAMS_PATH)[["team_name", "group_letter"]]
    final_strength_ranks = (
        ratings_df[["team_name", rating_col]]
        .assign(
            rank_by_final_strength=lambda df: df[rating_col]
            .rank(method="min", ascending=False)
            .astype("Int64")
        )[["team_name", "rank_by_final_strength"]]
    )
    diagnostics = teams.loc[teams["team_name"].isin(DIAGNOSTIC_TEAMS)].copy()

    rating_columns = ["team_name"]
    if "world_cup_elo_rating" in ratings_df.columns:
        rating_columns.append("world_cup_elo_rating")
    if rating_col in ratings_df.columns:
        rating_columns.append(rating_col)
    diagnostics = diagnostics.merge(ratings_df[rating_columns], on="team_name", how="left")

    if "world_cup_elo_rating" in diagnostics.columns:
        diagnostics = diagnostics.rename(columns={"world_cup_elo_rating": "raw_elo_rating"})
    else:
        diagnostics["raw_elo_rating"] = pd.NA

    if WEIGHTED_ELO_PATH.exists():
        weighted = pd.read_csv(WEIGHTED_ELO_PATH)
        if "weighted_elo_rating" in weighted.columns:
            diagnostics = diagnostics.merge(
                weighted[["team_name", "weighted_elo_rating"]],
                on="team_name",
                how="left",
            )
    if "weighted_elo_rating" not in diagnostics.columns:
        diagnostics["weighted_elo_rating"] = pd.NA

    if FIFA_ELO_PATH.exists():
        fifa = pd.read_csv(FIFA_ELO_PATH)
        if "fifa_elo_rating" in fifa.columns:
            diagnostics = diagnostics.merge(
                fifa[["team_name", "fifa_elo_rating"]],
                on="team_name",
                how="left",
            )
    if "fifa_elo_rating" not in diagnostics.columns:
        diagnostics["fifa_elo_rating"] = pd.NA

    diagnostics = diagnostics.rename(
        columns={rating_col: "final_strength_used_by_poisson"}
    )
    diagnostics = diagnostics.merge(final_strength_ranks, on="team_name", how="left")
    diagnostics = diagnostics[
        [
            "team_name",
            "group_letter",
            "raw_elo_rating",
            "weighted_elo_rating",
            "fifa_elo_rating",
            "final_strength_used_by_poisson",
            "rank_by_final_strength",
        ]
    ].sort_values("rank_by_final_strength")
    for key, value in metadata_columns(model_config).items():
        diagnostics[key] = value

    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(output_path, index=False)
    return diagnostics


def save_head_to_head_diagnostics(
    rating_lookup: Mapping[str, float],
    base_total_goals: float,
    output_path: Path = HEAD_TO_HEAD_DIAGNOSTICS_PATH,
    model_config: dict | None = None,
) -> pd.DataFrame:
    """Save neutral-site Poisson probabilities for selected diagnostic matchups."""
    model_config = load_model_config() if model_config is None else model_config
    model_kwargs = {
        **poisson_parameter_kwargs(model_config),
        **draw_calibration_kwargs(model_config),
    }
    model_kwargs.setdefault("base_total_goals", base_total_goals)
    rows = []
    for team_a, team_b in HEAD_TO_HEAD_MATCHUPS:
        prediction = predict_from_strength(
            team_a=team_a,
            team_b=team_b,
            strength_a=rating_lookup[team_a],
            strength_b=rating_lookup[team_b],
            **model_kwargs,
        )
        rows.append(
            {
                "team_a": team_a,
                "team_b": team_b,
                "strength_a": prediction["strength_a"],
                "strength_b": prediction["strength_b"],
                "strength_diff": prediction["strength_diff"],
                "lambda_a": prediction["lambda_a"],
                "lambda_b": prediction["lambda_b"],
                "p_team_a_win": prediction["p_team_a_win"],
                "p_draw": prediction["p_draw"],
                "p_team_b_win": prediction["p_team_b_win"],
                "top_5_scorelines": prediction["top_5_scorelines"],
            }
        )

    diagnostics = pd.DataFrame(rows)
    for key, value in metadata_columns(model_config).items():
        diagnostics[key] = value
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(output_path, index=False)
    return diagnostics


def run_tournament_sanity_checks(results: pd.DataFrame, n_simulations: int) -> None:
    """Validate team counts across group qualification and knockout rounds."""
    run_monte_carlo_sanity_checks(results=results, n_simulations=n_simulations)
    for column, teams_per_simulation in KNOCKOUT_STAGE_COUNTS.items():
        expected = teams_per_simulation * n_simulations
        observed = int(results[column].sum())
        if observed != expected:
            raise AssertionError(f"Expected {expected} {column} entries, found {observed}")

    if not (results["r32_count"] == results["qualified_count"]).all():
        raise AssertionError("R32 counts should match group qualification counts")
    if not (results["champion_count"] <= results["final_count"]).all():
        raise AssertionError("Champion count cannot exceed final count")
    if not (results["final_count"] <= results["sf_count"]).all():
        raise AssertionError("Final count cannot exceed semifinal count")
    if not (results["sf_count"] <= results["qf_count"]).all():
        raise AssertionError("Semifinal count cannot exceed quarterfinal count")
    if not (results["qf_count"] <= results["r16_count"]).all():
        raise AssertionError("Quarterfinal count cannot exceed R16 count")
    if not (results["r16_count"] <= results["r32_count"]).all():
        raise AssertionError("R16 count cannot exceed R32 count")


def assert_no_random_pairing_usage() -> None:
    """Guard against accidental random construction of knockout pairings."""
    pairing_functions = [
        build_fixed_r32_bracket,
        resolve_r32_slot,
        assign_third_place_slots,
        resolve_knockout_placeholder,
        resolve_knockout_match_slots,
        load_knockout_schedule,
        simulate_knockout_bracket,
    ]
    for function in pairing_functions:
        source = inspect.getsource(function)
        if "random.shuffle" in source or "random.sample" in source:
            raise AssertionError(
                f"{function.__name__} must not use random.shuffle or random.sample"
            )


def save_sample_knockout_bracket(
    bracket_rows: list[dict],
    output_path: Path = SAMPLE_KNOCKOUT_BRACKET_PATH,
    model_config: dict | None = None,
) -> pd.DataFrame:
    """Save one simulated fixed bracket for inspection."""
    model_config = load_model_config() if model_config is None else model_config
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bracket = pd.DataFrame(bracket_rows)
    for key, value in metadata_columns(model_config).items():
        bracket[key] = value
    bracket.to_csv(output_path, index=False)
    return bracket


def run_tournament_monte_carlo(
    n_simulations: int = DEFAULT_MONTE_CARLO_SIMULATIONS,
    seed: int = 2026,
    output_path: Path = TOURNAMENT_OUTPUT_PATH,
    save_diagnostics: bool = True,
    team_strength_diagnostics_path: Path = TEAM_STRENGTH_DIAGNOSTICS_PATH,
    head_to_head_diagnostics_path: Path = HEAD_TO_HEAD_DIAGNOSTICS_PATH,
    path_difficulty_diagnostics_path: Path = PATH_DIFFICULTY_DIAGNOSTICS_PATH,
    sample_bracket_path: Path = SAMPLE_KNOCKOUT_BRACKET_PATH,
    bracket_mode: str = BRACKET_MODE,
    model_config: dict | None = None,
) -> pd.DataFrame:
    """Run full tournament Monte Carlo simulation from group stage through Final."""
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive")
    if bracket_mode != BRACKET_MODE:
        raise ValueError("Only bracket_mode='fixed_approximation' is implemented")
    assert_no_random_pairing_usage()
    model_config = load_model_config() if model_config is None else model_config

    fixtures = load_group_stage_fixtures()
    teams_df = pd.read_csv(TEAMS_PATH)
    ratings_df, rating_col = load_default_ratings(model_config)
    base_total_goals = load_base_total_goals(model_config=model_config)
    model_kwargs = {
        **poisson_parameter_kwargs(model_config),
        **draw_calibration_kwargs(model_config),
    }
    run_sanity_checks(fixtures=fixtures, phase="fixtures")

    fixture_predictions = precompute_fixture_predictions(
        fixtures=fixtures,
        ratings_df=ratings_df,
        rating_col=rating_col,
        base_total_goals=base_total_goals,
        model_kwargs=model_kwargs,
    )
    knockout_predictions, rating_lookup = precompute_knockout_predictions(
        teams_df=teams_df,
        ratings_df=ratings_df,
        rating_col=rating_col,
        base_total_goals=base_total_goals,
        model_kwargs=model_kwargs,
    )
    grouped_fixtures = {
        group_letter: group.sort_values("match_number").to_dict("records")
        for group_letter, group in fixtures.groupby("group_letter")
    }
    grouped_teams = {
        group_letter: group.to_dict("records")
        for group_letter, group in teams_df.groupby("group_letter")
    }

    counts = initialize_tournament_counts(teams_df)
    path_counts = initialize_path_difficulty_counts(teams_df)
    sample_bracket_rows = None
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
        winners, runners_up, best_third = select_qualifier_records(ranked_groups)
        update_group_counts_from_records(
            counts=counts,
            ranked_groups=ranked_groups,
            best_third=best_third,
        )
        bracket = build_fixed_r32_bracket(
            winners=winners,
            runners_up=runners_up,
            best_third=best_third,
            bracket_mode=bracket_mode,
        )
        stages, bracket_rows = simulate_knockout_bracket(
            r32_bracket=bracket,
            knockout_predictions=knockout_predictions,
            rating_lookup=rating_lookup,
            rng=rng,
            path_counts=path_counts,
        )
        if sample_bracket_rows is None:
            sample_bracket_rows = bracket_rows
        update_knockout_counts(counts=counts, stages=stages)

    results = build_tournament_results(counts, model_config=model_config)
    run_tournament_sanity_checks(results=results, n_simulations=n_simulations)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    if sample_bracket_rows is not None:
        save_sample_knockout_bracket(
            sample_bracket_rows,
            output_path=sample_bracket_path,
            model_config=model_config,
        )
    if save_diagnostics:
        save_team_strength_diagnostics(
            ratings_df=ratings_df,
            rating_col=rating_col,
            output_path=team_strength_diagnostics_path,
            model_config=model_config,
        )
        save_head_to_head_diagnostics(
            rating_lookup=rating_lookup,
            base_total_goals=base_total_goals,
            output_path=head_to_head_diagnostics_path,
            model_config=model_config,
        )
        save_path_difficulty_diagnostics(
            teams_df=teams_df,
            path_counts=path_counts,
            output_path=path_difficulty_diagnostics_path,
            model_config=model_config,
        )
    return results


def print_tournament_sanity_report(results: pd.DataFrame, n_simulations: int) -> None:
    """Print pass/fail checks for full tournament Monte Carlo output."""
    checks = {
        "48 teams in tournament output": len(results) == 48,
        "32 R32 teams per simulation": int(results["r32_count"].sum()) == 32 * n_simulations,
        "16 R16 teams per simulation": int(results["r16_count"].sum()) == 16 * n_simulations,
        "8 quarterfinalists per simulation": int(results["qf_count"].sum()) == 8 * n_simulations,
        "4 semifinalists per simulation": int(results["sf_count"].sum()) == 4 * n_simulations,
        "2 finalists per simulation": int(results["final_count"].sum()) == 2 * n_simulations,
        "1 champion per simulation": int(results["champion_count"].sum()) == n_simulations,
    }
    print("\nTournament sanity checks")
    for label, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} - {label}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full 2026 World Cup tournament Monte Carlo simulation."
    )
    parser.add_argument(
        "--mode",
        default="default",
        choices=["default", "experimental", "test"],
        help="Model parameter mode.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=DEFAULT_MONTE_CARLO_SIMULATIONS,
        help="Number of full tournament simulations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TOURNAMENT_OUTPUT_PATH,
        help="CSV path for full tournament probability output.",
    )
    parser.add_argument(
        "--bracket-mode",
        default=BRACKET_MODE,
        choices=[BRACKET_MODE],
        help="Knockout bracket mapping mode.",
    )
    args = parser.parse_args()
    model_config = load_model_config(args.mode)
    if args.output == TOURNAMENT_OUTPUT_PATH:
        args.output = output_path("tournament_simulation_results", model_config)

    results = run_tournament_monte_carlo(
        n_simulations=args.n_simulations,
        seed=args.seed,
        output_path=args.output,
        bracket_mode=args.bracket_mode,
        model_config=model_config,
    )
    print(
        f"Ran {args.n_simulations} full tournament simulations with seed {args.seed}."
    )
    print(f"Saved tournament output to {args.output}")
    print(f"Saved sample knockout bracket to {SAMPLE_KNOCKOUT_BRACKET_PATH}")
    print(f"Bracket mode: {BRACKET_MODE}")
    print(f"Model version: {model_config['model_version']}")
    print(f"Parameter config: {model_config['parameter_config_path']}")
    print(f"Uses random pairing: {USES_RANDOM_PAIRING}")
    print(f"Official bracket: {OFFICIAL_BRACKET}")
    print(f"Bracket mapping note: {BRACKET_MAPPING_NOTE}")
    print_tournament_sanity_report(results=results, n_simulations=args.n_simulations)


if __name__ == "__main__":
    main()
