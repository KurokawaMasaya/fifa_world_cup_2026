from __future__ import annotations

import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"

sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))
from clean_results import (  # noqa: E402
    _clean_text,
    add_standardized_team_columns,
    load_team_name_mapping,
    read_csv_with_encoding_fallback,
)


START_DATE = pd.Timestamp("2014-01-01")
INITIAL_RATING = 1500.0
BASE_K = 30.0
HOME_ADVANTAGE = 50.0
ANNUAL_DECAY = 0.90
FORM_DECAY = 0.85
MAX_RATING_CHANGE = 60.0
FAVORITE_MIN_CONVINCING_WIN_REWARD = 2.0
MARGIN_DAMPING_DENOMINATOR = 1600.0
MARGIN_DAMPING_FLOOR = 0.65
ANCHOR_WEIGHT = 0.70
MODEL_WEIGHT = 0.30
RECENT_FORM_ADJUSTMENT_CAPPED = 0.0
STRENGTH_RECALIBRATION_DIAGNOSTICS_PATH = (
    OUTPUT_DIR / "diagnostics_strength_recalibration.csv"
)


# Current-team aliases that are specific to the latest teams.csv naming.
RATING_TEAM_ALIASES = {
    "Turkey": "Turkey",
    "Türkiye": "Turkey",
    "Czech Republic": "Czechia",
    "Curaçao": "Curacao",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "Ivory Coast": "Cote d'Ivoire",
}


CONTINENTAL_CHAMPIONSHIPS = {
    "AFC Asian Cup",
    "African Cup of Nations",
    "CONCACAF Gold Cup",
    "Copa América",
    "Gold Cup",
    "Oceania Nations Cup",
    "UEFA Euro",
}


# Confederation map is intentionally explicit and local. It supports the
# optional World Cup-oriented adjustment and reliability metrics without adding
# a new external data dependency.
CONFEDERATION_MAP = {
    "Algeria": "CAF",
    "Argentina": "CONMEBOL",
    "Australia": "AFC",
    "Austria": "UEFA",
    "Belgium": "UEFA",
    "Bosnia and Herzegovina": "UEFA",
    "Brazil": "CONMEBOL",
    "Cabo Verde": "CAF",
    "Canada": "CONCACAF",
    "Colombia": "CONMEBOL",
    "Croatia": "UEFA",
    "Curacao": "CONCACAF",
    "Cote d'Ivoire": "CAF",
    "Czechia": "UEFA",
    "DR Congo": "CAF",
    "Ecuador": "CONMEBOL",
    "Egypt": "CAF",
    "England": "UEFA",
    "France": "UEFA",
    "Germany": "UEFA",
    "Ghana": "CAF",
    "Haiti": "CONCACAF",
    "IR Iran": "AFC",
    "Iraq": "AFC",
    "Japan": "AFC",
    "Jordan": "AFC",
    "Mexico": "CONCACAF",
    "Morocco": "CAF",
    "Netherlands": "UEFA",
    "New Zealand": "OFC",
    "Norway": "UEFA",
    "Panama": "CONCACAF",
    "Paraguay": "CONMEBOL",
    "Portugal": "UEFA",
    "Qatar": "AFC",
    "Saudi Arabia": "AFC",
    "Scotland": "UEFA",
    "Senegal": "CAF",
    "South Africa": "CAF",
    "South Korea": "AFC",
    "Spain": "UEFA",
    "Sweden": "UEFA",
    "Switzerland": "UEFA",
    "Tunisia": "CAF",
    "Turkey": "UEFA",
    "USA": "CONCACAF",
    "Uruguay": "CONMEBOL",
    "Uzbekistan": "AFC",
}


CONFEDERATION_POOL_CORRECTION = {
    "UEFA": 20.0,
    "CONMEBOL": 25.0,
    "CAF": -15.0,
    "AFC": -20.0,
    "CONCACAF": -15.0,
    "OFC": -35.0,
}


STRENGTH_RECALIBRATION_TEAMS = [
    "France",
    "Morocco",
    "Senegal",
    "Brazil",
    "Germany",
    "Japan",
    "Spain",
    "Argentina",
    "England",
    "Portugal",
    "Netherlands",
]


@dataclass
class EloRun:
    final_ratings: dict[str, float]
    history: pd.DataFrame
    team_features: pd.DataFrame
    k_report: dict[str, float]


def standardize_current_name(team_name: str) -> str:
    team_name = _clean_text(team_name)
    return RATING_TEAM_ALIASES.get(team_name, team_name)


def expected_score(team_rating: float, opponent_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - team_rating) / 400.0))


def actual_scores(home_score: int, away_score: int) -> tuple[float, float]:
    if home_score > away_score:
        return 1.0, 0.0
    if home_score < away_score:
        return 0.0, 1.0
    return 0.5, 0.5


def tournament_multiplier(tournament: str) -> float:
    """World Cup-oriented K weighting by match importance."""
    if tournament == "Friendly":
        return 0.6
    if tournament == "FIFA World Cup":
        return 2.0
    if tournament == "FIFA World Cup qualification":
        return 1.2
    if tournament in CONTINENTAL_CHAMPIONSHIPS:
        return 1.5
    return 1.0


def margin_multiplier(
    goal_diff: int,
    team_rating: float,
    opponent_rating: float,
) -> float:
    """Scale Elo updates by margin with mismatch damping.

    Football scorelines carry useful performance signal, but margin information
    has diminishing value in mismatches. A heavy favorite winning 5-0 against a
    weak opponent is less informative than the same score between evenly matched
    teams, and huge margins can reflect red cards or game-state effects. The
    capped log base avoids linear goal-difference inflation, while rating-gap
    damping reduces the margin bonus when teams were far apart before kickoff.
    """
    base_margin = max(1.0, math.log(abs(goal_diff) + 1.0))
    base_margin = min(base_margin, 1.7)
    rating_gap = abs(team_rating - opponent_rating)
    gap_damping = 1.0 / (1.0 + rating_gap / MARGIN_DAMPING_DENOMINATOR)
    gap_damping = max(MARGIN_DAMPING_FLOOR, gap_damping)
    return 1.0 + (base_margin - 1.0) * gap_damping


def load_modern_matches() -> pd.DataFrame:
    """Load all completed international matches from the modern rating window.

    This intentionally does not filter to 2026 World Cup teams. Elo needs the
    full connected graph of international matches; filtering only happens in
    final rating outputs.
    """
    former_name_mapping = load_team_name_mapping(RAW_DIR / "former_names.csv")
    results = pd.read_csv(RAW_DIR / "results.csv")
    shootouts = pd.read_csv(RAW_DIR / "shootouts.csv")
    results, shootouts = add_standardized_team_columns(
        results, shootouts, former_name_mapping
    )

    matches = results.copy()
    matches["home_team"] = matches["home_team"].map(standardize_current_name)
    matches["away_team"] = matches["away_team"].map(standardize_current_name)
    matches = matches.dropna(subset=["date", "home_score", "away_score"])
    matches = matches.loc[matches["date"] >= START_DATE].copy()
    matches["home_score"] = matches["home_score"].astype(int)
    matches["away_score"] = matches["away_score"].astype(int)
    matches["neutral"] = matches["neutral"].astype(bool)
    matches["goal_diff"] = matches["home_score"] - matches["away_score"]
    matches["is_draw"] = matches["goal_diff"] == 0
    return matches.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


def load_world_cup_teams() -> pd.DataFrame:
    teams = read_csv_with_encoding_fallback(RAW_DIR / "teams.csv")
    teams["team_name"] = teams["team_name"].map(standardize_current_name)
    return teams[["id", "team_name", "fifa_code", "group_letter"]].copy()


def load_fifa_prior() -> pd.DataFrame | None:
    """Load FIFA points/ranks when present for Model 4 blending."""
    for path in [
        RAW_DIR / "qualifier_elo_rating.csv",
        PROCESSED_DIR / "qualifier_elo_rating.csv",
    ]:
        if path.exists():
            prior = read_csv_with_encoding_fallback(path)
            prior["team_name"] = prior["team_name"].map(standardize_current_name)
            return prior[["team_name", "fifa_elo_rating", "fifa_world_ranking"]]
    return None


def apply_annual_mean_reversion(
    ratings: dict[str, float],
    from_year: int | None,
    to_date: pd.Timestamp,
    teams: list[str],
) -> int:
    """Apply one 10% pull toward 1500 at each new calendar year."""
    if from_year is None:
        return to_date.year

    for year in range(from_year + 1, to_date.year + 1):
        for team in teams:
            ratings[team] = INITIAL_RATING + ANNUAL_DECAY * (
                ratings[team] - INITIAL_RATING
            )
    return to_date.year


def update_recent_form(
    form_state: dict[str, dict[str, float]],
    team: str,
    points: float,
    expected_points: float,
    goal_diff: int,
) -> None:
    """Track opponent-adjusted recent form.

    Recent form is measured against the Elo expectation for that opponent, so
    narrow wins as big favorites do not get the same boost as overperforming
    against a strong team. The final recent_form_score is capped to keep this
    short-term signal from overwhelming the rating.
    """
    state = form_state[team]
    state["recent_points"] = FORM_DECAY * state["recent_points"] + points
    state["recent_points_above_expected"] = (
        FORM_DECAY * state["recent_points_above_expected"]
        + (points - expected_points)
    )
    state["recent_goal_difference"] = (
        FORM_DECAY * state["recent_goal_difference"] + max(-5, min(5, goal_diff))
    )
    state["recent_matches"] = FORM_DECAY * state["recent_matches"] + 1.0


def form_score(state: dict[str, float]) -> float:
    """Convert decayed form state into a compact rating-point adjustment."""
    recent_matches = max(state["recent_matches"], 1e-9)
    points_above_expected = state["recent_points_above_expected"] / recent_matches
    goal_diff_per_match = state["recent_goal_difference"] / recent_matches
    score = 55.0 * points_above_expected + 6.0 * goal_diff_per_match
    return max(-25.0, min(25.0, score))


def run_elo_model(
    matches: pd.DataFrame,
    model_name: str,
    *,
    weighted: bool = False,
    adjusted: bool = False,
) -> EloRun:
    ratings: defaultdict[str, float] = defaultdict(lambda: INITIAL_RATING)
    matches_since_2014: defaultdict[str, int] = defaultdict(int)
    matches_last_4_years: defaultdict[str, int] = defaultdict(int)
    cross_confed_matches: defaultdict[str, int] = defaultdict(int)
    form_state: defaultdict[str, dict[str, float]] = defaultdict(
        lambda: {
            "recent_points": 0.0,
            "recent_points_above_expected": 0.0,
            "recent_goal_difference": 0.0,
            "recent_matches": 0.0,
        }
    )
    history_rows = []
    current_year = None

    cutoff_4_years = matches["date"].max() - pd.DateOffset(years=4)

    for row in matches.itertuples(index=False):
        home_team = row.home_team
        away_team = row.away_team
        teams = [home_team, away_team]

        # Model 3 adds annual mean reversion before the first match of each
        # calendar year, preventing old ratings from becoming permanently stale.
        if adjusted:
            current_year = apply_annual_mean_reversion(
                ratings, current_year, row.date, list(ratings.keys())
            )

        home_before = ratings[home_team]
        away_before = ratings[away_team]
        home_for_expectation = home_before

        if weighted and not row.neutral:
            home_for_expectation += HOME_ADVANTAGE

        expected_home = expected_score(home_for_expectation, away_before)
        expected_away = 1.0 - expected_home
        actual_home, actual_away = actual_scores(row.home_score, row.away_score)

        k_multiplier = 1.0
        mov_multiplier = 1.0
        if weighted:
            k_multiplier = tournament_multiplier(row.tournament)
            mov_multiplier = margin_multiplier(
                row.goal_diff, home_before, away_before
            )

        k_effective = BASE_K * k_multiplier * mov_multiplier
        delta_home = k_effective * (actual_home - expected_home)
        favorite_side = "even"
        if home_before > away_before:
            favorite_side = "home"
            if row.goal_diff >= 2 and actual_home > expected_home:
                delta_home = max(delta_home, FAVORITE_MIN_CONVINCING_WIN_REWARD)
        elif away_before > home_before:
            favorite_side = "away"
            if row.goal_diff <= -2 and actual_away > expected_away:
                delta_home = min(delta_home, -FAVORITE_MIN_CONVINCING_WIN_REWARD)

        delta_home = max(-MAX_RATING_CHANGE, min(MAX_RATING_CHANGE, delta_home))
        delta_away = -delta_home

        ratings[home_team] += delta_home
        ratings[away_team] += delta_away

        home_confed = CONFEDERATION_MAP.get(home_team)
        away_confed = CONFEDERATION_MAP.get(away_team)
        is_cross_confed = bool(home_confed and away_confed and home_confed != away_confed)

        for team in teams:
            matches_since_2014[team] += 1
            if row.date >= cutoff_4_years:
                matches_last_4_years[team] += 1
            if is_cross_confed:
                cross_confed_matches[team] += 1

        if adjusted:
            update_recent_form(
                form_state,
                home_team,
                actual_home,
                expected_home,
                row.goal_diff,
            )
            update_recent_form(
                form_state,
                away_team,
                actual_away,
                expected_away,
                -row.goal_diff,
            )

        history_rows.append(
            {
                "date": row.date.strftime("%Y-%m-%d"),
                "model": model_name,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": row.home_score,
                "away_score": row.away_score,
                "tournament": row.tournament,
                "neutral": row.neutral,
                "home_rating_before": round(home_before, 3),
                "away_rating_before": round(away_before, 3),
                "expected_home": round(expected_home, 4),
                "actual_home": actual_home,
                "expected_away": round(expected_away, 4),
                "actual_away": actual_away,
                "favorite_side": favorite_side,
                "favorite_status_home": (
                    "favorite"
                    if favorite_side == "home"
                    else "underdog"
                    if favorite_side == "away"
                    else "even"
                ),
                "favorite_status_away": (
                    "favorite"
                    if favorite_side == "away"
                    else "underdog"
                    if favorite_side == "home"
                    else "even"
                ),
                "k_effective": round(k_effective, 3),
                "home_rating_after": round(ratings[home_team], 3),
                "away_rating_after": round(ratings[away_team], 3),
                "home_delta": round(delta_home, 3),
                "away_delta": round(delta_away, 3),
            }
        )

    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    feature_rows = []
    for team in teams:
        state = form_state[team]
        feature_rows.append(
            {
                "team_name": team,
                "matches_since_2014": matches_since_2014[team],
                "matches_last_4_years": matches_last_4_years[team],
                "cross_confederation_matches": cross_confed_matches[team],
                "recent_points": round(state["recent_points"], 4),
                "recent_points_above_expected": round(
                    state["recent_points_above_expected"], 4
                ),
                "recent_goal_difference": round(state["recent_goal_difference"], 4),
                "recent_form_score": round(form_score(state), 3) if adjusted else 0.0,
            }
        )

    k_values = [row["k_effective"] for row in history_rows]
    k_report = {
        "base_k": BASE_K,
        "min_effective_k": min(k_values),
        "max_effective_k": max(k_values),
        "mean_effective_k": sum(k_values) / len(k_values),
    }

    return EloRun(
        final_ratings=dict(ratings),
        history=pd.DataFrame(history_rows),
        team_features=pd.DataFrame(feature_rows),
        k_report=k_report,
    )


def world_cup_rating_output(
    teams: pd.DataFrame,
    run: EloRun,
    model_name: str,
    *,
    include_adjustments: bool = False,
    fifa_prior: pd.DataFrame | None = None,
) -> pd.DataFrame:
    output = teams.copy()
    output[f"{model_name}_rating"] = output["team_name"].map(
        lambda team: round(run.final_ratings.get(team, INITIAL_RATING), 3)
    )
    output = output.merge(run.team_features, on="team_name", how="left")

    if include_adjustments:
        output["adjusted_elo_rating"] = output["adjusted_elo_rating"].astype(float)
        output["rating_with_recent_form"] = (
            output["adjusted_elo_rating"] + output["recent_form_score"]
        ).round(3)

    if fifa_prior is not None:
        output = output.merge(fifa_prior, on="team_name", how="left")

    return output


def build_world_cup_elo(
    teams: pd.DataFrame,
    adjusted_output: pd.DataFrame,
    fifa_prior: pd.DataFrame | None,
    adjusted_history: pd.DataFrame,
) -> pd.DataFrame:
    output = adjusted_output.copy()
    output["confederation"] = output["team_name"].map(CONFEDERATION_MAP)

    # Cross-confederation performance is optional because it is only reliable
    # when both teams have known confederations. We estimate confederation
    # strength from actual-vs-expected residuals in cross-confederation matches:
    # a confederation that beats Elo expectation gets a small positive nudge.
    residuals = defaultdict(list)
    for row in adjusted_history.itertuples(index=False):
        home_confed = CONFEDERATION_MAP.get(row.home_team)
        away_confed = CONFEDERATION_MAP.get(row.away_team)
        if not home_confed or not away_confed or home_confed == away_confed:
            continue

        home_residual = row.actual_home - row.expected_home
        residuals[home_confed].append(home_residual)
        residuals[away_confed].append(-home_residual)

    confed_strength = {
        confed: max(-25.0, min(25.0, 100.0 * (sum(values) / len(values))))
        for confed, values in residuals.items()
    }
    output["confederation_strength_adjustment"] = (
        output["confederation"].map(confed_strength).fillna(0.0).round(3)
    )

    output["model_rating_before_fifa_blend"] = (
        output["rating_with_recent_form"]
        + output["confederation_strength_adjustment"]
    ).round(3)

    if fifa_prior is not None and "fifa_elo_rating" not in output.columns:
        output = output.merge(fifa_prior, on="team_name", how="left")

    # FIFA points are interpreted as an Elo-style prior per project direction.
    # The model remains dominant, while FIFA points add a stabilizing prior.
    fifa_available = output["fifa_elo_rating"].notna()
    output["world_cup_elo_rating"] = output["model_rating_before_fifa_blend"]
    output.loc[fifa_available, "world_cup_elo_rating"] = (
        0.75 * output.loc[fifa_available, "model_rating_before_fifa_blend"]
        + 0.25 * output.loc[fifa_available, "fifa_elo_rating"]
    )
    output["world_cup_elo_rating"] = output["world_cup_elo_rating"].round(3)

    output["recent_form_adjustment_capped"] = RECENT_FORM_ADJUSTMENT_CAPPED
    output["anchored_final_strength_no_confed"] = output[
        "model_rating_before_fifa_blend"
    ]
    output.loc[fifa_available, "anchored_final_strength_no_confed"] = (
        ANCHOR_WEIGHT * output.loc[fifa_available, "fifa_elo_rating"]
        + MODEL_WEIGHT
        * output.loc[fifa_available, "model_rating_before_fifa_blend"]
        + output.loc[fifa_available, "recent_form_adjustment_capped"]
    )
    output["confederation_pool_correction"] = (
        output["confederation"].map(CONFEDERATION_POOL_CORRECTION).fillna(0.0)
    )
    output["anchored_final_strength"] = (
        output["anchored_final_strength_no_confed"]
        + output["confederation_pool_correction"]
    ).round(3)
    output["anchored_final_strength_no_confed"] = output[
        "anchored_final_strength_no_confed"
    ].round(3)

    output["reliability_score"] = (
        0.40 * (output["matches_since_2014"].clip(upper=80) / 80.0)
        + 0.35 * (output["matches_last_4_years"].clip(upper=35) / 35.0)
        + 0.25 * (output["cross_confederation_matches"].clip(upper=20) / 20.0)
    ).round(3)

    keep_columns = [
        "id",
        "team_name",
        "fifa_code",
        "group_letter",
        "world_cup_elo_rating",
        "model_rating_before_fifa_blend",
        "fifa_elo_rating",
        "fifa_world_ranking",
        "anchored_final_strength_no_confed",
        "confederation_pool_correction",
        "anchored_final_strength",
        "recent_form_adjustment_capped",
        "confederation",
        "confederation_strength_adjustment",
        "recent_form_score",
        "matches_since_2014",
        "matches_last_4_years",
        "cross_confederation_matches",
        "reliability_score",
    ]
    return output[keep_columns]


def save_history(run: EloRun, model_name: str) -> None:
    run.history.to_csv(PROCESSED_DIR / f"elo_history_{model_name}.csv", index=False)


def save_rating_change_diagnostics(run: EloRun) -> None:
    """Save average rating movement by favorite/underdog and tournament type."""
    history = run.history.copy()

    home_rows = history[
        ["tournament", "favorite_status_home", "home_delta"]
    ].rename(
        columns={
            "favorite_status_home": "favorite_status",
            "home_delta": "rating_change",
        }
    )
    away_rows = history[
        ["tournament", "favorite_status_away", "away_delta"]
    ].rename(
        columns={
            "favorite_status_away": "favorite_status",
            "away_delta": "rating_change",
        }
    )
    team_match_rows = pd.concat([home_rows, away_rows], ignore_index=True)
    team_match_rows["abs_rating_change"] = team_match_rows["rating_change"].abs()

    by_favorite = (
        team_match_rows.groupby("favorite_status")
        .agg(
            matches=("rating_change", "size"),
            avg_rating_change=("rating_change", "mean"),
            avg_abs_rating_change=("abs_rating_change", "mean"),
        )
        .reset_index()
    )
    by_favorite.insert(0, "diagnostic_type", "favorite_status")
    by_favorite = by_favorite.rename(columns={"favorite_status": "category"})

    by_tournament = (
        team_match_rows.groupby("tournament")
        .agg(
            matches=("rating_change", "size"),
            avg_rating_change=("rating_change", "mean"),
            avg_abs_rating_change=("abs_rating_change", "mean"),
        )
        .reset_index()
        .sort_values("matches", ascending=False)
    )
    by_tournament.insert(0, "diagnostic_type", "tournament")
    by_tournament = by_tournament.rename(columns={"tournament": "category"})

    diagnostics = pd.concat([by_favorite, by_tournament], ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(OUTPUT_DIR / "elo_rating_change_diagnostics.csv", index=False)


def save_strength_recalibration_diagnostics(world_cup_output: pd.DataFrame) -> None:
    """Print and save old-vs-anchored final strength diagnostics."""
    ranked = world_cup_output.copy()
    ranked["old_rank"] = (
        ranked["world_cup_elo_rating"].rank(method="min", ascending=False).astype(int)
    )
    ranked["new_rank"] = (
        ranked["anchored_final_strength"].rank(method="min", ascending=False).astype(int)
    )
    ranked["rank_change"] = ranked["old_rank"] - ranked["new_rank"]

    print("\nTop 20 teams by old world_cup_elo_rating:")
    print(
        ranked.sort_values("world_cup_elo_rating", ascending=False)
        .head(20)[["team_name", "world_cup_elo_rating", "old_rank"]]
        .to_string(index=False)
    )

    print("\nTop 20 teams by new anchored_final_strength:")
    print(
        ranked.sort_values("anchored_final_strength", ascending=False)
        .head(20)[["team_name", "anchored_final_strength", "new_rank"]]
        .to_string(index=False)
    )

    diagnostics = ranked.loc[
        ranked["team_name"].isin(STRENGTH_RECALIBRATION_TEAMS),
        [
            "team_name",
            "fifa_elo_rating",
            "model_rating_before_fifa_blend",
            "world_cup_elo_rating",
            "anchored_final_strength_no_confed",
            "confederation_pool_correction",
            "anchored_final_strength",
            "old_rank",
            "new_rank",
            "rank_change",
        ],
    ].rename(columns={"world_cup_elo_rating": "old_world_cup_elo_rating"})
    diagnostics = diagnostics.sort_values("new_rank")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(STRENGTH_RECALIBRATION_DIAGNOSTICS_PATH, index=False)
    print(f"\nWrote recalibration diagnostics to {STRENGTH_RECALIBRATION_DIAGNOSTICS_PATH}")


def test_k_sanity(run: EloRun, model_name: str) -> None:
    """Flag K settings that are likely too timid or too volatile."""
    max_single_delta = run.history["home_delta"].abs().max()
    rating_spread = max(run.final_ratings.values()) - min(run.final_ratings.values())
    if max_single_delta > 80:
        print(
            f"[WARN] {model_name}: max single-match move {max_single_delta:.2f} "
            "is high; consider lowering K or multipliers."
        )
    elif max_single_delta < 5:
        print(
            f"[WARN] {model_name}: max single-match move {max_single_delta:.2f} "
            "is very small; K may be too low."
        )
    else:
        print(
            f"[PASS] {model_name}: K sanity ok "
            f"(max move {max_single_delta:.2f}, rating spread {rating_spread:.1f})."
        )


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matches = load_modern_matches()
    teams = load_world_cup_teams()
    fifa_prior = load_fifa_prior()

    print(f"Loaded {len(matches):,} completed matches from {START_DATE.date()} onward.")

    simple_run = run_elo_model(matches, "simple_elo")
    simple_output = world_cup_rating_output(teams, simple_run, "simple_elo")
    simple_output.to_csv(PROCESSED_DIR / "team_ratings_simple_elo.csv", index=False)
    save_history(simple_run, "simple_elo")
    test_k_sanity(simple_run, "simple_elo")

    weighted_run = run_elo_model(matches, "weighted_elo", weighted=True)
    weighted_output = world_cup_rating_output(teams, weighted_run, "weighted_elo")
    weighted_output.to_csv(PROCESSED_DIR / "team_ratings_weighted_elo.csv", index=False)
    save_history(weighted_run, "weighted_elo")
    test_k_sanity(weighted_run, "weighted_elo")

    adjusted_run = run_elo_model(
        matches, "adjusted_elo", weighted=True, adjusted=True
    )
    adjusted_output = world_cup_rating_output(
        teams, adjusted_run, "adjusted_elo", include_adjustments=True
    )
    adjusted_output.to_csv(PROCESSED_DIR / "team_ratings_adjusted_elo.csv", index=False)
    save_history(adjusted_run, "adjusted_elo")
    save_rating_change_diagnostics(adjusted_run)
    test_k_sanity(adjusted_run, "adjusted_elo")

    world_cup_output = build_world_cup_elo(
        teams, adjusted_output, fifa_prior, adjusted_run.history
    )
    world_cup_output.to_csv(
        PROCESSED_DIR / "team_ratings_world_cup_elo.csv", index=False
    )
    save_strength_recalibration_diagnostics(world_cup_output)
    adjusted_run.history.assign(model="world_cup_elo").to_csv(
        PROCESSED_DIR / "elo_history_world_cup_elo.csv", index=False
    )

    print("Wrote staged Elo model outputs:")
    for filename in [
        "team_ratings_simple_elo.csv",
        "team_ratings_weighted_elo.csv",
        "team_ratings_adjusted_elo.csv",
        "team_ratings_world_cup_elo.csv",
        "diagnostics_strength_recalibration.csv",
        "elo_history_simple_elo.csv",
        "elo_history_weighted_elo.csv",
        "elo_history_adjusted_elo.csv",
        "elo_history_world_cup_elo.csv",
        "elo_rating_change_diagnostics.csv",
    ]:
        print(f"  - {PROCESSED_DIR / filename}")


if __name__ == "__main__":
    main()
