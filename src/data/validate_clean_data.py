from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TEAM_COLUMNS = ["home_team", "away_team", "shootout_winner", "shootout_loser"]
SUMMARY_COLUMNS = [
    "matches_played",
    "regulation_wins",
    "regulation_draws",
    "regulation_losses",
    "shootout_wins",
    "shootout_losses",
    "goals_for",
    "goals_against",
    "goal_difference",
]


def _clean_text(value):
    if pd.isna(value):
        return value
    return str(value).strip()


def read_csv_with_encoding_fallback(path):
    for encoding in ["utf-8", "utf-8-sig", "mac_roman", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    return pd.read_csv(path)


def print_check(passed, message, details=None):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {message}")
    if details:
        print(f"       {details}")
    return passed


def require_files(paths):
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        for path in missing:
            print_check(False, f"Required file exists: {path}")
        sys.exit(1)


def load_inputs():
    paths = [
        RAW_DIR / "teams.csv",
        RAW_DIR / "former_names.csv",
        PROCESSED_DIR / "qualifier_games_results.csv",
        PROCESSED_DIR / "qualifier_results_summary.csv",
    ]
    require_files(paths)

    teams = read_csv_with_encoding_fallback(RAW_DIR / "teams.csv")
    former_names = pd.read_csv(RAW_DIR / "former_names.csv")
    games = pd.read_csv(PROCESSED_DIR / "qualifier_games_results.csv")
    summary = pd.read_csv(PROCESSED_DIR / "qualifier_results_summary.csv")

    teams["team_name"] = teams["team_name"].map(_clean_text)
    former_names["former"] = former_names["former"].map(_clean_text)

    return teams, former_names, games, summary


def validate_summary_teams(summary, world_cup_teams):
    summary_teams = summary["team"].map(_clean_text)
    team_counts = summary_teams.value_counts()

    extra_teams = sorted(set(summary_teams) - world_cup_teams)
    missing_teams = sorted(world_cup_teams - set(summary_teams))
    duplicate_teams = sorted(team_counts[team_counts != 1].index.tolist())

    checks = [
        print_check(
            not extra_teams,
            "qualifier_results_summary.csv contains only teams from data/raw/teams.csv",
            f"Unexpected teams: {extra_teams[:10]}" if extra_teams else None,
        ),
        print_check(
            not missing_teams and not duplicate_teams,
            "Every team in data/raw/teams.csv appears exactly once in qualifier_results_summary.csv",
            (
                f"Missing: {missing_teams[:10]}, duplicates/non-single counts: "
                f"{duplicate_teams[:10]}"
                if missing_teams or duplicate_teams
                else None
            ),
        ),
    ]

    return all(checks)


def validate_game_team_filter(games, world_cup_teams):
    valid_rows = games["home_team"].isin(world_cup_teams) | games["away_team"].isin(
        world_cup_teams
    )
    invalid_rows = games.loc[~valid_rows, ["date", "home_team", "away_team"]]

    return print_check(
        invalid_rows.empty,
        "qualifier_games_results.csv only contains matches where at least one team is in teams.csv",
        invalid_rows.head(5).to_dict("records") if not invalid_rows.empty else None,
    )


def validate_no_former_names(games, former_names):
    former_team_names = set(former_names["former"].dropna())
    found = {}

    for column in TEAM_COLUMNS:
        if column not in games.columns:
            continue
        values = set(games[column].dropna().map(_clean_text))
        matches = sorted(values & former_team_names)
        if matches:
            found[column] = matches[:10]

    return print_check(
        not found,
        "No former team names from former_names.csv appear in cleaned team columns",
        str(found) if found else None,
    )


def compute_summary_from_games(games, world_cup_teams):
    records = {
        team: {
            "team": team,
            "matches_played": 0,
            "regulation_wins": 0,
            "regulation_draws": 0,
            "regulation_losses": 0,
            "shootout_wins": 0,
            "shootout_losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_difference": 0,
        }
        for team in sorted(world_cup_teams)
    }

    for row in games.itertuples(index=False):
        team_views = [
            (row.home_team, row.home_score, row.away_score, row.away_team),
            (row.away_team, row.away_score, row.home_score, row.home_team),
        ]
        for team, goals_for, goals_against, opponent in team_views:
            if team not in records:
                continue

            record = records[team]
            record["matches_played"] += 1
            record["goals_for"] += goals_for
            record["goals_against"] += goals_against

            if goals_for > goals_against:
                record["regulation_wins"] += 1
            elif goals_for < goals_against:
                record["regulation_losses"] += 1
            else:
                record["regulation_draws"] += 1
                if row.has_shootout:
                    if row.shootout_winner == team:
                        record["shootout_wins"] += 1
                    elif row.shootout_winner == opponent:
                        record["shootout_losses"] += 1

    computed = pd.DataFrame(records.values())
    computed["goal_difference"] = computed["goals_for"] - computed["goals_against"]
    return computed.sort_values("team").reset_index(drop=True)


def validate_shootout_logic(games, summary, world_cup_teams):
    games = games.copy()
    games["has_shootout"] = games["has_shootout"].astype(bool)

    equal_score = games["home_score"] == games["away_score"]
    draw_flag_ok = games["is_draw"].astype(bool) == equal_score
    non_draw_shootouts = games.loc[~equal_score & games["has_shootout"]]

    shootout_rows = games.loc[games["has_shootout"]].copy()
    winner_ok = (shootout_rows["shootout_winner"] == shootout_rows["home_team"]) | (
        shootout_rows["shootout_winner"] == shootout_rows["away_team"]
    )
    loser_ok = (shootout_rows["shootout_loser"] == shootout_rows["home_team"]) | (
        shootout_rows["shootout_loser"] == shootout_rows["away_team"]
    )
    winner_loser_distinct = (
        shootout_rows["shootout_winner"] != shootout_rows["shootout_loser"]
    )

    computed_summary = compute_summary_from_games(games, world_cup_teams)
    actual_summary = summary.sort_values("team").reset_index(drop=True)
    merged_summary = actual_summary.merge(
        computed_summary, on="team", how="outer", suffixes=("_actual", "_computed")
    )

    mismatches = []
    for column in SUMMARY_COLUMNS:
        actual = merged_summary[f"{column}_actual"]
        computed = merged_summary[f"{column}_computed"]
        bad_rows = merged_summary.loc[actual != computed, "team"].tolist()
        if bad_rows:
            mismatches.append(f"{column}: {bad_rows[:5]}")

    checks = [
        print_check(
            draw_flag_ok.all(),
            "Equal-score matches are marked as regulation draws, and non-equal scores are not",
            (
                games.loc[~draw_flag_ok, ["date", "home_team", "away_team"]]
                .head(5)
                .to_dict("records")
                if not draw_flag_ok.all()
                else None
            ),
        ),
        print_check(
            non_draw_shootouts.empty,
            "Shootout winners only appear on equal-score matches",
            (
                non_draw_shootouts[["date", "home_team", "away_team"]]
                .head(5)
                .to_dict("records")
                if not non_draw_shootouts.empty
                else None
            ),
        ),
        print_check(
            winner_ok.all() and loser_ok.all() and winner_loser_distinct.all(),
            "Each shootout has one winner and the other team as loser",
            (
                shootout_rows.loc[
                    ~(winner_ok & loser_ok & winner_loser_distinct),
                    ["date", "home_team", "away_team", "shootout_winner", "shootout_loser"],
                ]
                .head(5)
                .to_dict("records")
                if not (winner_ok.all() and loser_ok.all() and winner_loser_distinct.all())
                else None
            ),
        ),
        print_check(
            not mismatches,
            "qualifier_results_summary.csv shootout and match totals match qualifier_games_results.csv",
            "; ".join(mismatches[:5]) if mismatches else None,
        ),
    ]

    return all(checks)


def main():
    print("Clean data validation report")
    print("============================")

    teams, former_names, games, summary = load_inputs()
    world_cup_teams = set(teams["team_name"].dropna())

    checks = [
        validate_summary_teams(summary, world_cup_teams),
        validate_game_team_filter(games, world_cup_teams),
        validate_no_former_names(games, former_names),
        validate_shootout_logic(games, summary, world_cup_teams),
    ]

    print("============================")
    if all(checks):
        print("All validation checks passed.")
        sys.exit(0)

    print("One or more validation checks failed.")
    sys.exit(1)


if __name__ == "__main__":
    main()
