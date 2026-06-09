from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Manual aliases handle common dataset naming variants that are not historical
# former names, but still need to collapse into the current names in teams.csv.
TEAM_NAME_ALIASES = {
    "United States": "USA",
    "US": "USA",
    "Iran": "IR Iran",
    "Ir Iran": "IR Iran",
    "Cape Verde": "Cabo Verde",
    "Ivory Coast": "Cote d'Ivoire",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "Curaçao": "Curacao",
    "Turkey": "Turkey",
    "Türkiye": "Turkey",
    "Czech Republic": "Czechia",
}


def _clean_text(value):
    """Normalize whitespace while preserving missing values."""
    if pd.isna(value):
        return value
    return str(value).strip()


def read_csv_with_encoding_fallback(path):
    """Read CSV files saved by common spreadsheet tools.

    The project data is usually UTF-8, but teams.csv may be exported from
    PyCharm/Excel with Mac Roman encoding for names like Türkiye and Curaçao.
    """
    for encoding in ["utf-8", "utf-8-sig", "mac_roman", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    return pd.read_csv(path)


def load_team_name_mapping(former_names_path):
    """Load former-name rows for date-aware standardization.

    The former_names file maps historical team names to the modern/current
    names used by the project. Date ranges matter because some historical
    names only apply for a specific period, so callers pass the match date
    when standardizing match records.
    """
    former_names = pd.read_csv(former_names_path)
    for column in ["current", "former"]:
        former_names[column] = former_names[column].map(_clean_text)

    former_names["start_date"] = pd.to_datetime(
        former_names["start_date"], errors="coerce"
    )
    former_names["end_date"] = pd.to_datetime(former_names["end_date"], errors="coerce")

    mapping = {}
    for row in former_names.itertuples(index=False):
        mapping.setdefault(row.former, []).append(
            {
                "current": row.current,
                "start_date": row.start_date,
                "end_date": row.end_date,
            }
        )

    return mapping


def standardize_team_name(team_name, match_date, former_name_mapping):
    """Return the current team name for a raw team name.

    Manual aliases cover naming variants such as United States/USA and Iran/IR
    Iran. A date-aware former-name lookup is then used for historical names. If
    no row matches, the raw name is retained so non-mapped opponents are not
    accidentally renamed.
    """
    team_name = _clean_text(team_name)
    if pd.isna(team_name):
        return team_name

    if team_name in TEAM_NAME_ALIASES:
        return TEAM_NAME_ALIASES[team_name]

    candidates = former_name_mapping.get(team_name, [])
    for candidate in candidates:
        starts_before_match = (
            pd.isna(candidate["start_date"]) or candidate["start_date"] <= match_date
        )
        ends_after_match = (
            pd.isna(candidate["end_date"]) or match_date <= candidate["end_date"]
        )
        if starts_before_match and ends_after_match:
            # Former-name mappings can point to accented current names; apply
            # project aliases again so cleaned outputs stay ASCII-friendly.
            return TEAM_NAME_ALIASES.get(candidate["current"], candidate["current"])

    return team_name


def add_standardized_team_columns(results, shootouts, former_name_mapping):
    """Apply the former-name mapping consistently to result and shootout teams."""
    results = results.copy()
    shootouts = shootouts.copy()

    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    shootouts["date"] = pd.to_datetime(shootouts["date"], errors="coerce")

    for column in ["home_team", "away_team"]:
        results[column] = results[column].map(_clean_text)
        shootouts[column] = shootouts[column].map(_clean_text)

        results[column] = results.apply(
            lambda row: standardize_team_name(
                row[column], row["date"], former_name_mapping
            ),
            axis=1,
        )
        shootouts[column] = shootouts.apply(
            lambda row: standardize_team_name(
                row[column], row["date"], former_name_mapping
            ),
            axis=1,
        )

    shootouts["winner"] = shootouts.apply(
        lambda row: standardize_team_name(row["winner"], row["date"], former_name_mapping),
        axis=1,
    )

    return results, shootouts


def prepare_shootouts(shootouts):
    """Prepare shootout winner/loser columns for joining to drawn matches.

    Shootouts are extra information about a match that was drawn in regulation.
    They add shootout_wins/shootout_losses, but never change the regulation draw
    into a regulation win or loss.
    """
    shootouts = shootouts.copy()
    shootouts["shootout_winner"] = shootouts["winner"]
    shootouts["shootout_loser"] = shootouts.apply(
        lambda row: (
            row["away_team"] if row["winner"] == row["home_team"] else row["home_team"]
        ),
        axis=1,
    )

    return shootouts[
        ["date", "home_team", "away_team", "shootout_winner", "shootout_loser"]
    ].drop_duplicates(subset=["date", "home_team", "away_team"])


def build_clean_matches(results, shootouts, world_cup_teams):
    """Create cleaned match-level rows involving at least one 2026 team."""
    results = results.copy()
    results = results.dropna(subset=["date", "home_score", "away_score"])
    results["home_score"] = results["home_score"].astype(int)
    results["away_score"] = results["away_score"].astype(int)

    relevant_match = results["home_team"].isin(world_cup_teams) | results[
        "away_team"
    ].isin(world_cup_teams)
    matches = results.loc[relevant_match].copy()

    shootouts_for_join = prepare_shootouts(shootouts)
    matches = matches.merge(
        shootouts_for_join,
        how="left",
        on=["date", "home_team", "away_team"],
    )

    matches["is_draw"] = matches["home_score"] == matches["away_score"]
    non_draw_match = ~matches["is_draw"]
    matches.loc[non_draw_match, ["shootout_winner", "shootout_loser"]] = pd.NA
    matches["has_shootout"] = matches["shootout_winner"].notna()
    for location_column in ["city", "country"]:
        matches[location_column] = matches[location_column].map(_clean_text)
        matches[location_column] = matches[location_column].replace(TEAM_NAME_ALIASES)
    matches["date"] = matches["date"].dt.strftime("%Y-%m-%d")

    output_columns = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
        "is_draw",
        "has_shootout",
        "shootout_winner",
        "shootout_loser",
    ]

    return matches[output_columns].sort_values(
        ["date", "home_team", "away_team"]
    )


def summarize_team_results(matches_clean, world_cup_teams):
    """Aggregate match results separately for each 2026 World Cup team."""
    summary = {
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
        for team in world_cup_teams
    }

    for row in matches_clean.itertuples(index=False):
        team_views = [
            (row.home_team, row.home_score, row.away_score, row.away_team),
            (row.away_team, row.away_score, row.home_score, row.home_team),
        ]

        for team, goals_for, goals_against, opponent in team_views:
            if team not in summary:
                continue

            record = summary[team]
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

    summary_df = pd.DataFrame(summary.values())
    summary_df["goal_difference"] = (
        summary_df["goals_for"] - summary_df["goals_against"]
    )

    return summary_df.sort_values("team")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    teams = read_csv_with_encoding_fallback(RAW_DIR / "teams.csv")
    teams["team_name"] = teams["team_name"].map(_clean_text)
    world_cup_teams = teams["team_name"].tolist()

    former_name_mapping = load_team_name_mapping(RAW_DIR / "former_names.csv")
    results = pd.read_csv(RAW_DIR / "results.csv")
    shootouts = pd.read_csv(RAW_DIR / "shootouts.csv")

    results, shootouts = add_standardized_team_columns(
        results, shootouts, former_name_mapping
    )

    games_result_clean = build_clean_matches(results, shootouts, set(world_cup_teams))

    # Qualifier outputs keep any result where either side is one of the 2026
    # World Cup teams from teams.csv. For example, Germany vs India is kept if
    # Germany is in teams.csv, while China vs Hong Kong is excluded if neither
    # team is in teams.csv.
    qualifier_games_results = games_result_clean.copy()
    team_summary = summarize_team_results(games_result_clean, world_cup_teams)
    qualifier_summary = summarize_team_results(qualifier_games_results, world_cup_teams)

    games_result_clean.to_csv(PROCESSED_DIR / "games_result_clean.csv", index=False)
    qualifier_games_results.to_csv(
        PROCESSED_DIR / "qualifier_games_results.csv", index=False
    )
    team_summary.to_csv(PROCESSED_DIR / "team_results_summary.csv", index=False)
    qualifier_summary.to_csv(
        PROCESSED_DIR / "qualifier_results_summary.csv", index=False
    )

    legacy_matches_path = PROCESSED_DIR / "matches_clean.csv"
    if legacy_matches_path.exists():
        legacy_matches_path.unlink()

    print(
        f"Wrote {len(games_result_clean):,} rows to "
        f"{PROCESSED_DIR / 'games_result_clean.csv'}"
    )
    print(
        f"Wrote {len(qualifier_games_results):,} rows to "
        f"{PROCESSED_DIR / 'qualifier_games_results.csv'}"
    )
    print(
        f"Wrote {len(team_summary):,} rows to "
        f"{PROCESSED_DIR / 'team_results_summary.csv'}"
    )
    print(
        f"Wrote {len(qualifier_summary):,} rows to "
        f"{PROCESSED_DIR / 'qualifier_results_summary.csv'}"
    )


if __name__ == "__main__":
    main()
