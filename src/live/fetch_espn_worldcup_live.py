from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEAGUE_SLUG = "fifa.world"
SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/"
    f"{LEAGUE_SLUG}/scoreboard"
)
STANDINGS_URL = (
    "https://site.api.espn.com/apis/v2/sports/soccer/"
    f"{LEAGUE_SLUG}/standings"
)
LIVE_DIR = PROJECT_ROOT / "output" / "live"
RAW_DIR = LIVE_DIR / "raw"
DEFAULT_FIXTURES_OUTPUT = LIVE_DIR / "fixtures_results.csv"
STANDINGS_OUTPUT = LIVE_DIR / "group_standings.csv"
UPDATE_LOG_PATH = LIVE_DIR / "live_data_update_log.csv"
SOURCE = "espn_json"
LOG_COLUMNS = [
    "timestamp_utc",
    "task",
    "status",
    "source",
    "message",
    "rows_written",
    "output_path",
]
FIXTURE_COLUMNS = [
    "match_id",
    "espn_event_id",
    "date",
    "kickoff_time_utc",
    "group",
    "team_a",
    "team_b",
    "team_a_id",
    "team_b_id",
    "goals_a",
    "goals_b",
    "status",
    "status_detail",
    "actual_result",
    "venue",
    "city",
    "source",
    "last_updated",
]
STANDINGS_COLUMNS = [
    "group",
    "rank",
    "team",
    "team_id",
    "played",
    "wins",
    "draws",
    "losses",
    "goals_for",
    "goals_against",
    "goal_difference",
    "points",
    "source",
    "last_updated",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dirs() -> None:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def append_log(
    task: str,
    status: str,
    message: str,
    rows_written: int = 0,
    output_path: str | Path = "",
) -> None:
    ensure_dirs()
    row = pd.DataFrame(
        [
            {
                "timestamp_utc": utc_now_iso(),
                "task": task,
                "status": status,
                "source": SOURCE,
                "message": message,
                "rows_written": rows_written,
                "output_path": str(output_path),
            }
        ],
        columns=LOG_COLUMNS,
    )
    write_header = not UPDATE_LOG_PATH.exists()
    row.to_csv(UPDATE_LOG_PATH, mode="a", header=write_header, index=False)


def parse_yyyymmdd(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%d").replace(tzinfo=timezone.utc)


def date_range(start_date: str, end_date: str) -> list[str]:
    start = parse_yyyymmdd(start_date)
    end = parse_yyyymmdd(end_date)
    if end < start:
        raise ValueError("--end-date must be on or after --start-date")
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return dates


def requested_scoreboard_dates(args: argparse.Namespace) -> list[str | None]:
    if args.date:
        return [args.date]
    if args.start_date or args.end_date:
        if not args.start_date or not args.end_date:
            raise ValueError("--start-date and --end-date must be provided together")
        return date_range(args.start_date, args.end_date)
    return [None]


def scoreboard_url_for_date(date_value: str | None) -> str:
    if date_value:
        return f"{SCOREBOARD_URL}?dates={date_value}"
    return SCOREBOARD_URL


def raw_scoreboard_path(date_value: str | None) -> Path:
    raw_date = date_value or datetime.now(timezone.utc).strftime("%Y%m%d")
    return RAW_DIR / f"espn_scoreboard_{raw_date}.json"


def get_json(url: str, task: str) -> tuple[dict[str, Any] | None, str]:
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            message = f"HTTP {response.status_code}: {response.text[:200]}"
            append_log(task=task, status="failed", message=message)
            return None, message
        return response.json(), "ok"
    except Exception as exc:  # network and malformed JSON should not crash the whole script.
        message = str(exc)
        append_log(task=task, status="failed", message=message)
        return None, message


def save_raw_json(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def normalize_status(status_obj: dict[str, Any]) -> tuple[str, str]:
    status_type = status_obj.get("type", {}) if isinstance(status_obj, dict) else {}
    name = str(status_type.get("name", "")).lower()
    state = str(status_type.get("state", "")).lower()
    description = str(status_type.get("description", "")).lower()
    detail = str(status_type.get("detail") or status_type.get("shortDetail") or "")
    combined = " ".join([name, state, description, detail.lower()])

    if "postpon" in combined:
        return "postponed", detail
    if "cancel" in combined or "canceled" in combined:
        return "cancelled", detail
    if bool(status_type.get("completed")) or state == "post" or "final" in combined:
        return "final", detail
    if state == "in" or "progress" in combined or "halftime" in combined:
        return "in_progress", detail
    if state == "pre" or "scheduled" in combined:
        return "scheduled", detail
    return "unknown", detail


def parse_int_score(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def actual_result(status: str, goals_a: int | None, goals_b: int | None) -> str:
    if status == "final" and goals_a is not None and goals_b is not None:
        if goals_a > goals_b:
            return "team_a_win"
        if goals_a < goals_b:
            return "team_b_win"
        return "draw"
    if status == "scheduled":
        return "not_started"
    if status == "in_progress":
        return "in_progress"
    return "unknown"


def parse_group(event: dict[str, Any], competition: dict[str, Any]) -> str:
    candidates = [
        event.get("group", {}).get("name") if isinstance(event.get("group"), dict) else None,
        competition.get("group", {}).get("name") if isinstance(competition.get("group"), dict) else None,
        event.get("season", {}).get("name") if isinstance(event.get("season"), dict) else None,
        event.get("season", {}).get("slug") if isinstance(event.get("season"), dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.lower().startswith("group "):
            return candidate
    return ""


def normalize_event(event: dict[str, Any], last_updated: str) -> dict[str, Any]:
    competitions = event.get("competitions") or []
    competition = competitions[0] if competitions else {}
    competitors = competition.get("competitors") or []
    if len(competitors) < 2:
        raise ValueError("event has fewer than two competitors")

    team_a_competitor = competitors[0]
    team_b_competitor = competitors[1]
    team_a = team_a_competitor.get("team", {})
    team_b = team_b_competitor.get("team", {})
    status, status_detail = normalize_status(
        competition.get("status") or event.get("status") or {}
    )
    goals_a = parse_int_score(team_a_competitor.get("score"))
    goals_b = parse_int_score(team_b_competitor.get("score"))
    kickoff = event.get("date") or competition.get("date") or competition.get("startDate")
    venue = competition.get("venue") or event.get("venue") or {}
    address = venue.get("address") if isinstance(venue.get("address"), dict) else {}

    return {
        "match_id": event.get("id"),
        "espn_event_id": event.get("id"),
        "date": str(kickoff)[:10] if kickoff else "",
        "kickoff_time_utc": kickoff or "",
        "group": parse_group(event, competition),
        "team_a": team_a.get("displayName") or team_a.get("name") or "",
        "team_b": team_b.get("displayName") or team_b.get("name") or "",
        "team_a_id": team_a.get("id") or team_a_competitor.get("id") or "",
        "team_b_id": team_b.get("id") or team_b_competitor.get("id") or "",
        "goals_a": goals_a,
        "goals_b": goals_b,
        "status": status,
        "status_detail": status_detail,
        "actual_result": actual_result(status, goals_a, goals_b),
        "venue": venue.get("fullName") or venue.get("name") or "",
        "city": address.get("city") or venue.get("city") or "",
        "source": SOURCE,
        "last_updated": last_updated,
    }


def fetch_scoreboards(date_values: list[str | None]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    last_updated = utc_now_iso()
    for date_value in date_values:
        url = scoreboard_url_for_date(date_value)
        task = f"scoreboard:{date_value or 'default'}"
        data, message = get_json(url, task=task)
        if data is None:
            continue

        raw_path = raw_scoreboard_path(date_value)
        save_raw_json(data, raw_path)
        append_log(
            task=task,
            status="success",
            message=f"Fetched scoreboard JSON from {url}",
            rows_written=0,
            output_path=raw_path,
        )
        events = data.get("events") or []
        if not events:
            append_log(
                task=task,
                status="warning",
                message="Scoreboard returned zero events",
                rows_written=0,
                output_path=raw_path,
            )
            continue

        for event in events:
            try:
                rows.append(normalize_event(event, last_updated=last_updated))
            except Exception as exc:
                append_log(
                    task=task,
                    status="warning",
                    message=f"Skipped event {event.get('id', '')}: {exc}",
                    rows_written=0,
                    output_path=raw_path,
                )

    return pd.DataFrame(rows, columns=FIXTURE_COLUMNS)


def merge_with_existing_fixtures(new_fixtures: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Merge new ESPN rows into the existing live fixture file.

    ESPN scoreboard responses can be date/window specific. Keeping a cumulative
    file prevents completed matches from disappearing from live evaluation after
    the default scoreboard moves on to later fixtures.
    """
    if not output_path.exists():
        return new_fixtures
    try:
        existing = pd.read_csv(output_path)
    except Exception as exc:
        append_log(
            task="fixtures_results",
            status="warning",
            message=f"Could not read existing fixtures for cumulative merge: {exc}",
            output_path=output_path,
        )
        return new_fixtures

    if existing.empty:
        return new_fixtures
    combined = pd.concat([existing, new_fixtures], ignore_index=True)
    for column in FIXTURE_COLUMNS:
        if column not in combined.columns:
            combined[column] = pd.NA

    status_rank = {
        "unknown": 0,
        "scheduled": 1,
        "postponed": 1,
        "cancelled": 1,
        "in_progress": 2,
        "final": 3,
    }
    combined["_status_rank"] = combined["status"].map(status_rank).fillna(0)
    combined["_last_updated_sort"] = pd.to_datetime(
        combined["last_updated"],
        errors="coerce",
        utc=True,
    )
    combined["_dedupe_key"] = (
        combined["espn_event_id"].fillna(combined["match_id"]).astype(str)
    )
    combined = combined.sort_values(
        ["_dedupe_key", "_status_rank", "_last_updated_sort"],
        na_position="first",
    )
    combined = combined.drop_duplicates("_dedupe_key", keep="last")
    combined = combined.drop(columns=["_status_rank", "_last_updated_sort", "_dedupe_key"])
    return combined[FIXTURE_COLUMNS].sort_values(["date", "kickoff_time_utc", "match_id"])


def standings_stat_value(entry: dict[str, Any], *names: str) -> int | None:
    wanted = {name.lower() for name in names}
    for stat in entry.get("stats") or []:
        stat_names = {
            str(stat.get("name", "")).lower(),
            str(stat.get("type", "")).lower(),
            str(stat.get("abbreviation", "")).lower(),
        }
        if stat_names & wanted:
            value = stat.get("value")
            try:
                return int(float(value))
            except (TypeError, ValueError):
                display_value = stat.get("displayValue")
                try:
                    return int(float(display_value))
                except (TypeError, ValueError):
                    return None
    return None


def normalize_standings(data: dict[str, Any], last_updated: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    children = data.get("children") or []
    for child in children:
        group_name = child.get("name") or child.get("abbreviation") or ""
        entries = ((child.get("standings") or {}).get("entries") or [])
        for index, entry in enumerate(entries, start=1):
            team = entry.get("team") or {}
            note = entry.get("note") or {}
            rows.append(
                {
                    "group": group_name,
                    "rank": note.get("rank") or index,
                    "team": team.get("displayName") or team.get("name") or "",
                    "team_id": team.get("id") or "",
                    "played": standings_stat_value(entry, "gamesPlayed", "gamesplayed", "GP"),
                    "wins": standings_stat_value(entry, "wins", "W"),
                    "draws": standings_stat_value(entry, "ties", "draws", "D"),
                    "losses": standings_stat_value(entry, "losses", "L"),
                    "goals_for": standings_stat_value(entry, "pointsFor", "pointsfor", "F"),
                    "goals_against": standings_stat_value(
                        entry, "pointsAgainst", "pointsagainst", "A"
                    ),
                    "goal_difference": standings_stat_value(
                        entry, "pointDifferential", "pointdifferential", "GD"
                    ),
                    "points": standings_stat_value(entry, "points", "P"),
                    "source": SOURCE,
                    "last_updated": last_updated,
                }
            )
    return pd.DataFrame(rows, columns=STANDINGS_COLUMNS)


def fetch_standings() -> pd.DataFrame:
    last_updated = utc_now_iso()
    data, message = get_json(STANDINGS_URL, task="standings")
    if data is None:
        return pd.DataFrame(columns=STANDINGS_COLUMNS)

    raw_path = RAW_DIR / "espn_standings.json"
    save_raw_json(data, raw_path)
    append_log(
        task="standings",
        status="success",
        message=f"Fetched standings JSON from {STANDINGS_URL}",
        output_path=raw_path,
    )
    try:
        standings = normalize_standings(data, last_updated=last_updated)
        if standings.empty:
            append_log(
                task="standings",
                status="warning",
                message="Standings endpoint returned no parseable rows",
                output_path=STANDINGS_OUTPUT,
            )
        return standings
    except Exception as exc:
        append_log(
            task="standings",
            status="warning",
            message=f"Could not fully normalize standings: {exc}",
            output_path=STANDINGS_OUTPUT,
        )
        return pd.DataFrame(columns=STANDINGS_COLUMNS)


def print_fixture_quality_report(fixtures: pd.DataFrame, output_path: Path) -> None:
    if fixtures.empty:
        print("Rows: 0")
        print(f"Output path: {output_path}")
        return

    print(f"Rows: {len(fixtures)}")
    print(f"Final matches: {int(fixtures['status'].eq('final').sum())}")
    print(f"Scheduled matches: {int(fixtures['status'].eq('scheduled').sum())}")
    print(f"In-progress matches: {int(fixtures['status'].eq('in_progress').sum())}")
    print(f"Date range: {fixtures['date'].min()} to {fixtures['date'].max()}")
    print(f"Output path: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch FIFA World Cup live/scheduled/final data from ESPN JSON endpoints."
    )
    parser.add_argument("--date", help="Fetch a single scoreboard date, YYYYMMDD.")
    parser.add_argument("--start-date", help="Start date for inclusive date range, YYYYMMDD.")
    parser.add_argument("--end-date", help="End date for inclusive date range, YYYYMMDD.")
    parser.add_argument("--output", type=Path, default=DEFAULT_FIXTURES_OUTPUT)
    standings_group = parser.add_mutually_exclusive_group()
    standings_group.add_argument("--fetch-standings", action="store_true")
    standings_group.add_argument("--no-standings", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    date_values = requested_scoreboard_dates(args)
    fixtures = fetch_scoreboards(date_values)
    fixtures = merge_with_existing_fixtures(fixtures, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixtures.to_csv(args.output, index=False)
    append_log(
        task="fixtures_results",
        status="success" if not fixtures.empty else "warning",
        message=f"Wrote normalized fixtures/results rows for {len(date_values)} scoreboard request(s)",
        rows_written=len(fixtures),
        output_path=args.output,
    )

    if args.fetch_standings:
        standings = fetch_standings()
        if not standings.empty:
            standings.to_csv(STANDINGS_OUTPUT, index=False)
            append_log(
                task="group_standings",
                status="success",
                message="Wrote normalized standings",
                rows_written=len(standings),
                output_path=STANDINGS_OUTPUT,
            )
        else:
            append_log(
                task="group_standings",
                status="warning",
                message="No standings rows written",
                rows_written=0,
                output_path=STANDINGS_OUTPUT,
            )
    elif args.no_standings:
        append_log(
            task="standings",
            status="skipped",
            message="Standings fetch skipped by --no-standings",
            output_path=STANDINGS_OUTPUT,
        )

    print_fixture_quality_report(fixtures, args.output)


if __name__ == "__main__":
    main()
