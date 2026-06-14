from __future__ import annotations

"""Export daily group-stage prediction tables using Beijing calendar dates.

The live ESPN feed stores kickoff times in UTC. This script converts those
kickoffs to Asia/Shanghai and exports the matches whose Beijing date matches
the requested date. It is a presentation/export layer only: it reads existing
fixtures and clean prediction outputs and does not modify W/D/L probabilities
or model files.
"""

import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURES_PATH = PROJECT_ROOT / "output" / "live" / "fixtures_results.csv"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "output" / "predictions" / "group_stage_predictions_clean.csv"
DEFAULT_MATCHES_PATH = PROJECT_ROOT / "data" / "raw" / "matches.csv"
DEFAULT_TEAMS_PATH = PROJECT_ROOT / "data" / "raw" / "teams.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "predictions" / "daily_beijing"
BEIJING_TZ = ZoneInfo("Asia/Shanghai")


def normalize_team_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    collapsed = " ".join(text.split())
    aliases = {
        "united states": "usa",
        "us": "usa",
        "usa": "usa",
        "bosnia herzegovina": "bosnia and herzegovina",
        "bosnia and herzegovina": "bosnia and herzegovina",
        "czech republic": "czechia",
        "czechia": "czechia",
        "ir iran": "iran",
        "cape verde": "cabo verde",
        "cabo verde": "cabo verde",
        "ivory coast": "cote d ivoire",
        "cote d ivoire": "cote d ivoire",
        "turkiye": "turkey",
    }
    return aliases.get(collapsed, collapsed)


def match_key(team_a: object, team_b: object) -> str:
    return "::".join(sorted([normalize_team_name(team_a), normalize_team_name(team_b)]))


def parse_json_list(value: object) -> list:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def today_beijing() -> str:
    return datetime.now(BEIJING_TZ).strftime("%Y-%m-%d")


def build_daily_table(
    fixtures_path: Path,
    predictions_path: Path,
    matches_path: Path,
    teams_path: Path,
    beijing_date: str,
) -> pd.DataFrame:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    predictions = pd.read_csv(predictions_path)
    required_prediction_columns = {
        "match_id",
        "group",
        "team_a",
        "team_b",
        "team_a_win_pct",
        "draw_pct",
        "team_b_win_pct",
        "top_5_scorelines",
        "top_5_scoreline_probability_pct",
    }
    missing = required_prediction_columns - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    predictions = predictions.copy()
    predictions["join_key"] = predictions.apply(
        lambda row: match_key(row["team_a"], row["team_b"]), axis=1
    )

    if matches_path.exists() and teams_path.exists():
        schedule = load_schedule_base(matches_path=matches_path, teams_path=teams_path)
    elif fixtures_path.exists():
        schedule = load_live_fixture_base(fixtures_path=fixtures_path)
    else:
        raise FileNotFoundError(
            f"Neither schedule files nor live fixtures are available: {matches_path}, {teams_path}, {fixtures_path}"
        )

    selected = schedule[schedule["beijing_date"].eq(beijing_date)].copy()
    merged = selected.merge(
        predictions.add_prefix("prediction_"),
        left_on="join_key",
        right_on="prediction_join_key",
        how="left",
    )

    live_overlay = pd.DataFrame()
    if fixtures_path.exists():
        live_overlay = load_live_fixture_base(fixtures_path=fixtures_path)
        live_overlay = live_overlay[
            [
                "join_key",
                "espn_match_id",
                "beijing_date",
                "beijing_kickoff_time",
                "status",
                "venue",
                "city",
            ]
        ].drop_duplicates("join_key", keep="last")
        merged = merged.merge(live_overlay.add_prefix("live_"), left_on="join_key", right_on="live_join_key", how="left")

    rows = []
    for row in merged.itertuples(index=False):
        top5 = parse_json_list(getattr(row, "prediction_top_5_scorelines", None))
        probs = parse_json_list(getattr(row, "prediction_top_5_scoreline_probability_pct", None))
        top5_display = ", ".join(f"{score} ({prob}%)" for score, prob in zip(top5, probs))
        rows.append(
            {
                "beijing_date": beijing_date,
                "beijing_kickoff_time": getattr(row, "live_beijing_kickoff_time", pd.NA)
                if pd.notna(getattr(row, "live_beijing_kickoff_time", pd.NA))
                else getattr(row, "beijing_kickoff_time"),
                "espn_match_id": getattr(row, "live_espn_match_id", pd.NA),
                "group": getattr(row, "prediction_group", ""),
                "team_a": getattr(row, "prediction_team_a", None)
                if pd.notna(getattr(row, "prediction_team_a", None))
                else getattr(row, "team_a"),
                "team_b": getattr(row, "prediction_team_b", None)
                if pd.notna(getattr(row, "prediction_team_b", None))
                else getattr(row, "team_b"),
                "team_a_win_pct": getattr(row, "prediction_team_a_win_pct", pd.NA),
                "draw_pct": getattr(row, "prediction_draw_pct", pd.NA),
                "team_b_win_pct": getattr(row, "prediction_team_b_win_pct", pd.NA),
                "top_5_scorelines": json.dumps(top5, ensure_ascii=False),
                "top_5_scoreline_probability_pct": json.dumps(probs),
                "top_5_display": top5_display,
                "status": getattr(row, "live_status", "scheduled")
                if pd.notna(getattr(row, "live_status", pd.NA))
                else "scheduled",
                "venue": getattr(row, "live_venue", "")
                if pd.notna(getattr(row, "live_venue", pd.NA))
                else "",
                "city": getattr(row, "live_city", "")
                if pd.notna(getattr(row, "live_city", pd.NA))
                else "",
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "beijing_date",
                "beijing_kickoff_time",
                "espn_match_id",
                "group",
                "team_a",
                "team_b",
                "team_a_win_pct",
                "draw_pct",
                "team_b_win_pct",
                "top_5_scorelines",
                "top_5_scoreline_probability_pct",
                "top_5_display",
                "status",
                "venue",
                "city",
            ]
        )

    table = pd.DataFrame(rows).sort_values("beijing_kickoff_time").reset_index(drop=True)
    return table


def load_schedule_base(matches_path: Path, teams_path: Path) -> pd.DataFrame:
    """Use the full local World Cup schedule as the authoritative daily base."""
    matches = pd.read_csv(matches_path)
    teams = pd.read_csv(teams_path)
    team_lookup = teams.set_index("id")["team_name"].to_dict()
    schedule = matches[matches["stage_id"].eq(1)].copy()
    schedule["team_a"] = schedule["home_team_id"].map(team_lookup)
    schedule["team_b"] = schedule["away_team_id"].map(team_lookup)
    schedule = schedule[schedule[["team_a", "team_b"]].notna().all(axis=1)].copy()
    schedule["kickoff_beijing_dt"] = pd.to_datetime(
        schedule["kickoff_at"], utc=True, errors="coerce"
    ).dt.tz_convert(BEIJING_TZ)
    schedule["beijing_date"] = schedule["kickoff_beijing_dt"].dt.strftime("%Y-%m-%d")
    schedule["beijing_kickoff_time"] = schedule["kickoff_beijing_dt"].dt.strftime("%H:%M")
    schedule["join_key"] = schedule.apply(lambda row: match_key(row["team_a"], row["team_b"]), axis=1)
    schedule["schedule_match_id"] = schedule["match_number"]
    return schedule[
        [
            "schedule_match_id",
            "team_a",
            "team_b",
            "beijing_date",
            "beijing_kickoff_time",
            "join_key",
        ]
    ]


def load_live_fixture_base(fixtures_path: Path) -> pd.DataFrame:
    fixtures = pd.read_csv(fixtures_path)
    if "kickoff_time_utc" not in fixtures.columns:
        raise ValueError("Fixtures file is missing kickoff_time_utc")
    fixtures = fixtures.copy()
    fixtures["kickoff_beijing_dt"] = pd.to_datetime(
        fixtures["kickoff_time_utc"], utc=True, errors="coerce"
    ).dt.tz_convert(BEIJING_TZ)
    fixtures["beijing_date"] = fixtures["kickoff_beijing_dt"].dt.strftime("%Y-%m-%d")
    fixtures["beijing_kickoff_time"] = fixtures["kickoff_beijing_dt"].dt.strftime("%H:%M")
    fixtures["join_key"] = fixtures.apply(lambda row: match_key(row["team_a"], row["team_b"]), axis=1)
    return fixtures.rename(columns={"match_id": "espn_match_id"})[
        [
            "espn_match_id",
            "team_a",
            "team_b",
            "beijing_date",
            "beijing_kickoff_time",
            "join_key",
            "status",
            "venue",
            "city",
        ]
    ]


def markdown_table(table: pd.DataFrame) -> str:
    lines = [
        "| Beijing Time | Group | Match | W/D/L % | Top 5 scorelines | Status | Venue |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in table.itertuples(index=False):
        if pd.isna(row.team_a_win_pct):
            wdl = ""
        else:
            wdl = f"{int(row.team_a_win_pct)}/{int(row.draw_pct)}/{int(row.team_b_win_pct)}"
        lines.append(
            f"| {row.beijing_kickoff_time} | {row.group} | {row.team_a} vs {row.team_b} "
            f"| {wdl} | {row.top_5_display} | {row.status} | {row.venue}, {row.city} |"
        )
    return "\n".join(lines) + "\n"


def export_daily_table(args: argparse.Namespace) -> tuple[Path, Path, pd.DataFrame]:
    beijing_date = args.date or today_beijing()
    compact_date = beijing_date.replace("-", "_")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = build_daily_table(
        fixtures_path=Path(args.fixtures),
        predictions_path=Path(args.predictions),
        matches_path=Path(args.matches),
        teams_path=Path(args.teams),
        beijing_date=beijing_date,
    )
    csv_path = output_dir / f"group_stage_games_{compact_date}_beijing_top5.csv"
    md_dir = PROJECT_ROOT / "output" / "reports" / "daily_beijing"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_path = md_dir / f"group_stage_games_{compact_date}_beijing_top5.md"
    table.to_csv(csv_path, index=False)
    md_path.write_text(markdown_table(table))
    latest_csv = output_dir / "group_stage_games_today_beijing_top5.csv"
    latest_md = md_dir / "group_stage_games_today_beijing_top5.md"
    table.to_csv(latest_csv, index=False)
    latest_md.write_text(markdown_table(table))
    # Convenience "latest" files stay at the top level so the API/widget/user
    # can find today's table quickly; dated archives live in daily_beijing/.
    top_level_latest_csv = PROJECT_ROOT / "output" / "predictions" / "group_stage_games_today_beijing_top5.csv"
    top_level_latest_md = PROJECT_ROOT / "output" / "reports" / "group_stage_games_today_beijing_top5.md"
    table.to_csv(top_level_latest_csv, index=False)
    top_level_latest_md.write_text(markdown_table(table))
    return csv_path, md_path, table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Beijing date in YYYY-MM-DD. Defaults to today in Asia/Shanghai.")
    parser.add_argument("--fixtures", default=str(DEFAULT_FIXTURES_PATH))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS_PATH))
    parser.add_argument("--matches", default=str(DEFAULT_MATCHES_PATH))
    parser.add_argument("--teams", default=str(DEFAULT_TEAMS_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    csv_path, md_path, table = export_daily_table(parse_args())
    print(f"Saved Beijing daily prediction table to {csv_path}")
    print(f"Saved markdown table to {md_path}")
    print(f"Rows: {len(table)}")
    if not table.empty:
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
