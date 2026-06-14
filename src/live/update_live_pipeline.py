from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = PROJECT_ROOT / "output" / "live"
PIPELINE_LOG_PATH = LIVE_DIR / "live_pipeline_run_log.csv"
FIXTURES_RESULTS_PATH = LIVE_DIR / "fixtures_results.csv"
LIVE_SUMMARY_PATH = LIVE_DIR / "worldcup_group_stage_live_summary.csv"
LIVE_EVALUATION_PATH = LIVE_DIR / "worldcup_group_stage_live_evaluation.csv"
LIVE_CALIBRATION_PATH = LIVE_DIR / "worldcup_group_stage_live_calibration.csv"
LIVE_TOURNAMENT_SIMULATION_PATH = LIVE_DIR / "live_tournament_simulation.csv"
SIMULATIONS_DIR = PROJECT_ROOT / "output" / "simulations"
LIVE_TOURNAMENT_SIMULATION_FALLBACK_PATH = SIMULATIONS_DIR / "live_tournament_simulation.csv"
LIVE_GROUP_PROJECTION_PATH = LIVE_DIR / "live_group_projection.csv"
LIVE_PACKAGE_SYNC_PATHS = [
    ([FIXTURES_RESULTS_PATH], FIXTURES_RESULTS_PATH.name),
    ([LIVE_DIR / "group_standings.csv"], "group_standings.csv"),
    ([LIVE_GROUP_PROJECTION_PATH], LIVE_GROUP_PROJECTION_PATH.name),
    ([LIVE_SUMMARY_PATH], LIVE_SUMMARY_PATH.name),
    ([LIVE_EVALUATION_PATH], LIVE_EVALUATION_PATH.name),
    ([LIVE_CALIBRATION_PATH], LIVE_CALIBRATION_PATH.name),
    (
        [LIVE_TOURNAMENT_SIMULATION_PATH, LIVE_TOURNAMENT_SIMULATION_FALLBACK_PATH],
        LIVE_TOURNAMENT_SIMULATION_PATH.name,
    ),
]
LOG_COLUMNS = ["timestamp_utc", "step", "command", "status", "return_code", "message"]
BEIJING_TZ = ZoneInfo("Asia/Shanghai")


def local_api_package_live_dir() -> Path | None:
    configured = os.getenv("CUPCAST_API_PACKAGE_LIVE_DIR")
    if configured:
        return Path(configured).expanduser()

    candidates = [PROJECT_ROOT / "cupcast_api_package" / "output" / "live"]
    for ancestor in PROJECT_ROOT.parents:
        candidates.append(
            ancestor / "FIFAproject2026" / "cupcast_api_package" / "output" / "live"
        )
    return next((path for path in candidates if path.exists()), None)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def beijing_today_yyyymmdd() -> str:
    return datetime.now(BEIJING_TZ).strftime("%Y%m%d")


def append_pipeline_log(
    step: str,
    command: list[str] | str,
    status: str,
    return_code: int,
    message: str,
) -> None:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    command_text = " ".join(command) if isinstance(command, list) else command
    row = {
        "timestamp_utc": utc_now_iso(),
        "step": step,
        "command": command_text,
        "status": status,
        "return_code": return_code,
        "message": message,
    }
    if PIPELINE_LOG_PATH.exists():
        existing = pd.read_csv(PIPELINE_LOG_PATH)
        for column in LOG_COLUMNS:
            if column not in existing.columns:
                existing[column] = pd.NA
        output = pd.concat(
            [existing[LOG_COLUMNS], pd.DataFrame([row], columns=LOG_COLUMNS)],
            ignore_index=True,
        )
    else:
        output = pd.DataFrame([row], columns=LOG_COLUMNS)
    output.to_csv(PIPELINE_LOG_PATH, index=False)


def run_step(step: str, command: list[str], stop_on_failure: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    output = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
    if len(output) > 2000:
        output = output[:2000] + "... [truncated]"
    status = "success" if result.returncode == 0 else "failed"
    append_pipeline_log(
        step=step,
        command=command,
        status=status,
        return_code=result.returncode,
        message=output,
    )
    if stop_on_failure and result.returncode != 0:
        print(output)
        raise SystemExit(result.returncode)
    return result


def fixture_row_count() -> int | None:
    if not FIXTURES_RESULTS_PATH.exists():
        return None
    return len(pd.read_csv(FIXTURES_RESULTS_PATH))


def completed_match_count() -> int | None:
    if not LIVE_SUMMARY_PATH.exists():
        return None
    summary = pd.read_csv(LIVE_SUMMARY_PATH)
    if summary.empty:
        return None
    count_column = (
        "completed_matches"
        if "completed_matches" in summary.columns
        else "n_completed_matches"
        if "n_completed_matches" in summary.columns
        else None
    )
    if count_column is None:
        return None
    value = pd.to_numeric(summary[count_column].iloc[0], errors="coerce")
    if pd.isna(value):
        return None
    return int(value)


def sync_local_api_package() -> None:
    """Keep the local API package live outputs aligned with the source project."""
    package_live_dir = local_api_package_live_dir()
    if package_live_dir is None:
        append_pipeline_log(
            step="sync_local_api_package",
            command="",
            status="skipped",
            return_code=0,
            message="Package live directory not found",
        )
        return

    copied = []
    missing = []
    package_live_dir.mkdir(parents=True, exist_ok=True)
    for source_candidates, destination_name in LIVE_PACKAGE_SYNC_PATHS:
        source_path = next((path for path in source_candidates if path.exists()), None)
        if source_path is None:
            missing.append(" or ".join(str(path) for path in source_candidates))
            continue
        destination = package_live_dir / destination_name
        shutil.copy2(source_path, destination)
        copied.append(destination_name)

    status = "success" if not missing else "warning"
    message = f"Copied to local API package: {', '.join(copied) if copied else 'none'}"
    if missing:
        message += f"; missing: {', '.join(missing)}"
    append_pipeline_log(
        step="sync_local_api_package",
        command="",
        status=status,
        return_code=0,
        message=message,
    )
    print(message)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the lightweight ESPN live update and evaluation pipeline."
    )
    parser.add_argument("--skip-standings", action="store_true")
    parser.add_argument("--skip-cleanup", action="store_true")
    parser.add_argument("--skip-live-simulation", action="store_true")
    parser.add_argument("--skip-daily-beijing-table", action="store_true")
    parser.add_argument(
        "--live-sim-n-sims",
        "--live-simulations",
        dest="live_sim_n_sims",
        type=int,
        default=100000,
        help="Number of state-conditioned live tournament simulations to run.",
    )
    args = parser.parse_args()

    fetch_command = [
        sys.executable,
        "src/live/fetch_espn_worldcup_live.py",
        "--no-standings" if args.skip_standings else "--fetch-standings",
    ]
    fetch_result = run_step(
        step="fetch_espn_worldcup_live",
        command=fetch_command,
        stop_on_failure=True,
    )

    beijing_fetch_command = [
        sys.executable,
        "src/live/fetch_espn_worldcup_live.py",
        "--date",
        beijing_today_yyyymmdd(),
        "--no-standings",
    ]
    run_step(
        step="fetch_espn_worldcup_live_beijing_date",
        command=beijing_fetch_command,
        stop_on_failure=False,
    )

    if not args.skip_daily_beijing_table:
        run_step(
            step="export_daily_beijing_predictions",
            command=[sys.executable, "src/live/export_daily_beijing_predictions.py"],
            stop_on_failure=False,
        )
    else:
        append_pipeline_log(
            step="export_daily_beijing_predictions",
            command="",
            status="skipped",
            return_code=0,
            message="Skipped by --skip-daily-beijing-table",
        )

    eval_result = run_step(
        step="evaluate_live_group_stage",
        command=[sys.executable, "src/live/evaluate_live_group_stage.py"],
        stop_on_failure=False,
    )
    if eval_result.returncode != 0:
        print("Live evaluation failed; see output/live/live_pipeline_run_log.csv")
    else:
        completed = completed_match_count()
        if completed == 0:
            append_pipeline_log(
                step="evaluate_live_group_stage",
                command="",
                status="warning",
                return_code=0,
                message="Live evaluation completed with zero completed matches.",
            )

    if not args.skip_live_simulation:
        run_step(
            step="live_tournament_simulation",
            command=[
                sys.executable,
                "src/live/live_tournament_simulation.py",
                "--n-sims",
                str(args.live_sim_n_sims),
            ],
            stop_on_failure=False,
        )
    else:
        append_pipeline_log(
            step="live_tournament_simulation",
            command="",
            status="skipped",
            return_code=0,
            message="Skipped by --skip-live-simulation",
        )

    if not args.skip_cleanup:
        run_step(
            step="cleanup_legacy_outputs",
            command=[sys.executable, "scripts/cleanup_legacy_outputs.py"],
            stop_on_failure=False,
        )
    else:
        append_pipeline_log(
            step="cleanup_legacy_outputs",
            command="",
            status="skipped",
            return_code=0,
            message="Skipped by --skip-cleanup",
        )

    sync_local_api_package()

    rows = fixture_row_count()
    completed = completed_match_count()
    fixtures = pd.read_csv(FIXTURES_RESULTS_PATH) if FIXTURES_RESULTS_PATH.exists() else pd.DataFrame()
    scheduled_count = int(fixtures["status"].eq("scheduled").sum()) if "status" in fixtures else 0
    in_progress_count = int(fixtures["status"].eq("in_progress").sum()) if "status" in fixtures else 0
    final_count = int(fixtures["status"].eq("final").sum()) if "status" in fixtures else 0
    evaluation_status = "missing live summary"
    if LIVE_SUMMARY_PATH.exists():
        summary = pd.read_csv(LIVE_SUMMARY_PATH)
        if not summary.empty and "status" in summary.columns:
            evaluation_status = str(summary["status"].iloc[0])
    print(f"ESPN fetch succeeded: {fetch_result.returncode == 0}")
    print(f"fixtures_results rows: {rows if rows is not None else 'missing'}")
    print(f"scheduled matches: {scheduled_count}")
    print(f"in-progress matches: {in_progress_count}")
    print(f"final matches: {final_count}")
    print(
        "completed matches: "
        f"{completed if completed is not None else 'missing live summary'}"
    )
    print(f"evaluation status: {evaluation_status}")
    print("Output files:")
    for path in [
        FIXTURES_RESULTS_PATH,
        LIVE_SUMMARY_PATH,
        LIVE_EVALUATION_PATH,
        LIVE_CALIBRATION_PATH,
        PIPELINE_LOG_PATH,
        PROJECT_ROOT / "output" / "predictions" / "group_stage_games_today_beijing_top5.csv",
        LIVE_TOURNAMENT_SIMULATION_PATH,
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
