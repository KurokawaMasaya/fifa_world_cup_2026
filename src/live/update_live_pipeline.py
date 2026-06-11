from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

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
LOG_COLUMNS = ["timestamp_utc", "step", "command", "status", "return_code", "message"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the lightweight ESPN live update and evaluation pipeline."
    )
    parser.add_argument("--skip-standings", action="store_true")
    parser.add_argument("--skip-cleanup", action="store_true")
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
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
