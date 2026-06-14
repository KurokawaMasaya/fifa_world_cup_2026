from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.data_loader import (
    OUTPUT_DIR,
    dataframe_records,
    dataframe_response,
    dataset_response,
    existing_and_missing,
    file_metadata,
    freshness_metadata,
    latest_csv,
    read_csv_or_503,
)
from src.api.live_scores import DEFAULT_LEAGUE, get_live_score, get_live_scores
from src.api.schemas import DatasetResponse, HealthResponse, LiveScoreDetailResponse, LiveScoreResponse


PREDICTIONS_CLEAN_PATH = OUTPUT_DIR / "predictions" / "group_stage_predictions_clean.csv"
PREDICTIONS_DETAIL_PATH = (
    OUTPUT_DIR / "predictions" / "group_stage_predictions_v2_uncertainty_tuned.csv"
)
LIVE_RESULTS_PATH = OUTPUT_DIR / "live" / "fixtures_results.csv"
LIVE_STANDINGS_PATH = OUTPUT_DIR / "live" / "group_standings.csv"
LIVE_EVALUATION_PATH = OUTPUT_DIR / "live" / "worldcup_group_stage_live_summary.csv"
LIVE_EVALUATION_DETAIL_PATH = OUTPUT_DIR / "live" / "worldcup_group_stage_live_evaluation.csv"
LIVE_CALIBRATION_PATH = OUTPUT_DIR / "live" / "worldcup_group_stage_live_calibration.csv"
LIVE_GROUP_PROJECTION_PATH = OUTPUT_DIR / "live" / "live_group_projection.csv"
LIVE_TOURNAMENT_SIMULATION_PATH = OUTPUT_DIR / "live" / "live_tournament_simulation.csv"
MODEL_SUMMARY_PATH = OUTPUT_DIR / "diagnostics" / "model_performance_summary.csv"
SIMULATIONS_DIR = OUTPUT_DIR / "simulations"
BRACKETS_DIR = OUTPUT_DIR / "brackets"

HEALTH_FILES = {
    "group_stage_predictions_clean": PREDICTIONS_CLEAN_PATH,
    "group_stage_predictions_detail": PREDICTIONS_DETAIL_PATH,
    "live_results": LIVE_RESULTS_PATH,
    "live_standings": LIVE_STANDINGS_PATH,
    "live_evaluation": LIVE_EVALUATION_PATH,
    "live_evaluation_detail": LIVE_EVALUATION_DETAIL_PATH,
    "live_calibration": LIVE_CALIBRATION_PATH,
    "live_group_projection": LIVE_GROUP_PROJECTION_PATH,
    "live_tournament_simulation": LIVE_TOURNAMENT_SIMULATION_PATH,
    "model_summary": MODEL_SUMMARY_PATH,
}


app = FastAPI(
    title="CupCast Read-Only API",
    description="Read-only JSON API over existing FIFAproject2026 output CSV files.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _latest_simulation_path() -> Path:
    return latest_csv(SIMULATIONS_DIR, pattern="*.csv")


def _latest_bracket_path() -> Path:
    return latest_csv(BRACKETS_DIR, pattern="*.csv")


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    available, missing = existing_and_missing(HEALTH_FILES)
    for label, resolver in {
        "latest_simulation": _latest_simulation_path,
        "latest_bracket": _latest_bracket_path,
    }.items():
        try:
            path = resolver()
            available[label] = str(path.relative_to(OUTPUT_DIR.parent))
        except HTTPException:
            missing[label] = f"output/{label.replace('latest_', '')}s/*.csv"
    status = "ok" if not missing else "degraded"
    return {
        "status": status,
        "available_files": available,
        "missing_files": missing,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model/summary", response_model=DatasetResponse, response_model_exclude_none=True)
def model_summary() -> dict:
    return dataset_response(MODEL_SUMMARY_PATH)


@app.get("/predictions/group-stage", response_model=DatasetResponse, response_model_exclude_none=True)
def group_stage_predictions() -> dict:
    return dataset_response(PREDICTIONS_CLEAN_PATH)


@app.get("/predictions/group-stage/{match_id}")
def group_stage_prediction(match_id: str) -> dict:
    df = read_csv_or_503(PREDICTIONS_CLEAN_PATH)
    if "match_id" not in df.columns:
        raise HTTPException(status_code=503, detail="Prediction file is missing match_id column")
    match = df.loc[df["match_id"].astype(str) == str(match_id)]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"No group-stage prediction for match_id={match_id}")
    return {
        **file_metadata(PREDICTIONS_CLEAN_PATH),
        "data": dataframe_records(match)[0],
    }


@app.get("/predictions/matches", response_model=DatasetResponse, response_model_exclude_none=True)
def prediction_matches() -> dict:
    df = read_csv_or_503(PREDICTIONS_CLEAN_PATH)
    required = {"match_id", "group", "team_a", "team_b"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Prediction file is missing columns: {sorted(missing)}",
        )
    index = df[["match_id", "group", "team_a", "team_b"]].copy()
    index["match_id"] = index["match_id"].astype(str)
    index["label"] = index["team_a"].astype(str) + " vs " + index["team_b"].astype(str)
    return dataframe_response(PREDICTIONS_CLEAN_PATH, index)


@app.get("/predictions/teams", response_model=DatasetResponse, response_model_exclude_none=True)
def prediction_teams() -> dict:
    df = read_csv_or_503(PREDICTIONS_CLEAN_PATH)
    required = {"group", "team_a", "team_b"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Prediction file is missing columns: {sorted(missing)}",
        )
    teams = pd.concat(
        [
            df[["team_a", "group"]].rename(columns={"team_a": "team"}),
            df[["team_b", "group"]].rename(columns={"team_b": "team"}),
        ],
        ignore_index=True,
    )
    teams = teams.drop_duplicates().sort_values(["group", "team"]).reset_index(drop=True)
    return dataframe_response(PREDICTIONS_CLEAN_PATH, teams)


@app.get("/live/results", response_model=DatasetResponse, response_model_exclude_none=True)
def live_results() -> dict:
    return dataset_response(LIVE_RESULTS_PATH, freshness=True)


@app.get("/live/scores", response_model=LiveScoreResponse)
def live_scores(league: str = DEFAULT_LEAGUE, date: str | None = None) -> dict:
    """Return in-game scores from the live-score provider.

    `date` accepts YYYY-MM-DD or YYYYMMDD. Clients should poll this endpoint
    every 10-30 seconds for live match scoreboards.
    """
    return get_live_scores(league=league, date=date)


@app.get("/live/scores/{match_id}", response_model=LiveScoreDetailResponse)
def live_score(match_id: str, league: str = DEFAULT_LEAGUE, date: str | None = None) -> dict:
    return get_live_score(match_id=match_id, league=league, date=date)


@app.get("/live/standings", response_model=DatasetResponse, response_model_exclude_none=True)
def live_standings() -> dict:
    return dataset_response(LIVE_STANDINGS_PATH, freshness=True)


@app.get("/live/evaluation", response_model=DatasetResponse, response_model_exclude_none=True)
def live_evaluation() -> dict:
    return dataset_response(LIVE_EVALUATION_PATH, freshness=True)


@app.get("/live/evaluation/detail", response_model=DatasetResponse, response_model_exclude_none=True)
def live_evaluation_detail() -> dict:
    return dataset_response(LIVE_EVALUATION_DETAIL_PATH, freshness=True)


@app.get("/live/calibration", response_model=DatasetResponse, response_model_exclude_none=True)
def live_calibration() -> dict:
    return dataset_response(LIVE_CALIBRATION_PATH, freshness=True)


@app.get("/live/group-projection", response_model=DatasetResponse, response_model_exclude_none=True)
def live_group_projection() -> dict:
    return dataset_response(LIVE_GROUP_PROJECTION_PATH, freshness=True)


@app.get("/live/tournament-simulation", response_model=DatasetResponse, response_model_exclude_none=True)
def live_tournament_simulation() -> dict:
    return dataset_response(LIVE_TOURNAMENT_SIMULATION_PATH, freshness=True)


@app.get("/simulation/tournament", response_model=DatasetResponse, response_model_exclude_none=True)
def tournament_simulation() -> dict:
    return dataset_response(_latest_simulation_path())


@app.get("/simulation/teams", response_model=DatasetResponse, response_model_exclude_none=True)
def simulation_teams() -> dict:
    path = _latest_simulation_path()
    df = read_csv_or_503(path)
    if "team_name" not in df.columns:
        raise HTTPException(status_code=503, detail="Simulation file is missing team_name column")
    teams = (
        df[["team_name"]]
        .drop_duplicates()
        .sort_values("team_name")
        .rename(columns={"team_name": "team"})
        .reset_index(drop=True)
    )
    return dataframe_response(path, teams)


@app.get("/simulation/team/{team}")
def simulation_team(team: str) -> dict:
    path = _latest_simulation_path()
    df = read_csv_or_503(path)
    if "team_name" not in df.columns:
        raise HTTPException(status_code=503, detail="Simulation file is missing team_name column")
    team_key = team.strip().casefold()
    match = df.loc[df["team_name"].astype(str).str.casefold() == team_key]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"No simulation row found for team={team}")
    return {
        **file_metadata(path),
        "data": dataframe_records(match)[0],
    }


@app.get("/bracket/sample", response_model=DatasetResponse, response_model_exclude_none=True)
def bracket_sample() -> dict:
    return dataset_response(_latest_bracket_path())
