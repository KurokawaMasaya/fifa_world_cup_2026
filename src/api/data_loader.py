from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(os.getenv("CUPCAST_OUTPUT_DIR", PROJECT_ROOT / "output")).expanduser()


def _display_path(path: Path) -> str:
    for base in [PROJECT_ROOT, OUTPUT_DIR.parent]:
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def _ensure_output_path(path: Path) -> Path:
    """Keep the API read-only and scoped to generated output files."""
    resolved = path.resolve()
    output_root = OUTPUT_DIR.resolve()
    if output_root not in resolved.parents and resolved != output_root:
        raise HTTPException(status_code=403, detail="API can only read files under output/")
    return resolved


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def read_csv_or_503(path: Path) -> pd.DataFrame:
    """Read a CSV file or raise HTTP 503 with a clear missing/unreadable message."""
    safe_path = _ensure_output_path(path)
    if not safe_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Required output file is missing: {_display_path(safe_path)}",
        )
    try:
        return pd.read_csv(safe_path)
    except Exception as exc:  # pragma: no cover - defensive API boundary.
        raise HTTPException(
            status_code=503,
            detail=f"Could not read output file {_display_path(safe_path)}: {exc}",
        ) from exc


def csv_records(path: Path) -> list[dict[str, Any]]:
    """Read a CSV and return JSON-safe row dictionaries."""
    df = read_csv_or_503(path)
    return dataframe_records(df)


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return JSON-safe row dictionaries for an already-loaded DataFrame."""
    return [
        {key: _json_safe(value) for key, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def latest_csv(directory: Path, pattern: str = "*.csv") -> Path:
    """Return the latest CSV by modification time from an output subdirectory."""
    safe_dir = _ensure_output_path(directory)
    if not safe_dir.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Output directory is missing: {_display_path(safe_dir)}",
        )
    files = sorted(safe_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not files:
        raise HTTPException(
            status_code=503,
            detail=f"No CSV files found under {_display_path(safe_dir)}",
        )
    return files[0]


def file_metadata(path: Path) -> dict[str, Any]:
    """Return source path, modification time, and row count for a CSV."""
    safe_path = _ensure_output_path(path)
    df = read_csv_or_503(safe_path)
    modified = datetime.fromtimestamp(safe_path.stat().st_mtime, tz=timezone.utc)
    return {
        "source_file": _display_path(safe_path),
        "file_last_modified": modified.isoformat(),
        "row_count": int(len(df)),
    }


def freshness_metadata(path: Path, stale_after_hours: float = 6) -> dict[str, Any]:
    """Return live-data freshness metadata based on file modified time."""
    safe_path = _ensure_output_path(path)
    metadata = file_metadata(safe_path)
    modified = datetime.fromtimestamp(safe_path.stat().st_mtime, tz=timezone.utc)
    age_minutes = (datetime.now(timezone.utc) - modified).total_seconds() / 60
    df = read_csv_or_503(safe_path)
    last_updated = None
    if "last_updated" in df.columns and not df.empty:
        values = df["last_updated"].dropna().astype(str)
        if not values.empty:
            last_updated = values.max()
    return {
        **metadata,
        "last_updated": last_updated,
        "file_last_modified": metadata["file_last_modified"],
        "age_minutes": round(age_minutes, 2),
        "is_stale": age_minutes > stale_after_hours * 60,
    }


def dataset_response(path: Path, *, freshness: bool = False) -> dict[str, Any]:
    metadata = freshness_metadata(path) if freshness else file_metadata(path)
    return {**metadata, "data": csv_records(path)}


def dataframe_response(path: Path, df: pd.DataFrame, *, freshness: bool = False) -> dict[str, Any]:
    metadata = freshness_metadata(path) if freshness else file_metadata(path)
    return {**metadata, "row_count": int(len(df)), "data": dataframe_records(df)}


def existing_and_missing(paths: dict[str, Path]) -> tuple[dict[str, str], dict[str, str]]:
    """Return available and missing configured output files for health checks."""
    available = {}
    missing = {}
    for label, path in paths.items():
        safe_path = _ensure_output_path(path)
        rel_path = _display_path(safe_path)
        if safe_path.exists():
            available[label] = rel_path
        else:
            missing[label] = rel_path
    return available, missing
