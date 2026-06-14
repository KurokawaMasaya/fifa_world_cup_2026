from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class Metadata(BaseModel):
    source_file: str
    last_modified: str
    row_count: int


class FreshnessMetadata(Metadata):
    last_updated: str | None = None
    file_last_modified: str
    age_minutes: float
    is_stale: bool


class DatasetResponse(BaseModel):
    source_file: str
    row_count: int
    file_last_modified: str
    data: list[dict[str, Any]]
    last_modified: str | None = None
    last_updated: str | None = None
    age_minutes: float | None = None
    is_stale: bool | None = None


class HealthResponse(BaseModel):
    status: str
    available_files: dict[str, str]
    missing_files: dict[str, str]
    timestamp_utc: str


class LiveScoreResponse(BaseModel):
    provider: str
    provider_league: str
    provider_fetched_at_utc: str
    cache_seconds: int
    row_count: int
    data: list[dict[str, Any]]


class LiveScoreDetailResponse(BaseModel):
    provider: str
    provider_league: str
    provider_fetched_at_utc: str
    cache_seconds: int
    data: dict[str, Any]


class MessageResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: str | None = None
