from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import HTTPException


ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard"
DEFAULT_LEAGUE = os.getenv("LIVE_SCORE_ESPN_LEAGUE", "fifa.world")
DEFAULT_CACHE_SECONDS = int(os.getenv("LIVE_SCORE_CACHE_SECONDS", "15"))

_scoreboard_cache: dict[str, tuple[float, str, dict[str, Any]]] = {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "CupCastLiveScoreAPI/1.0"})
    try:
        with urlopen(request, timeout=10) as response:
            return json.load(response)
    except HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Live-score provider returned HTTP {exc.code}",
        ) from exc
    except (TimeoutError, URLError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Live-score provider request failed: {exc}",
        ) from exc


def _scoreboard_url(league: str, date: str | None = None) -> str:
    url = ESPN_SCOREBOARD_URL.format(league=league)
    if not date:
        return url
    compact_date = date.replace("-", "")
    return f"{url}?{urlencode({'dates': compact_date})}"


def _cached_provider_payload(league: str, date: str | None = None) -> tuple[str, dict[str, Any]]:
    cache_seconds = max(DEFAULT_CACHE_SECONDS, 0)
    cache_key = f"{league}:{date or 'today'}"
    cached = _scoreboard_cache.get(cache_key)
    if cached and time.time() - cached[0] < cache_seconds:
        return cached[1], cached[2]

    payload = _fetch_json(_scoreboard_url(league, date))
    provider_fetched_at = _utc_now_iso()
    _scoreboard_cache[cache_key] = (time.time(), provider_fetched_at, payload)
    return provider_fetched_at, payload


def _parse_score(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _team_payload(competitor: dict[str, Any]) -> dict[str, Any]:
    team = competitor.get("team") or {}
    return {
        "id": str(competitor.get("id") or team.get("id") or ""),
        "name": team.get("displayName") or team.get("name") or "",
        "abbreviation": team.get("abbreviation"),
        "logo": team.get("logo"),
        "score": _parse_score(competitor.get("score")),
    }


def _event_to_live_score(
    event: dict[str, Any],
    *,
    league: str,
    provider_fetched_at: str,
) -> dict[str, Any]:
    competition = (event.get("competitions") or [{}])[0]
    status = competition.get("status") or event.get("status") or {}
    status_type = status.get("type") or {}
    competitors = competition.get("competitors") or []
    home = next((item for item in competitors if item.get("homeAway") == "home"), None)
    away = next((item for item in competitors if item.get("homeAway") == "away"), None)

    if not home and competitors:
        home = competitors[0]
    if not away and len(competitors) > 1:
        away = competitors[1]

    home_team = _team_payload(home or {})
    away_team = _team_payload(away or {})
    venue = competition.get("venue") or event.get("venue") or {}

    return {
        "match_id": str(event.get("id") or competition.get("id") or ""),
        "provider": "espn",
        "provider_league": league,
        "kickoff_time_utc": event.get("date") or competition.get("date"),
        "name": event.get("name"),
        "short_name": event.get("shortName"),
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_team["score"],
        "away_score": away_team["score"],
        "status": status_type.get("state"),
        "status_detail": status_type.get("detail") or status_type.get("description"),
        "status_description": status_type.get("description"),
        "is_completed": bool(status_type.get("completed")),
        "period": status.get("period"),
        "clock_seconds": status.get("clock"),
        "minute": status.get("displayClock"),
        "venue": venue.get("fullName"),
        "city": (venue.get("address") or {}).get("city"),
        "country": (venue.get("address") or {}).get("country"),
        "provider_fetched_at_utc": provider_fetched_at,
    }


def get_live_scores(league: str = DEFAULT_LEAGUE, date: str | None = None) -> dict[str, Any]:
    provider_fetched_at, payload = _cached_provider_payload(league, date)
    scores = [
        _event_to_live_score(event, league=league, provider_fetched_at=provider_fetched_at)
        for event in payload.get("events", [])
    ]

    return {
        "provider": "espn",
        "provider_league": league,
        "provider_fetched_at_utc": provider_fetched_at,
        "cache_seconds": DEFAULT_CACHE_SECONDS,
        "row_count": len(scores),
        "data": scores,
    }


def get_live_score(match_id: str, league: str = DEFAULT_LEAGUE, date: str | None = None) -> dict[str, Any]:
    scores = get_live_scores(league=league, date=date)
    match = next((item for item in scores["data"] if item["match_id"] == str(match_id)), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"No live score found for match_id={match_id}")
    return {
        "provider": scores["provider"],
        "provider_league": scores["provider_league"],
        "provider_fetched_at_utc": scores["provider_fetched_at_utc"],
        "cache_seconds": scores["cache_seconds"],
        "data": match,
    }
