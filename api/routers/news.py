"""News endpoints — injuries, team changes, match context, and intel feed."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from api.services import news_service

router = APIRouter(prefix="/api/news", tags=["news"])


@router.get("/injuries")
def get_injuries():
    """Current injury list for all teams."""
    return news_service.get_injuries()


@router.get("/team-changes")
def get_team_changes(year: int = Query(2026, ge=2020, le=2030)):
    """Team selections / ins & outs for a season."""
    return news_service.get_team_changes(year)


@router.get("/match/{home_team}/{away_team}")
def get_match_news(home_team: str, away_team: str):
    """Combined news context for a specific matchup — injuries, changes, debutants."""
    return news_service.get_match_news(home_team, away_team)


@router.get("/round/{year}/{round_number}")
def get_round_news(year: int, round_number: int):
    """All news for a specific round — injuries + team changes for all teams playing."""
    return news_service.get_round_news(year, round_number)


# ---------------------------------------------------------------------------
# Intel feed endpoints
# ---------------------------------------------------------------------------


@router.get("/intel/feed")
def get_intel_feed(
    signal_type: Optional[str] = Query(None, description="Filter: injury, suspension, form, tactical, selection, prediction, general"),
    team: Optional[str] = Query(None, description="Filter by team name"),
    min_relevance: float = Query(0.0, ge=0.0, le=1.0, description="Min relevance score"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Paginated intel feed with optional filters."""
    return news_service.get_intel_feed(
        signal_type=signal_type,
        team=team,
        min_relevance=min_relevance,
        limit=limit,
        offset=offset,
    )


@router.get("/intel/summary")
def get_intel_summary():
    """High-level intel summary — breaking news, top signals, sentiment."""
    return news_service.get_intel_summary()


@router.get("/intel/team/{team}")
def get_team_intel(team: str):
    """All intel signals for a specific team."""
    return news_service.get_team_intel(team)
