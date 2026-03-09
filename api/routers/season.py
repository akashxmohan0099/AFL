"""Season endpoints."""
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException, Path, Query

from api.services import season_service

router = APIRouter(prefix="/api/season", tags=["season"])


@router.get("/{year}/summary")
def season_summary(year: int = Path(..., ge=2010, le=2030)):
    return season_service.get_season_summary(year)


@router.get("/{year}/matches")
def season_matches(year: int = Path(..., ge=2010, le=2030)):
    return season_service.get_season_matches(year)


@router.get("/{year}/upcoming")
def upcoming_matches(year: int = Path(..., ge=2010, le=2030)):
    return season_service.get_upcoming_matches(year)


@router.get("/{year}/schedule")
def season_schedule(year: int = Path(..., ge=2010, le=2030)):
    return season_service.get_season_schedule(year)


@router.get("/{year}/accuracy")
def round_accuracy(year: int = Path(..., ge=2010, le=2030)):
    return season_service.get_round_accuracy(year)


@router.get("/{year}/accuracy/breakdown")
def accuracy_breakdown(year: int = Path(..., ge=2010, le=2030)):
    return season_service.get_accuracy_breakdown(year)


@router.get("/{year}/predictions-history")
def predictions_history(year: int = Path(..., ge=2010, le=2030)):
    return season_service.get_predictions_history(year)


@router.get("/{year}/match/{match_id}")
def match_comparison(
    year: int = Path(..., ge=2010, le=2030),
    match_id: int = Path(...),
    round_number: Optional[int] = Query(None),
    home_team: Optional[str] = Query(None),
    away_team: Optional[str] = Query(None),
):
    result = season_service.get_match_comparison(
        match_id, year=year, round_number=round_number,
        home_team=home_team, away_team=away_team,
    )
    if result is None:
        raise HTTPException(404, "Match not found")
    return result
