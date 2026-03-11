"""League-wide endpoints: leaders, ladder, stats, team profiles."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.services import league_service

router = APIRouter(prefix="/api/league", tags=["league"])


@router.get("/leaders")
def league_leaders(
    stat: str = Query("GL", description="Stat column: GL, DI, MK, TK, KI, HB, HO, FF, FA, CP, UP, CL"),
    year: int = Query(2025),
    limit: int = Query(50, le=200),
    min_games: int = Query(5, ge=1),
):
    return league_service.get_leaders(stat=stat, year=year, limit=limit, min_games=min_games)


@router.get("/ladder")
def league_ladder(
    year: int = Query(2025, description="Season year"),
):
    """AFL ladder / standings for a given season."""
    return league_service.get_ladder(year=year)


@router.get("/team/{team_name}")
def team_profile(
    team_name: str,
    year: int = Query(2026, description="Season year"),
):
    """Full team profile: record, form, top players, averages, home/away split."""
    result = league_service.get_team_profile(team=team_name, year=year)
    if result is None:
        raise HTTPException(404, f"Team '{team_name}' not found")
    return result
