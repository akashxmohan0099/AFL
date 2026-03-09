"""League-wide endpoints: leaders, stats."""
from __future__ import annotations

from fastapi import APIRouter, Query

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
