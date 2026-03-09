"""Team endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.services import team_service

router = APIRouter(prefix="/api/teams", tags=["teams"])


@router.get("")
def list_teams(year: int = Query(2025)):
    return team_service.list_teams(year)


@router.get("/{team_name}")
def get_team(team_name: str, year: int = Query(2025)):
    result = team_service.get_team_detail(team_name, year)
    if result is None:
        raise HTTPException(404, f"Team '{team_name}' not found")
    return result
