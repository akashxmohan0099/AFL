"""Player endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.services import player_service

router = APIRouter(prefix="/api/players", tags=["players"])


@router.get("/directory")
def list_players(year: int = 2025):
    return player_service.list_players(year)


@router.get("/search")
def search_players(q: str = Query(..., min_length=1, max_length=100), limit: int = Query(20, le=100)):
    return player_service.search_players(q, limit)


@router.get("/{player_id}")
def get_player(player_id: str):
    profile = player_service.get_player_profile(player_id)
    if profile is None:
        raise HTTPException(404, "Player not found")
    return profile


@router.get("/{player_id}/games")
def get_player_games(
    player_id: str,
    year: Optional[int] = None,
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
):
    return player_service.get_player_games(player_id, year, limit, offset)


@router.get("/{player_id}/predictions")
def get_player_predictions(player_id: str, year: int = 2025):
    return player_service.get_player_predictions(player_id, year)
