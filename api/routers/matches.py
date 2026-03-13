"""Match endpoints."""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from api.services import match_service, simulation_service

router = APIRouter(prefix="/api/matches", tags=["matches"])


@router.get("/detail/{match_id}")
def get_match_detail(match_id: int):
    detail = match_service.get_match_detail(match_id)
    if detail is None:
        raise HTTPException(404, "Match not found")
    return detail


@router.get("/detail/{match_id}/simulation")
def get_match_simulation(
    match_id: int,
    round: Optional[int] = Query(None),
    home: Optional[str] = Query(None),
    away: Optional[str] = Query(None),
):
    result = simulation_service.get_match_simulation(
        match_id, home_team=home, away_team=away, round_num=round
    )
    if result is None:
        raise HTTPException(404, "Simulation not available for this match")
    return result


@router.get("/simulations/{year}/{round_num}")
def get_round_simulations(year: int, round_num: int):
    return simulation_service.get_round_simulations(year, round_num)


@router.get("/{year}/{round_num}")
def get_matches(year: int, round_num: int):
    return match_service.get_matches_for_round(year, round_num)
