"""Match endpoints."""

from fastapi import APIRouter, HTTPException

from api.services import match_service, simulation_service

router = APIRouter(prefix="/api/matches", tags=["matches"])


@router.get("/detail/{match_id}")
def get_match_detail(match_id: int):
    detail = match_service.get_match_detail(match_id)
    if detail is None:
        raise HTTPException(404, "Match not found")
    return detail


@router.get("/detail/{match_id}/simulation")
def get_match_simulation(match_id: int):
    result = simulation_service.get_match_simulation(match_id)
    if result is None:
        raise HTTPException(404, "Simulation not available for this match")
    return result


@router.get("/{year}/{round_num}")
def get_matches(year: int, round_num: int):
    return match_service.get_matches_for_round(year, round_num)
