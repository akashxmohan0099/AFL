"""Odds comparison endpoints."""

from fastapi import APIRouter

from api.services import odds_service

router = APIRouter(prefix="/api/odds", tags=["odds"])


@router.get("/game/{year}/{round_num}")
def get_game_odds(year: int, round_num: int):
    return odds_service.get_game_odds(year, round_num)


@router.get("/players/{year}/{round_num}")
def get_player_odds(year: int, round_num: int):
    return odds_service.get_player_odds(year, round_num)
