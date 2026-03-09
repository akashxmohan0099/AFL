"""Rounds and predictions endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from api.services import prediction_service

router = APIRouter(prefix="/api/rounds", tags=["rounds"])


@router.get("")
def list_rounds(year: Optional[int] = None):
    return prediction_service.list_available_rounds(year)


@router.get("/{year}/{round_num}")
def get_round(year: int, round_num: int):
    return prediction_service.get_round_predictions(year, round_num)
