"""Venue endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.services import venue_service

router = APIRouter(prefix="/api/venues", tags=["venues"])


@router.get("")
def list_venues():
    return venue_service.get_all_venues()


@router.get("/{venue_name}")
def venue_detail(venue_name: str):
    result = venue_service.get_venue_detail(venue_name)
    if result is None:
        raise HTTPException(404, f"Venue '{venue_name}' not found")
    return result
