"""Admin endpoints — cache reload, health diagnostics."""
from __future__ import annotations

import hmac
import logging
import os
import time

from fastapi import APIRouter, HTTPException, Request

from api.data_loader import DataCache

logger = logging.getLogger("api.admin")

router = APIRouter(prefix="/api/admin", tags=["admin"])


def _check_admin_key(request: Request) -> None:
    """Verify the admin key from header or env."""
    admin_key = os.getenv("AFL_ADMIN_KEY", "")
    if not admin_key:
        # No admin key configured — reject all requests
        raise HTTPException(status_code=403, detail="Admin endpoint not configured")

    provided = request.headers.get("X-Admin-Key", "")
    if not provided or not hmac.compare_digest(provided, admin_key):
        raise HTTPException(status_code=401, detail="Invalid admin key")


@router.post("/reload")
def reload_cache(request: Request):
    """Reload all cached data from disk. Called after pipeline updates."""
    _check_admin_key(request)

    logger.info("Admin reload triggered")
    start = time.perf_counter()

    cache = DataCache.get()
    cache.load_all()

    duration = time.perf_counter() - start
    logger.info("Cache reload completed in %.2fs", duration)

    return {
        "status": "reloaded",
        "duration_seconds": round(duration, 2),
        "player_games": len(cache.player_games),
        "matches": len(cache.matches),
    }
