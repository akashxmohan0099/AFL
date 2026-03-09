"""Security helpers for auth and rate limiting."""
from __future__ import annotations

import hmac
import math
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import Request

from api.settings import APISettings


class InMemoryRateLimiter:
    """Simple fixed-window rate limiter keyed by client identity."""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = int(max_requests)
        self.window_seconds = int(window_seconds)
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str) -> tuple[bool, int, int]:
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()

            if len(events) >= self.max_requests:
                retry_after = max(1, math.ceil(self.window_seconds - (now - events[0])))
                return False, 0, retry_after

            events.append(now)
            remaining = max(0, self.max_requests - len(events))
            return True, remaining, self.window_seconds


def get_client_identifier(request: Request, settings: APISettings) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if settings.trust_x_forwarded_for and forwarded_for:
        return forwarded_for.split(",", 1)[0].strip() or "unknown"

    if request.client and request.client.host:
        return request.client.host

    return "unknown"


def is_authorized(request: Request, settings: APISettings) -> bool:
    if not settings.auth_enabled:
        return True

    provided = request.headers.get(settings.api_key_header, "")
    if not provided:
        return False

    return hmac.compare_digest(provided, settings.api_key or "")
