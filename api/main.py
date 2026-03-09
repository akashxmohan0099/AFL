"""
AFL Prediction Platform — FastAPI Application
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api.data_loader import DataCache
from api.observability import RuntimeMetrics, configure_logging
from api.security import InMemoryRateLimiter, get_client_identifier, is_authorized
from api.settings import APISettings

logger = logging.getLogger("api.request")


def create_app(settings: APISettings | None = None) -> FastAPI:
    settings = settings or APISettings.from_env()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.runtime_metrics = RuntimeMetrics()
        app.state.rate_limiter = (
            InMemoryRateLimiter(
                max_requests=settings.rate_limit_requests,
                window_seconds=settings.rate_limit_window_seconds,
            )
            if settings.rate_limit_enabled
            else None
        )

        if settings.load_cache_on_startup:
            cache = DataCache.get()
            cache.load_all()
        yield

    app = FastAPI(
        title="AFL Prediction Platform",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allow_origins),
        allow_methods=list(settings.cors_allow_methods),
        allow_headers=list(settings.cors_allow_headers),
        allow_credentials=settings.cors_allow_credentials,
    )

    @app.middleware("http")
    async def protection_middleware(request: Request, call_next):
        path = request.url.path
        app_settings: APISettings = request.app.state.settings
        runtime_metrics: RuntimeMetrics | None = getattr(request.app.state, "runtime_metrics", None)
        rate_limiter: InMemoryRateLimiter | None = getattr(request.app.state, "rate_limiter", None)
        client_id = get_client_identifier(request, app_settings)
        request_id = request.headers.get("X-Request-ID", "").strip() or uuid.uuid4().hex
        start = time.perf_counter()
        request.state.request_id = request_id
        if runtime_metrics is not None:
            runtime_metrics.start_request()

        def _route_label() -> str:
            route = request.scope.get("route")
            route_path = getattr(route, "path", None) or path
            return f"{request.method} {route_path}"

        def _finalize(response, status_code: int):
            duration_ms = (time.perf_counter() - start) * 1000
            if runtime_metrics is not None:
                runtime_metrics.finish_request(_route_label(), status_code, duration_ms)
            level = logging.WARNING if status_code >= 500 else logging.INFO
            if path == "/api/health" and status_code < 500:
                level = logging.DEBUG
            logger.log(
                level,
                "request_complete method=%s path=%s status=%s duration_ms=%.2f client=%s request_id=%s",
                request.method,
                path,
                status_code,
                duration_ms,
                client_id,
                request_id,
            )
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
            return response

        rate_limit_headers = {}
        if path.startswith("/api/") and not app_settings.is_exempt_path(path):
            if app_settings.auth_enabled and not is_authorized(request, app_settings):
                return _finalize(JSONResponse(status_code=401, content={"detail": "Unauthorized"}), 401)

            if app_settings.rate_limit_enabled and rate_limiter is not None:
                allowed, remaining, reset_after = rate_limiter.check(client_id)
                rate_limit_headers = {
                    "X-RateLimit-Limit": str(app_settings.rate_limit_requests),
                    "X-RateLimit-Remaining": str(remaining),
                    "X-RateLimit-Window": str(app_settings.rate_limit_window_seconds),
                }
                if not allowed:
                    rate_limit_headers["Retry-After"] = str(reset_after)
                    return _finalize(
                        JSONResponse(
                            status_code=429,
                            content={"detail": "Rate limit exceeded"},
                            headers=rate_limit_headers,
                        ),
                        429,
                    )

        if path.startswith("/api/") and path != "/api/health":
            cache = DataCache.get()
            if not getattr(cache, "is_loaded", True):
                cache.load_all()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            if runtime_metrics is not None:
                runtime_metrics.finish_request(_route_label(), 500, duration_ms)
            logger.exception(
                "request_failed method=%s path=%s duration_ms=%.2f client=%s request_id=%s",
                request.method,
                path,
                duration_ms,
                client_id,
                request_id,
            )
            raise
        for header, value in rate_limit_headers.items():
            response.headers[header] = value
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("X-Frame-Options", "DENY")
        return _finalize(response, response.status_code)

    # Import and register routers
    from api.routers import players, rounds, matches, odds, metrics, season, venues, teams, league, news  # noqa: E402

    app.include_router(players.router)
    app.include_router(rounds.router)
    app.include_router(matches.router)
    app.include_router(odds.router)
    app.include_router(metrics.router)
    app.include_router(season.router)
    app.include_router(venues.router)
    app.include_router(teams.router)
    app.include_router(league.router)
    app.include_router(news.router)

    @app.get("/api/health")
    def health():
        cache = DataCache.get()
        pg_count = len(cache.player_games)
        m_count = len(cache.matches)
        status = "ok" if pg_count > 0 and m_count > 0 else "degraded"
        result = {
            "status": status,
            "player_games": pg_count,
            "matches": m_count,
            "experiments": len(cache.experiments),
            "cache_loaded": getattr(cache, "is_loaded", False),
            "auth_enabled": settings.auth_enabled,
            "rate_limit_enabled": settings.rate_limit_enabled,
        }
        # Latest data date from matches
        if m_count > 0:
            import pandas as pd

            latest = cache.matches["date"].dropna().max()
            if pd.notna(latest):
                result["latest_data"] = str(latest)[:10]
        return result

    return app


app = create_app()
