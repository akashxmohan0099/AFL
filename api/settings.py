"""Runtime settings for the FastAPI application."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _parse_csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    parts = tuple(part.strip() for part in raw.split(",") if part.strip())
    return parts or default


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class APISettings:
    cors_allow_origins: tuple[str, ...]
    cors_allow_methods: tuple[str, ...]
    cors_allow_headers: tuple[str, ...]
    cors_allow_credentials: bool
    api_key: str | None
    api_key_header: str
    rate_limit_enabled: bool
    rate_limit_requests: int
    rate_limit_window_seconds: int
    rate_limit_exempt_paths: tuple[str, ...]
    trust_x_forwarded_for: bool
    log_level: str
    load_cache_on_startup: bool = True

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)

    def is_exempt_path(self, path: str) -> bool:
        return any(path == prefix or path.startswith(prefix) for prefix in self.rate_limit_exempt_paths)

    @classmethod
    def from_env(cls) -> "APISettings":
        default_origins = (
            "http://localhost:3000",
            "http://localhost:3001",
            "http://localhost:3002",
            "http://localhost:3003",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
            "http://127.0.0.1:3002",
            "http://127.0.0.1:3003",
        )
        default_methods = ("GET", "POST", "OPTIONS")
        default_headers = ("Content-Type", "X-API-Key", "X-Admin-Key")
        default_exempt_paths = ("/api/health", "/api/admin", "/docs", "/openapi.json", "/redoc")

        api_key = os.getenv("AFL_API_KEY") or os.getenv("API_KEY")
        api_key = api_key.strip() if api_key else None

        return cls(
            cors_allow_origins=_parse_csv_env("AFL_API_ALLOW_ORIGINS", default_origins),
            cors_allow_methods=_parse_csv_env("AFL_API_ALLOW_METHODS", default_methods),
            cors_allow_headers=_parse_csv_env("AFL_API_ALLOW_HEADERS", default_headers),
            cors_allow_credentials=_parse_bool_env("AFL_API_ALLOW_CREDENTIALS", False),
            api_key=api_key,
            api_key_header=os.getenv("AFL_API_KEY_HEADER", "X-API-Key").strip() or "X-API-Key",
            rate_limit_enabled=_parse_bool_env("AFL_RATE_LIMIT_ENABLED", True),
            rate_limit_requests=max(1, _parse_int_env("AFL_RATE_LIMIT_REQUESTS", 240)),
            rate_limit_window_seconds=max(1, _parse_int_env("AFL_RATE_LIMIT_WINDOW_SECONDS", 60)),
            rate_limit_exempt_paths=_parse_csv_env("AFL_RATE_LIMIT_EXEMPT_PATHS", default_exempt_paths),
            trust_x_forwarded_for=_parse_bool_env("AFL_TRUST_X_FORWARDED_FOR", False),
            log_level=os.getenv("AFL_API_LOG_LEVEL", "INFO").strip().upper() or "INFO",
            load_cache_on_startup=_parse_bool_env("AFL_LOAD_CACHE_ON_STARTUP", True),
        )
