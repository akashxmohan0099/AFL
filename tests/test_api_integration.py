import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from api.main import create_app
from api.settings import APISettings


class _FakeStore:
    def __init__(self, rounds=None):
        self._rounds = rounds or []

    def list_rounds(self, subdir="predictions", year=None, **kwargs):
        return list(self._rounds)


class _LazyFakeCache(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_loaded = False
        self.load_calls = 0

    def load_all(self):
        self.is_loaded = True
        self.load_calls += 1


class TestAPIHardening(unittest.TestCase):
    def test_health_is_public_even_when_auth_enabled(self):
        settings = APISettings(
            cors_allow_origins=("http://localhost:3000",),
            cors_allow_methods=("GET", "OPTIONS"),
            cors_allow_headers=("Content-Type", "X-API-Key"),
            cors_allow_credentials=False,
            api_key="secret",
            api_key_header="X-API-Key",
            rate_limit_enabled=False,
            rate_limit_requests=10,
            rate_limit_window_seconds=60,
            rate_limit_exempt_paths=("/api/health", "/docs", "/openapi.json", "/redoc"),
            trust_x_forwarded_for=False,
            log_level="INFO",
            load_cache_on_startup=False,
        )
        fake_cache = SimpleNamespace(
            player_games=pd.DataFrame([{"player": "P"}]),
            matches=pd.DataFrame([{"date": "2026-03-09"}]),
            experiments=[],
        )

        app = create_app(settings=settings)
        with patch("api.main.DataCache.get", return_value=fake_cache):
            with TestClient(app) as client:
                response = client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["auth_enabled"], True)

    def test_rounds_requires_api_key_when_enabled(self):
        settings = APISettings(
            cors_allow_origins=("http://localhost:3000",),
            cors_allow_methods=("GET", "OPTIONS"),
            cors_allow_headers=("Content-Type", "X-API-Key"),
            cors_allow_credentials=False,
            api_key="secret",
            api_key_header="X-API-Key",
            rate_limit_enabled=False,
            rate_limit_requests=10,
            rate_limit_window_seconds=60,
            rate_limit_exempt_paths=("/api/health", "/docs", "/openapi.json", "/redoc"),
            trust_x_forwarded_for=False,
            log_level="INFO",
            load_cache_on_startup=False,
        )
        fake_cache = SimpleNamespace(
            sequential_store=_FakeStore(rounds=[(2026, 0)]),
            store=None,
        )

        app = create_app(settings=settings)
        with patch("api.services.prediction_service.DataCache.get", return_value=fake_cache):
            with TestClient(app) as client:
                response = client.get("/api/rounds")

        self.assertEqual(response.status_code, 401)

    def test_rounds_accepts_valid_api_key(self):
        settings = APISettings(
            cors_allow_origins=("http://localhost:3000",),
            cors_allow_methods=("GET", "OPTIONS"),
            cors_allow_headers=("Content-Type", "X-API-Key"),
            cors_allow_credentials=False,
            api_key="secret",
            api_key_header="X-API-Key",
            rate_limit_enabled=False,
            rate_limit_requests=10,
            rate_limit_window_seconds=60,
            rate_limit_exempt_paths=("/api/health", "/docs", "/openapi.json", "/redoc"),
            trust_x_forwarded_for=False,
            log_level="INFO",
            load_cache_on_startup=False,
        )
        fake_cache = SimpleNamespace(
            sequential_store=_FakeStore(rounds=[(2026, 0), (2026, 1)]),
            store=None,
        )

        app = create_app(settings=settings)
        with patch("api.services.prediction_service.DataCache.get", return_value=fake_cache):
            with TestClient(app) as client:
                response = client.get("/api/rounds", headers={"X-API-Key": "secret"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [{"year": 2026, "round_number": 0}, {"year": 2026, "round_number": 1}])

    def test_rounds_rate_limit_returns_429(self):
        settings = APISettings(
            cors_allow_origins=("http://localhost:3000",),
            cors_allow_methods=("GET", "OPTIONS"),
            cors_allow_headers=("Content-Type", "X-API-Key"),
            cors_allow_credentials=False,
            api_key=None,
            api_key_header="X-API-Key",
            rate_limit_enabled=True,
            rate_limit_requests=2,
            rate_limit_window_seconds=60,
            rate_limit_exempt_paths=("/api/health", "/docs", "/openapi.json", "/redoc"),
            trust_x_forwarded_for=False,
            log_level="INFO",
            load_cache_on_startup=False,
        )
        fake_cache = SimpleNamespace(
            sequential_store=_FakeStore(rounds=[(2026, 0)]),
            store=None,
        )

        app = create_app(settings=settings)
        with patch("api.services.prediction_service.DataCache.get", return_value=fake_cache):
            with TestClient(app) as client:
                first = client.get("/api/rounds")
                second = client.get("/api/rounds")
                third = client.get("/api/rounds")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(third.status_code, 429)
        self.assertEqual(third.json()["detail"], "Rate limit exceeded")
        self.assertEqual(third.headers["X-RateLimit-Limit"], "2")

    def test_rounds_lazy_loads_cache_when_startup_load_disabled(self):
        settings = APISettings(
            cors_allow_origins=("http://localhost:3000",),
            cors_allow_methods=("GET", "OPTIONS"),
            cors_allow_headers=("Content-Type", "X-API-Key"),
            cors_allow_credentials=False,
            api_key=None,
            api_key_header="X-API-Key",
            rate_limit_enabled=False,
            rate_limit_requests=10,
            rate_limit_window_seconds=60,
            rate_limit_exempt_paths=("/api/health", "/docs", "/openapi.json", "/redoc"),
            trust_x_forwarded_for=False,
            log_level="INFO",
            load_cache_on_startup=False,
        )
        fake_cache = _LazyFakeCache(
            sequential_store=_FakeStore(rounds=[(2026, 0)]),
            store=None,
        )

        app = create_app(settings=settings)
        with patch("api.main.DataCache.get", return_value=fake_cache):
            with patch("api.services.prediction_service.DataCache.get", return_value=fake_cache):
                with TestClient(app) as client:
                    response = client.get("/api/rounds")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(fake_cache.load_calls, 1)
        self.assertEqual(fake_cache.is_loaded, True)

    def test_runtime_metrics_expose_request_ids_and_counts(self):
        settings = APISettings(
            cors_allow_origins=("http://localhost:3000",),
            cors_allow_methods=("GET", "OPTIONS"),
            cors_allow_headers=("Content-Type", "X-API-Key"),
            cors_allow_credentials=False,
            api_key=None,
            api_key_header="X-API-Key",
            rate_limit_enabled=False,
            rate_limit_requests=10,
            rate_limit_window_seconds=60,
            rate_limit_exempt_paths=("/api/health", "/docs", "/openapi.json", "/redoc"),
            trust_x_forwarded_for=False,
            log_level="INFO",
            load_cache_on_startup=False,
        )
        fake_cache = _LazyFakeCache(
            sequential_store=_FakeStore(rounds=[(2026, 0)]),
            store=None,
            player_games=pd.DataFrame([{"player": "P"}]),
            matches=pd.DataFrame([{"date": "2026-03-09"}]),
            experiments=[],
        )

        app = create_app(settings=settings)
        with patch("api.main.DataCache.get", return_value=fake_cache):
            with patch("api.services.prediction_service.DataCache.get", return_value=fake_cache):
                with patch("api.services.metrics_service.DataCache.get", return_value=fake_cache):
                    with TestClient(app) as client:
                        rounds = client.get("/api/rounds")
                        runtime = client.get("/api/metrics/runtime")

        self.assertEqual(rounds.status_code, 200)
        self.assertIn("X-Request-ID", rounds.headers)
        self.assertIn("X-Process-Time-Ms", rounds.headers)
        self.assertEqual(runtime.status_code, 200)
        payload = runtime.json()
        self.assertEqual(payload["cache_loaded"], True)
        self.assertGreaterEqual(payload["requests_total"], 1)
        self.assertEqual(payload["paths"]["GET /api/rounds"]["requests"], 1)


if __name__ == "__main__":
    unittest.main()
