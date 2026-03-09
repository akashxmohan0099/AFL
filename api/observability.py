"""Logging and lightweight runtime metrics for the API."""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from threading import Lock


def configure_logging(level_name: str) -> None:
    """Ensure API logs are emitted with a predictable format."""
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    logging.getLogger("api").setLevel(level)


class RuntimeMetrics:
    """Tracks basic request counters and latency statistics in-process."""

    def __init__(self) -> None:
        self.started_at = time.time()
        self._lock = Lock()
        self._requests_total = 0
        self._requests_in_flight = 0
        self._status_groups = defaultdict(int)
        self._path_stats: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {
                "requests": 0,
                "errors": 0,
                "total_latency_ms": 0.0,
                "last_status": 0,
            }
        )

    def start_request(self) -> None:
        with self._lock:
            self._requests_in_flight += 1

    def finish_request(self, path: str, status_code: int, duration_ms: float) -> None:
        status_group = f"{status_code // 100}xx"
        with self._lock:
            self._requests_in_flight = max(0, self._requests_in_flight - 1)
            self._requests_total += 1
            self._status_groups[status_group] += 1

            path_stats = self._path_stats[path]
            path_stats["requests"] += 1
            path_stats["total_latency_ms"] += float(duration_ms)
            path_stats["last_status"] = int(status_code)
            if status_code >= 500:
                path_stats["errors"] += 1

    def snapshot(self) -> dict:
        with self._lock:
            total_latency_ms = sum(
                float(stats["total_latency_ms"]) for stats in self._path_stats.values()
            )
            paths = {}
            for path, stats in sorted(self._path_stats.items()):
                requests = int(stats["requests"])
                paths[path] = {
                    "requests": requests,
                    "errors": int(stats["errors"]),
                    "avg_latency_ms": round(float(stats["total_latency_ms"]) / requests, 2)
                    if requests
                    else 0.0,
                    "last_status": int(stats["last_status"]),
                }

            return {
                "uptime_seconds": round(time.time() - self.started_at, 2),
                "requests_total": int(self._requests_total),
                "requests_in_flight": int(self._requests_in_flight),
                "avg_latency_ms": round(total_latency_ms / self._requests_total, 2)
                if self._requests_total
                else 0.0,
                "responses": dict(sorted(self._status_groups.items())),
                "paths": paths,
            }
