"""Thread-safe in-memory metrics collector.

Resets on restart. No external dependencies.
"""

from __future__ import annotations

import threading
import time


class MetricsCollector:
    """Collects API call, experiment, and error metrics in-memory."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._api_call_count = 0
        self._experiments_completed = 0
        self._total_experiment_duration = 0.0
        self._error_count = 0
        self._api_error_count = 0

    def record_api_call(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._api_call_count += 1

    def record_experiment_complete(self, duration_seconds: float) -> None:
        with self._lock:
            self._experiments_completed += 1
            self._total_experiment_duration += duration_seconds

    def record_error(self, is_api_error: bool = False) -> None:
        with self._lock:
            self._error_count += 1
            if is_api_error:
                self._api_error_count += 1

    def snapshot(self) -> dict:
        """Return a JSON-serializable summary of all metrics."""
        with self._lock:
            uptime = time.time() - self._start_time
            avg_duration = (
                self._total_experiment_duration / self._experiments_completed
                if self._experiments_completed > 0
                else 0.0
            )
            experiments_per_hour = (
                self._experiments_completed / (uptime / 3600)
                if uptime > 0
                else 0.0
            )
            return {
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "total_tokens": self._total_input_tokens + self._total_output_tokens,
                "api_call_count": self._api_call_count,
                "experiments_completed": self._experiments_completed,
                "avg_experiment_duration_s": round(avg_duration, 2),
                "experiments_per_hour": round(experiments_per_hour, 2),
                "error_count": self._error_count,
                "api_error_count": self._api_error_count,
                "uptime_seconds": round(uptime, 2),
            }
