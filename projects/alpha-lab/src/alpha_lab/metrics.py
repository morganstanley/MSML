"""Thread-safe in-memory metrics collector.

Resets on restart. No external dependencies.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace


@dataclass
class _MetricsCounter:
    """The raw additive counters a :class:`MetricsCollector` accumulates."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    api_call_count: int = 0
    experiments_completed: int = 0
    total_experiment_duration: float = 0.0
    error_count: int = 0
    api_error_count: int = 0

    def plus(self, other: _MetricsCounter) -> _MetricsCounter:
        """Field-wise sum with another counter."""
        return _MetricsCounter(
            **{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)}
        )

    def minus(self, baseline: _MetricsCounter) -> _MetricsCounter:
        """Field-wise increase of this counter over ``baseline``."""
        return _MetricsCounter(
            **{f.name: getattr(self, f.name) - getattr(baseline, f.name) for f in fields(self)}
        )

    def is_zero(self) -> bool:
        """True when every counter is zero (nothing to flush)."""
        return all(getattr(self, f.name) == 0 for f in fields(self))

    @classmethod
    def from_mapping(cls, data: Mapping) -> _MetricsCounter:
        """Build from a (possibly wire-deserialized) mapping. Non-numeric or missing
        values — a malformed/version-skewed payload crossing the process boundary — are
        skipped (left at the field default) rather than raising."""
        values: dict[str, int | float] = {}
        for f in fields(cls):
            value = data.get(f.name, 0)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            values[f.name] = value
        return cls(**values)


class MetricsCollector:
    """Collects API call, experiment, and error metrics in-memory."""

    def __init__(self, on_record: Callable[[], None] | None = None) -> None:
        self._on_record = on_record
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._counters = _MetricsCounter()

    def record_api_call(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        with self._lock:
            self._counters.total_input_tokens += input_tokens
            self._counters.total_output_tokens += output_tokens
            self._counters.total_cache_read_tokens += cache_read_tokens
            self._counters.total_cache_write_tokens += cache_write_tokens
            self._counters.api_call_count += 1
        self._notify()

    def record_experiment_complete(self, duration_seconds: float) -> None:
        with self._lock:
            self._counters.experiments_completed += 1
            self._counters.total_experiment_duration += duration_seconds
        self._notify()

    def record_error(self, is_api_error: bool = False) -> None:
        with self._lock:
            self._counters.error_count += 1
            if is_api_error:
                self._counters.api_error_count += 1
        self._notify()

    def _notify(self) -> None:
        """Run the on-record hook, if any (called after the lock is released so the
        hook can safely call back into counters()/delta_since())."""
        if self._on_record is not None:
            self._on_record()

    def counters(self) -> _MetricsCounter:
        """A snapshot copy of the raw additive counters (safe to keep, compare, or merge)."""
        with self._lock:
            return replace(self._counters)

    def delta_since(self, baseline: _MetricsCounter) -> tuple[_MetricsCounter, _MetricsCounter]:
        """Return ``(current, increase-since-baseline)`` counters from one snapshot."""
        current = self.counters()
        return current, current.minus(baseline)

    def merge_counters(self, counters: _MetricsCounter | Mapping) -> None:
        """Add another collector's counters into this one (e.g. a sandboxed child's).

        Accepts a :class:`_MetricsCounter` or the raw mapping a child emits over the wire;
        a mapping is normalized best-effort via :meth:`_MetricsCounter.from_mapping`.
        """
        other = (
            counters if isinstance(counters, _MetricsCounter)
            else _MetricsCounter.from_mapping(counters)
        )
        with self._lock:
            self._counters = self._counters.plus(other)

    def snapshot(self) -> dict:
        """Return a JSON-serializable summary of all metrics (rates/averages derived)."""
        with self._lock:
            counters = self._counters
            uptime = time.time() - self._start_time
            avg_duration = (
                counters.total_experiment_duration / counters.experiments_completed
                if counters.experiments_completed > 0
                else 0.0
            )
            experiments_per_hour = (
                counters.experiments_completed / (uptime / 3600)
                if uptime > 0
                else 0.0
            )
            input_tokens = counters.total_input_tokens
            return {
                "total_input_tokens": input_tokens,
                "total_output_tokens": counters.total_output_tokens,
                "total_tokens": input_tokens + counters.total_output_tokens,
                "cache_read_input_tokens": counters.total_cache_read_tokens,
                "cache_write_input_tokens": counters.total_cache_write_tokens,
                "cache_hit_rate": round(counters.total_cache_read_tokens / input_tokens, 3)
                if input_tokens > 0
                else 0.0,
                "api_call_count": counters.api_call_count,
                "experiments_completed": counters.experiments_completed,
                "avg_experiment_duration_s": round(avg_duration, 2),
                "experiments_per_hour": round(experiments_per_hour, 2),
                "error_count": counters.error_count,
                "api_error_count": counters.api_error_count,
                "uptime_seconds": round(uptime, 2),
            }
