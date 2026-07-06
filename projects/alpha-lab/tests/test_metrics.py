"""Tests for the MetricsCollector."""

from __future__ import annotations

import threading
import time

from alpha_lab.metrics import MetricsCollector, _MetricsCounter


class TestMetricsInitialState:
    def test_snapshot_initial(self) -> None:
        m = MetricsCollector()
        snap = m.snapshot()
        assert snap["total_input_tokens"] == 0
        assert snap["total_output_tokens"] == 0
        assert snap["total_tokens"] == 0
        assert snap["api_call_count"] == 0
        assert snap["experiments_completed"] == 0
        assert snap["avg_experiment_duration_s"] == 0.0
        assert snap["error_count"] == 0
        assert snap["api_error_count"] == 0
        assert snap["uptime_seconds"] >= 0


class TestRecordApiCall:
    def test_single_call(self) -> None:
        m = MetricsCollector()
        m.record_api_call(100, 50)
        snap = m.snapshot()
        assert snap["total_input_tokens"] == 100
        assert snap["total_output_tokens"] == 50
        assert snap["total_tokens"] == 150
        assert snap["api_call_count"] == 1

    def test_multiple_calls(self) -> None:
        m = MetricsCollector()
        m.record_api_call(100, 50)
        m.record_api_call(200, 100)
        snap = m.snapshot()
        assert snap["total_input_tokens"] == 300
        assert snap["total_output_tokens"] == 150
        assert snap["total_tokens"] == 450
        assert snap["api_call_count"] == 2


class TestRecordExperiment:
    def test_experiment_complete(self) -> None:
        m = MetricsCollector()
        m.record_experiment_complete(120.0)
        m.record_experiment_complete(60.0)
        snap = m.snapshot()
        assert snap["experiments_completed"] == 2
        assert snap["avg_experiment_duration_s"] == 90.0


class TestRecordError:
    def test_general_error(self) -> None:
        m = MetricsCollector()
        m.record_error()
        snap = m.snapshot()
        assert snap["error_count"] == 1
        assert snap["api_error_count"] == 0

    def test_api_error(self) -> None:
        m = MetricsCollector()
        m.record_error(is_api_error=True)
        snap = m.snapshot()
        assert snap["error_count"] == 1
        assert snap["api_error_count"] == 1


class TestThreadSafety:
    def test_concurrent_api_calls(self) -> None:
        m = MetricsCollector()
        n_threads = 4
        calls_per_thread = 1000

        def worker() -> None:
            for _ in range(calls_per_thread):
                m.record_api_call(10, 5)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = m.snapshot()
        expected = n_threads * calls_per_thread
        assert snap["api_call_count"] == expected
        assert snap["total_input_tokens"] == expected * 10
        assert snap["total_output_tokens"] == expected * 5


class TestCacheMetrics:
    def test_initial_cache_fields(self) -> None:
        m = MetricsCollector()
        snap = m.snapshot()
        assert snap["cache_read_input_tokens"] == 0
        assert snap["cache_write_input_tokens"] == 0
        assert snap["cache_hit_rate"] == 0.0

    def test_cache_tokens_accumulated(self) -> None:
        m = MetricsCollector()
        m.record_api_call(1000, 200, cache_read_tokens=800, cache_write_tokens=100)
        m.record_api_call(1000, 200, cache_read_tokens=900, cache_write_tokens=0)
        snap = m.snapshot()
        assert snap["cache_read_input_tokens"] == 1700
        assert snap["cache_write_input_tokens"] == 100
        assert snap["cache_hit_rate"] == 0.85

    def test_cache_hit_rate_zero_when_no_reads(self) -> None:
        m = MetricsCollector()
        m.record_api_call(500, 100, cache_read_tokens=0, cache_write_tokens=500)
        snap = m.snapshot()
        assert snap["cache_hit_rate"] == 0.0

    def test_cache_defaults_when_omitted(self) -> None:
        m = MetricsCollector()
        m.record_api_call(100, 50)
        snap = m.snapshot()
        assert snap["cache_read_input_tokens"] == 0
        assert snap["cache_write_input_tokens"] == 0
        assert snap["cache_hit_rate"] == 0.0


class TestThroughputCalculation:
    def test_experiments_per_hour(self) -> None:
        m = MetricsCollector()
        # Backdate start time by 1 hour
        m._start_time = time.time() - 3600
        m.record_experiment_complete(100.0)
        m.record_experiment_complete(200.0)
        snap = m.snapshot()
        # ~2 experiments in ~1 hour = ~2.0/hr
        assert 1.9 <= snap["experiments_per_hour"] <= 2.1


class TestCountersMerge:
    def test_counters_round_trip_and_merge(self) -> None:
        a = MetricsCollector()
        a.record_api_call(10, 5, cache_read_tokens=2, cache_write_tokens=1)
        a.record_error(is_api_error=True)
        a.record_experiment_complete(3.0)

        b = MetricsCollector()
        b.merge_counters(a.counters())
        assert b.counters() == a.counters()  # merge into empty == source

        b.merge_counters(a.counters())  # additive
        snap = b.snapshot()
        assert snap["total_input_tokens"] == 20
        assert snap["api_call_count"] == 2
        assert snap["error_count"] == 2
        assert snap["api_error_count"] == 2
        assert snap["experiments_completed"] == 2


class TestOnRecordHookAndDelta:
    def test_on_record_fires_after_every_counter_change(self) -> None:
        seen: list[int] = []
        collector = MetricsCollector(on_record=lambda: seen.append(collector.counters().api_call_count))
        collector.record_api_call(1, 1)
        collector.record_error()
        collector.record_experiment_complete(1.0)  # a non-API counter still triggers the hook
        assert len(seen) == 3
        assert seen[0] == 1  # counter already applied when the hook runs

    def test_delta_since_returns_current_and_increase(self) -> None:
        collector = MetricsCollector()
        collector.record_api_call(10, 5)
        current, delta = collector.delta_since(_MetricsCounter())
        assert current.total_input_tokens == 10
        assert delta.total_input_tokens == 10

        collector.record_api_call(3, 2)
        current2, delta2 = collector.delta_since(current)
        assert current2.total_input_tokens == 13  # cumulative
        assert delta2.total_input_tokens == 3      # only the increase since `current`
