"""Tests for the MetricsCollector."""

from __future__ import annotations

import threading
import time

from alpha_lab.metrics import MetricsCollector


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
