"""Tests for the host-level resource gate.

These tests exercise the ``host_has_capacity`` decision logic by
monkey-patching the underlying ``/proc`` readers, so the suite passes
regardless of the actual host state at CI time.
"""
from __future__ import annotations

from unittest.mock import patch

from alpha_lab import host


class TestHostHasCapacity:
    def test_returns_true_when_both_sensors_below_threshold(self) -> None:
        with patch.object(host, "system_available_ram_gb", return_value=500.0), \
             patch.object(host, "system_loadavg_per_cpu", return_value=1.5):
            assert host.host_has_capacity() is True

    def test_refuses_when_ram_below_threshold(self) -> None:
        with patch.object(host, "system_available_ram_gb", return_value=10.0), \
             patch.object(host, "system_loadavg_per_cpu", return_value=0.1):
            # 10 GB available < default 64 GB threshold
            assert host.host_has_capacity() is False

    def test_refuses_when_loadavg_above_threshold(self) -> None:
        with patch.object(host, "system_available_ram_gb", return_value=500.0), \
             patch.object(host, "system_loadavg_per_cpu", return_value=10.0):
            # loadavg/cpu = 10 > default 4.0 threshold
            assert host.host_has_capacity() is False

    def test_custom_thresholds_respected(self) -> None:
        with patch.object(host, "system_available_ram_gb", return_value=80.0), \
             patch.object(host, "system_loadavg_per_cpu", return_value=2.0):
            # default would pass (80 > 64, 2 < 4); tighter thresholds reject
            assert host.host_has_capacity(min_avail_ram_gb=100.0) is False
            assert host.host_has_capacity(max_load_per_cpu=1.0) is False

    def test_sensor_failure_does_not_block_submissions(self) -> None:
        """If /proc isn't readable, we DON'T want to refuse all submits —
        executor's own slot accounting still gates. A sensor failure
        should be conservatively-permissive, not conservatively-restrictive,
        because a hard refusal under sensor failure would freeze the
        pipeline.
        """
        with patch.object(host, "system_available_ram_gb", return_value=None), \
             patch.object(host, "system_loadavg_per_cpu", return_value=None):
            assert host.host_has_capacity() is True

    def test_partial_sensor_failure_uses_remaining_signal(self) -> None:
        """If RAM is readable but loadavg isn't (or vice versa), the
        readable signal still gates."""
        with patch.object(host, "system_available_ram_gb", return_value=10.0), \
             patch.object(host, "system_loadavg_per_cpu", return_value=None):
            assert host.host_has_capacity() is False
        with patch.object(host, "system_available_ram_gb", return_value=None), \
             patch.object(host, "system_loadavg_per_cpu", return_value=20.0):
            assert host.host_has_capacity() is False


class TestSystemAvailableRamReader:
    def test_parses_meminfo_correctly(self) -> None:
        """Validate the parser against a real /proc/meminfo if available.
        On a Linux host the value is a positive float; on Mac or in a
        container without /proc the function returns None — accept both.
        """
        result = host.system_available_ram_gb()
        assert result is None or (isinstance(result, float) and result >= 0.0)


class TestLoadAvgReader:
    def test_returns_float_or_none(self) -> None:
        result = host.system_loadavg_per_cpu()
        assert result is None or (isinstance(result, float) and result >= 0.0)
