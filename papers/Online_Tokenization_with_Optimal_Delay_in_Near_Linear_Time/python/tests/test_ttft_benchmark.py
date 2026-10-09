from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import benchmark_common as common
from tools import ttft_bench


class FakeSuite:
    suite_id = "fake"
    item_label = "encoding"

    def construct(
        self,
        implementation: str,
        item: object,
        *,
        dfa: bool,
        profile: bool,
    ) -> object:
        del item, dfa
        self.construct_calls.append((implementation, profile))
        assert implementation == "hiriluk"
        return FakeChopper()

    def __init__(self) -> None:
        self.construct_calls: list[tuple[str, bool]] = []


class FakeChopper:
    last_profile: object | None = None

    def chop_file(self, path: Path, *, output: str) -> list[int]:
        assert path.name == "input.txt"
        assert output == "array"
        self.last_profile = SimpleNamespace(
            total_tokens=3,
            ttft_seconds=0.002,
            elapsed_seconds=0.005,
        )
        return [1, 2, 3]


class EmptyChopper:
    last_profile: object | None = None

    def chop_file(self, path: Path, *, output: str) -> list[int]:
        del path, output
        self.last_profile = SimpleNamespace(
            total_tokens=0,
            ttft_seconds=None,
            elapsed_seconds=0.001,
        )
        return []


def test_hiriluk_observation_uses_native_profile_ttft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter((10.0, 10.006))
    monkeypatch.setattr(
        ttft_bench.time,
        "perf_counter",
        lambda: next(timestamps),
    )

    observation = ttft_bench.observe_once(
        FakeChopper(),
        Path("/tmp/input.txt"),
    )
    assert observation == ttft_bench.TtftObservation(
        tokens=3,
        ttft_seconds=0.002,
        completion_seconds=pytest.approx(0.006),
    )


def test_empty_hiriluk_output_has_no_ttft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter((1.0, 1.001))
    monkeypatch.setattr(
        ttft_bench.time,
        "perf_counter",
        lambda: next(timestamps),
    )
    observation = ttft_bench.observe_once(
        EmptyChopper(),
        Path("/tmp/empty.txt"),
    )
    assert observation.tokens == 0
    assert observation.ttft_seconds is None


def test_measure_case_uses_fresh_profiled_hiriluk_encoders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(common, "WARMUP_RUNS", 1)
    monkeypatch.setattr(common, "TTFT_RUNS", 2)
    timestamps = iter((1.0, 1.1, 2.0, 2.1, 3.0, 3.1))
    monkeypatch.setattr(
        ttft_bench.time,
        "perf_counter",
        lambda: next(timestamps),
    )
    input_path = tmp_path / "input.txt"
    input_path.write_text("text", encoding="utf-8")
    suite = FakeSuite()
    item = SimpleNamespace(name="r50k")

    result = ttft_bench.measure_case(
        suite,
        item,
        "english",
        input_path,
        dfa=False,
    )

    assert suite.construct_calls == [
        ("hiriluk", True),
        ("hiriluk", True),
        ("hiriluk", True),
    ]
    assert result["tokens"] == 3
    assert result["median_ttft_seconds"] == 0.002
    assert result["ttft_p25_seconds"] == 0.002
    assert result["ttft_p75_seconds"] == 0.002
    assert result["impl"] == "hiriluk"
