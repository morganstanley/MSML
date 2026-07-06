"""Tests for the tracing module."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from opentelemetry.trace import Link

from alpha_lab.tracing import (
    TRACE_INFO_FILENAME,
    _read_trace_info,
    _write_trace_info,
    resolve_run_id,
)


class TestResolveRunId:
    """Validate the 4-tier resolution order for run IDs."""

    def test_explicit_run_id_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tier 1: explicit run_id takes precedence over everything."""
        monkeypatch.setenv("ALPHALAB_RUN_ID", "from-env")
        result = resolve_run_id(
            run_id="explicit-123",
            run_id_prefix="some-prefix",
            workspace="/workspace/my_run",
        )
        assert result == "explicit-123"

    def test_prefix_beats_env_and_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tier 2: prefix generates a composite ID when no explicit run_id."""
        monkeypatch.setenv("ALPHALAB_RUN_ID", "from-env")
        result = resolve_run_id(
            run_id=None,
            run_id_prefix="nightly",
            workspace="/workspace/my_run",
        )
        assert result.startswith("nightly-")
        assert result != "from-env"

    def test_prefix_format(self) -> None:
        """Tier 2: prefix result has format {prefix}-{YYYYMMDD-HHMM}-{hex8}."""
        result = resolve_run_id(
            run_id=None,
            run_id_prefix="ci",
            workspace="/workspace/x",
        )
        parts = result.split("-")
        # ci-YYYYMMDD-HHMM-hex8 → 4 parts
        assert len(parts) == 4
        assert parts[0] == "ci"
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 4  # HHMM
        assert len(parts[3]) == 8  # short uuid hex

    def test_env_var_beats_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tier 3: ALPHALAB_RUN_ID env var used when no explicit args."""
        monkeypatch.setenv("ALPHALAB_RUN_ID", "env-run-42")
        result = resolve_run_id(
            run_id=None,
            run_id_prefix=None,
            workspace="/workspace/my_run",
        )
        assert result == "env-run-42"

    def test_fallback_uses_workspace_basename(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tier 4: falls back to workspace basename + timestamp + uuid."""
        monkeypatch.delenv("ALPHALAB_RUN_ID", raising=False)
        result = resolve_run_id(
            run_id=None,
            run_id_prefix=None,
            workspace="/some/deep/path/my_workspace",
        )
        assert result.startswith("my_workspace-")

    def test_fallback_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tier 4: fallback has format {basename}-{YYYYMMDD-HHMM}-{hex8}."""
        monkeypatch.delenv("ALPHALAB_RUN_ID", raising=False)
        result = resolve_run_id(
            run_id=None,
            run_id_prefix=None,
            workspace="/ws/test_run",
        )
        parts = result.split("-")
        # test_run-YYYYMMDD-HHMM-hex8 → 4 parts
        # (test_run contains underscore, not hyphen, so split is clean)
        assert len(parts) == 4
        assert parts[0] == "test_run"
        assert len(parts[3]) == 8  # short uuid

    def test_fallback_uniqueness(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tier 4: consecutive calls produce distinct IDs."""
        monkeypatch.delenv("ALPHALAB_RUN_ID", raising=False)
        ids = {
            resolve_run_id(None, None, "/ws/x")
            for _ in range(20)
        }
        assert len(ids) == 20

    def test_empty_string_run_id_is_falsy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty string run_id falls through to next tier."""
        monkeypatch.setenv("ALPHALAB_RUN_ID", "from-env")
        result = resolve_run_id(
            run_id="",
            run_id_prefix=None,
            workspace="/ws/x",
        )
        assert result == "from-env"

    def test_empty_string_prefix_is_falsy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty string prefix falls through to next tier."""
        monkeypatch.setenv("ALPHALAB_RUN_ID", "from-env")
        result = resolve_run_id(
            run_id=None,
            run_id_prefix="",
            workspace="/ws/x",
        )
        assert result == "from-env"

    def test_env_var_not_set_falls_to_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ALPHALAB_RUN_ID is unset, tier 4 fallback is used."""
        monkeypatch.delenv("ALPHALAB_RUN_ID", raising=False)
        result = resolve_run_id(
            run_id=None,
            run_id_prefix=None,
            workspace="/ws/demo",
        )
        assert result.startswith("demo-")


class TestWriteTraceInfo:
    """Tests for _write_trace_info persistence."""

    def test_creates_file(self, tmp_path: Path) -> None:
        ws = str(tmp_path / "workspace")
        os.makedirs(ws)
        _write_trace_info(
            workspace=ws,
            run_id="run-1",
            trace_id="0" * 32,
            span_id="a1b2c3d4e5f60718",
            attempt=1,
            previous=None,
            config_path="cfg.json",
        )
        path = Path(ws) / TRACE_INFO_FILENAME
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["run_id"] == "run-1"
        assert data["attempt"] == 1
        assert data["trace_id"] == "0" * 32
        assert data["span_id"] == "a1b2c3d4e5f60718"
        assert data["config"] == "cfg.json"

    def test_records_user(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("USER", "testuser")
        ws = str(tmp_path)
        _write_trace_info(
            workspace=ws, run_id="r", trace_id="ab" * 16,
            span_id="1234567890abcdef", attempt=1, previous=None,
        )
        data = json.loads((Path(ws) / TRACE_INFO_FILENAME).read_text())
        assert data["user"] == "testuser"

    def test_includes_started_at(self, tmp_path: Path) -> None:
        ws = str(tmp_path)
        _write_trace_info(
            workspace=ws, run_id="r", trace_id="ab" * 16,
            span_id="1234567890abcdef", attempt=1, previous=None,
        )
        data = json.loads((Path(ws) / TRACE_INFO_FILENAME).read_text())
        assert "started_at" in data
        # ISO-ish format: YYYY-MM-DDTHH:MM:SSZ
        assert data["started_at"].endswith("Z")
        assert "T" in data["started_at"]

    def test_includes_previous_trace_id(self, tmp_path: Path) -> None:
        ws = str(tmp_path)
        previous = {"trace_id": "aa" * 16, "attempt": 1}
        _write_trace_info(
            workspace=ws, run_id="r", trace_id="bb" * 16,
            span_id="fedcba9876543210", attempt=2, previous=previous,
        )
        data = json.loads((Path(ws) / TRACE_INFO_FILENAME).read_text())
        assert data["previous_trace_id"] == "aa" * 16

    def test_omits_previous_trace_id_on_first_run(
        self, tmp_path: Path
    ) -> None:
        ws = str(tmp_path)
        _write_trace_info(
            workspace=ws, run_id="r", trace_id="cc" * 16,
            span_id="0102030405060708", attempt=1, previous=None,
        )
        data = json.loads((Path(ws) / TRACE_INFO_FILENAME).read_text())
        assert "previous_trace_id" not in data

    def test_omits_previous_trace_id_when_missing_key(
        self, tmp_path: Path
    ) -> None:
        """previous dict exists but has no trace_id key."""
        ws = str(tmp_path)
        _write_trace_info(
            workspace=ws, run_id="r", trace_id="dd" * 16,
            span_id="aabbccddeeff0011", attempt=2, previous={"attempt": 1},
        )
        data = json.loads((Path(ws) / TRACE_INFO_FILENAME).read_text())
        assert "previous_trace_id" not in data

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        ws = str(tmp_path / "deep" / "nested" / "workspace")
        _write_trace_info(
            workspace=ws, run_id="r", trace_id="ee" * 16,
            span_id="9f8e7d6c5b4a3210", attempt=1, previous=None,
        )
        assert (Path(ws) / TRACE_INFO_FILENAME).exists()

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        ws = str(tmp_path)
        _write_trace_info(
            workspace=ws, run_id="first", trace_id="11" * 16,
            span_id="1111111111111111", attempt=1, previous=None,
        )
        _write_trace_info(
            workspace=ws, run_id="second", trace_id="22" * 16,
            span_id="2222222222222222", attempt=2,
            previous={"trace_id": "11" * 16, "attempt": 1},
        )
        data = json.loads((Path(ws) / TRACE_INFO_FILENAME).read_text())
        assert data["run_id"] == "second"
        assert data["attempt"] == 2
        assert data["previous_trace_id"] == "11" * 16


class TestReadTraceInfo:
    """Tests for _read_trace_info deserialization and error handling."""

    def test_returns_none_when_no_file(self, tmp_path: Path) -> None:
        assert _read_trace_info(str(tmp_path)) is None

    def test_reads_written_data(self, tmp_path: Path) -> None:
        ws = str(tmp_path)
        _write_trace_info(
            workspace=ws, run_id="run-42", trace_id="ab" * 16,
            span_id="0a1b2c3d4e5f6789", attempt=3, previous=None,
            config_path="my.json",
        )
        result = _read_trace_info(ws)
        assert result is not None
        assert result["run_id"] == "run-42"
        assert result["attempt"] == 3
        assert result["config"] == "my.json"

    def test_reconstructs_span_context(self, tmp_path: Path) -> None:
        ws = str(tmp_path)
        trace_id_hex = "0123456789abcdef" * 2  # 32 hex chars
        span_id_hex = "fedcba9876543210"
        _write_trace_info(
            workspace=ws, run_id="r", trace_id=trace_id_hex,
            span_id=span_id_hex, attempt=1, previous=None,
        )
        result = _read_trace_info(ws)
        assert result is not None
        sc = result["span_context"]
        assert sc is not None
        assert sc.trace_id == int(trace_id_hex, 16)
        assert sc.span_id == int(span_id_hex, 16)
        assert sc.is_remote is True

    def test_span_context_none_when_trace_id_empty(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / TRACE_INFO_FILENAME
        path.write_text(json.dumps({"attempt": 1, "trace_id": ""}))
        result = _read_trace_info(str(tmp_path))
        assert result is not None
        assert result["span_context"] is None

    def test_span_context_none_when_trace_id_missing(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / TRACE_INFO_FILENAME
        path.write_text(json.dumps({"attempt": 1}))
        result = _read_trace_info(str(tmp_path))
        assert result is not None
        assert result["span_context"] is None

    def test_span_context_none_when_span_id_missing(
        self, tmp_path: Path
    ) -> None:
        """Old trace_info.json files without span_id yield span_context=None."""
        path = tmp_path / TRACE_INFO_FILENAME
        path.write_text(json.dumps({"attempt": 1, "trace_id": "ab" * 16}))
        result = _read_trace_info(str(tmp_path))
        assert result is not None
        assert result["span_context"] is None

    def test_span_context_none_on_invalid_hex(self, tmp_path: Path) -> None:
        path = tmp_path / TRACE_INFO_FILENAME
        path.write_text(json.dumps({"attempt": 1, "trace_id": "not-hex"}))
        result = _read_trace_info(str(tmp_path))
        assert result is not None
        assert result["span_context"] is None

    def test_returns_none_on_corrupt_json(self, tmp_path: Path) -> None:
        path = tmp_path / TRACE_INFO_FILENAME
        path.write_text("{bad json")
        assert _read_trace_info(str(tmp_path)) is None

    def test_roundtrip_preserves_previous_trace_id(
        self, tmp_path: Path
    ) -> None:
        ws = str(tmp_path)
        _write_trace_info(
            workspace=ws, run_id="r1", trace_id="aa" * 16,
            span_id="1234123412341234", attempt=1, previous=None,
        )
        first = _read_trace_info(ws)
        _write_trace_info(
            workspace=ws, run_id="r2", trace_id="bb" * 16,
            span_id="5678567856785678", attempt=2, previous=first,
        )
        second = _read_trace_info(ws)
        assert second is not None
        assert second["previous_trace_id"] == "aa" * 16

    def test_roundtrip_span_context_linkable(self, tmp_path: Path) -> None:
        """Span context from read can be used to create an OTel Link."""
        ws = str(tmp_path)
        span_id_hex = "aabbccddeeff0011"
        _write_trace_info(
            workspace=ws, run_id="r", trace_id="ff" * 16,
            span_id=span_id_hex, attempt=1, previous=None,
        )
        result = _read_trace_info(ws)
        assert result is not None
        link = Link(result["span_context"])
        assert link.context.trace_id == int("ff" * 16, 16)
        assert link.context.span_id == int(span_id_hex, 16)
