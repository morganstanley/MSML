"""Tests for run.py's canonical-config resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alpha_lab.run import resolve_canonical_config


def _write_config(path: Path, **fields: object) -> Path:
    """Write a minimal valid config JSON at ``path`` (data_path/description defaulted)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"data_path": "data.csv", "description": "d", **fields}
    path.write_text(json.dumps(payload))
    return path


class TestResolveCanonicalConfig:
    def test_seeds_from_config_when_absent(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        src = _write_config(tmp_path / "src.json")
        canonical = resolve_canonical_config(str(src), str(ws), False)
        assert canonical == ws / ".alpha_lab" / "config.json"
        assert canonical.exists()
        # data_path is persisted absolute
        assert Path(json.loads(canonical.read_text())["data_path"]).is_absolute()

    def test_resume_without_config(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        canonical = _write_config(
            ws / ".alpha_lab" / "config.json", data_path=str(ws / "data.csv")
        )
        assert resolve_canonical_config(None, str(ws), False) == canonical

    def test_requires_config_when_absent(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        with pytest.raises(SystemExit):
            resolve_canonical_config(None, str(ws), False)

    def test_overwrite_requires_config(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        with pytest.raises(SystemExit):
            resolve_canonical_config(None, str(ws), True)

    def test_conflict_when_config_differs(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        _write_config(
            ws / ".alpha_lab" / "config.json",
            data_path=str(ws / "data.csv"),
            description="existing",
        )
        src = _write_config(
            tmp_path / "src.json", data_path=str(ws / "data.csv"), description="different"
        )
        with pytest.raises(SystemExit):
            resolve_canonical_config(str(src), str(ws), False)

    def test_no_conflict_when_config_matches(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        canonical = _write_config(
            ws / ".alpha_lab" / "config.json", data_path=str(ws / "data.csv")
        )
        src = _write_config(tmp_path / "src.json", data_path=str(ws / "data.csv"))
        assert resolve_canonical_config(str(src), str(ws), False) == canonical

    def test_match_reconciles_relative_and_absolute_data_path(self, tmp_path: Path) -> None:
        # canonical stores an absolute data_path; --config gives it relative to a dir
        # that resolves to the same workspace-internal file -> no false conflict.
        ws = tmp_path / "ws"
        _write_config(ws / ".alpha_lab" / "config.json", data_path=str(ws / "data.csv"))
        src = _write_config(ws / "src.json", data_path="data.csv")
        # must not raise
        resolve_canonical_config(str(src), str(ws), False)

    def test_match_reconciles_relative_canonical_data_path(self, tmp_path: Path) -> None:
        # Canonical stores a workspace-relative data_path; an incoming --config with the
        # equivalent absolute path must not be a false conflict. The existing config is
        # normalized against the workspace root (not .alpha_lab/).
        ws = tmp_path / "ws"
        _write_config(ws / ".alpha_lab" / "config.json", data_path="data.csv")
        src = _write_config(tmp_path / "src.json", data_path=str(ws / "data.csv"))
        # must not raise
        resolve_canonical_config(str(src), str(ws), False)

    def test_overwrite_replaces_mismatched(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        canonical = _write_config(
            ws / ".alpha_lab" / "config.json",
            data_path=str(ws / "data.csv"),
            description="old",
        )
        src = _write_config(
            tmp_path / "src.json", data_path=str(ws / "data.csv"), description="new"
        )
        resolve_canonical_config(str(src), str(ws), True)
        assert json.loads(canonical.read_text())["description"] == "new"
