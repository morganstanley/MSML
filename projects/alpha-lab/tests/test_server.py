"""Tests for the FastAPI server endpoints.

Uses HTTPX TestClient — no actual server started.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Set workspace before importing the app
_test_workspace = None


@pytest.fixture(autouse=True)
def setup_server(tmp_workspace: str) -> None:
    """Set server globals before each test."""
    global _test_workspace
    _test_workspace = tmp_workspace

    import alpha_lab.server as srv
    srv._workspace = tmp_workspace
    srv._config_path = None
    srv.event_history.clear()
    srv.manager = srv.AgentManager()
    srv.manager.db = None


def get_client():
    """Create a fresh TestClient."""
    from fastapi.testclient import TestClient

    from alpha_lab.server import app
    return TestClient(app)


class TestStatusEndpoint:
    def test_get_status(self, tmp_workspace: str) -> None:
        client = get_client()
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "running" in data
        assert data["running"] is False
        assert data["has_config"] is False


class TestStartStopEndpoints:
    def test_start_without_config(self, tmp_workspace: str) -> None:
        client = get_client()
        resp = client.post("/api/start")
        assert resp.status_code == 400
        assert "view-only" in resp.json()["error"].lower()

    def test_stop(self, tmp_workspace: str) -> None:
        client = get_client()
        resp = client.post("/api/stop")
        assert resp.status_code == 200


class TestFilesEndpoint:
    def test_list_files(self, tmp_workspace: str) -> None:
        # Create some files
        (Path(tmp_workspace) / "learnings.md").write_text("# Findings")
        scripts_dir = Path(tmp_workspace) / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "analysis.py").write_text("print('hi')")

        client = get_client()
        resp = client.get("/api/files")
        assert resp.status_code == 200
        data = resp.json()
        assert "children" in data
        names = [c["name"] for c in data["children"]]
        assert "learnings.md" in names
        assert "scripts" in names

    def test_list_files_empty(self, tmp_workspace: str) -> None:
        client = get_client()
        resp = client.get("/api/files")
        assert resp.status_code == 200
        data = resp.json()
        assert data["children"] == []


class TestFileEndpoint:
    def test_read_text_file(self, tmp_workspace: str) -> None:
        (Path(tmp_workspace) / "readme.md").write_text("# Hello")
        client = get_client()
        resp = client.get("/api/file/readme.md")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "text"
        assert "Hello" in data["content"]

    def test_file_not_found(self, tmp_workspace: str) -> None:
        client = get_client()
        resp = client.get("/api/file/nope.txt")
        assert resp.status_code == 404

    def test_path_traversal_blocked(self, tmp_workspace: str) -> None:
        # Starlette normalizes path params, so ../../ gets stripped.
        # Use URL-encoded %2e%2e%2f to bypass, or test with absolute path logic.
        # The real protection is in the resolve().relative_to() check.
        # Create a file outside workspace and try to reach it via symlink.
        import os
        outside = Path(tmp_workspace).parent / "secret.txt"
        outside.write_text("secret data")
        link = Path(tmp_workspace) / "escape"
        try:
            link.symlink_to(outside.parent)
        except OSError:
            pytest.skip("Cannot create symlinks")
        client = get_client()
        resp = client.get("/api/file/escape/secret.txt")
        # Should be blocked (403) because resolved path is outside workspace
        assert resp.status_code == 403

    def test_read_image_file(self, tmp_workspace: str) -> None:
        import base64
        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        (Path(tmp_workspace) / "plot.png").write_bytes(png_data)
        client = get_client()
        resp = client.get("/api/file/plot.png")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "image"
        assert data["media_type"] == "image/png"


class TestHealthEndpoint:
    def test_health_basic(self, tmp_workspace: str) -> None:
        client = get_client()
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "uptime_seconds" in data
        assert "components" in data
        assert "dispatcher" in data["components"]
        assert "strategist" in data["components"]
        assert "database" in data["components"]
        assert "metrics" in data["components"]

    def test_health_with_db(self, tmp_workspace: str) -> None:
        from alpha_lab.experiment_db import ExperimentDB
        import alpha_lab.server as srv

        db = ExperimentDB(str(Path(tmp_workspace) / "health.db"))
        db.create("health_exp", "D", "H", "{}")
        srv.manager.db = db

        client = get_client()
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["components"]["database"]["status"] == "ok"
        assert "board" in data["components"]["database"]


class TestMetricsEndpoint:
    def test_metrics_basic(self, tmp_workspace: str) -> None:
        client = get_client()
        resp = client.get("/api/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_tokens" in data
        assert "api_call_count" in data
        assert "uptime_seconds" in data


class TestExperimentEndpoints:
    def test_experiments_no_db(self, tmp_workspace: str) -> None:
        client = get_client()
        resp = client.get("/api/experiments")
        assert resp.status_code == 200
        assert resp.json()["experiments"] == []

    def test_experiments_with_db(self, tmp_workspace: str) -> None:
        from alpha_lab.experiment_db import ExperimentDB
        import alpha_lab.server as srv

        db = ExperimentDB(str(Path(tmp_workspace) / "test.db"))
        db.create("exp_a", "Desc A", "Hyp A", "{}")
        db.create("exp_b", "Desc B", "Hyp B", "{}")
        srv.manager.db = db

        client = get_client()
        resp = client.get("/api/experiments")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["experiments"]) == 2
        assert "summary" in data

    def test_experiment_by_id(self, tmp_workspace: str) -> None:
        from alpha_lab.experiment_db import ExperimentDB
        import alpha_lab.server as srv

        db = ExperimentDB(str(Path(tmp_workspace) / "test.db"))
        db.create("exp_detail", "D", "H", '{"model": "lstm"}')
        srv.manager.db = db

        client = get_client()
        resp = client.get("/api/experiments/1")
        assert resp.status_code == 200
        assert resp.json()["name"] == "exp_detail"

    def test_experiment_not_found(self, tmp_workspace: str) -> None:
        from alpha_lab.experiment_db import ExperimentDB
        import alpha_lab.server as srv

        db = ExperimentDB(str(Path(tmp_workspace) / "test.db"))
        srv.manager.db = db

        client = get_client()
        resp = client.get("/api/experiments/999")
        assert resp.status_code == 404

    def test_leaderboard_no_db(self, tmp_workspace: str) -> None:
        client = get_client()
        resp = client.get("/api/leaderboard")
        assert resp.status_code == 200
        assert resp.json()["leaderboard"] == []

    def test_leaderboard_with_results(self, tmp_workspace: str) -> None:
        from alpha_lab.experiment_db import ExperimentDB
        import alpha_lab.server as srv

        db = ExperimentDB(str(Path(tmp_workspace) / "test.db"))
        eid = db.create("leader_exp", "D", "H", "{}")
        db.set_results(eid, '{"sharpe": 2.0, "max_drawdown": -0.1}')
        srv.manager.db = db

        client = get_client()
        resp = client.get("/api/leaderboard")
        assert resp.status_code == 200
        leaders = resp.json()["leaderboard"]
        assert len(leaders) == 1
        assert leaders[0]["metrics"]["sharpe"] == 2.0
