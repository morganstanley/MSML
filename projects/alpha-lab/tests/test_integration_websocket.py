"""Integration tests for WebSocket connections."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import alpha_lab.server as srv
from alpha_lab.server import app


@pytest.fixture(autouse=True)
def setup_ws_server(tmp_workspace: str) -> None:
    """Reset server globals before each test."""
    srv._workspace = tmp_workspace
    srv._config_path = None
    srv.event_history.clear()
    srv.connected_websockets.clear()
    srv.manager = srv.AgentManager()
    srv.manager.db = None


def get_client():
    from fastapi.testclient import TestClient
    return TestClient(app)


class TestWebSocketConnect:
    def test_connect_and_disconnect(self, tmp_workspace: str) -> None:
        client = get_client()
        with client.websocket_connect("/ws") as ws:
            # Connection should succeed — just disconnect cleanly
            pass
        # No error means success


class TestWebSocketHistory:
    def test_receives_history_on_connect(self, tmp_workspace: str) -> None:
        # Pre-populate event_history
        srv.event_history.extend([
            {"type": "status", "status": "starting", "detail": "Agent starting"},
            {"type": "status", "status": "thinking", "detail": "Calling API..."},
            {"type": "agent_text", "delta": "Hello", "full_text": "Hello"},
        ])

        client = get_client()
        with client.websocket_connect("/ws") as ws:
            received = []
            # Receive all 3 history events
            for _ in range(3):
                data = ws.receive_json()
                received.append(data)

            assert len(received) == 3
            assert received[0]["type"] == "status"
            assert received[0]["status"] == "starting"
            assert received[2]["type"] == "agent_text"


class TestWebSocketCommands:
    def test_send_stop_command(self, tmp_workspace: str) -> None:
        client = get_client()
        with client.websocket_connect("/ws") as ws:
            # Send stop command — should not error
            ws.send_json({"type": "stop"})
            # If we get here without exception, the command was handled
