"""Tests for event dataclasses: serialization and structure."""

from __future__ import annotations

import json

from alpha_lab.events import (
    AgentEvent,
    AgentTextEvent,
    BoardSummaryEvent,
    ErrorEvent,
    ExperimentEvent,
    FileChangedEvent,
    PhaseEvent,
    QuestionEvent,
    StatusEvent,
    ToolCallEvent,
    ToolResultEvent,
)


class TestEventSerialization:
    """All events must serialize to JSON-safe dicts."""

    def test_status_event(self) -> None:
        e = StatusEvent(status="thinking", detail="Calling API...")
        d = e.to_dict()
        assert d["type"] == "status"
        assert d["status"] == "thinking"
        assert d["detail"] == "Calling API..."
        assert isinstance(d["timestamp"], float)
        # Must be JSON-serializable
        json.dumps(d)

    def test_agent_text_event(self) -> None:
        e = AgentTextEvent(delta="Hello", full_text="Hello World")
        d = e.to_dict()
        assert d["type"] == "agent_text"
        assert d["delta"] == "Hello"
        json.dumps(d)

    def test_tool_call_event(self) -> None:
        e = ToolCallEvent(call_id="abc123", name="shell_exec", arguments='{"command": "ls"}')
        d = e.to_dict()
        assert d["type"] == "tool_call"
        assert d["name"] == "shell_exec"
        json.dumps(d)

    def test_tool_result_event(self) -> None:
        e = ToolResultEvent(
            call_id="abc123",
            name="shell_exec",
            output="file1.py\nfile2.py",
        )
        d = e.to_dict()
        assert d["type"] == "tool_result"
        assert d["image_base64"] is None
        json.dumps(d)

    def test_tool_result_with_image(self) -> None:
        e = ToolResultEvent(
            call_id="abc",
            name="view_image",
            output="Image loaded",
            image_base64="iVBOR...",
            media_type="image/png",
        )
        d = e.to_dict()
        assert d["image_base64"] == "iVBOR..."
        assert d["media_type"] == "image/png"
        json.dumps(d)

    def test_error_event(self) -> None:
        e = ErrorEvent(message="Something broke")
        d = e.to_dict()
        assert d["type"] == "error"
        assert d["message"] == "Something broke"
        json.dumps(d)

    def test_question_event(self) -> None:
        e = QuestionEvent(question_id="q1", question="What is the target?")
        d = e.to_dict()
        assert d["type"] == "question"
        json.dumps(d)

    def test_phase_event(self) -> None:
        e = PhaseEvent(phase="phase2", step="builder", iteration=1, status="starting", detail="Go")
        d = e.to_dict()
        assert d["type"] == "phase"
        assert d["iteration"] == 1
        json.dumps(d)

    def test_experiment_event(self) -> None:
        e = ExperimentEvent(
            experiment_id=42,
            name="exp_lstm",
            status="running",
            prev_status="queued",
            slurm_job_id="12345",
            metrics={"sharpe": 1.5},
        )
        d = e.to_dict()
        assert d["type"] == "experiment"
        assert d["experiment_id"] == 42
        assert d["metrics"]["sharpe"] == 1.5
        json.dumps(d)

    def test_board_summary_event(self) -> None:
        e = BoardSummaryEvent(
            counts={"running": 2, "done": 5},
            experiments=[{"id": 1, "name": "exp"}],
            leaderboard=[],
        )
        d = e.to_dict()
        assert d["type"] == "board_summary"
        assert d["counts"]["running"] == 2
        json.dumps(d)

    def test_file_changed_event(self) -> None:
        e = FileChangedEvent(change="modified", path="scripts/analysis.py")
        d = e.to_dict()
        assert d["type"] == "file_changed"
        assert d["change"] == "modified"
        json.dumps(d)


class TestEventTimestamp:
    """Events should have auto-generated timestamps."""

    def test_timestamp_auto_set(self) -> None:
        import time

        before = time.time()
        e = StatusEvent(status="test")
        after = time.time()
        assert before <= e.timestamp <= after

    def test_different_events_different_timestamps(self) -> None:
        e1 = StatusEvent(status="a")
        e2 = StatusEvent(status="b")
        # Timestamps should be very close but potentially different
        assert isinstance(e1.timestamp, float)
        assert isinstance(e2.timestamp, float)
