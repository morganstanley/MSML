"""Typed event dataclasses emitted by the agent loop."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AgentEvent:
    """Base class for all agent events."""

    type: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StatusEvent(AgentEvent):
    """Agent status change: starting, thinking, tool_executing, done, error."""

    type: str = "status"
    status: str = ""  # starting | thinking | tool_executing | done | error
    detail: str = ""


@dataclass
class AgentTextEvent(AgentEvent):
    """Streaming text delta from the agent."""

    type: str = "agent_text"
    delta: str = ""
    full_text: str = ""


@dataclass
class ApiRequestEvent(AgentEvent):
    """Full API request payload sent to OpenAI."""

    type: str = "api_request"
    model: str = ""
    instructions: str = ""
    input: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    previous_response_id: str | None = None
    reasoning_effort: str = ""


@dataclass
class ApiResponseEvent(AgentEvent):
    """Full API response received from OpenAI."""

    type: str = "api_response"
    response_id: str = ""
    output: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallEvent(AgentEvent):
    """Agent is about to call a tool."""

    type: str = "tool_call"
    call_id: str = ""
    name: str = ""
    arguments: str = ""  # raw JSON


@dataclass
class ToolResultEvent(AgentEvent):
    """Result from a tool execution."""

    type: str = "tool_result"
    call_id: str = ""
    name: str = ""
    output: str = ""
    image_base64: str | None = None
    media_type: str | None = None


@dataclass
class QuestionEvent(AgentEvent):
    """Agent is asking the user a question."""

    type: str = "question"
    question_id: str = ""
    question: str = ""


@dataclass
class ErrorEvent(AgentEvent):
    """An error occurred."""

    type: str = "error"
    message: str = ""


@dataclass
class PhaseEvent(AgentEvent):
    """Pipeline phase transition event."""

    type: str = "phase"
    phase: str = ""        # "phase1", "phase2"
    step: str = ""         # "builder", "critic", "tester", "complete"
    iteration: int = 0
    status: str = ""       # "starting", "completed"
    detail: str = ""


@dataclass
class ExperimentEvent(AgentEvent):
    """An experiment changed kanban state."""

    type: str = "experiment"
    experiment_id: int = 0
    name: str = ""
    status: str = ""
    prev_status: str = ""
    detail: str = ""
    worker_id: str = ""
    slurm_job_id: str = ""
    metrics: dict = field(default_factory=dict)


@dataclass
class BoardSummaryEvent(AgentEvent):
    """Periodic board state snapshot."""

    type: str = "board_summary"
    counts: dict = field(default_factory=dict)
    experiments: list = field(default_factory=list)
    leaderboard: list = field(default_factory=list)


@dataclass
class FileChangedEvent(AgentEvent):
    """A file in the workspace changed (from file watcher)."""

    type: str = "file_changed"
    change: str = ""  # added | modified | deleted
    path: str = ""  # relative to workspace
