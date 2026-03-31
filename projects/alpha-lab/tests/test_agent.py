"""Tests for AgentLoop: counter resets, nudge limits, stop behavior, tool dispatch."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from alpha_lab.agent import (
    CONTINUE_MESSAGE,
    MAX_CONSECUTIVE_NUDGES,
    MAX_CONSECUTIVE_TOOL_CALLS,
    AgentLoop,
)
from alpha_lab.context import ContextManager
from alpha_lab.events import (
    AgentEvent,
    AgentTextEvent,
    ErrorEvent,
    StatusEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from alpha_lab.provider import Response, ToolCall


def _make_mock_provider():
    """Create a mock provider that passes through build_tool_result_items."""
    provider = MagicMock()

    def _build_tool_result_items(results, images=None):
        # Mimic OpenAI format so tests can inspect output
        items = []
        for r in results:
            items.append({
                "type": "function_call_output",
                "call_id": r["call_id"],
                "output": r["output"],
            })
        return items

    provider.build_tool_result_items.side_effect = _build_tool_result_items
    provider.build_user_items.side_effect = lambda msg: [{"role": "user", "content": msg}]
    return provider


@pytest.fixture()
def mock_provider() -> MagicMock:
    return _make_mock_provider()


@pytest.fixture()
def ctx(tmp_workspace: str, mock_provider: MagicMock) -> ContextManager:
    return ContextManager(
        provider=mock_provider,
        model="gpt-4o",
        workspace=tmp_workspace,
    )


@pytest.fixture()
def events() -> list[AgentEvent]:
    return []


@pytest.fixture()
def agent(mock_provider: MagicMock, ctx: ContextManager, events: list[AgentEvent]) -> AgentLoop:
    return AgentLoop(
        provider=mock_provider,
        model="gpt-4o",
        context=ctx,
        event_callback=lambda e: events.append(e),
        min_report_attempts=1,  # allow quick finish for tests
    )


class TestAgentInit:
    def test_initial_state(self, agent: AgentLoop) -> None:
        assert agent._done is False
        assert agent._stop_requested is False
        assert agent._consecutive_tool_calls == 0
        assert agent._consecutive_nudges == 0
        assert agent._report_attempts == 0


class TestAgentStop:
    def test_stop_sets_flag(self, agent: AgentLoop) -> None:
        agent.stop()
        assert agent._stop_requested is True

    def test_stop_unblocks_question(self, agent: AgentLoop) -> None:
        """Calling stop should unblock _ask_user_fn."""
        agent._question_event.clear()

        def ask_in_thread():
            result = agent._ask_user_fn("Blocked?")
            return result

        t = threading.Thread(target=ask_in_thread)
        t.start()
        agent.stop()
        t.join(timeout=2)
        assert not t.is_alive()


class TestAgentProvideAnswer:
    def test_provide_answer(self, agent: AgentLoop) -> None:
        agent._question_event.clear()
        answers = []
        started = threading.Event()

        def ask_in_thread():
            started.set()
            answers.append(agent._ask_user_fn("question"))

        t = threading.Thread(target=ask_in_thread)
        t.start()
        started.wait(timeout=2)  # Ensure thread is running before providing answer
        import time; time.sleep(0.05)  # Small delay for _ask_user_fn to reach wait()
        agent.provide_answer("the answer")
        t.join(timeout=2)
        assert answers == ["the answer"]


class TestAgentToolCallHandling:
    def test_report_to_user_sets_done(self, agent: AgentLoop) -> None:
        """report_to_user tool call should set _done=True."""
        tool_calls = [
            ToolCall(call_id="c1", name="report_to_user", arguments='{"summary": "done"}')
        ]
        agent._report_attempts = 0  # min_report_attempts=1
        result = agent._handle_tool_calls(tool_calls)
        assert agent._done is True

    def test_report_first_attempt_nudge(self, agent: AgentLoop) -> None:
        """With min_report_attempts=2, first report should be nudged."""
        agent.min_report_attempts = 2
        tool_calls = [
            ToolCall(call_id="c1", name="report_to_user", arguments='{"summary": "done"}')
        ]
        result = agent._handle_tool_calls(tool_calls)
        assert agent._done is False  # Not done yet — first attempt
        assert "review your plan.md" in result[0]["output"].lower()

    def test_tool_call_counter_tracks(self, agent: AgentLoop) -> None:
        """Each tool call increments the counter."""
        tool_calls = [
            ToolCall(call_id="c1", name="shell_exec", arguments='{"command": "echo hi"}')
        ]
        agent._handle_tool_calls(tool_calls)
        assert agent._consecutive_tool_calls == 1

    def test_runaway_tool_calls_capped(self, agent: AgentLoop) -> None:
        """After MAX_CONSECUTIVE_TOOL_CALLS, the agent gets a stop message."""
        agent._consecutive_tool_calls = MAX_CONSECUTIVE_TOOL_CALLS
        tool_calls = [
            ToolCall(call_id="c1", name="shell_exec", arguments='{"command": "echo hi"}')
        ]
        result = agent._handle_tool_calls(tool_calls)
        assert "many consecutive tool calls" in result[0]["output"].lower()

    def test_tool_exception_handled(self, agent: AgentLoop) -> None:
        """execute_tool exceptions should be caught and returned as error text."""
        tool_calls = [
            ToolCall(call_id="c1", name="view_image", arguments='{"path": "/nonexistent.png"}')
        ]
        result = agent._handle_tool_calls(tool_calls)
        # Should not raise, should have error in output
        assert any("[ERROR]" in item.get("output", "") for item in result if isinstance(item, dict))


class TestAgentEventEmission:
    def test_emits_starting_event(self, agent: AgentLoop, events: list[AgentEvent]) -> None:
        """run() should emit a 'starting' StatusEvent."""
        # Mock API to return None (no response)
        agent._call_api = MagicMock(return_value=None)
        agent.run("test message")

        status_events = [e for e in events if isinstance(e, StatusEvent)]
        assert any(e.status == "starting" for e in status_events)

    def test_emits_done_only_when_done(self, agent: AgentLoop, events: list[AgentEvent]) -> None:
        """'done' StatusEvent should only be emitted when _done is True."""
        agent._call_api = MagicMock(return_value=None)
        agent.run("test message")

        # Agent didn't finish successfully — should NOT emit 'done'
        status_events = [e for e in events if isinstance(e, StatusEvent)]
        done_events = [e for e in status_events if e.status == "done"]
        assert len(done_events) == 0  # API returned None, agent didn't complete

    def test_emits_error_on_unexpected_stop(self, agent: AgentLoop, events: list[AgentEvent]) -> None:
        """If agent stops without _done or _stop_requested, should emit error."""
        agent._call_api = MagicMock(return_value=None)
        agent.run("test message")

        status_events = [e for e in events if isinstance(e, StatusEvent)]
        # Should have an error or unexpected-stop status
        assert any(e.status == "error" for e in status_events)

    def test_emits_stopped_on_stop(self, agent: AgentLoop, events: list[AgentEvent]) -> None:
        """If stop() is called, should emit 'stopped' status."""
        agent._stop_requested = True
        agent._call_api = MagicMock(return_value=None)
        agent.run("test message")

        status_events = [e for e in events if isinstance(e, StatusEvent)]
        assert any(e.status == "stopped" for e in status_events)


class TestAgentNudgeLimit:
    def test_nudge_limit_stops_agent(self, agent: AgentLoop, events: list[AgentEvent]) -> None:
        """After MAX_CONSECUTIVE_NUDGES nudges, the agent should stop."""
        agent._consecutive_nudges = MAX_CONSECUTIVE_NUDGES - 1

        # Create a normalized Response (what _call_api returns)
        mock_response = Response(
            id="resp_test",
            text="Some text without tool calls",
            tool_calls=[],
            has_web_search=False,
            input_tokens=100,
            output_tokens=50,
            raw_output_items=[{"text": "Some text without tool calls"}],
        )

        agent._call_api = MagicMock(side_effect=[mock_response, None])
        agent.send_user_message("test")

        # After the nudge limit, the agent should be done
        assert agent._done is True
        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert any("stuck" in e.message.lower() for e in error_events)


class TestAgentBuildInstructions:
    def test_includes_summary_when_available(self, agent: AgentLoop) -> None:
        agent.context.summary = "Previous conversation about data analysis"
        instructions = agent._build_system_instructions()
        assert "Previous conversation about data analysis" in instructions
        assert "Conversation Summary" in instructions

    def test_no_summary_section_when_none(self, agent: AgentLoop) -> None:
        agent.context.summary = None
        instructions = agent._build_system_instructions()
        assert "Conversation Summary" not in instructions
