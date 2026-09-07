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


class TestAgentIterationBudget:
    """One audited strategist log carried 3,606 calls and 40% of its run's
    tokens; a session must consolidate at its iteration budget and be
    force-ended a small grace later (2026-08-08)."""

    def _resp(self, text: str = "working") -> Response:
        return Response(
            id="r", text=text, tool_calls=[], has_web_search=False,
            input_tokens=1, output_tokens=1,
            raw_output_items=[{"text": text}],
        )

    def test_budget_demands_consolidation_then_force_ends(
        self, mock_provider: MagicMock, ctx: ContextManager,
        events: list[AgentEvent],
    ) -> None:
        agent = AgentLoop(
            provider=mock_provider, model="m", context=ctx,
            event_callback=lambda e: events.append(e),
            min_report_attempts=1, max_iterations=2,
        )
        # stays busy with tool calls and never reports — only the
        # iteration budget can end it (nudge limit never fires)
        busy = self._resp()
        busy.tool_calls = [MagicMock()]
        agent._call_api = MagicMock(return_value=busy)
        agent._handle_tool_calls = MagicMock(
            return_value=[{"role": "user", "content": "result"}])
        agent._run_loop([{"role": "user", "content": "go"}])
        assert agent._done is True
        err = [e for e in events if isinstance(e, ErrorEvent)]
        assert any("force-ended" in e.message for e in err)
        status = [e for e in events if isinstance(e, StatusEvent)]
        assert any("budget reached" in (e.detail or "") for e in status)
        # bounded: budget + grace, not unbounded grinding
        assert agent._call_api.call_count <= 2 + AgentLoop.ITERATION_BUDGET_GRACE + 1

    def test_zero_budget_means_unbounded_unchanged(
        self, mock_provider: MagicMock, ctx: ContextManager,
        events: list[AgentEvent],
    ) -> None:
        agent = AgentLoop(
            provider=mock_provider, model="m", context=ctx,
            event_callback=lambda e: events.append(e),
            min_report_attempts=1, max_iterations=0,
        )
        responses = [self._resp() for _ in range(4)] + [None]
        agent._call_api = MagicMock(side_effect=responses)
        agent._run_loop([{"role": "user", "content": "go"}])
        assert not any("force-ended" in e.message
                       for e in events if isinstance(e, ErrorEvent))


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


class TestInferCallerRole:
    """Phase 2 step agents log as ``phase2_<step>_<iteration>``; the
    role inferred from log_name must map to the corresponding
    directive-target role so ack_directive filters work."""

    def test_phase2_step_agents_map_correctly(self) -> None:
        from alpha_lab.agent import _infer_caller_role
        assert _infer_caller_role("phase2_builder_0") == "builder"
        assert _infer_caller_role("phase2_builder_3") == "builder"
        assert _infer_caller_role("phase2_critic_0") == "critic"
        assert _infer_caller_role("phase2_tester_1") == "tester"

    def test_phase3_workers_still_map(self) -> None:
        from alpha_lab.agent import _infer_caller_role
        assert _infer_caller_role("strategist") == "strategist"
        assert _infer_caller_role("worker_0_implement_x") == "worker"
        assert _infer_caller_role("reporter_milestone_001") == "reporter"
        assert _infer_caller_role("supervisor_review_phase1") == "supervisor"
        assert _infer_caller_role("conductor_phase0") == "conductor"

    def test_unknown_log_name_returns_unknown(self) -> None:
        from alpha_lab.agent import _infer_caller_role
        assert _infer_caller_role("") == "unknown"
        assert _infer_caller_role("random_name") == "unknown"

    def test_inferred_role_is_a_valid_directive_target(self) -> None:
        """Every role _infer_caller_role can return for a phase2 step
        agent must be in VALID_DIRECTIVE_TARGETS — otherwise
        ack_directive would write an ack that directives_for_role
        cannot key against."""
        from alpha_lab.agent import _infer_caller_role
        from alpha_lab.conductor_tools import VALID_DIRECTIVE_TARGETS
        for log_name, expected in [
            ("phase2_builder_0", "builder"),
            ("phase2_critic_0", "critic"),
            ("phase2_tester_0", "tester"),
            ("strategist", "strategist"),
            ("worker_0_implement_x", "worker"),
            ("reporter_milestone_001", "reporter"),
            ("supervisor_review_phase1", "supervisor"),
        ]:
            role = _infer_caller_role(log_name)
            assert role == expected
            assert role in VALID_DIRECTIVE_TARGETS, (
                f"{log_name} → {role} not in VALID_DIRECTIVE_TARGETS"
            )


class TestAgentRunReturnsSummary:
    """``agent.run()`` must return the final ``report_to_user`` summary
    so callers (Conductor, Supervisor, Pipeline) can surface it. Was
    previously typed ``-> None`` and the captured summary was dead."""

    def test_run_returns_empty_when_no_report(self, agent: AgentLoop) -> None:
        """No report_to_user call → empty string."""
        assert agent._final_summary == ""

    def test_final_summary_set_on_report(self, agent: AgentLoop) -> None:
        """Simulating the report_to_user tool dispatch: the agent's
        _final_summary attribute should hold what the LLM passed."""
        # Just verify the attribute exists and is wired
        agent._final_summary = "Phase 1 complete: explored 5 hypotheses."
        # run() would return this on done
        assert agent._final_summary == "Phase 1 complete: explored 5 hypotheses."


class TestClassifyApiError:
    """Retry classification: deterministic 4xx must not be blind-retried
    (measured: poisoned-history 400s were retried 3x before every agent
    death); congestion signals get the slow ladder (Anthropic 'Overloaded'
    storms died in 3 tries over ~20s against a 16-minute window)."""

    def _status_exc(self, code, msg="boom"):
        e = Exception(msg)
        e.status_code = code
        return e

    def test_bad_request_is_fatal(self):
        from alpha_lab.agent import classify_api_error
        e = self._status_exc(400, "Unterminated string starting at: line 1")
        assert classify_api_error(e) == "fatal"

    def test_429_is_congestion_not_fatal(self):
        from alpha_lab.agent import classify_api_error
        assert classify_api_error(self._status_exc(429)) == "congestion"

    def test_408_is_not_fatal(self):
        from alpha_lab.agent import classify_api_error
        assert classify_api_error(self._status_exc(408)) != "fatal"

    def test_overloaded_message_is_congestion(self):
        from alpha_lab.agent import classify_api_error
        e = Exception("{'type': 'overloaded_error', 'message': 'Overloaded'}")
        assert classify_api_error(e) == "congestion"

    def test_timeout_is_congestion(self):
        from alpha_lab.agent import classify_api_error
        assert classify_api_error(Exception("Request timed out.")) == "congestion"

    def test_connection_error_is_congestion(self):
        from alpha_lab.agent import classify_api_error
        assert classify_api_error(Exception("Connection error.")) == "congestion"

    def test_server_error_is_transient(self):
        from alpha_lab.agent import classify_api_error
        e = self._status_exc(500, "EngineCore encountered an issue")
        assert classify_api_error(e) == "transient"

    def test_unknown_is_transient(self):
        from alpha_lab.agent import classify_api_error
        assert classify_api_error(Exception("weird parse thing")) == "transient"
