"""Tests for AgentLoop: counter resets, nudge limits, stop behavior, tool dispatch."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider as _TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from alpha_lab.agent import (
    CONTINUE_MESSAGE,
    MAX_CONSECUTIVE_NUDGES,
    MAX_CONSECUTIVE_TOOL_CALLS,
    AgentLoop,
    build_loop_kwargs,
)
from alpha_lab.agents.agent_definition import AgentDefinition
from alpha_lab.context import ContextManager
from alpha_lab.events import (
    AgentEvent,
    AgentTextEvent,
    ApiResponseEvent,
    ErrorEvent,
    StatusEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from alpha_lab.provider import Response, StreamEvent, ToolCall
from alpha_lab.tools import load_tool
from alpha_lab.tools.tool_definition import ToolDefinition


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

    def test_phase1_adapter_phase1_md_is_loaded(
        self, mock_provider: MagicMock, ctx: ContextManager,
    ) -> None:
        """Regression: the Phase 1 AgentLoop path uses the default
        build_system_prompt, and that builder must receive ``self.adapter``
        so the customized ``adapter/phase1.md`` actually reaches the LLM.

        Before this fix, ``_build_system_instructions`` called
        ``self.prompt_builder(workspace, learnings, config)`` with no
        adapter, so ``build_system_prompt`` saw ``adapter=None`` and
        always fell back to ``SYSTEM_PROMPT_BASE`` — silently ignoring
        Phase 0's customized phase1 prompt.
        """
        from alpha_lab.adapter import (
            DomainAdapter, ExperimentStructure, MetricConfig,
        )

        sentinel = "PHASE1_CUSTOMIZED_PROMPT_SENTINEL"
        adapter = DomainAdapter(
            domain_name="test_domain",
            domain_description="d",
            prompts={"phase1": sentinel + "\n\n## Tools\n- shell_exec"},
            metric=MetricConfig(
                primary_metric="sharpe", direction="maximize",
                extract_key="sharpe", display_name="Sharpe",
                secondary_metrics=[],
            ),
            experiment=ExperimentStructure(
                required_files=["strategy.py", "run_experiment.py"],
                entry_point="run_experiment.py",
                results_dir="results",
                results_file="metrics.json",
                framework_dir="backtest",
                framework_files=[],
            ),
            phase2_framework_description="f",
            domain_knowledge="",
            phase2_review_file="review.md",
        )
        agent = AgentLoop(
            provider=mock_provider,
            model="gpt-4o",
            context=ctx,
            event_callback=lambda e: None,
            min_report_attempts=1,
            adapter=adapter,
        )
        instructions = agent._build_system_instructions()
        assert sentinel in instructions, (
            "Default build_system_prompt did not receive the adapter; "
            "the customized phase1.md is being silently dropped at "
            "agent.py:_build_system_instructions."
        )

    def test_closure_prompt_builder_receives_adapter_as_noop_kwarg(
        self, mock_provider: MagicMock, ctx: ContextManager,
    ) -> None:
        """Closure prompt_builders (worker/strategist/pipeline/supervisor/
        phase0) declare a 4th ``adapter`` parameter for the uniform call
        contract. They typically capture adapter from enclosing scope and
        ignore the kwarg, but the parameter must be present so the
        AgentLoop's unconditional ``adapter=self.adapter`` call doesn't
        TypeError.
        """
        sentinel_adapter = object()
        seen: dict = {}

        def closure_builder(workspace, learnings, config, adapter=None):
            seen["called_with_workspace"] = workspace
            seen["received_adapter"] = adapter
            return "closure prompt content"

        agent = AgentLoop(
            provider=mock_provider,
            model="gpt-4o",
            context=ctx,
            event_callback=lambda e: None,
            prompt_builder=closure_builder,
            adapter=sentinel_adapter,
            min_report_attempts=1,
        )
        out = agent._build_system_instructions()
        assert "closure prompt content" in out
        assert seen["called_with_workspace"] == agent.context.workspace
        assert seen["received_adapter"] is sentinel_adapter


# ---------------------------------------------------------------------------
# Tracing instrumentation tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def trace_exporter():
    """Wire up an in-memory OTel TracerProvider for test inspection.

    Patches `alpha_lab.agent.get_tracer` so that any AgentLoop constructed
    during the test (including sub-agents) emits spans to the in-memory
    exporter.  Yields the exporter so tests can call
    `trace_exporter.get_finished_spans()`.
    """
    exporter = InMemorySpanExporter()
    provider = _TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("alpha_lab", "0.0.0-test")

    with patch("alpha_lab.agent.get_tracer", return_value=tracer):
        yield exporter

    provider.shutdown()


@pytest.fixture()
def traced_agent(
    mock_provider: MagicMock,
    ctx: ContextManager,
    events: list[AgentEvent],
    trace_exporter: InMemorySpanExporter,
) -> AgentLoop:
    """An AgentLoop wired to the in-memory trace exporter."""
    return AgentLoop(
        provider=mock_provider,
        model="gpt-4o",
        context=ctx,
        event_callback=lambda e: events.append(e),
        min_report_attempts=1,
    )


class TestAgentTracing:
    """Validates that OTel spans are emitted with correct names, kinds, and attributes."""

    def test_run_creates_invoke_agent_span(
        self, traced_agent: AgentLoop, trace_exporter: InMemorySpanExporter
    ) -> None:
        """run() should create an invoke_agent span with GenAI semconv attributes."""
        traced_agent._call_api = MagicMock(return_value=None)
        traced_agent.run("hello")

        spans = trace_exporter.get_finished_spans()
        agent_spans = [s for s in spans if s.name.startswith("invoke_agent")]
        assert len(agent_spans) == 1

        span = agent_spans[0]
        assert span.attributes["gen_ai.operation.name"] == "invoke_agent"
        assert span.attributes["gen_ai.agent.name"] == "conversation"
        assert span.attributes["gen_ai.request.model"] == "gpt-4o"
        assert span.attributes["alpha_lab.agent.depth"] == 0

    def test_invoke_agent_span_records_error_on_exception(
        self, traced_agent: AgentLoop, trace_exporter: InMemorySpanExporter
    ) -> None:
        """Unhandled exception in run() should set ERROR status and record exception."""
        from opentelemetry.trace import StatusCode

        traced_agent._call_api = MagicMock(side_effect=RuntimeError("boom"))
        traced_agent.run("trigger error")

        spans = trace_exporter.get_finished_spans()
        agent_spans = [s for s in spans if s.name.startswith("invoke_agent")]
        assert len(agent_spans) == 1

        span = agent_spans[0]
        assert span.status.status_code == StatusCode.ERROR
        # Exception should be recorded as a span event
        exception_events = [e for e in span.events if e.name == "exception"]
        assert len(exception_events) == 1
        assert "boom" in exception_events[0].attributes["exception.message"]

    def test_call_api_creates_chat_span_with_tokens(
        self, traced_agent: AgentLoop, trace_exporter: InMemorySpanExporter
    ) -> None:
        """_call_api() should create a chat span and record token usage."""
        from alpha_lab.provider import StreamEvent, Response as ProviderResponse

        mock_response = ProviderResponse(
            id="resp_1",
            text="Hello",
            tool_calls=[],
            has_web_search=False,
            input_tokens=150,
            output_tokens=42,
            raw_output_items=[],
        )

        def fake_stream(*args, **kwargs):
            yield StreamEvent(type="text_delta", delta="Hello")
            yield StreamEvent(type="done", response=mock_response)

        traced_agent.provider.stream_response.side_effect = fake_stream
        traced_agent.provider.append_response_to_history = MagicMock()

        # Call _call_api directly to isolate the chat span
        result = traced_agent._call_api([{"role": "user", "content": "hi"}])

        assert result is not None
        spans = trace_exporter.get_finished_spans()
        chat_spans = [s for s in spans if s.name.startswith("chat")]
        assert len(chat_spans) == 1

        span = chat_spans[0]
        assert span.attributes["gen_ai.operation.name"] == "chat"
        assert span.attributes["gen_ai.request.model"] == "gpt-4o"
        assert span.attributes["gen_ai.usage.input_tokens"] == 150
        assert span.attributes["gen_ai.usage.output_tokens"] == 42

    def test_call_api_retry_creates_multiple_chat_spans(
        self, traced_agent: AgentLoop, trace_exporter: InMemorySpanExporter
    ) -> None:
        """Each retry attempt should produce its own chat span."""
        from alpha_lab.provider import StreamEvent, Response as ProviderResponse

        mock_response = ProviderResponse(
            id="resp_2",
            text="recovered",
            tool_calls=[],
            has_web_search=False,
            input_tokens=80,
            output_tokens=20,
            raw_output_items=[],
        )

        call_count = 0

        def fake_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("transient failure")
            yield StreamEvent(type="text_delta", delta="recovered")
            yield StreamEvent(type="done", response=mock_response)

        traced_agent.provider.stream_response.side_effect = fake_stream
        traced_agent.provider.append_response_to_history = MagicMock()

        # Patch sleep to avoid test delay
        with patch("alpha_lab.agent.time.sleep"):
            result = traced_agent._call_api([{"role": "user", "content": "retry me"}])

        assert result is not None
        spans = trace_exporter.get_finished_spans()
        chat_spans = [s for s in spans if s.name.startswith("chat")]
        assert len(chat_spans) == 2  # one failed, one succeeded

        from opentelemetry.trace import StatusCode

        # First span should have ERROR status
        assert chat_spans[0].status.status_code == StatusCode.ERROR
        # Second span should be OK (unset = success)
        assert chat_spans[1].status.status_code == StatusCode.UNSET

    def test_tool_execution_creates_tool_span(
        self, traced_agent: AgentLoop, trace_exporter: InMemorySpanExporter
    ) -> None:
        """_handle_tool_calls() should create execute_tool spans with tool metadata."""
        tool_calls = [
            ToolCall(call_id="tc_1", name="shell_exec", arguments='{"command": "echo hi"}')
        ]
        traced_agent._handle_tool_calls(tool_calls)

        spans = trace_exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.name.startswith("execute_tool")]
        assert len(tool_spans) == 1

        span = tool_spans[0]
        assert span.attributes["gen_ai.operation.name"] == "execute_tool"
        assert span.attributes["gen_ai.tool.name"] == "shell_exec"
        assert span.attributes["gen_ai.tool.call.id"] == "tc_1"

    def test_tool_error_records_exception_on_span(
        self, traced_agent: AgentLoop, trace_exporter: InMemorySpanExporter
    ) -> None:
        """Tool exceptions should set ERROR status and record the exception."""
        from opentelemetry.trace import StatusCode

        tool_calls = [
            ToolCall(
                call_id="tc_err",
                name="shell_exec",
                arguments='{"command": "echo hi"}',
            )
        ]
        # Force execute_tool to raise an unhandled exception
        with patch("alpha_lab.agent.execute_tool", side_effect=RuntimeError("unexpected")):
            traced_agent._handle_tool_calls(tool_calls)

        spans = trace_exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.name.startswith("execute_tool")]
        assert len(tool_spans) == 1

        span = tool_spans[0]
        assert span.status.status_code == StatusCode.ERROR
        exception_events = [e for e in span.events if e.name == "exception"]
        assert len(exception_events) == 1

    def test_multiple_tools_create_multiple_spans(
        self, traced_agent: AgentLoop, trace_exporter: InMemorySpanExporter
    ) -> None:
        """Multiple tool calls should each get their own span."""
        tool_calls = [
            ToolCall(call_id="tc_a", name="shell_exec", arguments='{"command": "echo a"}'),
            ToolCall(call_id="tc_b", name="shell_exec", arguments='{"command": "echo b"}'),
        ]
        traced_agent._handle_tool_calls(tool_calls)

        spans = trace_exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.name.startswith("execute_tool")]
        assert len(tool_spans) == 2
        assert tool_spans[0].attributes["gen_ai.tool.call.id"] == "tc_a"
        assert tool_spans[1].attributes["gen_ai.tool.call.id"] == "tc_b"

    def test_full_turn_span_hierarchy(
        self, traced_agent: AgentLoop, trace_exporter: InMemorySpanExporter
    ) -> None:
        """A full run() with one API call + tool calls should produce nested spans."""
        from alpha_lab.provider import StreamEvent, Response as ProviderResponse

        mock_response = ProviderResponse(
            id="resp_full",
            text="",
            tool_calls=[
                ToolCall(call_id="tc_h1", name="report_to_user", arguments='{"summary": "done"}')
            ],
            has_web_search=False,
            input_tokens=200,
            output_tokens=30,
            raw_output_items=[],
        )

        def fake_stream(*args, **kwargs):
            yield StreamEvent(type="done", response=mock_response)

        traced_agent.provider.stream_response.side_effect = fake_stream
        traced_agent.provider.append_response_to_history = MagicMock()
        traced_agent.run("do something")

        spans = trace_exporter.get_finished_spans()
        agent_spans = [s for s in spans if s.name.startswith("invoke_agent")]
        chat_spans = [s for s in spans if s.name.startswith("chat")]
        tool_spans = [s for s in spans if s.name.startswith("execute_tool")]

        assert len(agent_spans) == 1
        assert len(chat_spans) == 1
        assert len(tool_spans) == 1

        # Verify parent-child: chat and tool spans should be children of invoke_agent
        parent_ctx = agent_spans[0].context
        assert chat_spans[0].parent.trace_id == parent_ctx.trace_id
        assert chat_spans[0].parent.span_id == parent_ctx.span_id
        assert tool_spans[0].parent.trace_id == parent_ctx.trace_id

class TestApiResponseUsageCacheFields:
    def test_api_response_usage_omits_cache_fields_when_zero(
        self,
        agent: AgentLoop,
        events: list[AgentEvent],
        mock_provider: MagicMock,
    ) -> None:
        response = Response(
            id="resp_zero_cache",
            text="ok",
            tool_calls=[],
            has_web_search=False,
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=0,
            cache_write_input_tokens=0,
            raw_output_items=[{"text": "ok"}],
        )
        mock_provider.stream_response.return_value = iter([
            StreamEvent(type="done", response=response),
        ])

        result = agent._call_api([{"role": "user", "content": "hello"}])
        assert result is response

        api_response_events = [e for e in events if isinstance(e, ApiResponseEvent)]
        assert len(api_response_events) == 1
        usage = api_response_events[0].usage
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50
        assert "cache_read_input_tokens" not in usage
        assert "cache_write_input_tokens" not in usage

    def test_api_response_usage_includes_cache_fields_when_positive(
        self,
        agent: AgentLoop,
        events: list[AgentEvent],
        mock_provider: MagicMock,
    ) -> None:
        response = Response(
            id="resp_positive_cache",
            text="ok",
            tool_calls=[],
            has_web_search=False,
            input_tokens=1000,
            output_tokens=200,
            cache_read_input_tokens=850,
            cache_write_input_tokens=120,
            raw_output_items=[{"text": "ok"}],
        )
        mock_provider.stream_response.return_value = iter([
            StreamEvent(type="done", response=response),
        ])

        result = agent._call_api([{"role": "user", "content": "hello"}])
        assert result is response

        api_response_events = [e for e in events if isinstance(e, ApiResponseEvent)]
        assert len(api_response_events) == 1
        usage = api_response_events[0].usage
        assert usage["input_tokens"] == 1000
        assert usage["output_tokens"] == 200
        assert usage["cache_read_input_tokens"] == 850
        assert usage["cache_write_input_tokens"] == 120


def _make_agent_definition(
    *,
    tools: tuple[ToolDefinition, ...] | None = None,
    reasoning_effort: str | None = None,
    log_name: str = "conversation",
    min_report_attempts: int = 2,
) -> AgentDefinition:
    return AgentDefinition(
        name="explorer",
        description="d",
        tools=tools if tools is not None else (),
        include_web_search=False,
        reasoning_effort=reasoning_effort,
        log_name=log_name,
        min_report_attempts=min_report_attempts,
        prompt_source="adapter:phase1",
        prompt_body="",
    )


class TestBuildLoopKwargs:
    def test_carries_log_name_and_min_report_attempts(self) -> None:
        agent_definition = _make_agent_definition(
            log_name="phase1", min_report_attempts=4
        )

        kwargs = build_loop_kwargs(agent_definition)

        assert kwargs["log_name"] == "phase1"
        assert kwargs["min_report_attempts"] == 4
        assert kwargs["tools"] == []

    def test_omits_reasoning_effort_when_unset_and_no_default(self) -> None:
        agent_definition = _make_agent_definition(reasoning_effort=None)

        kwargs = build_loop_kwargs(agent_definition)

        assert "reasoning_effort" not in kwargs

    def test_definition_reasoning_effort_wins_over_default(self) -> None:
        agent_definition = _make_agent_definition(reasoning_effort="medium")

        kwargs = build_loop_kwargs(agent_definition, reasoning_effort_default="low")

        assert kwargs["reasoning_effort"] == "medium"

    def test_falls_back_to_default_when_definition_unset(self) -> None:
        agent_definition = _make_agent_definition(reasoning_effort=None)

        kwargs = build_loop_kwargs(agent_definition, reasoning_effort_default="high")

        assert kwargs["reasoning_effort"] == "high"

    def test_log_name_and_tools_overrides(self) -> None:
        agent_definition = _make_agent_definition(
            tools=(load_tool("shell_exec"),), log_name="base"
        )

        kwargs = build_loop_kwargs(
            agent_definition, log_name="base_1", tools=()
        )

        assert kwargs["log_name"] == "base_1"
        assert kwargs["tools"] == []
