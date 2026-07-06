"""Core agent loop for alpha-lab.

Provider-agnostic: talks to the Provider protocol instead of the OpenAI SDK
directly. Backed by the OpenAI Responses API.

Features:
  - ZDR-compatible mode: local conversation history tracking
  - Streaming for live token rendering
  - Configurable reasoning effort

Event-based: emits AgentEvent objects via a callback instead of rendering
to a terminal. Designed to run in a background thread.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from alpha_lab.context import ContextManager
from alpha_lab.events import (
    AgentEvent,
    AgentTextEvent,
    ApiRequestEvent,
    ApiResponseEvent,
    ErrorEvent,
    QuestionEvent,
    StatusEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from alpha_lab.agents.agent_definition import AgentDefinition
from alpha_lab.prompts import build_system_prompt
from alpha_lab.provider import Provider, ToolCall
from alpha_lab.tools import (
    DEFAULT_TIMEOUT,
    default_tool_schemas,
    execute_tool,
    get_tool_schemas,
    parse_tool_args,
)
from alpha_lab.tools.tool_definition import ToolDefinition
from alpha_lab.tracing import (
    GEN_AI_AGENT_NAME,
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_TOOL_NAME,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GenAiOperationNameValues,
    get_tracer,
    provider_to_system,
)
from opentelemetry import trace

logger = logging.getLogger("alpha_lab.agent")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
RETRY_DELAYS = [1, 5, 15]  # seconds

MAX_CONSECUTIVE_TOOL_CALLS = 50
MAX_CONSECUTIVE_NUDGES = 5

# If the model outputs text without calling report_to_user, nudge it to keep going
CONTINUE_MESSAGE = (
    "[SYSTEM] You output text without calling a tool. Do not stop. "
    "Continue your analysis by calling tools. If you are fully done, "
    "call report_to_user with your summary."
)


# TODO: Fold into AgentLoop initialization or turn into a factory function
def build_loop_kwargs(
    agent_definition: AgentDefinition,
    *,
    reasoning_effort_default: str | None = None,
    log_name: str | None = None,
    tools: Sequence[ToolDefinition] | None = None,
) -> dict[str, Any]:
    """Build the AgentLoop kwargs derived from an AgentDefinition.

    Omits ``reasoning_effort`` when it resolves to ``None`` so callers without a
    config fallback keep AgentLoop's own default.
    """
    kwargs: dict[str, Any] = {
        "tools": get_tool_schemas(
            agent_definition.tools if tools is None else tools,
            include_web_search=agent_definition.include_web_search,
        ),
        "log_name": agent_definition.log_name if log_name is None else log_name,
        "min_report_attempts": agent_definition.min_report_attempts,
    }
    reasoning_effort = agent_definition.reasoning_effort or reasoning_effort_default
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    return kwargs


class AgentLoop:
    """The main agent loop: send messages, stream responses, dispatch tools.

    Emits events via event_callback. Supports ask_user blocking via
    threading primitives. Can be stopped externally via stop().

    ``prompt_builder`` contract
    ----------------------------
    Any callable supplied as ``prompt_builder`` MUST accept four
    arguments:

        prompt_builder(
            workspace: str | None,
            learnings: str | None,
            config: Any | None = None,
            adapter: Any | None = None,
        ) -> str

    The 4th ``adapter`` parameter is required even for closures that
    capture ``adapter`` from their enclosing scope and don't read the
    kwarg — declaring ``adapter=None`` in the signature is enough.
    ``AgentLoop._build_system_instructions`` always passes
    ``adapter=self.adapter``; a builder that omits the parameter will
    raise ``TypeError`` at runtime. The default ``build_system_prompt``
    in ``prompts.py`` accepts ``adapter`` and uses it to load the
    customized ``phase1`` prompt from ``workspace/adapter/phase1.md``.
    """

    def __init__(
        self,
        provider: Provider,
        model: str,
        context: ContextManager,
        event_callback: Callable[[AgentEvent], None],
        reasoning_effort: str = "low",
        config: Any | None = None,
        tools: list[dict] | None = None,
        prompt_builder: Callable[..., str] | None = None,
        log_name: str = "conversation",
        min_report_attempts: int = 2,
        db: Any | None = None,
        metrics: Any | None = None,
        adapter: Any | None = None,
        mlflow_run_target: str | None = None,
        memory_store: Any | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.context = context
        self.event_callback = event_callback
        self.reasoning_effort = reasoning_effort
        self.config = config
        self.tools = tools if tools is not None else default_tool_schemas()
        self.prompt_builder = prompt_builder or build_system_prompt
        self.log_name = log_name
        self.min_report_attempts = min_report_attempts
        self.db = db
        self.metrics = metrics
        self.adapter = adapter
        self.memory_store = memory_store
        # MLflow Run UUID this agent's trace should attach to. ``None`` →
        # the active pipeline Run. Phase 3 workers pass their experiment's
        # sub-run UUID. Honored only when MLflow is the active backend.
        self._mlflow_run_target = mlflow_run_target

        self._depth = 0  # 0 = top-level agent, 1 = sub-agent (max)
        self._consecutive_tool_calls = 0
        self._consecutive_nudges = 0
        self._report_attempts = 0
        self._done = False
        self._stop_requested = False

        # OTel tracer — resolved once here so we don't call get_tracer() per span
        self._tracer = get_tracer()
        # Pre-compute the gen_ai.system value from config.provider
        self._gen_ai_system = provider_to_system(
            getattr(config, "provider", "") if config else ""
        )

        # For ask_user blocking
        self._question_event = threading.Event()
        self._question_answer: str = ""

        # JSONL conversation log
        self._log_file = None

        # Track conversation history locally (provider-native format)
        self._input_history: list[dict[str, Any]] = []

    def _init_log(self) -> None:
        """Open the JSONL log file in the workspace."""
        if self.context.workspace:
            log_dir = Path(self.context.workspace) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = open(log_dir / f"{self.log_name}.jsonl", "a")

    def emit(self, event: AgentEvent) -> None:
        """Emit an event via the callback and write to JSONL log."""
        self.event_callback(event)
        if self._log_file is not None:
            try:
                self._log_file.write(json.dumps(event.to_dict(), default=str) + "\n")
                self._log_file.flush()
            except (OSError, TypeError, ValueError) as e:
                logger.warning("Failed to write event to JSONL log: %s", e)

    def run(self, initial_message: str) -> None:
        """Top-level entry point for running in a background thread."""
        # When MLflow is the active backend, layer an MLflow agent-trace ctx
        # mgr around the OTel span. agent_trace is a no-op in Tempo mode, so
        # the OTel span is the only thing that runs there.
        from alpha_lab import mlflow_logger
        mlflow_ctx = mlflow_logger.agent_trace(
            f"invoke_agent {self.log_name}",
            target_run_uuid=self._mlflow_run_target,
            attributes={
                "alpha_lab.agent.name": self.log_name,
                "alpha_lab.agent.depth": self._depth,
                "alpha_lab.agent.model": self.model,
            },
        )
        with mlflow_ctx, self._tracer.start_as_current_span(
            f"invoke_agent {self.log_name}",
            kind=trace.SpanKind.INTERNAL,
            attributes={
                GEN_AI_OPERATION_NAME: GenAiOperationNameValues.INVOKE_AGENT.value,
                GEN_AI_AGENT_NAME: self.log_name,
                GEN_AI_SYSTEM: self._gen_ai_system,
                GEN_AI_REQUEST_MODEL: self.model,
                # alpha_lab.* prefix: custom namespace to avoid collision with
                # future GenAI semconv additions (Decision #6).
                "alpha_lab.agent.depth": self._depth,
            },
        ) as agent_span:
            self._init_log()
            self.emit(StatusEvent(status="starting", detail="Agent starting"))
            try:
                self.send_user_message(initial_message)
                if self._done:
                    self.emit(StatusEvent(status="done", detail="Agent finished"))
                elif self._stop_requested:
                    self.emit(StatusEvent(status="stopped", detail="Agent stopped by user"))
                else:
                    self.emit(StatusEvent(status="error", detail="Agent stopped unexpectedly"))
            except Exception as e:
                agent_span.set_status(trace.StatusCode.ERROR)
                agent_span.record_exception(e)
                if not self._stop_requested:
                    self.emit(ErrorEvent(message=f"Agent error: {e}"))
                    self.emit(StatusEvent(status="error", detail=str(e)))
            finally:
                if self._log_file is not None:
                    try:
                        self._log_file.close()
                    except OSError as e:
                        logger.warning("Failed to close JSONL log file: %s", e)

    def stop(self) -> None:
        """Request the agent to stop. Unblocks any waiting ask_user."""
        self._stop_requested = True
        self._question_event.set()  # Unblock if waiting for user answer

    def provide_answer(self, text: str) -> None:
        """Provide an answer to a pending ask_user question."""
        self._question_answer = text
        self._question_event.set()

    def _ask_user_fn(self, question: str) -> str:
        """Ask the user a question and block until answered or stopped."""
        question_id = str(uuid.uuid4())
        self._question_event.clear()
        self._question_answer = ""

        self.emit(QuestionEvent(question_id=question_id, question=question))

        # Block until answer or stop
        while not self._stop_requested:
            if self._question_event.wait(timeout=1.0):
                break

        if self._stop_requested:
            return "(agent stopped)"
        return self._question_answer or "(no response)"

    def _spawn_sub_agent(self, task: str, context_info: str = "") -> str:
        """Spawn a child agent to handle a focused sub-task.

        The child gets its own conversation context and tool access (minus
        spawn_sub_agent) so it can work independently. Blocks until the
        child completes and returns its report_to_user summary.

        Max nesting depth is 1 (top-level agent can spawn sub-agents,
        but sub-agents cannot spawn further sub-agents).
        """
        if self._depth >= 1:
            return (
                "[ERROR] Maximum sub-agent depth reached. "
                "Sub-agents cannot spawn further sub-agents."
            )

        # Build tools for the child: everything the parent has, minus spawn_sub_agent
        child_tools = [t for t in self.tools if t.get("name") != "spawn_sub_agent"]

        # Fresh context manager — isolated conversation history, same workspace
        child_context = ContextManager(
            provider=self.provider,
            model=self.model,
            workspace=self.context.workspace,
        )

        # Simple prompt builder that uses the task as the system prompt
        def sub_agent_prompt_builder(workspace: str, learnings: str, config: Any = None, adapter: Any = None) -> str:
            parts = [
                "You are a focused sub-agent. Complete the task described below, "
                "then call report_to_user with a summary of your findings and results.",
                f"\n## Task\n{task}",
            ]
            if context_info:
                parts.append(f"\n## Context\n{context_info}")
            if workspace:
                parts.append(f"\n## Workspace\n{workspace}")
            if learnings:
                parts.append(f"\n## Accumulated Learnings\n{learnings}")
            return "\n".join(parts)

        # Capture the sub-agent's report_to_user summary
        captured_summary: list[str] = []

        def child_event_callback(event: AgentEvent) -> None:
            # Forward all events to the parent's callback so they appear in logs/dashboard
            self.emit(event)
            # Capture the report_to_user output
            if isinstance(event, ToolResultEvent) and event.name == "report_to_user":
                captured_summary.append(event.output)

        child = AgentLoop(
            provider=self.provider,
            model=self.model,
            context=child_context,
            event_callback=child_event_callback,
            reasoning_effort=self.reasoning_effort,
            config=self.config,
            tools=child_tools,
            prompt_builder=sub_agent_prompt_builder,
            log_name=f"sub_agent_{uuid.uuid4().hex[:8]}",
            min_report_attempts=1,
            db=self.db,
            metrics=self.metrics,
            adapter=self.adapter,
        )
        child._depth = self._depth + 1

        # Build the initial message for the child
        initial_message = task
        if context_info:
            initial_message = f"{task}\n\nContext:\n{context_info}"

        # Run the child agent (blocks until completion)
        child.run(initial_message)

        if captured_summary:
            return captured_summary[-1]
        return "[Sub-agent completed but did not produce a report_to_user summary.]"

    def send_user_message(self, message: str) -> None:
        """Process a user message through the full agent loop."""
        self.context.add_entry("user", message)
        self._consecutive_tool_calls = 0
        self._done = False

        # Check if we need to summarize and fork
        if self.context.should_summarize():
            self.emit(StatusEvent(status="thinking", detail="Summarizing context..."))
            _summary, trimmed = self.context.summarize_and_fork(
                history=self._input_history,
            )
            if trimmed is not None:
                self._input_history = trimmed

        input_items = self.provider.build_user_items(message)
        self._run_loop(input_items)

    def _build_system_instructions(self) -> str:
        """Build system instructions with current learnings.

        Every prompt_builder in the codebase (default ``build_system_prompt``
        and the closure-style builders in worker/strategist/pipeline/
        supervisor/phase0) accepts a 4th ``adapter`` parameter. Closures
        ignore it (they capture adapter from their own enclosing scope);
        the default builder uses it to pick up the customized ``phase1``
        prompt from ``workspace/adapter/phase1.md``. The uniform contract
        lets us call every builder the same way.
        """
        learnings = self.context.get_learnings()
        prompt = self.prompt_builder(
            self.context.workspace, learnings, self.config,
            adapter=self.adapter,
        )

        if self.context.summary:
            prompt += (
                "\n\n## Conversation Summary (from earlier context)\n"
                f"{self.context.summary}"
            )

        return prompt

    def _run_loop(self, input_items: list[dict[str, Any]]) -> None:
        """Run the agent loop until done or stopped."""
        while not self._done and not self._stop_requested:
            response = self._call_api(input_items)
            if response is None:
                return

            # Append response to local history (provider handles format)
            self.provider.append_response_to_history(
                self._input_history, response
            )

            # Update context tracking
            if response.input_tokens or response.output_tokens:
                self.context.update_usage(
                    response.input_tokens,
                    response.output_tokens,
                )
                if self.metrics is not None:
                    self.metrics.record_api_call(
                        response.input_tokens,
                        response.output_tokens,
                        cache_read_tokens=response.cache_read_input_tokens,
                        cache_write_tokens=response.cache_write_input_tokens,
                    )

            if response.text:
                self.context.add_entry("assistant", response.text)

            # Emit web search events if applicable
            if response.has_web_search:
                self.emit(ToolCallEvent(
                    call_id="web_search",
                    name="web_search",
                    arguments="{}",
                ))
                self.emit(ToolResultEvent(
                    call_id="web_search",
                    name="web_search",
                    output="Web search completed",
                ))

            if response.tool_calls:
                self._consecutive_nudges = 0
                self._consecutive_tool_calls = 0
                input_items = self._handle_tool_calls(response.tool_calls)
                if self._done:
                    return
            elif response.has_web_search:
                # Web search is server-side work — don't count as stuck
                self._consecutive_nudges = 0
                input_items = self.provider.build_user_items(CONTINUE_MESSAGE)
            else:
                self._consecutive_nudges += 1
                if self._consecutive_nudges >= MAX_CONSECUTIVE_NUDGES:
                    self.emit(ErrorEvent(
                        message=f"Agent stuck: {MAX_CONSECUTIVE_NUDGES} consecutive nudges without tool calls. Stopping.",
                    ))
                    self._done = True
                    return
                input_items = self.provider.build_user_items(CONTINUE_MESSAGE)

    def _call_api(self, input_items: list[dict[str, Any]]) -> Any | None:
        """Call the provider with retries and streaming."""
        self.emit(StatusEvent(status="thinking", detail="Calling API..."))
        instructions = self._build_system_instructions()

        # Add new input items to local history
        self._input_history.extend(input_items)

        # Emit the full request payload
        self.emit(ApiRequestEvent(
            model=self.model,
            instructions=instructions,
            input=self._input_history,
            tools=self.tools,
            previous_response_id=None,
            reasoning_effort=self.reasoning_effort,
        ))

        for attempt in range(MAX_RETRIES):
            if self._stop_requested:
                return None

            # Decision #3: one chat span per attempt — each retry is a distinct
            # API call with its own latency and token counts.
            from alpha_lab import mlflow_logger
            mlflow_chat_ctx = mlflow_logger.child_span(
                f"chat {self.model}", span_type=mlflow_logger.SpanType.LLM,
            )
            with mlflow_chat_ctx as mlflow_chat_span, self._tracer.start_as_current_span(
                f"chat {self.model}",
                kind=trace.SpanKind.CLIENT,
                attributes={
                    GEN_AI_OPERATION_NAME: GenAiOperationNameValues.CHAT.value,
                    GEN_AI_SYSTEM: self._gen_ai_system,
                    GEN_AI_REQUEST_MODEL: self.model,
                },
            ) as chat_span:
                mlflow_logger.set_inputs(mlflow_chat_span, {
                    "model": self.model,
                    "system": instructions,
                    "messages": self._input_history,
                    "tools": [t.get("name") for t in (self.tools or []) if isinstance(t, dict)],
                    "reasoning_effort": self.reasoning_effort,
                })
                try:
                    full_text = ""
                    response = None

                    for event in self.provider.stream_response(
                        model=self.model,
                        system=instructions,
                        history=self._input_history,
                        tools=self.tools,
                        reasoning_effort=self.reasoning_effort,
                    ):
                        if self._stop_requested:
                            return None

                        if event.type == "text_delta":
                            full_text += event.delta
                            self.emit(AgentTextEvent(
                                delta=event.delta,
                                full_text=full_text,
                            ))
                        elif event.type == "done":
                            response = event.response

                    # Emit the full response payload
                    if response is not None:
                        # NOTE: GEN_AI_RESPONSE_MODEL intentionally omitted —
                        # ProviderResponse does not expose response_model yet.
                        # Add when provider protocol gains it (Decision #4).
                        chat_span.set_attribute(
                            GEN_AI_USAGE_INPUT_TOKENS, response.input_tokens
                        )
                        chat_span.set_attribute(
                            GEN_AI_USAGE_OUTPUT_TOKENS, response.output_tokens
                        )
                        usage = {
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                        }
                        if response.cache_read_input_tokens > 0:
                            usage["cache_read_input_tokens"] = response.cache_read_input_tokens
                        if response.cache_write_input_tokens > 0:
                            usage["cache_write_input_tokens"] = response.cache_write_input_tokens

                        mlflow_logger.set_outputs(mlflow_chat_span, {
                            "id": response.id,
                            "text": response.text,
                            "tool_calls": [
                                {"call_id": tc.call_id, "name": tc.name, "arguments": tc.arguments}
                                for tc in response.tool_calls
                            ],
                            **usage,
                        })

                        self.emit(ApiResponseEvent(
                            response_id=response.id,
                            output=response.raw_output_items,
                            usage=usage,
                        ))

                    return response

                except Exception as e:
                    chat_span.set_status(trace.StatusCode.ERROR)
                    chat_span.record_exception(e)
                    if self.metrics is not None:
                        self.metrics.record_error(is_api_error=True)
                    delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                    if attempt < MAX_RETRIES - 1:
                        self.emit(StatusEvent(
                            status="error",
                            detail=f"API error: {e}. Retrying in {delay}s...",
                        ))
                        time.sleep(delay)
                    else:
                        self.emit(ErrorEvent(
                            message=f"API error after {MAX_RETRIES} retries: {e}",
                        ))
                        return None

        return None

    def _handle_tool_calls(
        self, tool_calls: list[ToolCall]
    ) -> list[dict[str, Any]]:
        """Process tool calls and return input for the next API turn."""
        tool_outputs: list[dict[str, Any]] = []
        images_to_inject: list[tuple[str, str]] = []

        for tc in tool_calls:
            if self._stop_requested:
                break

            name = tc.name
            call_id = tc.call_id
            args = parse_tool_args(tc.arguments)

            self._consecutive_tool_calls += 1

            # Check runaway loop
            if self._consecutive_tool_calls > MAX_CONSECUTIVE_TOOL_CALLS:
                self.emit(StatusEvent(
                    status="error",
                    detail="Too many consecutive tool calls, forcing summary",
                ))
                tool_outputs.append({
                    "call_id": call_id,
                    "output": (
                        "[SYSTEM] You've made many consecutive tool calls. "
                        "Summarize progress and call report_to_user."
                    ),
                })
                self._consecutive_tool_calls = 0
                continue

            # Emit tool call event
            self.emit(ToolCallEvent(
                call_id=call_id,
                name=name,
                arguments=tc.arguments,
            ))

            # Build a human-readable status detail
            if name == "shell_exec":
                cmd = args.get("command", "")
                if len(cmd) > 100:
                    cmd = cmd[:100] + "..."
                status_detail = f"$ {cmd}"
            elif name == "view_image":
                status_detail = f"Viewing {args.get('path', '')}"
            elif name == "report_to_user":
                status_detail = "Writing final report..."
            elif name == "spawn_sub_agent":
                status_detail = f"Spawning sub-agent: {args.get('task', '')[:80]}..."
            else:
                status_detail = f"Executing {name}..."

            self.emit(StatusEvent(
                status="tool_executing",
                detail=status_detail,
            ))

            # One span per tool execution (GenAI semconv: execute_tool)
            from alpha_lab import mlflow_logger
            mlflow_tool_ctx = mlflow_logger.child_span(
                f"execute_tool {name}", span_type=mlflow_logger.SpanType.TOOL,
            )
            with mlflow_tool_ctx as mlflow_tool_span, self._tracer.start_as_current_span(
                f"execute_tool {name}",
                kind=trace.SpanKind.INTERNAL,
                attributes={
                    GEN_AI_OPERATION_NAME: GenAiOperationNameValues.EXECUTE_TOOL.value,
                    GEN_AI_SYSTEM: self._gen_ai_system,
                    GEN_AI_TOOL_NAME: name,
                    GEN_AI_TOOL_CALL_ID: call_id,
                },
            ) as tool_span:
                mlflow_logger.set_inputs(mlflow_tool_span, {
                    "tool_name": name, "call_id": call_id, "arguments": args,
                })
                # Handle spawn_sub_agent within AgentLoop (needs self.provider, etc.)
                if name == "spawn_sub_agent":
                    task = args.get("task", "")
                    context_info = args.get("context", "")
                    try:
                        output_text = self._spawn_sub_agent(task, context_info)
                        result = {"output": output_text}
                    except Exception as e:
                        tool_span.set_status(trace.StatusCode.ERROR)
                        tool_span.record_exception(e)
                        result = {"output": f"[ERROR] Sub-agent failed: {e}"}
                else:
                    # Execute the tool
                    _openai_client = getattr(self.provider, "openai_client", None)
                    try:
                        result = execute_tool(
                            name=name,
                            arguments=args,
                            workspace=self.context.workspace or ".",
                            ask_user_fn=self._ask_user_fn,
                            db=self.db,
                            openai_client=_openai_client,
                            adapter=self.adapter,
                            shell_timeout=getattr(self.config, "shell_timeout", DEFAULT_TIMEOUT),
                            memory_store=self.memory_store,
                        )
                    except Exception as e:
                        tool_span.set_status(trace.StatusCode.ERROR)
                        tool_span.record_exception(e)
                        result = {"output": f"[TOOL ERROR] {name} raised an exception: {e}"}
                mlflow_logger.set_outputs(mlflow_tool_span, {
                    "output": result.get("output"),
                    "image_media_type": result.get("image", (None, None))[1],
                    "done": result.get("done", False),
                })

            output = result["output"]

            # Emit tool result event
            image_b64 = None
            image_media = None
            if "image" in result:
                image_b64 = result["image"][0]
                image_media = result["image"][1]

            self.emit(ToolResultEvent(
                call_id=call_id,
                name=name,
                output=output,
                image_base64=image_b64,
                media_type=image_media,
            ))

            # Check if agent is done — require confirmation on first attempt
            if result.get("done"):
                self._report_attempts += 1
                if self._report_attempts < self.min_report_attempts:
                    # First attempt: override output, nudge to verify plan
                    tool_outputs.append({
                        "call_id": call_id,
                        "output": (
                            "[SYSTEM] Before finishing, review your plan.md. "
                            "Are ALL items checked off? Have you written the "
                            "data_report/ files? If anything is incomplete, "
                            "continue working. If truly done, call report_to_user "
                            "again to confirm."
                        ),
                    })
                else:
                    tool_outputs.append({
                        "call_id": call_id,
                        "output": output,
                    })
                    self._done = True
            else:
                # Compact large tool outputs to keep history bounded.
                # Cap is config-driven so runs that see heavy truncation-induced
                # follow-up round-trips can raise it without a code change. Type
                # and range of `tool_output_max_chars` are enforced at config
                # load time (TaskConfig.__post_init__), so this is an int >= 100.
                max_chars = getattr(self.config, "tool_output_max_chars", 8000)
                compacted = ContextManager.compact_tool_output(
                    output, name, max_chars=max_chars
                )
                tool_outputs.append({
                    "call_id": call_id,
                    "output": compacted,
                })

            # Track images for injection
            if "image" in result:
                images_to_inject.append(result["image"])

            self.context.add_entry("tool", f"[{name}] {output[:500]}")

        # Build next input using provider (handles format differences)
        return self.provider.build_tool_result_items(
            tool_outputs,
            images_to_inject if images_to_inject else None,
        )
