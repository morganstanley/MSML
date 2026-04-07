"""Core agent loop for alpha-lab.

Provider-agnostic: talks to the Provider protocol instead of the OpenAI SDK
directly. Supports OpenAI Responses API and Anthropic Messages API.

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
from collections.abc import Callable
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
from alpha_lab.prompts import build_system_prompt
from alpha_lab.provider import Provider, ToolCall
from alpha_lab.tools import ALL_TOOL_SCHEMAS, execute_tool, get_tool_schemas, parse_tool_args

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


class AgentLoop:
    """The main agent loop: send messages, stream responses, dispatch tools.

    Emits events via event_callback. Supports ask_user blocking via
    threading primitives. Can be stopped externally via stop().
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
    ) -> None:
        self.provider = provider
        self.model = model
        self.context = context
        self.event_callback = event_callback
        self.reasoning_effort = reasoning_effort
        self.config = config
        self.tools = tools if tools is not None else ALL_TOOL_SCHEMAS
        self.prompt_builder = prompt_builder or build_system_prompt
        self.log_name = log_name
        self.min_report_attempts = min_report_attempts
        self.db = db
        self.metrics = metrics
        self.adapter = adapter

        self._depth = 0  # 0 = top-level agent, 1 = sub-agent (max)
        self._consecutive_tool_calls = 0
        self._consecutive_nudges = 0
        self._report_attempts = 0
        self._done = False
        self._stop_requested = False

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
                self._log_file.write(json.dumps(event.to_dict()) + "\n")
                self._log_file.flush()
            except (OSError, TypeError, ValueError) as e:
                logger.warning("Failed to write event to JSONL log: %s", e)

    def run(self, initial_message: str) -> None:
        """Top-level entry point for running in a background thread."""
        self._init_log()
        self.emit(StatusEvent(status="starting", detail="Agent starting"))
        try:
            self.send_user_message(initial_message)
            if self._done:
                self.emit(StatusEvent(status="done", detail="Agent finished"))
            elif self._stop_requested:
                self.emit(StatusEvent(status="stopped", detail="Agent stopped by user"))
            else:
                # API returned None (retries exhausted) or nudge limit hit
                self.emit(StatusEvent(status="error", detail="Agent stopped unexpectedly"))
        except Exception as e:
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
        def sub_agent_prompt_builder(workspace: str, learnings: str, config: Any = None) -> str:
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
            self.context.summarize_and_fork()
            # Clear the actual API history — the summary is injected into
            # the system prompt, so old messages no longer need to be sent.
            self._input_history.clear()

        input_items = self.provider.build_user_items(message)
        self._run_loop(input_items)

    def _build_system_instructions(self) -> str:
        """Build system instructions with current learnings."""
        learnings = self.context.get_learnings()
        prompt = self.prompt_builder(self.context.workspace, learnings, self.config)

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
                    self.emit(ApiResponseEvent(
                        response_id=response.id,
                        output=response.raw_output_items,
                        usage={
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                        },
                    ))

                return response

            except Exception as e:
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

            # Handle spawn_sub_agent within AgentLoop (needs self.provider, etc.)
            if name == "spawn_sub_agent":
                task = args.get("task", "")
                context_info = args.get("context", "")
                try:
                    output_text = self._spawn_sub_agent(task, context_info)
                    result = {"output": output_text}
                except Exception as e:
                    result = {"output": f"[ERROR] Sub-agent failed: {e}"}
            else:
                # Execute the tool
                # Pass OpenAI client for web_search proxy (Anthropic provider has one)
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
                    )
                except Exception as e:
                    result = {"output": f"[TOOL ERROR] {name} raised an exception: {e}"}

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
                tool_outputs.append({
                    "call_id": call_id,
                    "output": output,
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
