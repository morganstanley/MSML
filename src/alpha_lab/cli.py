"""Command-line interface surfaces for alpha-lab.

Hosts the interactive entry points: the open-ended chat REPL (``main``) and the
pipeline's intake session (``run_intake``). Both construct an ``AgentLoop`` and
run it against an interactive user; they share Rich rendering and the
prompt_toolkit input surface. Uses ``CliEventHandler`` as an adapter between
the event-based ``AgentLoop`` and Rich.
"""

from __future__ import annotations

import logging
import os
import select
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import confirm
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from alpha_lab import __version__
from alpha_lab.agent import AgentLoop, build_loop_kwargs
from alpha_lab.config import TaskConfig
from alpha_lab.agents import load_agent
from alpha_lab.context import ContextManager
from alpha_lab.events import (
    AgentEvent,
    AgentTextEvent,
    ErrorEvent,
    PhaseEvent,
    QuestionEvent,
    StatusEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from alpha_lab.prompts import build_step_prompt
from alpha_lab.providers import Provider

logger = logging.getLogger("alpha_lab.cli")


def _drain_pending_input() -> str:
    """Return input already buffered before this call (type-ahead), non-blocking.

    Reads whatever the terminal has queued *right now* — i.e. lines the user submitted
    while the agent was working, before a question was posed — without waiting for new
    input. Returns "" if nothing is pending or stdin isn't a readable fd.
    """
    try:
        fd = sys.stdin.fileno()
    except (AttributeError, ValueError, OSError):
        return ""
    buf = bytearray()
    # select can raise on a non-selectable/out-of-range stdin (some pipes/platforms);
    # treat any such failure as "nothing buffered" rather than breaking prompting.
    try:
        while select.select([fd], [], [], 0)[0]:
            data = os.read(fd, 4096)
            if not data:
                break
            buf.extend(data)
    except (OSError, ValueError):
        pass
    # Decode once, so a multi-byte UTF-8 char split across reads isn't mangled by
    # errors="replace".
    return buf.decode("utf-8", errors="replace")


def _new_prompt_session() -> PromptSession:
    """A reusable interactive input surface for the chat REPL and intake.

    Reusing one session per REPL/intake run keeps prompt_toolkit's bracketed-paste
    mode active across turns (so pasted multi-line text arrives as one atomic input
    rather than one line per prompt). The key binding lets a trailing backslash before
    Return continue onto a new line instead of submitting.
    """
    kb = KeyBindings()

    @kb.add("enter")
    def _(event: Any) -> None:
        buffer = event.current_buffer
        if buffer.document.text.endswith("\\"):
            buffer.delete_before_cursor(1)  # drop the trailing backslash
            buffer.insert_text("\n")
        else:
            buffer.validate_and_handle()  # normal submit

    return PromptSession(history=InMemoryHistory(), key_bindings=kb)


# ---------------------------------------------------------------------------
# CLI Event Handler — adapts AgentEvents to Rich output
# ---------------------------------------------------------------------------


class CliEventHandler:
    """Renders AgentEvents to the terminal via Rich console.

    Also handles ask_user by prompting interactively.
    """

    def __init__(self, console: Console, prompt_session: PromptSession) -> None:
        self.console = console
        self.prompt_session = prompt_session
        self._last_text = ""

    def __call__(self, event: AgentEvent) -> None:
        """Handle an event — this is passed as event_callback to AgentLoop."""
        if isinstance(event, AgentTextEvent):
            # Only print when we get the final accumulated text
            # (we skip deltas and print the full text at the end via status=done or tool_call)
            self._last_text = event.full_text

        elif isinstance(event, StatusEvent):
            if event.status == "thinking":
                self.console.print(f"[dim]{event.detail}[/dim]")
            elif event.status == "starting":
                self.console.print(f"[green]{event.detail}[/green]")
            elif event.status == "done":
                # Print any accumulated text
                if self._last_text:
                    self.console.print(
                        Panel(
                            Markdown(self._last_text),
                            title="Alpha Lab",
                            border_style="green",
                            padding=(1, 2),
                        )
                    )
                    self._last_text = ""
                self.console.print("[green]Agent finished.[/green]")
            elif event.status == "error":
                self.console.print(f"[red]{event.detail}[/red]")
            elif event.status == "tool_executing":
                # Flush accumulated text before showing tool execution
                if self._last_text:
                    self.console.print(
                        Panel(
                            Markdown(self._last_text),
                            title="Alpha Lab",
                            border_style="green",
                            padding=(1, 2),
                        )
                    )
                    self._last_text = ""

        elif isinstance(event, ToolCallEvent):
            if event.name == "shell_exec":
                import json
                try:
                    args = json.loads(event.arguments)
                    command = args.get("command", event.arguments)
                except (json.JSONDecodeError, AttributeError):
                    command = event.arguments
                self.console.print(
                    Panel(
                        Syntax(command, "bash", theme="monokai"),
                        title="Shell Command",
                        border_style="yellow",
                        padding=(0, 1),
                    )
                )
            elif event.name == "view_image":
                import json
                try:
                    args = json.loads(event.arguments)
                    path = args.get("path", "")
                except (json.JSONDecodeError, AttributeError):
                    path = ""
                self.console.print(f"[dim]Viewing image: {path}[/dim]")

        elif isinstance(event, ToolResultEvent):
            if event.name == "shell_exec":
                output = event.output
                if len(output) > 2000:
                    output = output[:1000] + "\n...\n" + output[-1000:]
                self.console.print(
                    Panel(
                        output,
                        title="Output",
                        border_style="dim",
                        padding=(0, 1),
                    )
                )
            elif event.name == "report_to_user":
                self.console.print(
                    Panel(
                        event.output,
                        title="Analysis Complete",
                        border_style="bold green",
                        padding=(1, 2),
                    )
                )

        elif isinstance(event, QuestionEvent):
            self.console.print(
                Panel(
                    event.question,
                    title="Agent Question",
                    border_style="blue",
                    padding=(1, 2),
                )
            )
            # The actual answer is provided through the agent's _ask_user_fn
            # which blocks. We need to provide the answer via prompt.
            try:
                answer = self.prompt_session.prompt("Your answer: ")
                answer = answer.strip() or "(no response)"
            except (EOFError, KeyboardInterrupt):
                answer = "(user declined to answer)"
            # The agent is blocking on _ask_user_fn waiting for provide_answer
            # We need to store this to be provided back
            self._pending_answer = answer

        elif isinstance(event, ErrorEvent):
            self.console.print(f"[red]Error: {event.message}[/red]")


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Welcome Banner
# ---------------------------------------------------------------------------


def make_interactive_ask_user(
    console: Console,
    prompt_session: PromptSession,
) -> Callable[[str], str]:
    """Build an interactive ``ask_user`` function bound to the given I/O surface.

    Used by both the chat REPL and the intake session to render agent questions
    via Rich and read user answers via prompt_toolkit.
    """
    def ask(question: str) -> str:
        # Process-first: text typed before this question was posed (buffered while the
        # agent was working) is handed to the agent to process instead of posing the
        # question — it can then decide whether to re-ask. Whitespace-only buffered
        # input is ignored so a stray newline can't auto-submit.
        pre_existing = _drain_pending_input().strip()
        if pre_existing:
            return pre_existing
        console.print(
            Panel(
                question,
                title="Agent Question",
                border_style="blue",
                padding=(1, 2),
            )
        )
        try:
            answer = prompt_session.prompt("Your answer: ")
            return answer.strip() or "(no response)"
        except (EOFError, KeyboardInterrupt):
            return "(user declined to answer)"

    return ask


def print_banner(console: Console, model: str) -> None:
    """Print the welcome banner."""
    banner = Text()
    banner.append("Alpha Lab", style="bold cyan")
    banner.append(f"  v{__version__}\n")
    banner.append("Autonomous Quant Research Agent\n", style="dim")
    banner.append(f"Model: {model}", style="dim")

    console.print(
        Panel(
            banner,
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------


@click.command(name="alpha-lab", help="Alpha Lab — Autonomous Quant Research Agent", context_settings={"show_default": True})
@click.option("--workspace", type=str, default=None, help="Workspace directory path")
@click.option("--model", type=str, default="gpt-5.2", help="Model to use")
@click.option(
    "--provider",
    "provider_name",
    type=click.Choice(["openai", "bedrock"]),
    default="openai",
    help="LLM provider",
)
@click.option("--auto-approve", is_flag=True, help="Auto-approve all shell commands (always on now)")
@click.option("--resume", is_flag=True, help="Resume previous session (loads learnings.md)")
def main(
    workspace: str | None,
    model: str,
    provider_name: str,
    auto_approve: bool,
    resume: bool,
) -> None:
    """Main entry point for alpha-lab CLI."""
    # Check for API key (skip if on-prem or using bedrock)
    from alpha_lab.providers import get_provider, ON_PREM_AVAILABLE
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not ON_PREM_AVAILABLE and provider_name == "openai":
        print(
            "Error: OPENAI_API_KEY environment variable not set.\n"
            "Set it with: export OPENAI_API_KEY=your-key-here",
            file=sys.stderr,
        )
        sys.exit(1)

    # Initialize components
    console = Console()
    provider = get_provider(provider_name, api_key=api_key, model=model)
    prompt_session: PromptSession = _new_prompt_session()

    # Print banner
    print_banner(console, model)

    # Initialize context manager
    context = ContextManager(
        provider=provider,
        model=model,
        workspace=workspace,
    )

    # If resuming, load learnings
    if resume and workspace:
        learnings = context.get_learnings()
        if learnings:
            console.print("[dim]Loaded learnings from previous session.[/dim]\n")
        else:
            console.print("[dim]No previous learnings found.[/dim]\n")

    # Create event handler
    event_handler = CliEventHandler(console, prompt_session)

    # Initialize agent loop with event callback
    agent_definition = load_agent("cli/interactive")
    agent = AgentLoop(
        provider=provider,
        model=model,
        context=context,
        event_callback=event_handler,
        **build_loop_kwargs(agent_definition),
    )

    # For CLI mode, override _ask_user_fn to prompt interactively
    agent._ask_user_fn = make_interactive_ask_user(console, prompt_session)  # type: ignore[assignment]

    # Initial message
    console.print("[dim]Ctrl+C to interrupt, /exit to quit.[/dim]\n")

    initial_message = (
        "Start. Ask me for my data path and workspace location."
        if not workspace
        else f"Start. Workspace: {workspace}. Go."
    )

    if resume and workspace:
        initial_message = (
            f"Resume. Workspace: {workspace}. "
            "Review learnings.md and continue where you left off."
        )

    try:
        # Send initial message to kick off the conversation
        agent.send_user_message(initial_message)

        # REPL loop
        while True:
            try:
                user_input = prompt_session.prompt("\nyou> ").strip()
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted.[/yellow]")
                continue
            except EOFError:
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q", "/quit", "/exit", "/q"):
                console.print("[dim]Goodbye![/dim]")
                break

            agent.send_user_message(user_input)

    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        sys.exit(1)
    finally:
        # Force exit — httpx connection pool threads can keep the process alive
        if hasattr(provider, 'openai_client'):
            try:
                provider.openai_client.close()
            except Exception:
                pass
        os._exit(0)


# ---------------------------------------------------------------------------
# Intake — interactive user-proxy session that runs before phase 0.
#
# Surfaces the user's purpose, success criteria, and constraints in
# conversation, then writes ``{workspace}/agenda.md`` and seeds
# ``{workspace}/private/proxy_state.md`` so downstream phases inherit a
# concrete description of user intent.
# ---------------------------------------------------------------------------


def run_intake(
    provider: Provider,
    config: TaskConfig,
    config_path: str,
    workspace: str,
    event_callback: Callable[[AgentEvent], None],
) -> None:
    """Run the interactive intake session.

    Runs the registry-backed intake prompt. The agent may edit the workspace
    config, write ``agenda.md``, and seed ``proxy_state.md``.

    Args:
        provider: LLM provider for the agent.
        config: Loaded task configuration.
        config_path: Filesystem path the config was loaded from. Passed to
            the agent so it can offer to edit it directly.
        workspace: Workspace root directory.
        event_callback: Callback for emitting agent and phase events.

    Raises:
        RuntimeError: if ``sys.stdin`` is not a TTY (intake is interactive).
    """
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Intake requires an interactive terminal; stdin is not a TTY. "
            "Pass --no-enable-intake to skip, or run from an interactive shell."
        )

    logger.info("Intake: starting")
    event_callback(PhaseEvent(
        phase="intake", step="session", status="starting",
        detail="Running user intake session",
    ))

    agent_definition = load_agent("proxy/intake")

    context = ContextManager(
        provider=provider,
        model=config.model,
        workspace=workspace,
    )

    def prompt_builder(
        workspace_arg: str | None,
        learnings: str | None,
        config_arg: Any | None = None,
        adapter: Any | None = None,
    ) -> str:
        return build_step_prompt(
            agent_definition.adapter_prompt_key,
            workspace_arg,
            learnings,
            config_arg,
            extra_context=f"Config path: `{config_path}`",
            adapter=adapter,
        )

    # Interactive I/O surface for the user-proxy session. We compose the
    # caller's event_callback (pipeline logging) with a CliEventHandler so
    # the user actually sees what the agent says during intake.
    console = Console()
    prompt_session: PromptSession = _new_prompt_session()
    cli_handler = CliEventHandler(console, prompt_session)

    def composed_callback(event: AgentEvent) -> None:
        event_callback(event)
        # Drop per-call progress noise ("Calling API…"/"Summarizing…") from the intake UI;
        # it stacks one line per model call. event_callback still records it (JSONL/log).
        if isinstance(event, StatusEvent) and event.status == "thinking":
            return
        cli_handler(event)

    agent = AgentLoop(
        provider=provider,
        model=config.model,
        context=context,
        event_callback=composed_callback,
        config=config,
        prompt_builder=prompt_builder,
        **build_loop_kwargs(agent_definition, reasoning_effort_default=config.reasoning_effort),
    )
    ask = make_interactive_ask_user(console, prompt_session)
    agent._ask_user_fn = ask  # type: ignore[assignment]

    # Termination is user-controlled: when the agent calls report_to_user, confirm with the
    # user rather than ending outright. "no" hands the user's follow-up back to the agent so
    # it keeps going; only "yes" ends intake. (The report_to_user narrative is rendered just
    # before this by the report_to_user tool-result panel.)
    def confirm_report_to_user(report: str) -> str | None:
        # An interrupt at the gate (Ctrl-C/Ctrl-D) is treated as "end the session" — the
        # agent already called report_to_user, so bailing here means ending, not looping.
        try:
            end_session = confirm("End the intake session?")
        except (EOFError, KeyboardInterrupt):
            return None
        if end_session:
            return None
        return f"[User] {ask('What would you like to add or adjust before we finish?')}"

    agent._on_report_to_user = confirm_report_to_user  # type: ignore[assignment]

    # For the intake window only, raise the root logger to WARNING so background INFO logs
    # (httpx/openai/httpcore/urllib3/botocore plus the app's own one-liners) don't drown or
    # clobber the conversation — the interactive Rich UI is the channel here. This is safe
    # and complete because the sole terminal sink is the root StreamHandler from run.py's
    # basicConfig, every logger propagates to it, and none set their own level — so raising
    # root suppresses INFO from all sources. Restored afterward so the headless pipeline
    # logs exactly as before. patch_stdout keeps the remaining console output (the Rich
    # conversation) above the live input line instead of over what you're typing.
    root_logger = logging.getLogger()
    saved_level = root_logger.level
    root_logger.setLevel(logging.WARNING)
    try:
        # raw=True so Rich's ANSI color codes pass through instead of being escaped
        # (patch_stdout strips vt100 sequences by default), while still routing output
        # above the live input line.
        with patch_stdout(raw=True):
            agent.run("Begin the intake session.")
    finally:
        root_logger.setLevel(saved_level)

    event_callback(PhaseEvent(
        phase="intake", step="session", status="completed",
        detail="Intake session finished",
    ))
    logger.info("Intake: complete")

if __name__ == "__main__":
    main()
