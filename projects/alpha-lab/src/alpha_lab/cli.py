"""CLI entry point for alpha-lab.

Handles argument parsing, Rich rendering, REPL loop, and permissions.
Uses CliEventHandler as an adapter between the event-based AgentLoop and Rich.
"""

from __future__ import annotations

import argparse
import os
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from alpha_lab import __version__
from alpha_lab.agent import AgentLoop
from alpha_lab.context import ContextManager
from alpha_lab.events import (
    AgentEvent,
    AgentTextEvent,
    ErrorEvent,
    QuestionEvent,
    StatusEvent,
    ToolCallEvent,
    ToolResultEvent,
)


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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="alpha-lab",
        description="Alpha Lab — Autonomous Quant Research Agent",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace directory path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2",
        help="Model to use (default: gpt-5.2)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic"],
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve all shell commands (always on now)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume previous session (loads learnings.md)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Welcome Banner
# ---------------------------------------------------------------------------


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


def main() -> None:
    """Main entry point for alpha-lab CLI."""
    args = parse_args()

    # Check for API key
    from alpha_lab.client import get_provider
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and args.provider == "openai":
        print(
            "Error: OPENAI_API_KEY environment variable not set.\n"
            "Set it with: export OPENAI_API_KEY=your-key-here",
            file=sys.stderr,
        )
        sys.exit(1)

    # Initialize components
    console = Console()
    provider = get_provider(args.provider, api_key=api_key)
    prompt_session: PromptSession = PromptSession(history=InMemoryHistory())

    # Print banner
    print_banner(console, args.model)

    # Initialize context manager
    context = ContextManager(
        provider=provider,
        model=args.model,
        workspace=args.workspace,
    )

    # If resuming, load learnings
    if args.resume and args.workspace:
        learnings = context.get_learnings()
        if learnings:
            console.print("[dim]Loaded learnings from previous session.[/dim]\n")
        else:
            console.print("[dim]No previous learnings found.[/dim]\n")

    # Create event handler
    event_handler = CliEventHandler(console, prompt_session)

    # Initialize agent loop with event callback
    agent = AgentLoop(
        provider=provider,
        model=args.model,
        context=context,
        event_callback=event_handler,
    )

    # For CLI mode, override _ask_user_fn to prompt interactively
    original_ask_user = agent._ask_user_fn

    def cli_ask_user(question: str) -> str:
        """Synchronous ask_user for CLI — prompts directly."""
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

    agent._ask_user_fn = cli_ask_user  # type: ignore[assignment]

    # Initial message
    console.print("[dim]Ctrl+C to interrupt, /exit to quit.[/dim]\n")

    initial_message = (
        "Start. Ask me for my data path and workspace location."
        if not args.workspace
        else f"Start. Workspace: {args.workspace}. Go."
    )

    if args.resume and args.workspace:
        initial_message = (
            f"Resume. Workspace: {args.workspace}. "
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


if __name__ == "__main__":
    main()
