"""Single construction site for an ``AgentLoop`` from an ``AgentDefinition``.

``build_agent`` is the one place an ``AgentLoop`` is assembled. In-process callers,
the no-bwrap fallback in ``sandboxing.sandbox``, and the bwrap child in ``sandboxing.runner``
all go through it, so a sandboxed agent is byte-identical to an in-process one.

Static properties (tools, log name, min report attempts, reasoning effort) are
derived from the ``AgentDefinition`` via :func:`alpha_lab.agent.build_loop_kwargs`;
only genuinely per-call values are accepted as arguments.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from alpha_lab import deps
from alpha_lab.adapter_loader import load_adapter
from alpha_lab.agent import AgentLoop, build_loop_kwargs
from alpha_lab.agents.agent_definition import AgentDefinition
from alpha_lab.client import get_provider
from alpha_lab.context import ContextManager
from alpha_lab.experiment_db import ExperimentDB
from alpha_lab.prompts import build_step_prompt


def _make_prompt_builder(
    agent_definition: AgentDefinition, extra_context: str | None
) -> Callable[..., str]:
    """Build the prompt closure that matches the agent's ``prompt_source``."""
    if agent_definition.prompt_source == "inline":
        body = agent_definition.prompt_body

        def inline_prompt(
            workspace: str | None,
            learnings: str | None,
            config: Any | None = None,
            adapter: Any | None = None,
        ) -> str:
            parts = [body]
            if workspace:
                parts.append(f"\n## Workspace\n`{workspace}`")
            return "\n".join(parts)

        return inline_prompt

    prompt_key = agent_definition.adapter_prompt_key

    def adapter_prompt(
        workspace: str | None,
        learnings: str | None,
        config: Any | None = None,
        adapter: Any | None = None,
    ) -> str:
        return build_step_prompt(
            prompt_key, workspace, learnings, config, extra_context, adapter=adapter
        )

    return adapter_prompt


def _load_workspace_adapter(adapter_dir: str | None) -> Any | None:
    """Load the workspace adapter if its manifest exists, else None.

    Phase 0 generation runs before a manifest is written, so a missing manifest
    is expected rather than an error.
    """
    if not adapter_dir:
        return None
    if not (Path(adapter_dir) / "manifest.json").is_file():
        return None
    return load_adapter(adapter_dir)


def build_agent(
    agent_definition: AgentDefinition,
    *,
    event_callback: Callable[..., None],
    db_path: str | None = None,
    extra_context: str | None = None,
    log_name: str | None = None,
    tools_include: tuple[str, ...] | None = None,
    mlflow_run_target: str | None = None,
    provider: Any | None = None,
    db: Any | None = None,
    adapter: Any | None = None,
    metrics: Any | None = None,
    memory_store: Any | None = None,
) -> AgentLoop:
    """Assemble an ``AgentLoop``, reconstructing any collaborators not supplied.

    ``config`` (live resolved ``TaskConfig``; model/provider/reasoning_effort), ``workspace``
    (run root; the adapter directory ``<workspace>/adapter`` is read from it), and ``api_key``
    come from the active :class:`~alpha_lab.deps.RunDeps` via ``deps.get(strict=True)`` — so this must run
    within a ``with RunDeps(...)`` scope. Run/sandbox-only values (``initial_message`` for
    ``agent.run`` and the dataset ``data_path`` mount) are handled by the caller, not here.
    """
    run = deps.get(strict=True)
    config = run.config
    if provider is None:
        provider = get_provider(config.provider, run.api_key)
    if adapter is None:
        adapter = _load_workspace_adapter(str(Path(run.workspace) / "adapter"))
    if db is None and db_path:
        db = ExperimentDB(db_path)

    context = ContextManager(provider=provider, model=config.model, workspace=str(run.workspace))

    tools = agent_definition.tools
    if tools_include is not None:
        tools = tuple(tool for tool in tools if tool.name in tools_include)

    loop_kwargs = build_loop_kwargs(
        agent_definition,
        reasoning_effort_default=config.reasoning_effort,
        log_name=log_name,
        tools=tools,
    )

    return AgentLoop(
        provider=provider,
        model=config.model,
        context=context,
        event_callback=event_callback,
        config=config,
        prompt_builder=_make_prompt_builder(agent_definition, extra_context),
        db=db,
        metrics=metrics,
        adapter=adapter,
        mlflow_run_target=mlflow_run_target,
        memory_store=memory_store,
        **loop_kwargs,
    )
