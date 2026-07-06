"""Phase 0 — Domain Adapter resolution and customization.

Four paths:
  1. Resume path: workspace adapter already exists → load and return
  2. Built-in match: copy template → run customization agent → return
  3. No domain specified: copy time_series → run customization agent → return
  4. Free-text domain: run full generation agent → return
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from alpha_lab.adapter import DomainAdapter
from alpha_lab.adapter_loader import (
    _resolve_adapter_path,
    copy_adapter_to_workspace,
    load_adapter,
)
from alpha_lab.sandboxing import sandbox
from alpha_lab.config import TaskConfig
from alpha_lab.events import AgentEvent, PhaseEvent
from alpha_lab.provider import Provider

logger = logging.getLogger("alpha_lab.phase0")


def _run_customization_agent(
    provider: Provider,
    config: TaskConfig,
    event_callback: Callable[[AgentEvent], None],
) -> None:
    """Run a lightweight agent to customize a built-in adapter template."""
    logger.info("Phase 0: running customization agent")
    event_callback(PhaseEvent(
        phase="phase0", step="adapter", status="customizing",
        detail="Customizing adapter for task-specific data",
    ))

    initial_message = (
        f"Customize the installed adapter for this specific task.\n\n"
        f"**Domain:** {config.domain}\n"
        f"**Data path:** {config.data_path}\n"
    )
    if config.description:
        initial_message += f"**Description:** {config.description}\n"
    if config.target:
        initial_message += f"**Target:** {config.target}\n"
    initial_message += (
        "\nStart by reading the current adapter, then explore the data, "
        "then patch any files that should be more task-specific. Go."
    )

    sandbox.run_agent(
        "phase0/customization",
        event_callback,
        initial_message=initial_message,
        provider=provider,
    )


def run_phase0(
    provider: Provider,
    config: TaskConfig,
    workspace: str,
    event_callback: Callable[[AgentEvent], None],
) -> DomainAdapter:
    """Run Phase 0: resolve or generate a domain adapter.

    Returns the loaded DomainAdapter.
    """
    domain = config.domain
    adapter_dir = Path(workspace) / "adapter"

    # 1. Resume path: adapter already exists in workspace
    if (adapter_dir / "manifest.json").exists():
        logger.info("Phase 0: loading existing workspace adapter")
        event_callback(PhaseEvent(
            phase="phase0", step="adapter", status="completed",
            detail="Loaded existing workspace adapter",
        ))
        return load_adapter(adapter_dir)

    # 2. Known adapter (name or absolute path): copy template -> customize.
    # _resolve_adapter_path raises FileNotFoundError if the domain doesn't
    # resolve as either a packaged adapter name or an absolute path.
    # We deliberately do NOT swallow that error: a non-None domain that
    # doesn't resolve is a configuration mistake, not an invitation to
    # silently spawn a generation agent. Only ``domain is None`` triggers
    # the generation path.
    if domain is not None:
        src = _resolve_adapter_path(domain)
        logger.info("Phase 0: copying adapter '%s'", domain)
        copy_adapter_to_workspace(src, adapter_dir)
        _run_customization_agent(provider, config, event_callback)
        event_callback(PhaseEvent(
            phase="phase0", step="adapter", status="completed",
            detail=f"Customized adapter: {domain}",
        ))
        return load_adapter(adapter_dir)

    # 3. Generation path: domain is None -> synthesize an adapter from scratch.
    logger.info("Phase 0: no domain specified; generating adapter")
    event_callback(PhaseEvent(
        phase="phase0", step="adapter", status="starting",
        detail="Generating adapter from task description",
    ))

    initial_message = (
        f"Generate a domain adapter for the following task.\n\n"
        f"**Domain:** {domain}\n"
        f"**Data path:** {config.data_path}\n"
    )
    if config.description:
        initial_message += f"**Description:** {config.description}\n"
    if config.target:
        initial_message += f"**Target:** {config.target}\n"
    initial_message += (
        "\nStart by reading the 'time_series' reference adapter to understand "
        "the format, then explore the data, then generate all adapter files. Go."
    )

    sandbox.run_agent(
        "phase0/generation",
        event_callback,
        initial_message=initial_message,
        provider=provider,
    )

    # Load the generated adapter
    if not (adapter_dir / "manifest.json").exists():
        raise RuntimeError(
            f"Phase 0 agent did not generate manifest.json for domain '{domain}'"
        )

    adapter = load_adapter(adapter_dir)

    event_callback(PhaseEvent(
        phase="phase0", step="adapter", status="completed",
        detail=f"Generated adapter: {adapter.domain_name}",
    ))

    return adapter
