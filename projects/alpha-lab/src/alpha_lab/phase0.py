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
from typing import Any

from alpha_lab.adapter import ADAPTER_FILES, PROMPT_KEYS, DomainAdapter
from alpha_lab.adapter_loader import (
    BUILTIN_ADAPTERS,
    copy_builtin_to_workspace,
    load_adapter,
    load_builtin_adapter,
    resolve_adapter,
)
from alpha_lab.agent import AgentLoop
from alpha_lab.config import TaskConfig
from alpha_lab.context import ContextManager
from alpha_lab.events import AgentEvent, PhaseEvent
from alpha_lab.provider import Provider
from alpha_lab.tools import get_tool_schemas

logger = logging.getLogger("alpha_lab.phase0")

# ---------------------------------------------------------------------------
# Phase 0 System Prompt
# ---------------------------------------------------------------------------

PHASE0_SYSTEM_PROMPT = """\
You are **Alpha Lab Phase 0 — Domain Adapter Generator**.

Your job: create a complete domain adapter for the given task so the
Alpha Lab pipeline can work on ANY domain, not just time-series forecasting.

## What is an Adapter?

An adapter is a directory of files that configures the Alpha Lab pipeline:

- **manifest.json** — metric config, experiment structure, metadata
- **9 prompt .md files** — one per pipeline phase/role:
  phase1.md, phase2_builder.md, phase2_critic.md, phase2_tester.md,
  phase3_strategist.md, phase3_worker_implement.md, phase3_worker_analyze.md,
  phase3_reporter.md, phase3_fixer.md
- **domain_knowledge.md** — domain expertise injected into all prompts

## Process

1. **Read a reference adapter** to understand the format and style
   (use `read_reference_adapter` with name "time_series")
2. **Explore the data/benchmark** using `shell_exec` and `read_file`
   to understand what we're optimizing
3. **Generate all adapter files** using `write_adapter_file`:
   - Start with manifest.json (metric, experiment structure)
   - Write domain_knowledge.md
   - Write all 9 prompt files

## Prompt Writing Rules

Each prompt .md file should:
- Be 200-500 words
- Start with "You are **Alpha Lab [Role]**..."
- Include clear instructions for that phase/role
- Reference the domain's metrics, file structure, and goals
- Include tool usage instructions
- Match the style of the reference adapter

## manifest.json Structure

```json
{
  "domain_name": "...",
  "domain_description": "...",
  "phase2_framework_description": "...",
  "phase2_review_file": "review.md",
  "metric": {
    "primary_metric": "...",
    "direction": "maximize|minimize",
    "extract_key": "...",
    "display_name": "...",
    "secondary_metrics": [...]
  },
  "experiment": {
    "required_files": [...],
    "entry_point": "...",
    "results_dir": "results",
    "results_file": "metrics.json",
    "framework_dir": "...",
    "framework_files": [...]
  }
}
```

## Rules

- Write ALL 11 files (manifest + 9 prompts + domain_knowledge)
- Make prompts specific to the domain — don't write generic prompts
- The primary_metric must be extractable from a JSON results file
- The entry_point must be a Python script
- Call `report_to_user` when done with a summary of the adapter
"""

# ---------------------------------------------------------------------------
# Phase 0 Customization Prompt (for built-in adapter templates)
# ---------------------------------------------------------------------------

PHASE0_CUSTOMIZE_PROMPT = """\
You are **Alpha Lab Phase 0 — Adapter Customizer**.

A working domain adapter template has already been installed in the workspace.
Your job: examine the actual data/benchmark/task and **customize** the adapter
to be specific to this particular problem rather than generic.

## Process

1. **Read the current adapter** using `read_adapter` to see what's installed
2. **Explore the data** using `shell_exec` and `read_file`:
   - Check column names, data types, shape, date ranges
   - Look at distributions, missing values, unique values
   - Identify key features and patterns
3. **Patch adapter files** using `patch_adapter_file` to make them task-specific:
   - **domain_knowledge.md** — Add findings: actual column names, data
     characteristics, known domain patterns, feature descriptions
   - **Prompt files** — Reference actual data features, suggest domain-specific
     strategies, adjust focus areas based on what you found in the data
   - **manifest.json** — Tweak domain_description to be task-specific, adjust
     secondary_metrics if appropriate
4. Call `report_to_user` with a summary of what you customized and why

## Guidelines

- The built-in defaults for primary_metric, direction, and experiment structure
  are usually correct for the domain category — only change them if the task
  clearly demands it
- Focus on making generic prompts task-specific: replace placeholder language
  with references to actual columns, features, and data characteristics
- domain_knowledge.md is the highest-value file to customize — it gets injected
  into every phase's prompt
- Keep prompts the same length/style, just make them more specific
- Do NOT rewrite files that are already specific enough
"""


def _run_customization_agent(
    provider: Provider,
    config: TaskConfig,
    workspace: str,
    event_callback: Callable[[AgentEvent], None],
) -> None:
    """Run a lightweight agent to customize a built-in adapter template."""
    logger.info("Phase 0: running customization agent")
    event_callback(PhaseEvent(
        phase="phase0", step="adapter", status="customizing",
        detail="Customizing adapter for task-specific data",
    ))

    tools = get_tool_schemas(
        [
            "shell_exec", "read_file", "read_adapter",
            "patch_adapter_file", "report_to_user",
        ],
        include_web_search=False,
    )

    context = ContextManager(
        provider=provider,
        model=config.model,
        workspace=workspace,
    )

    def prompt_builder(
        workspace_arg: str | None,
        learnings: str | None,
        config_arg: Any | None = None,
    ) -> str:
        parts = [PHASE0_CUSTOMIZE_PROMPT]
        if workspace_arg:
            parts.append(f"\n## Workspace\n`{workspace_arg}`")
        return "\n".join(parts)

    agent = AgentLoop(
        provider=provider,
        model=config.model,
        context=context,
        event_callback=event_callback,
        reasoning_effort="medium",
        config=config,
        tools=tools,
        prompt_builder=prompt_builder,
        log_name="phase0_customize",
        min_report_attempts=1,
    )

    initial_message = (
        f"Customize the installed adapter for this specific task.\n\n"
        f"**Domain:** {config.domain or 'time_series'}\n"
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

    agent.run(initial_message)


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

    # 2. Built-in match: copy template → customize for task
    if domain in BUILTIN_ADAPTERS:
        logger.info("Phase 0: copying built-in adapter '%s'", domain)
        copy_builtin_to_workspace(domain, adapter_dir)
        _run_customization_agent(provider, config, workspace, event_callback)
        event_callback(PhaseEvent(
            phase="phase0", step="adapter", status="completed",
            detail=f"Customized built-in adapter: {domain}",
        ))
        return load_adapter(adapter_dir)

    # 3. No domain specified → use time_series default → customize for task
    if not domain:
        logger.info("Phase 0: no domain specified, using time_series default")
        copy_builtin_to_workspace("time_series", adapter_dir)
        _run_customization_agent(provider, config, workspace, event_callback)
        event_callback(PhaseEvent(
            phase="phase0", step="adapter", status="completed",
            detail="Customized time_series default adapter for task",
        ))
        return load_adapter(adapter_dir)

    # 4. Generation path: free-text domain description → run agent
    logger.info("Phase 0: generating adapter for domain: %s", domain)
    event_callback(PhaseEvent(
        phase="phase0", step="adapter", status="starting",
        detail=f"Generating adapter for: {domain}",
    ))

    tools = get_tool_schemas(
        [
            "shell_exec", "read_file", "write_adapter_file",
            "read_reference_adapter", "report_to_user",
        ],
        include_web_search=True,
    )

    context = ContextManager(
        provider=provider,
        model=config.model,
        workspace=workspace,
    )

    def prompt_builder(
        workspace_arg: str | None,
        learnings: str | None,
        config_arg: Any | None = None,
    ) -> str:
        parts = [PHASE0_SYSTEM_PROMPT]
        if workspace_arg:
            parts.append(f"\n## Workspace\n`{workspace_arg}`")
        return "\n".join(parts)

    agent = AgentLoop(
        provider=provider,
        model=config.model,
        context=context,
        event_callback=event_callback,
        reasoning_effort="medium",
        config=config,
        tools=tools,
        prompt_builder=prompt_builder,
        log_name="phase0",
        min_report_attempts=1,
    )

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

    agent.run(initial_message)

    # Load the generated adapter
    if not (adapter_dir / "manifest.json").exists():
        logger.warning("Phase 0 agent did not generate manifest.json, falling back to time_series")
        copy_builtin_to_workspace("time_series", adapter_dir)

    adapter = load_adapter(adapter_dir)

    event_callback(PhaseEvent(
        phase="phase0", step="adapter", status="completed",
        detail=f"Generated adapter: {adapter.domain_name}",
    ))

    return adapter
