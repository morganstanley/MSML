"""Supervisory agent for alpha-lab.

Monitors pipeline phases, catches problems, and can patch the domain adapter.
Each review method runs a short-lived AgentLoop with tools to inspect workspace
artifacts and optionally patch adapter files.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from alpha_lab.adapter import DomainAdapter
from alpha_lab.agent import AgentLoop
from alpha_lab.config import TaskConfig
from alpha_lab.context import ContextManager
from alpha_lab.events import AgentEvent, PhaseEvent
from alpha_lab.provider import Provider
from alpha_lab.tools import get_tool_schemas

logger = logging.getLogger("alpha_lab.supervisor")


# ---------------------------------------------------------------------------
# Supervisor system prompts
# ---------------------------------------------------------------------------

VALIDATE_ADAPTER_PROMPT = """\
You are the **Alpha Lab Supervisor** reviewing a newly generated domain adapter.

## Checks
1. **Completeness**: All 11 files present (manifest.json, 9 prompt .md files, domain_knowledge.md)
2. **Manifest validity**: Valid JSON with required fields (metric, experiment)
3. **Prompts substantive**: Each prompt .md file is >100 characters and contains domain-specific content
4. **Metric sensible**: primary_metric, direction, and extract_key are consistent
5. **Experiment structure**: required_files and entry_point are specified

## Actions
- Use `read_adapter` to read the current adapter files
- If issues found, use `patch_adapter_file` to fix them
- Call `report_to_user` with your assessment (PASS/NEEDS_FIXES + details)

Be strict but practical. Minor style issues are OK. Missing files or broken JSON are not.
"""

REVIEW_PHASE1_PROMPT = """\
You are the **Alpha Lab Supervisor** reviewing Phase 1 (exploration) output.

## Checks
1. **learnings.md** exists and contains substantive findings
2. **data_report/** directory has findings.md and/or schema.md
3. **scripts/** directory has exploration scripts
4. **plots/** directory has visualization outputs
5. No obvious errors or empty files

## Actions
- Use `read_file` to inspect key files
- Use `shell_exec` to check file sizes and directory contents
- If the adapter prompts seem misaligned with the data, use `patch_adapter_file`
- Call `report_to_user` with PASS/NEEDS_ATTENTION + details

Don't block progress — Phase 1 doesn't need to be perfect.
"""

REVIEW_PHASE2_PROMPT = """\
You are the **Alpha Lab Supervisor** reviewing Phase 2 (framework) output.

## Checks
1. **Framework directory** exists with expected files
2. **Tests exist** and pass (check test output)
3. **Review verdict** is PASS (check review.md)
4. **No obvious bugs** in framework code

## Actions
- Use `read_file` to inspect framework files and review.md
- Use `shell_exec` to run tests if needed
- If the adapter's framework config is wrong, use `patch_adapter_file`
- Call `report_to_user` with PASS/NEEDS_FIXES + details
"""

PHASE3_HEALTH_CHECK_PROMPT = """\
You are the **Alpha Lab Supervisor** checking Phase 3 experiment health.

The error rate has exceeded 40%, indicating systemic issues.

## Diagnosis Checklist
1. **Read recent errors**: Use `read_board` and `grep_file` on experiment logs
2. **Identify patterns**: Are failures due to the same root cause?
3. **Common systemic issues**:
   - Wrong entry_point in adapter (experiments can't run)
   - Missing dependency in experiment template
   - Incorrect results JSON format (metric key mismatch)
   - GPU memory issues from bad default configs
   - Framework bugs causing all experiments to fail

## Actions
- Use `read_adapter` to check current adapter config
- Use `read_file` and `grep_file` to inspect failed experiments
- If the adapter needs fixing, use `patch_adapter_file` (creates git checkpoint)
- Call `report_to_user` with diagnosis and what you patched (if anything)

Focus on fixes that will prevent future failures, not retrospective analysis.
"""


class Supervisor:
    """Meta-agent that monitors pipeline phases and patches the adapter."""

    def __init__(
        self,
        provider: Provider,
        config: TaskConfig,
        workspace: str,
        adapter: DomainAdapter,
        event_callback: Callable[[AgentEvent], None],
    ) -> None:
        self.provider = provider
        self.config = config
        self.workspace = workspace
        self.adapter = adapter
        self.event_callback = event_callback

    def _run_review(
        self,
        system_prompt: str,
        initial_message: str,
        tools: list[dict],
        log_name: str,
        phase_name: str,
    ) -> str:
        """Run a short-lived review agent and return its final report."""
        context = ContextManager(
            provider=self.provider,
            model=self.config.model,
            workspace=self.workspace,
        )

        def prompt_builder(
            workspace: str | None,
            learnings: str | None,
            config: Any | None = None,
        ) -> str:
            parts = [system_prompt]
            if workspace:
                parts.append(f"\n## Workspace\n`{workspace}`")
            return "\n".join(parts)

        agent = AgentLoop(
            provider=self.provider,
            model=self.config.model,
            context=context,
            event_callback=self.event_callback,
            reasoning_effort="low",
            config=self.config,
            tools=tools,
            prompt_builder=prompt_builder,
            log_name=log_name,
            min_report_attempts=1,
            adapter=self.adapter,
        )

        self.event_callback(PhaseEvent(
            phase=phase_name, step="supervisor", status="starting",
            detail=f"Supervisor review: {log_name}",
        ))

        report = agent.run(initial_message)

        self.event_callback(PhaseEvent(
            phase=phase_name, step="supervisor", status="completed",
            detail=f"Supervisor review complete: {log_name}",
        ))

        return report or ""

    def validate_adapter(self) -> str:
        """After Phase 0: check all adapter files present and valid."""
        logger.info("Supervisor: validating adapter")
        tools = get_tool_schemas([
            "read_file", "grep_file", "shell_exec",
            "read_adapter", "patch_adapter_file", "report_to_user",
        ])
        return self._run_review(
            system_prompt=VALIDATE_ADAPTER_PROMPT,
            initial_message=(
                "Review the domain adapter in the workspace. "
                "Check completeness, validity, and quality. Go."
            ),
            tools=tools,
            log_name="supervisor_validate_adapter",
            phase_name="phase0",
        )

    def review_phase1(self) -> str:
        """After Phase 1: check exploration artifacts."""
        logger.info("Supervisor: reviewing Phase 1")
        tools = get_tool_schemas([
            "read_file", "grep_file", "shell_exec",
            "read_adapter", "patch_adapter_file", "report_to_user",
        ])
        return self._run_review(
            system_prompt=REVIEW_PHASE1_PROMPT,
            initial_message=(
                "Review Phase 1 exploration output. "
                "Check learnings.md, data_report/, scripts/, plots/. Go."
            ),
            tools=tools,
            log_name="supervisor_review_phase1",
            phase_name="phase1",
        )

    def review_phase2(self) -> str:
        """After Phase 2: check framework, tests, review verdict."""
        logger.info("Supervisor: reviewing Phase 2")
        framework_dir = "backtest"
        if self.adapter:
            framework_dir = self.adapter.experiment.framework_dir
        tools = get_tool_schemas([
            "read_file", "grep_file", "shell_exec",
            "read_adapter", "patch_adapter_file", "report_to_user",
        ])
        return self._run_review(
            system_prompt=REVIEW_PHASE2_PROMPT,
            initial_message=(
                f"Review Phase 2 framework output in {framework_dir}/. "
                f"Check files, tests, and review verdict. Go."
            ),
            tools=tools,
            log_name="supervisor_review_phase2",
            phase_name="phase2",
        )

    def phase3_health_check(self) -> str:
        """During Phase 3: diagnose high error rate."""
        logger.info("Supervisor: Phase 3 health check")
        tools = get_tool_schemas([
            "read_file", "grep_file", "shell_exec", "read_board",
            "read_adapter", "patch_adapter_file", "report_to_user",
        ])
        return self._run_review(
            system_prompt=PHASE3_HEALTH_CHECK_PROMPT,
            initial_message=(
                "The Phase 3 experiment error rate has exceeded 40%. "
                "Diagnose the systemic issue and patch the adapter if needed. Go."
            ),
            tools=tools,
            log_name="supervisor_health_check",
            phase_name="phase3",
        )
