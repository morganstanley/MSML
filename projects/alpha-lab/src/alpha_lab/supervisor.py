"""Supervisory agent for alpha-lab.

Monitors pipeline phases, catches problems, and can patch the domain adapter.
Each review method runs a short-lived agent via to inspect workspace artifacts and optionally patch
adapter files.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from alpha_lab.adapter import DomainAdapter
from alpha_lab.agents import load_agent
from alpha_lab.sandboxing import sandbox
from alpha_lab.config import TaskConfig
from alpha_lab.events import AgentEvent, PhaseEvent
from alpha_lab.provider import Provider

logger = logging.getLogger("alpha_lab.supervisor")


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
        agent_id: str,
        initial_message: str,
        phase_name: str,
    ) -> str:
        """Run a short-lived review agent (loaded from agent file) and return its final report."""
        agent_definition = load_agent(agent_id)

        self.event_callback(PhaseEvent(
            phase=phase_name, step="supervisor", status="starting",
            detail=f"Supervisor review: {agent_definition.log_name}",
        ))

        sandbox.run_agent(
            agent_id,
            self.event_callback,
            initial_message=initial_message,
            provider=self.provider,
            adapter=self.adapter,
        )

        self.event_callback(PhaseEvent(
            phase=phase_name, step="supervisor", status="completed",
            detail=f"Supervisor review complete: {agent_definition.log_name}",
        ))

        # The review verdict is surfaced via report_to_user events, not a return value.
        return ""

    def validate_adapter(self) -> str:
        """After Phase 0: check all adapter files present and valid."""
        logger.info("Supervisor: validating adapter")
        return self._run_review(
            agent_id="supervisor/adapter_validator",
            initial_message=(
                "Review the domain adapter in the workspace. "
                "Check completeness, validity, and quality. Go."
            ),
            phase_name="phase0",
        )

    def review_phase1(self) -> str:
        """After Phase 1: check exploration artifacts."""
        logger.info("Supervisor: reviewing Phase 1")
        return self._run_review(
            agent_id="supervisor/phase1_reviewer",
            initial_message=(
                "Review Phase 1 exploration output. "
                "Check learnings.md, data_report/, scripts/, plots/. Go."
            ),
            phase_name="phase1",
        )

    def review_phase2(self) -> str:
        """After Phase 2: check framework, tests, review verdict."""
        logger.info("Supervisor: reviewing Phase 2")
        framework_dir = "backtest"
        if self.adapter:
            framework_dir = self.adapter.experiment.framework_dir
        return self._run_review(
            agent_id="supervisor/phase2_reviewer",
            initial_message=(
                f"Review Phase 2 framework output in {framework_dir}/. "
                f"Check files, tests, and review verdict. Go."
            ),
            phase_name="phase2",
        )

    def phase3_health_check(self) -> str:
        """During Phase 3: diagnose high error rate."""
        logger.info("Supervisor: Phase 3 health check")
        return self._run_review(
            agent_id="supervisor/phase3_health_check",
            initial_message=(
                "The Phase 3 experiment error rate has exceeded 40%. "
                "Diagnose the systemic issue and patch the adapter if needed. Go."
            ),
            phase_name="phase3",
        )
