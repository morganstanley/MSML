"""Strategist agent for Phase 3 — proposes experiments and maintains playbook."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from alpha_lab.agent import AgentLoop
from alpha_lab.config import TaskConfig
from alpha_lab.context import ContextManager
from alpha_lab.events import AgentEvent
from alpha_lab.experiment_db import ExperimentDB
from alpha_lab.prompts import build_step_prompt
from alpha_lab.provider import Provider
from alpha_lab.tools import WEB_SEARCH_TOOL, get_tool_schemas

logger = logging.getLogger("alpha_lab.strategist")


class Strategist:
    """Periodically runs a strategist turn to propose experiments and update playbook."""

    def __init__(
        self,
        provider: Provider,
        config: TaskConfig,
        workspace: str,
        db: ExperimentDB,
        event_callback: Callable[[AgentEvent], None],
        adapter: Any = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.workspace = workspace
        self.db = db
        self.event_callback = event_callback
        self.adapter = adapter
        self._agent: AgentLoop | None = None

    def stop(self) -> None:
        if self._agent is not None:
            self._agent.stop()

    @staticmethod
    def _resource_snapshot() -> str:
        """Gather a lightweight snapshot of machine resource utilization."""
        import os
        import subprocess as sp

        lines = ["\n## Machine Resource Snapshot"]
        try:
            n_cores = os.cpu_count() or 0
            load_1, load_5, load_15 = os.getloadavg()
            lines.append(f"  CPU cores: {n_cores}")
            lines.append(
                f"  Load average (1/5/15 min): {load_1:.0f} / {load_5:.0f} / {load_15:.0f}"
            )
            if n_cores:
                lines.append(
                    f"  Load-to-core ratio: {load_1 / n_cores:.1f}x "
                    f"({'overloaded' if load_1 > n_cores * 1.5 else 'ok'})"
                )
        except Exception:
            lines.append("  CPU load: unavailable")

        try:
            with open("/proc/meminfo") as f:
                meminfo = f.read()
            for key in ("MemTotal", "MemAvailable"):
                for line in meminfo.splitlines():
                    if line.startswith(key):
                        kb = int(line.split()[1])
                        lines.append(f"  {key}: {kb // (1024 * 1024)} GB")
                        break
        except Exception:
            pass

        try:
            result = sp.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                lines.append("  GPUs:")
                for row in result.stdout.strip().splitlines():
                    parts = [p.strip() for p in row.split(",")]
                    if len(parts) == 4:
                        idx, util, used, total = parts
                        lines.append(
                            f"    GPU {idx}: {util}% util, "
                            f"{used}/{total} MB VRAM"
                        )
        except Exception:
            pass

        # Count experiment processes
        try:
            result = sp.run(
                ["ps", "-u", os.environ.get("USER", ""), "-o", "args"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                procs = result.stdout.splitlines()
                full_runs = sum(
                    1 for p in procs
                    if "run_experiment" in p and "--smoke" not in p
                )
                smoke_runs = sum(
                    1 for p in procs if "run_experiment" in p and "--smoke" in p
                )
                lines.append(
                    f"  Running experiments: {full_runs} full + {smoke_runs} smoke"
                )
        except Exception:
            pass

        return "\n".join(lines)

    def _build_context(self) -> str:
        """Build rich context for the strategist from DB and workspace files."""
        parts: list[str] = []

        # Use adapter metric if available
        _metric = "sharpe"
        _metric_display = "Sharpe"
        if self.adapter is not None:
            _metric = self.adapter.metric.primary_metric
            _metric_display = self.adapter.metric.display_name

        # Budget tracking — cancelled experiments do not consume budget slots
        max_experiments = self.config.pipeline.phase3.max_experiments
        summary = self.db.board_summary()
        total_proposed = sum(v for k, v in summary.items() if k != "cancelled")
        analyzed_count = summary.get("analyzed", 0)
        remaining_budget = max(0, max_experiments - total_proposed)

        parts.append("## Experiment Budget")
        parts.append(f"  Max experiments: {max_experiments}")
        parts.append(f"  Already proposed: {total_proposed}")
        parts.append(f"  Fully analyzed: {analyzed_count}")
        parts.append(f"  **Remaining budget: {remaining_budget}**")
        if remaining_budget < 10:
            parts.append(f"  ⚠️ LOW BUDGET — be very selective, focus on highest-value experiments")
        if remaining_budget == 0:
            parts.append(f"  🛑 BUDGET EXHAUSTED — no more experiments can be proposed")

        # Board summary
        parts.append("\n## Board Summary")
        for col, cnt in sorted(summary.items()):
            parts.append(f"  {col}: {cnt}")

        # Recent experiments
        recent = self.db.list_all()[-10:]
        if recent:
            parts.append("\n## Recent Experiments")
            for exp in recent:
                metrics_str = ""
                if exp.results_json:
                    try:
                        m = json.loads(exp.results_json)
                        pieces = [f"{k}={v}" for k, v in m.items()]
                        metrics_str = f" [{', '.join(pieces[:5])}]"
                    except (json.JSONDecodeError, TypeError):
                        pass
                err = f" ERROR: {exp.error}" if exp.error else ""
                parts.append(
                    f"  #{exp.id} {exp.name} [{exp.status}]{metrics_str}{err}"
                )

        # Leaderboard
        leaders = self.db.leaderboard(_metric, 10)
        if leaders:
            parts.append(f"\n## Leaderboard (by {_metric_display})")
            for i, exp in enumerate(leaders, 1):
                try:
                    m = json.loads(exp.results_json or "{}")
                    primary_val = m.get(_metric, "?")
                except (json.JSONDecodeError, TypeError):
                    primary_val = "?"
                parts.append(f"  {i}. #{exp.id} {exp.name} — {_metric_display}: {primary_val}")

        # Machine resource snapshot
        parts.append(self._resource_snapshot())

        # Latest milestone report (feedback from Reporter)
        reports_dir = Path(self.workspace) / "reports"
        if reports_dir.is_dir():
            milestone_dirs = sorted(
                (d for d in reports_dir.iterdir()
                 if d.is_dir() and d.name.startswith("milestone_")),
                key=lambda d: d.name,
            )
            if milestone_dirs:
                latest_report = milestone_dirs[-1] / "report.md"
                if latest_report.exists():
                    content = latest_report.read_text().strip()
                    if content:
                        parts.append(
                            f"\n## Latest Milestone Report "
                            f"({milestone_dirs[-1].name})\n"
                            f"{content[:6000]}"
                        )

        # Playbook (suppressed in no_playbook ablation mode)
        if not self.config.pipeline.phase3.no_playbook:
            playbook_path = Path(self.workspace) / "playbook.md"
            if playbook_path.exists():
                content = playbook_path.read_text().strip()
                if content:
                    parts.append(f"\n## Current Playbook\n{content}")
            else:
                parts.append("\n## Current Playbook\nNo playbook yet — this is your first turn.")

        # Phase 1 learnings
        learnings_path = Path(self.workspace) / "learnings.md"
        if learnings_path.exists():
            content = learnings_path.read_text().strip()
            if content:
                parts.append(f"\n## Phase 1 Learnings\n{content[:3000]}")

        # Task config
        if self.config:
            parts.append(f"\n## Task Config")
            parts.append(f"Data: {self.config.data_path}")
            parts.append(f"Description: {self.config.description}")
            if self.config.target:
                parts.append(f"Target: {self.config.target}")

        return "\n".join(parts)

    def run_turn(self) -> None:
        """Run a single strategist turn."""
        logger.info("Strategist turn starting")

        extra_context = self._build_context()

        def prompt_builder(
            workspace: str | None,
            learnings: str | None,
            config: Any | None = None,
        ) -> str:
            return build_step_prompt(
                "phase3_strategist",
                workspace,
                learnings,
                config,
                extra_context,
                adapter=self.adapter,
            )

        tool_names = [
            "read_board", "propose_experiment", "cancel_experiments",
            "update_playbook", "read_file", "grep_file", "report_to_user",
        ]
        # Remove playbook tool in no_playbook ablation mode
        if self.config.pipeline.phase3.no_playbook:
            tool_names.remove("update_playbook")

        tools = get_tool_schemas(tool_names, include_web_search=True)

        context = ContextManager(
            provider=self.provider,
            model=self.config.model,
            workspace=self.workspace,
        )

        agent = AgentLoop(
            provider=self.provider,
            model=self.config.model,
            context=context,
            event_callback=self.event_callback,
            reasoning_effort=self.config.reasoning_effort,
            config=self.config,
            tools=tools,
            prompt_builder=prompt_builder,
            log_name="strategist",
            min_report_attempts=1,
            db=self.db,
            adapter=self.adapter,
        )

        self._agent = agent
        try:
            agent.run(
                "Review the board and propose new experiments. "
                "Read the context above for current state. Go."
            )
        finally:
            self._agent = None
            logger.info("Strategist turn complete")
