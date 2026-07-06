"""Strategist agent for Phase 3 — proposes experiments and maintains playbook."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alpha_lab.adapter import DomainAdapter

from alpha_lab import deps, utils
from alpha_lab.agents import load_agent
from alpha_lab.sandboxing import sandbox
from alpha_lab.config import TaskConfig
from alpha_lab.events import AgentEvent
from alpha_lab.experiment_db import ExperimentDB, is_smoke_result
from alpha_lab.provider import Provider

logger = logging.getLogger("alpha_lab.strategist")


class StallError(RuntimeError):
    """Strategist proposed nothing into an idle board with free capacity — fail the run."""


class Strategist:
    """Periodically runs a strategist turn to propose experiments and update playbook."""

    def __init__(
        self,
        provider: Provider,
        workspace: str,
        db: ExperimentDB,
        event_callback: Callable[[AgentEvent], None],
        adapter: DomainAdapter,
    ) -> None:
        self.provider = provider
        self.workspace = workspace
        self.db = db
        self.event_callback = event_callback
        self.adapter = adapter
        self._run_handle: sandbox.AgentRunHandle | None = None

    def stop(self) -> None:
        if self._run_handle is not None:
            self._run_handle.stop()

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

    def _proposal_bounds(self) -> tuple[int, int, dict[str, dict[str, int]]]:
        """(min_proposals, max_proposals, slot_states) for this turn.

        max = min(free slots, free workers, remaining budget) — what a worker can pick up
        immediately without exceeding max_experiments. min = 1 only when the board is idle
        (no busy slots) with capacity and budget free, else 0.
        """
        states = utils.slot_states(self.db)
        free_total = sum(s["free"] for s in states.values())
        busy_total = sum(s["busy"] for s in states.values())
        free_workers = utils.worker_states(self.db)["free"]

        # Global experiment budget: never propose past max_experiments (cancelled
        # experiments don't consume budget — matches _build_context).
        config = deps.get().config
        summary = self.db.board_summary()
        total_proposed = sum(v for k, v in summary.items() if k != "cancelled")
        remaining_budget = max(0, config.pipeline.phase3.max_experiments - total_proposed)

        min_proposals = 1 if (
            busy_total == 0 and free_total > 0 and remaining_budget > 0
        ) else 0
        max_proposals = min(free_total, free_workers, remaining_budget)
        return min_proposals, max_proposals, states

    def _build_context(self, jit_bounds: tuple[int, int] | None = None) -> str:
        """Build rich context for the strategist from DB and workspace files."""
        parts: list[str] = []
        config = deps.get().config

        _metric = self.adapter.metric.primary_metric
        _metric_display = self.adapter.metric.display_name
        _direction = self.adapter.metric.direction

        # Budget tracking — cancelled experiments do not consume budget slots
        max_experiments = config.pipeline.phase3.max_experiments
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
                smoke = is_smoke_result(exp.results_json)
                metrics_str = ""
                if smoke:
                    metrics_str = " [SMOKE — metrics redacted]"
                elif exp.results_json:
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
        leaders = self.db.leaderboard(_metric, 10, _direction)
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
        if not config.pipeline.phase3.no_playbook:
            playbook_path = Path(self.workspace) / "playbook.md"
            if playbook_path.exists():
                content = playbook_path.read_text().strip()
                if content:
                    parts.append(f"\n## Current Playbook\n{content}")
            else:
                parts.append("\n## Current Playbook\nNo playbook yet — this is your first turn.")

        # Submission rules: proposal directives + allowed resources + resource state +
        # in-flight list. Rendered every turn; directives and the allowed `resource` set
        # branch on JIT mode. Resource state is identical in both modes (all enabled
        # devices, including fully-occupied ones JIT won't let you propose for). Only
        # `running` rows carry runtime duration/limit; earlier states haven't started.
        states = utils.slot_states(self.db)
        allowed = (
            list(states)
            if jit_bounds is None
            else [t for t, s in states.items() if s["free"] > 0]
        )

        parts += [
            "\n## Submission Rules",
            "Generate experiment proposals based on your current strategy.",
            f"STRICT: Each proposal's requested `resource` must be exactly one of: {allowed}.",
            "STRICT: Only propose an experiment if it makes more sense to do so now than later.",
            "STRICT: Explain how many experiments you proposed and why when calling `report_to_user`.",
        ]
        if jit_bounds is not None:
            min_proposals, max_proposals = jit_bounds
            parts.append(
                f"STRICT: Generate {min_proposals} to {max_proposals} just-in-time"
                " proposals for immediate execution based on the available slots."
            )

        parts += ["", "### Resource State"]
        for rtype, s in states.items():
            parts.append(f"{rtype.upper()}: {s['free']} of {s['total']} slots free")

        parts += ["", "### In-flight experiments:"]
        p3 = config.pipeline.phase3
        in_flight = self.db.list_by_status(
            "to_implement", "implemented", "checked", "queued", "running"
        )
        now = time.time()
        for exp in in_flight:
            if exp.status == "running" and exp.started_at:
                rd = int(now - exp.started_at)
                limit = (
                    p3.cpu_time_limit_seconds
                    if utils.experiment_resource(exp) == "cpu"
                    else p3.time_limit_seconds
                )
                parts.append(
                    f"{exp.name}: {exp.status}, runtime duration: {rd}s, runtime limit: {limit}s"
                )
            else:
                parts.append(f"{exp.name}: {exp.status}")
        if not in_flight:
            parts.append("(none)")

        return "\n".join(parts)

    def run_turn(self) -> None:
        """Run a single strategist turn."""
        logger.info("Strategist turn starting")
        config = deps.get().config

        # JIT: compute this turn's proposal bounds. Skip the turn entirely if nothing can
        # be implemented right now (no free slot or no free worker) rather than design
        # stale work — the trigger re-fires once capacity frees up.
        jit_bounds = None
        forcing = False
        if config.pipeline.phase3.jit:
            min_p, max_p, _states = self._proposal_bounds()
            if max_p == 0:
                logger.info("Strategist turn skipped: no free slot/worker to fill now")
                return
            jit_bounds = (min_p, max_p)
            forcing = min_p == 1

        extra_context = self._build_context(jit_bounds)

        agent_definition = load_agent("phase3/strategist")
        # Remove the playbook tool in no_playbook ablation mode.
        tools_include = None
        if config.pipeline.phase3.no_playbook:
            tools_include = tuple(
                tool.name
                for tool in agent_definition.tools
                if tool.name != "update_playbook"
            )

        try:
            sandbox.run_agent(
                "phase3/strategist",
                self.event_callback,
                initial_message=(
                    "Review the board and propose new experiments. "
                    "Read the context above for current state. Go."
                ),
                extra_context=extra_context,
                tools_include=tools_include,
                db_path=self.db.db_path,
                provider=self.provider,
                db=self.db,
                adapter=self.adapter,
                owner=self,
            )
        finally:
            logger.info("Strategist turn complete")

        # JIT fail-loud: if the board was idle with free capacity (forcing) and the
        # strategist still proposed nothing, fail the run rather than silently stall.
        if forcing and (
            sum(s["busy"] for s in utils.slot_states(self.db).values()) == 0
        ):
            raise StallError(
                "Strategist proposed nothing into an idle board with free capacity"
            )
