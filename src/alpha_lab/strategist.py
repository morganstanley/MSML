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
        _direction = "maximize"
        if self.adapter is not None:
            _metric = self.adapter.metric.primary_metric
            _metric_display = self.adapter.metric.display_name
            _direction = self.adapter.metric.direction

        # Budget tracking. Two caps apply in parallel:
        #   1. ``max_pending_proposals`` (PRIMARY signal): the strategist
        #      should only refill the ``to_implement`` queue up to this many
        #      pending rows. This is the sliding cap that keeps the
        #      strategist actively re-engaging with new evidence each turn,
        #      instead of dumping the entire lifetime budget in the first
        #      session and then sitting idle as debriefs land.
        #   2. ``max_experiments`` (SAFETY ceiling): the long-run lifetime
        #      cap. Stays as a hard upper bound; the Conductor can request
        #      ``request_run_end`` before it fires.
        max_experiments = self.config.pipeline.phase3.max_experiments
        max_pending = getattr(
            self.config.pipeline.phase3, "max_pending_proposals", 12
        )
        summary = self.db.board_summary()
        total_proposed = sum(v for k, v in summary.items() if k != "cancelled")
        analyzed_count = summary.get("analyzed", 0)
        # Parked rows (Conductor soft-cancelled) don't consume the sliding cap.
        pending_count = self.db.active_to_implement_count()
        lifetime_remaining = max(0, max_experiments - total_proposed)
        # The sliding-cap headroom: how many *new* proposals the strategist
        # may add this turn so the pending queue stays at or below the cap.
        # Genuinely informed proposals win over filling slots — if pending is
        # already at the cap, the strategist should re-read recent debriefs
        # and (maybe) prune the queue rather than mechanically refilling.
        slots_open_this_turn = max(0, max_pending - pending_count)

        parts.append("## Experiment Budget")
        parts.append(f"  Pending (to_implement): {pending_count}")
        parts.append(f"  Pending cap (max_pending_proposals): {max_pending}")
        parts.append(
            f"  **Slots open this turn: {slots_open_this_turn}** "
            f"(propose at most this many new experiments per turn so the "
            f"queue stays responsive to fresh debriefs)"
        )
        parts.append("")
        parts.append(f"  Lifetime cap (max_experiments, safety ceiling): {max_experiments}")
        parts.append(f"  Total proposed lifetime: {total_proposed}")
        parts.append(f"  Fully analyzed: {analyzed_count}")
        parts.append(f"  Lifetime remaining: {lifetime_remaining}")
        if slots_open_this_turn == 0:
            parts.append(
                "  📋 PENDING QUEUE AT CAP — do not propose new experiments "
                "this turn. Instead: read recent debriefs, update the "
                "playbook with what you learned, and (if appropriate) "
                "consider whether any queued rows have been invalidated "
                "by new evidence — write a note_to_conductor asking for "
                "them to be parked. The queue will refill on the next "
                "turn after the dispatcher works through implements."
            )
        if lifetime_remaining == 0:
            parts.append(
                "  🛑 LIFETIME CAP REACHED — no more experiments can be "
                "proposed. The Conductor may request a graceful run end."
            )

        # Board summary
        parts.append("\n## Board Summary")
        for col, cnt in sorted(summary.items()):
            parts.append(f"  {col}: {cnt}")

        shared_paths = (
            "playbook.md", "research_state.md", "verify/feedback_stream.md",
            "verify/feedback_to_system.md",
        )
        available_shared = [
            rel for rel in shared_paths
            if (Path(self.workspace) / rel).is_file()
        ]
        parts.append(
            "\n## Available Shared Artifacts\n  "
            + (", ".join(available_shared) if available_shared else "none")
        )

        # Experiment state and artifact readiness. Listing actual files for a
        # bounded recent window avoids speculative reads without growing the
        # prompt throughout a long run.
        experiments = self.db.list_all()[-20:]
        if experiments:
            parts.append("\n## Experiments and Available Artifacts")
            for exp in experiments:
                metrics_str = ""
                if exp.results_json:
                    try:
                        m = json.loads(exp.results_json)
                        pieces = [f"{k}={v}" for k, v in m.items()]
                        metrics_str = f" [{', '.join(pieces[:5])}]"
                    except (json.JSONDecodeError, TypeError):
                        pass
                err = f" ERROR: {exp.error}" if exp.error else ""
                exp_dir = Path(self.workspace) / "experiments" / exp.name
                ready = [
                    rel for rel in (
                        "config.yaml", "results/metrics.json", "debrief.md",
                        ".variant_intent.md",
                    )
                    if (exp_dir / rel).is_file()
                ]
                parent = f" parent=#{exp.parent_id}" if exp.parent_id is not None else ""
                parts.append(
                    f"  #{exp.id} {exp.name} [{exp.status}]{parent}{metrics_str}{err}; "
                    f"available: {', '.join(ready) if ready else 'none'}"
                )

        # Leaderboard
        leaders = self.db.leaderboard(_metric, 10, direction=_direction)
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
                            f"(reports/{milestone_dirs[-1].name}/report.md)\n"
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

        # Phase 1 learnings (summary — use memory_search for details)
        learnings_path = Path(self.workspace) / "learnings.md"
        if learnings_path.exists():
            content = learnings_path.read_text().strip()
            if content:
                truncated = content[:1500]
                if len(content) > 1500:
                    truncated += "\n\n[...use memory_search for detailed findings]"
                parts.append(f"\n## Phase 1 Learnings (summary)\n{truncated}")

        # Conductor directives — filtered for this role, with one-shot
        # claims by same-role peers already removed. Tolerant of missing
        # files so the no_conductor=True mode is unchanged.
        try:
            from alpha_lab import conductor_tools as _ct
            active = _ct.directives_for_role(self.workspace, "strategist")
            acks = _ct.read_directive_acks(self.workspace)
            if active or acks:
                rendered = _ct.render_directives_for_prompt(active, acks=acks)
                parts.append("\n" + rendered[:6000])
            # Zero-uptake escalation: directives that stay unacknowledged
            # turn after turn get a mandatory banner at the very top
            # (zero-ack runs lost every audited pair, 2026-08-08).
            banner = _ct.directive_uptake_escalation(
                self.workspace, active, acks, actor_role="strategist")
            if banner:
                parts.insert(0, banner)
            details = _ct.read_annotation_details(self.workspace)
            if details:
                # Surface the Conductor's reason inline so the strategist
                # doesn't have to grep meta_log.jsonl to learn why a row
                # is labeled the way it is. Truncate per-reason to keep
                # the section bounded.
                rows = []
                for eid, rec in details.items():
                    label = rec.get("label", "")
                    reason = (rec.get("reason") or "").strip()
                    if reason:
                        rows.append(f"- #{eid} [{label}] — {reason[:240]}")
                    else:
                        rows.append(f"- #{eid} [{label}]")
                parts.append(
                    "\n## Leaderboard annotations (Conductor-applied labels)\n"
                    + "\n".join(rows)[:4000]
                )
        except Exception as e:
            logger.debug("Skipping Conductor directive injection: %s", e)

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
        # Lifetime budget gate (mirrors msml's turn-skip in strategist
        # run_turn): once max_experiments non-cancelled rows exist, skip
        # the turn entirely instead of asking the model not to propose.
        # The prompt banner alone proved advisory — measured 2026-07-31,
        # glm-5.2 proposed 72 experiments past it in one run; the
        # propose_experiment tool accepts rows without a budget check.
        max_experiments = self.config.pipeline.phase3.max_experiments
        summary = self.db.board_summary()
        total_proposed = sum(v for k, v in summary.items() if k != "cancelled")
        if total_proposed >= max_experiments:
            logger.info(
                "Strategist turn skipped: lifetime budget exhausted "
                f"({total_proposed}/{max_experiments} proposed)"
            )
            return

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
            "read_board", "propose_experiment", "propose_variant",
            "update_playbook", "read_file", "grep_file", "report_to_user",
            "memory_store", "memory_search", "memory_read",
            "note_to_conductor", "ack_directive",
        ]
        # Conductor administers parking; the strategist asks via
        # note_to_conductor instead of cancelling directly. In NOOP mode
        # (no_conductor=True) we restore cancel_experiments so the
        # strategist retains its pre-Conductor capability — the system
        # reverts to its prior behavior end-to-end.
        if self.config.pipeline.phase3.no_conductor:
            tool_names.append("cancel_experiments")
            for n in ("note_to_conductor", "ack_directive"):
                try:
                    tool_names.remove(n)
                except ValueError:
                    pass
        # Remove playbook tool in no_playbook ablation mode
        if self.config.pipeline.phase3.no_playbook:
            tool_names.remove("update_playbook")

        tools = get_tool_schemas(tool_names, include_web_search=True)

        context = ContextManager(
            provider=self.provider,
            model=self.config.model,
            workspace=self.workspace,
            summarization_threshold_tokens=self.config.context_summarization_threshold_tokens,
            learnings_summary_threshold_tokens=self.config.learnings_summary_threshold_tokens,
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
            max_iterations=getattr(
                self.config.pipeline.phase3, "strategist_max_iterations", 0),
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
