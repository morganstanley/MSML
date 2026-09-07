"""Worker agent for Phase 3 — implements and analyzes experiments."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from alpha_lab.agent import AgentLoop
from alpha_lab.config import TaskConfig
from alpha_lab.context import ContextManager
from alpha_lab.events import AgentEvent
from alpha_lab.experiment_db import Experiment, ExperimentDB
from alpha_lab.prompts import build_step_prompt
from alpha_lab.provider import Provider
from alpha_lab.tools import get_tool_schemas

logger = logging.getLogger("alpha_lab.worker")


class Worker:
    """On-demand worker that creates fresh AgentLoops for implement/analyze tasks."""

    def __init__(
        self,
        worker_id: str,
        provider: Provider,
        config: TaskConfig,
        workspace: str,
        db: ExperimentDB,
        event_callback: Callable[[AgentEvent], None],
        metrics: Any | None = None,
        adapter: Any | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.provider = provider
        self.config = config
        self.workspace = workspace
        self.db = db
        self.event_callback = event_callback
        self.metrics = metrics
        self.adapter = adapter
        self.last_event_at: float = 0.0
        self._current_experiment: Experiment | None = None
        self._thread: threading.Thread | None = None
        self._agent: AgentLoop | None = None
        self._stop_requested = False

    def _tracked_event_callback(self, event: AgentEvent) -> None:
        """Wrap event_callback to update last_event_at for watchdog."""
        self.last_event_at = time.time()
        self.event_callback(event)

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def current_experiment(self) -> Experiment | None:
        return self._current_experiment

    def implement(self, experiment: Experiment) -> None:
        """Start implementing an experiment in a background thread."""
        if self.busy:
            raise RuntimeError(f"Worker {self.worker_id} is busy")
        self._current_experiment = experiment
        self.last_event_at = time.time()
        self.db.assign_worker(experiment.id, self.worker_id)
        self._thread = threading.Thread(
            target=self._run_implement,
            args=(experiment,),
            daemon=True,
        )
        self._thread.start()

    def analyze(self, experiment: Experiment) -> None:
        """Start analyzing an experiment in a background thread."""
        if self.busy:
            raise RuntimeError(f"Worker {self.worker_id} is busy")
        self._current_experiment = experiment
        self.last_event_at = time.time()
        self.db.assign_worker(experiment.id, self.worker_id)
        self._thread = threading.Thread(
            target=self._run_analyze,
            args=(experiment,),
            daemon=True,
        )
        self._thread.start()

    def generate_report(self, milestone_number: int, done_count: int) -> None:
        """Start generating a milestone report in a background thread."""
        if self.busy:
            raise RuntimeError(f"Worker {self.worker_id} is busy")
        self.last_event_at = time.time()
        self._thread = threading.Thread(
            target=self._run_report,
            args=(milestone_number, done_count),
            daemon=True,
        )
        self._thread.start()

    def fix(self, experiment: Experiment) -> None:
        """Start fixing a failed experiment in a background thread."""
        if self.busy:
            raise RuntimeError(f"Worker {self.worker_id} is busy")
        self._current_experiment = experiment
        self.last_event_at = time.time()
        self.db.assign_worker(experiment.id, self.worker_id)
        self._thread = threading.Thread(
            target=self._run_fix,
            args=(experiment,),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_requested = True
        if self._agent is not None:
            self._agent.stop()

    def _build_experiment_context(self, experiment: Experiment) -> str:
        """Build extra context describing the experiment, with optional
        Conductor directive injection at the top so the worker reads
        steering instructions before the experiment description."""
        lines: list[str] = []

        # Conductor directives — filtered for this role + experiment, so
        # one-shot directives already claimed by another worker do not
        # show up, and per-experiment directives only appear for the
        # matching experiment id. Tolerant of missing files
        # (no_conductor mode is a no-op here).
        try:
            from alpha_lab import conductor_tools as _ct
            active = _ct.directives_for_role(
                self.workspace, "worker", experiment_id=experiment.id,
            )
            acks = _ct.read_directive_acks(self.workspace)
            if active or acks:
                rendered = _ct.render_directives_for_prompt(active, acks=acks)
                lines.append(rendered[:4000])
                lines.append("")
        except Exception as e:
            logger.debug("Skipping Conductor directive injection: %s", e)

        lines.extend([
            f"## Experiment #{experiment.id}: {experiment.name}",
            f"**Description:** {experiment.description}",
            f"**Hypothesis:** {experiment.hypothesis}",
            f"**Config:** {experiment.config_json}",
            f"**Status:** {experiment.status}",
        ])
        if experiment.parent_id is None:
            lines.append("**Variant:** no")
        else:
            lines.append(f"**Variant of:** experiment #{experiment.parent_id}")
        exp_dir = Path(self.workspace) / "experiments" / experiment.name
        ready = [
            rel for rel in (
                "config.yaml", "results/metrics.json", "debrief.md",
                ".variant_intent.md",
            )
            if (exp_dir / rel).is_file()
        ]
        lines.append(f"**Available artifacts:** {', '.join(ready) if ready else 'none'}")
        if experiment.error:
            lines.append(f"**Error:** {experiment.error}")
        return "\n".join(lines)

    def _run_implement(self, experiment: Experiment) -> None:
        """Run the implement task."""
        try:
            experiment = _ensure_mlflow_run(self.db, experiment)
            extra_context = self._build_experiment_context(experiment)

            def prompt_builder(
                workspace: str | None,
                learnings: str | None,
                config: Any | None = None,
            ) -> str:
                return build_step_prompt(
                    "phase3_worker_implement",
                    workspace,
                    learnings,
                    config,
                    extra_context,
                    adapter=self.adapter,
                )

            tools = get_tool_schemas(
                [
                    "shell_exec", "read_file", "grep_file", "view_image",
                    "update_experiment", "report_to_user",
                    "memory_store", "memory_search", "memory_read",
                    "note_to_conductor", "ack_directive",
                ],
            )

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
                event_callback=self._tracked_event_callback,
                reasoning_effort=self.config.reasoning_effort,
                config=self.config,
                tools=tools,
                prompt_builder=prompt_builder,
                log_name=f"worker_{self.worker_id}_implement_{experiment.name}",
                min_report_attempts=1,
                db=self.db,
                metrics=self.metrics,
                adapter=self.adapter,
                mlflow_run_target=experiment.mlflow_run_uuid,
            )

            self._agent = agent
            agent.run(
                f"Implement experiment #{experiment.id} '{experiment.name}'. "
                f"Read the experiment details above. Go."
            )
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Worker {self.worker_id} implement error: {e}\n{tb}")
            self.db.set_error(experiment.id, f"{e}\n{tb}")
        finally:
            self._agent = None
            self._current_experiment = None
            # Auto-transition: if LLM didn't update status, do it based on filesystem
            try:
                self._ensure_implement_status(experiment)
            except Exception as e:
                logger.error(f"_ensure_implement_status failed: {e}")
            try:
                self.db.release_worker(experiment.id)
            except Exception as e:
                logger.error(f"release_worker failed: {e}")

    def _ensure_implement_status(self, experiment: Experiment) -> None:
        """Ensure experiment progresses after implement, even if LLM didn't call update_experiment."""
        current = self.db.get(experiment.id)
        if current is None:
            return

        # If already past to_implement, LLM did its job
        if current.status != "to_implement":
            return

        # If there's already an error set (agent crashed), don't overwrite it
        if current.error:
            return

        exp_dir = Path(self.workspace) / "experiments" / experiment.name

        # Use adapter-defined required files or defaults
        required_files = ["strategy.py", "run_experiment.py"]
        if self.adapter is not None:
            required_files = self.adapter.experiment.required_files

        file_status = {f: (exp_dir / f).exists() for f in required_files}
        all_present = all(file_status.values())

        if all_present:
            logger.warning(
                f"Experiment #{experiment.id} {experiment.name}: auto-transitioning "
                f"to_implement → checked (files exist but LLM didn't update status)"
            )
            self.db.update_status(experiment.id, "checked")
        else:
            missing = [f for f, exists in file_status.items() if not exists]
            logger.warning(
                f"Experiment #{experiment.id} {experiment.name}: worker finished but "
                f"files missing: {missing}. Setting error."
            )
            self.db.set_error(
                experiment.id,
                f"Worker {self.worker_id} finished without creating required files: {missing}"
            )

    def _run_analyze(self, experiment: Experiment) -> None:
        """Run the analyze task."""
        try:
            extra_context = self._build_experiment_context(experiment)

            def prompt_builder(
                workspace: str | None,
                learnings: str | None,
                config: Any | None = None,
            ) -> str:
                return build_step_prompt(
                    "phase3_worker_analyze",
                    workspace,
                    learnings,
                    config,
                    extra_context,
                    adapter=self.adapter,
                )

            tools = get_tool_schemas(
                [
                    "read_file", "grep_file", "shell_exec", "view_image",
                    "read_board", "update_experiment", "report_to_user",
                    "memory_store", "memory_search", "memory_read",
                    "note_to_conductor", "ack_directive",
                ],
            )

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
                event_callback=self._tracked_event_callback,
                reasoning_effort=self.config.reasoning_effort,
                config=self.config,
                tools=tools,
                prompt_builder=prompt_builder,
                log_name=f"worker_{self.worker_id}_analyze_{experiment.name}",
                min_report_attempts=1,
                db=self.db,
                metrics=self.metrics,
                adapter=self.adapter,
                mlflow_run_target=experiment.mlflow_run_uuid,
            )

            self._agent = agent
            agent.run(
                f"Analyze experiment #{experiment.id} '{experiment.name}'. "
                f"Read the experiment details above. Go."
            )
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Worker {self.worker_id} analyze error: {e}\n{tb}")
            self.db.set_error(experiment.id, f"{e}\n{tb}")
        finally:
            self._agent = None
            self._current_experiment = None
            # Auto-transition: if LLM didn't update status, do it
            try:
                self._ensure_analyze_status(experiment)
            except Exception as e:
                logger.error(f"_ensure_analyze_status failed: {e}")
            try:
                self.db.release_worker(experiment.id)
            except Exception as e:
                logger.error(f"release_worker failed: {e}")

    def _ensure_analyze_status(self, experiment: Experiment) -> None:
        """Ensure a completed experiment reaches the terminal 'done' state,
        even if the analyze agent didn't advance it via update_experiment.

        The kanban terminal is 'done' (… → finished → analyzed → done). This
        system has no separate handoff step, so once analysis is complete the
        experiment is finalized straight to 'done'. Previously the worker
        stopped at 'analyzed' and nothing ever advanced past it, leaving the
        'done' column permanently empty."""
        current = self.db.get(experiment.id)
        if current is None:
            return

        # Only finalize experiments sitting at the tail of the pipeline.
        if current.status not in ("finished", "analyzed"):
            return

        # If analysis crashed (error set by the except block) without producing
        # results, leave it at 'finished' for retry rather than finalizing.
        if current.status == "finished" and current.error and not current.results_json:
            logger.warning(
                f"Experiment #{experiment.id} {experiment.name}: analysis failed with error, "
                f"keeping at 'finished' for retry"
            )
            return

        debrief_path = (
            Path(self.workspace) / "experiments" / experiment.name / "debrief.md"
        )
        if not debrief_path.is_file():
            logger.warning(
                f"Experiment #{experiment.id} {experiment.name}: analysis produced no debrief; "
                f"keeping at '{current.status}' for retry"
            )
            return

        # finished → analyzed (auto-transition if the analyze agent didn't).
        if current.status == "finished":
            logger.warning(
                f"Experiment #{experiment.id} {experiment.name}: auto-transitioning "
                f"finished → analyzed (debrief exists, LLM didn't update status)"
            )
            self.db.update_status(experiment.id, "analyzed")

        # analyzed → done: finalize to the terminal kanban state.
        self.db.update_status(experiment.id, "done")

    def _run_fix(self, experiment: Experiment) -> None:
        """Run the fixer task to diagnose and fix a failed experiment."""
        try:
            extra_context = self._build_experiment_context(experiment)
            extra_context += f"\n**Fix attempts so far:** {experiment.fix_attempts}"

            def prompt_builder(
                workspace: str | None,
                learnings: str | None,
                config: Any | None = None,
            ) -> str:
                return build_step_prompt(
                    "phase3_fixer",
                    workspace,
                    learnings,
                    config,
                    extra_context,
                    adapter=self.adapter,
                )

            tools = get_tool_schemas(
                [
                    "shell_exec", "read_file", "grep_file", "view_image",
                    "update_experiment", "report_to_user",
                    "memory_store", "memory_search", "memory_read",
                    "note_to_conductor", "ack_directive",
                ],
            )

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
                event_callback=self._tracked_event_callback,
                reasoning_effort=self.config.reasoning_effort,
                config=self.config,
                tools=tools,
                prompt_builder=prompt_builder,
                log_name=f"worker_{self.worker_id}_fix_{experiment.name}",
                min_report_attempts=1,
                db=self.db,
                metrics=self.metrics,
                adapter=self.adapter,
                mlflow_run_target=experiment.mlflow_run_uuid,
            )

            self._agent = agent
            agent.run(
                f"Fix failed experiment #{experiment.id} '{experiment.name}'. "
                f"Error: {experiment.error}. Read the logs and fix it."
            )
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Worker {self.worker_id} fix error: {e}\n{tb}")
            self.db.set_error(experiment.id, f"Fixer failed: {e}")
        finally:
            self._agent = None
            self._current_experiment = None
            try:
                self.db.release_worker(experiment.id)
            except Exception as e:
                logger.error(f"release_worker failed: {e}")

    def _build_report_context(self, milestone_number: int, done_count: int) -> str:
        """Build context for the reporter with board state and top experiments."""
        parts: list[str] = []

        # Use adapter metric if available
        _metric = "sharpe"
        _metric_display = "Sharpe"
        _direction = "maximize"
        if self.adapter is not None:
            _metric = self.adapter.metric.primary_metric
            _metric_display = self.adapter.metric.display_name
            _direction = self.adapter.metric.direction

        parts.append(f"## Milestone #{milestone_number}")
        parts.append(f"**Total experiments completed:** {done_count}")
        parts.append(f"**Report directory:** reports/milestone_{milestone_number:03d}/")

        # Conductor directives + annotations — auto-injected so the
        # reporter doesn't have to spend tool calls reading the files
        # itself, and so one-shot directives already acked by another
        # role are filtered out. Mirrors strategist/worker behavior.
        try:
            from alpha_lab import conductor_tools as _ct
            active = _ct.directives_for_role(self.workspace, "reporter")
            acks = _ct.read_directive_acks(self.workspace)
            if active or acks:
                rendered = _ct.render_directives_for_prompt(active, acks=acks)
                parts.append("\n" + rendered[:4000])
            details = _ct.read_annotation_details(self.workspace)
            if details:
                parts.append(
                    "\n## Leaderboard annotations (Conductor-applied labels)"
                )
                for eid, rec in details.items():
                    label = rec.get("label", "")
                    reason = (rec.get("reason") or "").strip()
                    if reason:
                        parts.append(f"- #{eid} [{label}] — {reason[:240]}")
                    else:
                        parts.append(f"- #{eid} [{label}]")
        except Exception as e:
            logger.debug("Skipping Conductor context for reporter: %s", e)

        # Board summary
        summary = self.db.board_summary()
        parts.append("\n## Board Summary")
        for col, cnt in sorted(summary.items()):
            parts.append(f"  {col}: {cnt}")

        # Leaderboard
        leaders = self.db.leaderboard(_metric, 15, direction=_direction)
        if leaders:
            parts.append(f"\n## Leaderboard (top 15 by {_metric_display})")
            for i, exp in enumerate(leaders, 1):
                try:
                    m = json.loads(exp.results_json or "{}")
                    if isinstance(m, dict) and m:
                        metrics_str = ", ".join(f"{k}={v}" for k, v in m.items())
                    else:
                        metrics_str = "no metrics"
                except (json.JSONDecodeError, TypeError):
                    metrics_str = "no metrics"
                parts.append(f"  {i}. #{exp.id} {exp.name} [{metrics_str}]")

        # All done experiments for the report
        done = self.db.list_by_status("done", "analyzed")
        parts.append(f"\n## All completed experiments ({len(done)})")
        for exp in done:
            try:
                m = json.loads(exp.results_json or "{}")
                if isinstance(m, dict) and m:
                    metrics_str = ", ".join(f"{k}={v}" for k, v in list(m.items())[:5])
                else:
                    metrics_str = "no metrics"
            except (json.JSONDecodeError, TypeError):
                metrics_str = "no metrics"
            err = f" ERROR: {exp.error}" if exp.error else ""
            debrief = f"experiments/{exp.name}/debrief.md"
            if not (Path(self.workspace) / debrief).is_file():
                debrief = "none"
            parts.append(
                f"  #{exp.id} {exp.name} [{metrics_str}]{err}; debrief: {debrief}"
            )

        return "\n".join(parts)

    def _run_report(self, milestone_number: int, done_count: int) -> None:
        """Run the milestone report generation task."""
        try:
            extra_context = self._build_report_context(milestone_number, done_count)

            def prompt_builder(
                workspace: str | None,
                learnings: str | None,
                config: Any | None = None,
            ) -> str:
                return build_step_prompt(
                    "phase3_reporter",
                    workspace,
                    learnings,
                    config,
                    extra_context,
                    adapter=self.adapter,
                )

            tools = get_tool_schemas(
                [
                    "shell_exec", "read_file", "grep_file", "view_image",
                    "read_board", "report_to_user",
                    "memory_store", "memory_search", "memory_read",
                    "note_to_conductor", "ack_directive",
                ],
            )

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
                event_callback=self._tracked_event_callback,
                reasoning_effort=self.config.reasoning_effort,
                config=self.config,
                tools=tools,
                prompt_builder=prompt_builder,
                log_name=f"reporter_milestone_{milestone_number:03d}",
                min_report_attempts=1,
                db=self.db,
                metrics=self.metrics,
                adapter=self.adapter,
            )

            self._agent = agent
            agent.run(
                f"Generate milestone report #{milestone_number}. "
                f"{done_count} experiments completed so far. "
                f"Write to reports/milestone_{milestone_number:03d}/. Go."
            )
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Worker {self.worker_id} report error: {e}\n{tb}")
        finally:
            self._agent = None


def _ensure_mlflow_run(db: ExperimentDB, experiment: Experiment) -> Experiment:
    """Create the MLflow sub-run lazily when implement first picks up the experiment. Idempotent."""
    if experiment.mlflow_run_uuid:
        return experiment
    from alpha_lab import mlflow_logger
    info = mlflow_logger.create_experiment_run(
        experiment.name,
        params={
            "description": experiment.description,
            "hypothesis": experiment.hypothesis,
            "config": experiment.config_json,
            "experiment_id": experiment.id,
        },
    )
    if info is None:
        return experiment
    run_uuid, artifact_uri = info
    try:
        db.set_mlflow_run(experiment.id, run_uuid, artifact_uri)
    except Exception:
        # Persistence failed; the run exists in MLflow but we lost the
        # back-reference. Not worth crashing the worker over.
        return experiment
    refreshed = db.get(experiment.id)
    return refreshed if refreshed is not None else experiment
