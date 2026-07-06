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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alpha_lab.adapter import DomainAdapter

from alpha_lab.agents import load_agent
from alpha_lab.sandboxing import sandbox
from alpha_lab.config import TaskConfig
from alpha_lab.events import AgentEvent
from alpha_lab.experiment_db import (
    Experiment,
    ExperimentDB,
    is_smoke_result,
)
from alpha_lab.provider import Provider
from alpha_lab.tracing import attach_context, detach_context

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
        adapter: DomainAdapter,
        metrics: Any | None = None,
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
        self._run_handle: sandbox.AgentRunHandle | None = None
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

    def implement(self, experiment: Experiment, otel_context: Any = None) -> None:
        """Start implementing an experiment in a background thread."""
        if self.busy:
            raise RuntimeError(f"Worker {self.worker_id} is busy")
        self._current_experiment = experiment
        self.last_event_at = time.time()
        self.db.assign_worker(experiment.id, self.worker_id)
        self._thread = threading.Thread(
            target=self._run_implement,
            args=(experiment, otel_context),
            daemon=True,
        )
        self._thread.start()

    def analyze(self, experiment: Experiment, otel_context: Any = None) -> None:
        """Start analyzing an experiment in a background thread."""
        if self.busy:
            raise RuntimeError(f"Worker {self.worker_id} is busy")
        self._current_experiment = experiment
        self.last_event_at = time.time()
        self.db.assign_worker(experiment.id, self.worker_id)
        self._thread = threading.Thread(
            target=self._run_analyze,
            args=(experiment, otel_context),
            daemon=True,
        )
        self._thread.start()

    def handoff(self, experiment: Experiment, otel_context: Any = None) -> None:
        """Start the user-proxy handoff task in a background thread."""
        if self.busy:
            raise RuntimeError(f"Worker {self.worker_id} is busy")
        self._current_experiment = experiment
        self.last_event_at = time.time()
        self.db.assign_worker(experiment.id, self.worker_id)
        self._thread = threading.Thread(
            target=self._run_handoff,
            args=(experiment, otel_context),
            daemon=True,
        )
        self._thread.start()

    def generate_report(self, milestone_number: int, done_count: int, otel_context: Any = None) -> None:
        """Start generating a milestone report in a background thread."""
        if self.busy:
            raise RuntimeError(f"Worker {self.worker_id} is busy")
        self.last_event_at = time.time()
        self._thread = threading.Thread(
            target=self._run_report,
            args=(milestone_number, done_count, otel_context),
            daemon=True,
        )
        self._thread.start()

    def fix(self, experiment: Experiment, otel_context: Any = None) -> None:
        """Start fixing a failed experiment in a background thread."""
        if self.busy:
            raise RuntimeError(f"Worker {self.worker_id} is busy")
        self._current_experiment = experiment
        self.last_event_at = time.time()
        self.db.assign_worker(experiment.id, self.worker_id)
        self._thread = threading.Thread(
            target=self._run_fix,
            args=(experiment, otel_context),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_requested = True
        if self._run_handle is not None:
            self._run_handle.stop()

    def _run_agent(
        self,
        agent_id: str,
        *,
        initial_message: str,
        extra_context: str,
        log_name: str,
        mlflow_run_target: str | None = None,
    ) -> None:
        """Run one worker-role agent (sandboxed when bwrap is available)."""
        sandbox.run_agent(
            agent_id,
            self._tracked_event_callback,
            initial_message=initial_message,
            extra_context=extra_context,
            log_name=log_name,
            db_path=str(Path(self.workspace) / "experiments.db"),
            mlflow_run_target=mlflow_run_target,
            provider=self.provider,
            db=self.db,
            adapter=self.adapter,
            metrics=self.metrics,
            owner=self,
        )

    def _build_experiment_context(self, experiment: Experiment) -> str:
        """Build extra context describing the experiment."""
        lines = [
            f"## Experiment #{experiment.id}: {experiment.name}",
            f"**Description:** {experiment.description}",
            f"**Hypothesis:** {experiment.hypothesis}",
            f"**Config:** {experiment.config_json}",
            f"**Status:** {experiment.status}",
        ]
        if experiment.error:
            lines.append(f"**Error:** {experiment.error}")
        return "\n".join(lines)

    def _run_implement(self, experiment: Experiment, otel_context: Any = None) -> None:
        """Run the implement task."""
        token = attach_context(otel_context) if otel_context else None
        try:
            experiment = _ensure_mlflow_run(self.db, experiment)
            agent_definition = load_agent("phase3/worker_implement")
            self._run_agent(
                "phase3/worker_implement",
                initial_message=(
                    f"Implement experiment #{experiment.id} '{experiment.name}'. "
                    f"Read the experiment details above. Go."
                ),
                extra_context=self._build_experiment_context(experiment),
                log_name=agent_definition.log_name.format(
                    worker_id=self.worker_id, experiment=experiment
                ),
                mlflow_run_target=experiment.mlflow_run_uuid,
            )
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Worker {self.worker_id} implement error: {e}\n{tb}")
            self.db.set_error(experiment.id, f"{e}\n{tb}")
        finally:
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
            if token is not None:
                detach_context(token)

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
        required_files: list[str] = []
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

    def _run_analyze(self, experiment: Experiment, otel_context: Any = None) -> None:
        """Run the analyze task."""
        token = attach_context(otel_context) if otel_context else None
        try:
            agent_definition = load_agent("phase3/worker_analyze")
            self._run_agent(
                "phase3/worker_analyze",
                initial_message=(
                    f"Analyze experiment #{experiment.id} '{experiment.name}'. "
                    f"Read the experiment details above. Go."
                ),
                extra_context=self._build_experiment_context(experiment),
                log_name=agent_definition.log_name.format(
                    worker_id=self.worker_id, experiment=experiment
                ),
                mlflow_run_target=experiment.mlflow_run_uuid,
            )
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Worker {self.worker_id} analyze error: {e}\n{tb}")
            self.db.set_error(experiment.id, f"{e}\n{tb}")
        finally:
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
            if token is not None:
                detach_context(token)

    def _ensure_analyze_status(self, experiment: Experiment) -> None:
        """Ensure experiment progresses after analyze, even if LLM didn't call update_experiment."""
        current = self.db.get(experiment.id)
        if current is None:
            return

        # If already past finished, LLM did its job
        if current.status != "finished":
            return

        # If analysis crashed (error set by except block), don't promote to analyzed
        if current.error and not current.results_json:
            logger.warning(
                f"Experiment #{experiment.id} {experiment.name}: analysis failed with error, "
                f"keeping at 'finished' for retry"
            )
            return

        exp_dir = Path(self.workspace) / "experiments" / experiment.name
        has_debrief = (exp_dir / "debrief.md").exists()

        if has_debrief:
            logger.warning(
                f"Experiment #{experiment.id} {experiment.name}: auto-transitioning "
                f"finished → analyzed (debrief exists but LLM didn't update status)"
            )
            try:
                from alpha_lab.memory import remember_workspace_file
                remember_workspace_file(
                    self.workspace,
                    f"experiments/{experiment.name}/debrief.md",
                    tags=["phase3", "debrief", experiment.name],
                    summary=f"Experiment {experiment.name} debrief",
                    kind="result",
                    phase="phase3",
                    agent="worker",
                )
            except Exception as e:
                logger.warning("Failed to ingest debrief into memory: %s", e, exc_info=True)
            self.db.update_status(experiment.id, "analyzed")
        else:
            logger.warning(
                f"Experiment #{experiment.id} {experiment.name}: auto-transitioning "
                f"finished → analyzed (no debrief but worker finished)"
            )
            self.db.update_status(experiment.id, "analyzed")

    def _run_handoff(self, experiment: Experiment, otel_context: Any = None) -> None:
        """Run the user-proxy handoff task."""
        token = attach_context(otel_context) if otel_context else None
        try:
            agent_definition = load_agent("proxy/handoff")
            self._run_agent(
                "proxy/handoff",
                initial_message=(
                    f"Hand off experiment #{experiment.id} '{experiment.name}'. "
                    f"Read the experiment details above. Go."
                ),
                extra_context=self._build_experiment_context(experiment),
                log_name=agent_definition.log_name.format(
                    worker_id=self.worker_id, experiment=experiment.name
                ),
            )
            self.db.update_status(experiment.id, "done")
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Worker {self.worker_id} handoff error: {e}\n{tb}")
            self.db.set_error(experiment.id, f"{e}\n{tb}")
        finally:
            self._current_experiment = None
            try:
                self.db.release_worker(experiment.id)
            except Exception as e:
                logger.error(f"release_worker failed: {e}")
            if token is not None:
                detach_context(token)

    def _run_fix(self, experiment: Experiment, otel_context: Any = None) -> None:
        """Run the fixer task to diagnose and fix a failed experiment."""
        token = attach_context(otel_context) if otel_context else None
        try:
            extra_context = self._build_experiment_context(experiment)
            extra_context += f"\n**Fix attempts so far:** {experiment.fix_attempts}"
            agent_definition = load_agent("phase3/worker_fixer")
            self._run_agent(
                "phase3/worker_fixer",
                initial_message=(
                    f"Fix failed experiment #{experiment.id} '{experiment.name}'. "
                    f"Error: {experiment.error}. Read the logs and fix it."
                ),
                extra_context=extra_context,
                log_name=agent_definition.log_name.format(
                    worker_id=self.worker_id, experiment=experiment
                ),
                mlflow_run_target=experiment.mlflow_run_uuid,
            )
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Worker {self.worker_id} fix error: {e}\n{tb}")
            self.db.set_error(experiment.id, f"Fixer failed: {e}")
        finally:
            self._current_experiment = None
            try:
                self.db.release_worker(experiment.id)
            except Exception as e:
                logger.error(f"release_worker failed: {e}")
            if token is not None:
                detach_context(token)

    def _build_report_context(self, milestone_number: int, done_count: int) -> str:
        """Build context for the reporter with board state and top experiments."""
        parts: list[str] = []

        _metric = self.adapter.metric.primary_metric
        _metric_display = self.adapter.metric.display_name
        _direction = self.adapter.metric.direction

        parts.append(f"## Milestone #{milestone_number}")
        parts.append(f"**Report directory:** reports/milestone_{milestone_number:03d}/")

        # Board summary
        summary = self.db.board_summary()
        parts.append("\n## Board Summary")
        for col, cnt in sorted(summary.items()):
            parts.append(f"  {col}: {cnt}")

        # Leaderboard
        leaders = self.db.leaderboard(_metric, 15, _direction)
        if leaders:
            parts.append(f"\n## Leaderboard (top 15 by {_metric_display})")
            for i, exp in enumerate(leaders, 1):
                try:
                    m = json.loads(exp.results_json or "{}")
                    metrics_str = ", ".join(f"{k}={v}" for k, v in m.items()) if isinstance(m, dict) else "no metrics"
                except (json.JSONDecodeError, TypeError):
                    metrics_str = "no metrics"
                parts.append(f"  {i}. #{exp.id} {exp.name} [{metrics_str}]")

        done = self.db.list_by_status("done", "analyzed")
        done = [e for e in done if not is_smoke_result(e.results_json)]
        parts.insert(1, f"**Total experiments completed:** {len(done)}")
        parts.append(f"\n## All completed experiments ({len(done)})")
        for exp in done:
            try:
                m = json.loads(exp.results_json or "{}")
                metrics_str = ", ".join(f"{k}={v}" for k, v in list(m.items())[:5]) if isinstance(m, dict) else "no metrics"
            except (json.JSONDecodeError, TypeError):
                metrics_str = "no metrics"
            err = f" ERROR: {exp.error}" if exp.error else ""
            parts.append(f"  #{exp.id} {exp.name} [{metrics_str}]{err}")

        return "\n".join(parts)

    def _run_report(self, milestone_number: int, done_count: int, otel_context: Any = None) -> None:
        """Run the milestone report generation task."""
        token = attach_context(otel_context) if otel_context else None
        try:
            agent_definition = load_agent("phase3/reporter")
            self._run_agent(
                "phase3/reporter",
                initial_message=(
                    f"Generate milestone report #{milestone_number}. "
                    f"{done_count} experiments completed so far. "
                    f"Write to reports/milestone_{milestone_number:03d}/. Go."
                ),
                extra_context=self._build_report_context(milestone_number, done_count),
                log_name=agent_definition.log_name.format(
                    milestone_number=milestone_number
                ),
            )
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Worker {self.worker_id} report error: {e}\n{tb}")
        finally:
            if token is not None:
                detach_context(token)


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