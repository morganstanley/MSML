"""Dispatcher — the main orchestration loop for Phase 3.

Pure Python (no LLM). Manages strategist turns, worker assignments,
job submission/polling, and kanban state transitions.

Supports executors:
- SlurmManager: submits jobs via sbatch
- LocalGPUManager: spawns subprocesses directly on local GPUs
- LocalCPUManager: runs tree-based models on CPU in parallel with GPU jobs
"""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path

from opentelemetry import trace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alpha_lab.adapter import DomainAdapter

from alpha_lab.adapter_loader import resolve_adapter
from alpha_lab.events import (
    AgentEvent,
    BoardSummaryEvent,
    ExperimentEvent,
    PhaseEvent,
)
from alpha_lab.experiment_db import Experiment, ExperimentDB
from alpha_lab.provider import Provider
from alpha_lab import deps, utils
from alpha_lab.strategist import StallError, Strategist
from alpha_lab.tracing import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SYSTEM,
    GenAiOperationNameValues,
    attach_context,
    copy_context,
    detach_context,
    get_tracer,
    provider_to_system,
)
from alpha_lab.worker import Worker

logger = logging.getLogger("alpha_lab.dispatcher")

POLL_INTERVAL = 10  # seconds


class Dispatcher:
    """Main orchestration loop for Phase 3 experiment system."""

    def __init__(
        self,
        provider: Provider,
        workspace: str,
        db: ExperimentDB,
        event_callback: Callable[[AgentEvent], None],
        adapter: DomainAdapter,
        metrics: object | None = None,
        supervisor: object | None = None,
    ) -> None:
        # Config + executors come from the active run deps (published by run.py around the
        # dispatcher's lifecycle); they're read at point of use, not stored on self.
        config = deps.get().config
        self.provider = provider
        self.workspace = workspace
        self.db = db
        self.event_callback = event_callback
        self.metrics = metrics
        self.adapter = adapter
        self.supervisor = supervisor
        self._stop_requested = False
        self._crashed = False
        # Track which jobs are CPU vs GPU for polling
        self._cpu_job_ids: set[str] = set()
        # Set once when KILL_SIGNAL is first observed and queued/running
        # work has been cancelled; prevents repeated cancellation.
        self._kill_signal_actioned: bool = False

        # OTel tracer — resolved once (same pattern as AgentLoop)
        self._tracer = get_tracer()
        self._gen_ai_system = provider_to_system(config.provider)

        p3 = config.pipeline.phase3

        # Create workers
        self.workers = [
            Worker(
                worker_id=f"worker_{i}",
                provider=provider,
                config=config,
                workspace=workspace,
                db=db,
                event_callback=event_callback,
                metrics=metrics,
                adapter=adapter,
            )
            for i in range(p3.worker_count)
        ]

        # Create strategist
        self.strategist = Strategist(
            provider=provider,
            workspace=workspace,
            db=db,
            event_callback=event_callback,
            adapter=adapter,
        )

        # Strategist scheduling (protected by _state_lock)
        self._state_lock = threading.Lock()
        self._strategist_interval = p3.strategist_interval
        self._last_strategist_time = 0.0
        self._analyzed_since_strategist = 0
        self._last_analyzed_count = 0
        self._strategist_running = False
        self._strategist_thread: threading.Thread | None = None

        # Report scheduling
        self._report_interval = p3.report_interval
        self._last_report_at_done_count = 0
        self._report_number = 0
        self._current_report_number = 0
        self._report_in_progress = False
        self._report_worker: Worker | None = None

        # Limits
        self._max_experiments = p3.max_experiments

        # Convergence tracking
        self._convergence_threshold = p3.convergence_threshold
        if p3.convergence_metric:
            self._convergence_metric = p3.convergence_metric
        else:
            self._convergence_metric = adapter.metric.primary_metric
        self._metric_direction = adapter.metric.direction
        if self._metric_direction == "minimize":
            self._best_metric_value: float = float("inf")
        else:
            self._best_metric_value: float = float("-inf")
        self._experiments_since_improvement: int = 0

        # Dispatcher JSONL log
        self._log_file = None

    def _init_log(self) -> None:
        """Open the dispatcher JSONL log."""
        log_dir = Path(self.workspace) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = open(log_dir / "dispatcher.jsonl", "a")

    def _log(self, action: str, **details: object) -> None:
        """Write a structured log entry for dispatcher decisions."""
        if self._log_file is not None:
            entry = {"t": time.time(), "action": action, **details}
            try:
                self._log_file.write(json.dumps(entry, default=str) + "\n")
                self._log_file.flush()
            except (OSError, TypeError, ValueError) as e:
                logger.warning("Failed to write dispatcher log entry: %s", e)

    def emit(self, event: AgentEvent) -> None:
        self.event_callback(event)
        # Also log to dispatcher JSONL
        if self._log_file is not None:
            try:
                self._log_file.write(json.dumps(event.to_dict(), default=str) + "\n")
                self._log_file.flush()
            except (OSError, TypeError, ValueError) as e:
                logger.warning("Failed to write event to dispatcher log: %s", e)

    @property
    def crashed(self) -> bool:
        """True if run() exited via an unhandled exception.

        Callers (CLI, supervisor, tests) can read this after run() returns
        instead of wrapping the call in a try/except; the dispatcher does
        not re-raise so that the CLI driver doesn't abort the finally
        cleanup block in run.py.
        """
        return self._crashed

    def stop(self, join_timeout: float = 30) -> None:
        """Stop the dispatcher and all workers, waiting for threads to finish."""
        self._stop_requested = True
        self.strategist.stop()
        for w in self.workers:
            w.stop()

        # Join worker threads
        for w in self.workers:
            if w._thread is not None and w._thread.is_alive():
                w._thread.join(timeout=join_timeout)

        # Join strategist thread
        if self._strategist_thread is not None and self._strategist_thread.is_alive():
            self._strategist_thread.join(timeout=join_timeout)

        # Executor teardown (killing orphaned jobs) is owned by RunDeps.close(), which
        # fires when run.py exits the `with RunDeps(...)` block around this lifecycle.
        self._cleanup()

    def _cleanup(self) -> None:
        """Release all remaining worker assignments and close log file."""
        for status in ("to_implement", "implemented", "finished", "analyzed"):
            try:
                assigned = self.db.list_by_status(status)
                for exp in assigned:
                    if exp.worker_id is not None:
                        self.db.release_worker(exp.id)
            except Exception as e:
                logger.warning("Failed to release workers for status '%s' during cleanup: %s", status, e)

        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError as e:
                logger.warning("Failed to close dispatcher log file: %s", e)
            self._log_file = None

    def _scan_existing_milestones(self) -> int:
        """Return the highest N found among workspace/reports/milestone_NNN dirs, or 0."""
        reports_dir = Path(self.workspace) / "reports"
        # is_dir() covers "doesn't exist" and "exists but is a file" in one check; the
        # try/except guards against permission errors or a TOCTOU race where the dir
        # disappears between the is_dir check and iterdir.
        if not reports_dir.is_dir():
            return 0
        max_n = 0
        try:
            for child in reports_dir.iterdir():
                if not child.is_dir() or not child.name.startswith("milestone_"):
                    continue
                suffix = child.name[len("milestone_"):]
                if suffix.isdigit():
                    n = int(suffix)
                    if n > max_n:
                        max_n = n
        except OSError as e:
            logger.warning("Failed to scan milestone reports dir '%s': %s", reports_dir, e)
            return 0
        return max_n

    def recover(self) -> dict:
        """Recover from a crash: release orphaned workers and reconcile SLURM jobs.

        Returns a summary dict of recovery actions taken.
        """
        summary: dict = {"released_workers": 0, "slurm_reconciled": 0, "report_counter": 0}

        # 1. Release orphaned worker assignments (no workers exist yet at startup)
        for status in ("to_implement", "implemented", "finished", "analyzed"):
            assigned = self.db.list_by_status(status)
            for exp in assigned:
                if exp.worker_id is not None:
                    logger.info(
                        f"Recovery: releasing orphaned worker {exp.worker_id} "
                        f"from experiment #{exp.id} {exp.name}"
                    )
                    self.db.release_worker(exp.id)
                    summary["released_workers"] += 1

        # 2. Reconcile SLURM jobs
        slurm_exps = self.db.list_by_status("queued", "running")
        job_ids = [exp.slurm_job_id for exp in slurm_exps if exp.slurm_job_id]
        if job_ids:
            statuses = deps.get().gpu_executor.poll_jobs(job_ids)
            for exp in slurm_exps:
                if not exp.slurm_job_id or exp.slurm_job_id not in statuses:
                    continue
                slurm_status = statuses[exp.slurm_job_id]

                if slurm_status == "RUNNING" and exp.status == "queued":
                    self.db.update_status(exp.id, "running", started_at=time.time())
                    logger.info(f"Recovery: #{exp.id} queued -> running (SLURM RUNNING)")
                    summary["slurm_reconciled"] += 1

                elif slurm_status == "COMPLETED":
                    self.db.update_status(exp.id, "finished", finished_at=time.time())
                    logger.info(f"Recovery: #{exp.id} {exp.status} -> finished (SLURM COMPLETED)")
                    summary["slurm_reconciled"] += 1

                elif slurm_status in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"):
                    self.db.set_error_and_finish(exp.id, f"SLURM {slurm_status}")
                    logger.info(f"Recovery: #{exp.id} {exp.status} -> finished (SLURM {slurm_status})")
                    summary["slurm_reconciled"] += 1

                elif slurm_status == "PENDING":
                    pass  # Leave as queued

                elif slurm_status == "UNKNOWN":
                    self.db.set_error_and_finish(exp.id, "SLURM job lost")
                    logger.info(f"Recovery: #{exp.id} {exp.status} -> finished (SLURM job lost)")
                    summary["slurm_reconciled"] += 1

        # 3. Recover milestone-report counter from filesystem so a restart doesn't overwrite
        #    milestone_001. Also align _last_report_at_done_count to current done_count so we
        #    don't immediately fire a redundant report at resume.
        max_milestone = self._scan_existing_milestones()
        if max_milestone > 0:
            self._report_number = max_milestone
            self._current_report_number = max_milestone
            self._last_report_at_done_count = len(self.db.list_by_status("done", "analyzed"))
            summary["report_counter"] = max_milestone
            logger.info(
                f"Recovery: milestone counter restored to {max_milestone} "
                f"(next report will be milestone_{max_milestone + 1:03d})"
            )

        self._log("recovery", **summary)
        logger.info(f"Recovery complete: {summary}")
        return summary

    def run(self) -> None:
        """Main loop — blocks until stopped or max_experiments reached."""
        logger.info("Dispatcher starting")
        self._init_log()
        self._log("dispatcher_start", max_experiments=self._max_experiments,
                  worker_count=len(self.workers), report_interval=self._report_interval)
        self.emit(PhaseEvent(
            phase="phase3",
            step="dispatcher",
            status="starting",
            detail="Phase 3 experiment loop starting",
        ))

        # Ensure experiments directory exists
        Path(self.workspace, "experiments").mkdir(parents=True, exist_ok=True)

        with self._tracer.start_as_current_span(
            "phase3.dispatcher",
            kind=trace.SpanKind.INTERNAL,
            attributes={
                # alpha_lab.* prefix: custom namespace to avoid collision with
                # future GenAI semconv additions (Decision #6).
                "alpha_lab.dispatcher.max_experiments": self._max_experiments,
                "alpha_lab.dispatcher.worker_count": len(self.workers),
                "alpha_lab.dispatcher.convergence_metric": self._convergence_metric,
                "alpha_lab.dispatcher.metric_direction": self._metric_direction,
            },
        ) as dispatcher_span:
            # Crash recovery before entering main loop
            try:
                self.recover()
            except Exception as e:
                logger.error(f"Recovery failed: {e}")
                self._log("recovery_error", error=str(e))

            try:
                while not self._stop_requested:
                    # 0. KILL_SIGNAL drain: cancel queued/running once, then
                    #    only analyze remaining finished work to termination.
                    if self._kill_signal_active():
                        self._kill_signal_drain_in_flight()

                    # 1. Strategist turn (if due)
                    if self._should_run_strategist():
                        self._run_strategist()

                    # 2. Poll SLURM jobs
                    self._poll_slurm()

                    # 3. Submit checked experiments to SLURM
                    self._submit_checked()

                    # 4. Milestone report (if due) — checked BEFORE assigning
                    #    workers so a report can claim an idle slot.
                    self._maybe_generate_report()

                    # 5. Assign idle workers
                    self._assign_workers()

                    # 6. Track newly analyzed experiments (for strategist trigger)
                    self._track_analyzed()

                    # 7. Detect stale workers
                    self._check_stale()

                    # 7b. Detect stuck workers (observability)
                    self._check_stuck_workers()

                    # 7c. Supervisor health check (if error rate high)
                    self._maybe_supervisor_check()

                    # 8. Emit board summary
                    self._emit_board_summary()

                    # 9. Check termination
                    if self._should_terminate():
                        logger.info("Max experiments reached, stopping")
                        break

                    # Sleep (interruptible)
                    for _ in range(POLL_INTERVAL):
                        if self._stop_requested:
                            break
                        time.sleep(1)

            except Exception as e:
                # Swallow the exception so the CLI caller (run.py) doesn't need a
                # wrapping try/except; the crash is still visible via the error
                # PhaseEvent emitted in the finally block and via `self.crashed`,
                # which tests and supervisors can inspect after run() returns.
                dispatcher_span.set_status(trace.StatusCode.ERROR)
                dispatcher_span.record_exception(e)
                logger.error(f"Dispatcher error: {e}")
                self._log("dispatcher_error", error=str(e), traceback=traceback.format_exc())
                self._crashed = True
            finally:
                self._log("dispatcher_stop")
                if getattr(self, "_crashed", False):
                    status = "error"
                    detail = "Phase 3 dispatcher crashed unexpectedly"
                else:
                    status = "completed"
                    detail = "Phase 3 experiment loop finished"
                self.emit(PhaseEvent(
                    phase="phase3",
                    step="dispatcher",
                    status=status,
                    detail=detail,
                ))
                if self._log_file is not None:
                    try:
                        self._log_file.close()
                    except OSError as e:
                        logger.warning("Failed to close dispatcher log on shutdown: %s", e)
                logger.info("Dispatcher stopped")

    def _kill_signal_active(self) -> bool:
        """True when ``{workspace}/KILL_SIGNAL`` exists.

        Existence-only sentinel written by benchmark objectives when their
        evaluation budget is exhausted. Strict drain: queued and running
        experiments are cancelled in-place (no further work performed);
        only already-finished experiments may still be analyzed.
        """
        return (Path(self.workspace) / "KILL_SIGNAL").exists()

    def _kill_signal_drain_in_flight(self) -> None:
        """Cancel queued/running jobs once when KILL_SIGNAL is first observed.

        Idempotent; ``_kill_signal_actioned`` gates re-entry. Sets each
        affected experiment to ``cancelled`` so the subsequent poll path
        cannot transition it to ``finished`` (which would otherwise make
        it eligible for analyze).
        """
        if self._kill_signal_actioned:
            return
        d = deps.get()
        for status in ("queued", "running"):
            for exp in self.db.list_by_status(status):
                job_id = exp.slurm_job_id
                if job_id:
                    executor = (
                        d.cpu_executor
                        if d.cpu_executor is not None and job_id in self._cpu_job_ids
                        else d.gpu_executor
                    )
                    try:
                        executor.cancel(job_id)
                    except Exception as e:
                        logger.warning("KILL_SIGNAL: cancel %s failed: %s", job_id, e)
                self.db.update_status(exp.id, "cancelled")
                logger.info("KILL_SIGNAL: cancelled experiment #%d (%s)", exp.id, exp.name)
        self._kill_signal_actioned = True

    def _should_run_strategist(self) -> bool:
        """Determine if it's time for a strategist turn."""
        if self._kill_signal_active():
            return False

        # JIT: capacity-driven trigger — fire when a slot is free or the interval
        # elapsed (the old count/idle conditions are subsumed). Flag off leaves the
        # original trigger below untouched, so existing behavior/tests are unchanged.
        if deps.get().config.pipeline.phase3.jit:
            with self._state_lock:
                if self._strategist_running:
                    return False
            # Never wake the strategist when every worker is busy — there'd be no one to
            # implement a new proposal, so it would only design stale work. (The strategist
            # is not itself a worker.)
            if utils.worker_states(self.db)["free"] <= 0:
                return False
            with self._state_lock:
                if self._last_strategist_time == 0:
                    return True
                if time.time() - self._last_strategist_time >= self._strategist_interval:
                    return True
            return any(s["free"] > 0 for s in utils.slot_states(self.db).values())

        with self._state_lock:
            if self._strategist_running:
                return False

            now = time.time()
            elapsed = now - self._last_strategist_time

            # First turn: immediately
            if self._last_strategist_time == 0:
                return True

            # After N experiments analyzed
            if self._analyzed_since_strategist >= 3:
                return True

        # DB/worker checks don't need the lock
        pending = self.db.list_by_status("to_implement")
        idle_workers = [w for w in self.workers if not w.busy]

        with self._state_lock:
            elapsed = time.time() - self._last_strategist_time
            if not pending and idle_workers and elapsed > 60:
                return True
            if elapsed >= self._strategist_interval:
                return True

        return False

    def _run_strategist(self) -> None:
        """Run a strategist turn in a background thread (non-blocking).

        In no_strategist ablation mode, proposes experiments via a simple
        one-shot LLM call with no feedback about what's working — testing
        the value of strategic planning vs naive proposals.
        """
        with self._state_lock:
            self._strategist_running = True

        if deps.get().config.pipeline.phase3.no_strategist:
            self._log("random_proposer_start")
            self.emit(ExperimentEvent(
                name="random_proposer",
                status="running",
                detail="Random proposer generating experiments (no strategist)",
            ))

            ctx = copy_context()

            def _random_proposer_thread() -> None:
                token = attach_context(ctx)
                try:
                    self._propose_random_experiments()
                    with self._state_lock:
                        self._last_strategist_time = time.time()
                        self._analyzed_since_strategist = 0
                    self._log("random_proposer_done")
                except Exception as e:
                    logger.error(f"Random proposer error: {e}")
                    self._log("random_proposer_error", error=str(e),
                              traceback=traceback.format_exc())
                finally:
                    detach_context(token)
                    with self._state_lock:
                        self._strategist_running = False

            t = threading.Thread(target=_random_proposer_thread, daemon=True)
            self._strategist_thread = t
            t.start()
            return

        self._log("strategist_start")
        self.emit(ExperimentEvent(
            name="strategist",
            status="running",
            detail="Strategist proposing experiments",
        ))

        ctx = copy_context()

        def _strategist_thread() -> None:
            token = attach_context(ctx)
            try:
                self.strategist.run_turn()
                with self._state_lock:
                    self._last_strategist_time = time.time()
                    self._analyzed_since_strategist = 0
                self._log("strategist_done")
            except StallError as e:
                # Idle board + free capacity + no proposal ⇒ no progress possible.
                # Fail the whole run loudly instead of spinning.
                logger.error(f"Strategist stall: {e}")
                self._log("jit_stall", error=str(e))
                self._crashed = True
                self._stop_requested = True
            except Exception as e:
                logger.error(f"Strategist error: {e}")
                self._log("strategist_error", error=str(e), traceback=traceback.format_exc())
            finally:
                detach_context(token)
                with self._state_lock:
                    self._strategist_running = False

        t = threading.Thread(target=_strategist_thread, daemon=True)
        self._strategist_thread = t
        t.start()

    def _propose_random_experiments(self) -> None:
        """Propose experiments without strategic context (ablation mode).

        Uses a one-shot LLM call with only framework description and learnings
        from Phase 1 — no board state, no leaderboard, no playbook. This tests
        the value of the strategist's iterative learning and planning.
        """
        # Check budget — cancelled experiments do not consume budget slots
        summary = self.db.board_summary()
        total_proposed = sum(v for k, v in summary.items() if k != "cancelled")
        remaining = self._max_experiments - total_proposed
        if remaining <= 0:
            logger.info("Random proposer: budget exhausted")
            return

        # Propose in batches of 5 (or remaining, whichever is less)
        batch_size = min(5, remaining)

        # Read Phase 1 learnings for minimal domain context
        learnings = ""
        learnings_path = Path(self.workspace) / "learnings.md"
        if learnings_path.exists():
            learnings = learnings_path.read_text()[:3000]

        # Read framework code for context on what configs are valid
        framework_code = ""
        adapter = self.adapter
        if adapter is not None:
            fw_name = adapter.experiment.framework_dir or "backtest"
            fw_dir = Path(self.workspace) / fw_name
            if fw_dir.is_dir():
                snippets = []
                for f in sorted(fw_dir.rglob("*.py"))[:10]:
                    try:
                        snippets.append(f"### {f.name}\n```python\n{f.read_text()[:2000]}\n```")
                    except OSError:
                        pass
                framework_code = "\n\n".join(snippets)

        metric_name = "metric"
        direction = "maximize"
        domain = "unknown"
        if adapter is not None:
            metric_name = adapter.metric.primary_metric
            direction = adapter.metric.direction
            domain = adapter.domain_name

        prompt = f"""You are proposing experiment configs for domain '{domain}'.
Goal: {direction} '{metric_name}'.

## Learnings
{learnings if learnings else "No prior exploration."}

## Framework Code (for valid config fields)
{framework_code[:8000] if framework_code else "No framework code available."}

## Instructions
Propose {batch_size} DIVERSE experiment configs. Vary architectures, hyperparameters,
and strategies broadly. Each experiment should be meaningfully different.

Return a JSON array of objects, each with:
- "name": short unique name (alphanumeric + underscores)
- "description": one-line description
- "hypothesis": what you expect
- "config": the experiment config JSON object

Return ONLY the JSON array. No explanation."""

        import re as _re
        model = deps.get().config.model
        with self._tracer.start_as_current_span(
            f"chat {model}",
            kind=trace.SpanKind.CLIENT,
            attributes={
                GEN_AI_OPERATION_NAME: GenAiOperationNameValues.CHAT.value,
                GEN_AI_REQUEST_MODEL: model,
                GEN_AI_SYSTEM: self._gen_ai_system,
            },
        ):
            response = self.provider.complete(
                model=model,
                system="Return only valid JSON.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8000,
            )

        # Parse response
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(l for l in lines if not l.strip().startswith("```"))

        try:
            proposals = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Random proposer: failed to parse LLM response")
            return

        if not isinstance(proposals, list):
            proposals = [proposals]

        for p in proposals[:batch_size]:
            name = _re.sub(r"[^a-zA-Z0-9_\-]", "_", str(p.get("name", "random")))[:80]
            desc = str(p.get("description", ""))
            hyp = str(p.get("hypothesis", ""))
            config = p.get("config", {})
            config_str = json.dumps(config) if isinstance(config, dict) else str(config)
            try:
                exp_id = self.db.create(name, desc, hyp, config_str)
                logger.info(f"Random proposer: created experiment #{exp_id} '{name}'")
            except Exception as e:
                logger.warning(f"Random proposer: failed to create experiment: {e}")

    def _poll_slurm(self) -> None:
        """Poll GPU and CPU executors for job status updates."""
        # Under KILL_SIGNAL, queued/running have been cancelled in-place;
        # polling would race the cancellation by flipping them back to
        # ``finished`` via ``set_error_and_finish``.
        if self._kill_signal_active():
            return
        # Get all queued/running experiments with job IDs
        active = self.db.list_by_status("queued", "running")
        job_ids = [exp.slurm_job_id for exp in active if exp.slurm_job_id]
        if not job_ids:
            return

        # Split into CPU and GPU jobs
        cpu_job_ids = [jid for jid in job_ids if jid in self._cpu_job_ids]
        gpu_job_ids = [jid for jid in job_ids if jid not in self._cpu_job_ids]

        # Poll both executors
        d = deps.get()
        statuses: dict[str, str] = {}
        if gpu_job_ids:
            statuses.update(d.gpu_executor.poll_jobs(gpu_job_ids))
        if cpu_job_ids and d.cpu_executor:
            statuses.update(d.cpu_executor.poll_jobs(cpu_job_ids))

        for exp in active:
            if not exp.slurm_job_id or exp.slurm_job_id not in statuses:
                continue

            slurm_status = statuses[exp.slurm_job_id]

            if exp.status == "queued" and slurm_status == "RUNNING":
                self.db.update_status(
                    exp.id, "running",
                    started_at=time.time(),
                )
                self.emit(ExperimentEvent(
                    experiment_id=exp.id,
                    name=exp.name,
                    status="running",
                    prev_status="queued",
                    slurm_job_id=exp.slurm_job_id,
                    detail=f"SLURM job {exp.slurm_job_id} running",
                ))
                logger.info(f"Experiment #{exp.id} {exp.name}: queued -> running")

            elif exp.status in ("queued", "running") and slurm_status == "COMPLETED":
                self.db.update_status(
                    exp.id, "finished",
                    finished_at=time.time(),
                )
                self.emit(ExperimentEvent(
                    experiment_id=exp.id,
                    name=exp.name,
                    status="finished",
                    prev_status=exp.status,
                    slurm_job_id=exp.slurm_job_id,
                    detail=f"SLURM job {exp.slurm_job_id} completed",
                ))
                logger.info(f"Experiment #{exp.id} {exp.name}: {exp.status} -> finished")

            elif exp.status in ("queued", "running") and slurm_status in (
                "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "UNKNOWN"
            ):
                self.db.set_error_and_finish(exp.id, f"SLURM {slurm_status}")
                self.emit(ExperimentEvent(
                    experiment_id=exp.id,
                    name=exp.name,
                    status="finished",
                    prev_status=exp.status,
                    slurm_job_id=exp.slurm_job_id,
                    detail=f"SLURM job {exp.slurm_job_id} {slurm_status}",
                ))
                logger.warning(
                    f"Experiment #{exp.id} {exp.name}: SLURM {slurm_status}"
                )

    def _is_cpu_experiment(self, exp: Experiment) -> bool:
        """Check if an experiment should run on CPU.

        Priority:
        1. Explicit resource tag in config ("cpu"/"gpu") — always honored.
        2. Source code scan — read the adapter-declared source files for this
           experiment (``adapter.experiment.required_files`` plus
           ``entry_point``, falling back to ``strategy.py`` + ``run_experiment.py``
           only if the adapter declares neither) and search them for GPU
           markers such as ``torch.cuda``, ``.cuda()``, ``torch.device``,
           etc. If any declared file references a GPU marker, route to GPU.
           If a file exists but is unreadable (OSError / UnicodeDecodeError)
           we conservatively route to GPU rather than risk a false-negative
           CPU routing.
        """
        try:
            config = json.loads(exp.config_json or "{}")
            resource = ""
            if isinstance(config, dict):
                raw_resource = config.get("resource", "")
                # Callers have been known to set resource to non-string values
                # (null, numeric tag, etc.); only a str is usable here, and
                # anything else falls through to the source-scan heuristic.
                if isinstance(raw_resource, str):
                    resource = raw_resource.lower()

            # Explicit resource tag takes precedence
            if resource == "cpu":
                return True
            if resource == "gpu":
                return False
        except (json.JSONDecodeError, TypeError):
            pass

        # Scan experiment source files for GPU usage indicators. Explicit
        # UTF-8 + UnicodeDecodeError catch: the dispatcher loop must survive
        # a single odd-encoded source file instead of tearing the pipeline
        # down on a decode error. If a source file exists but can't be read,
        # we conservatively treat it as "unknown" and default to GPU — routing
        # an unreadable file to CPU risks silently running a GPU experiment on
        # CPU, which would crash at runtime.
        exp_dir = Path(self.workspace) / "experiments" / exp.name
        gpu_markers = {"torch.device", ".cuda()", "CUDA", "torch.cuda", ".to(device",
                       ".to(self.device", "gpu_id", "accelerator"}
        # Adapters vary in required filenames (time_series: strategy.py,
        # cuda_kernel: kernel.cu, nanogpt: run_training.py + train_config.py,
        # llm_speedrun: train.py). We scan the adapter-declared files
        # (``required_files`` + ``entry_point``) so non-time_series adapters
        # aren't silently misrouted to CPU. If the adapter declares neither,
        # we fall back to the legacy ``strategy.py`` + ``run_experiment.py``
        # pair. We deliberately do NOT glob every ``.py`` in the experiment
        # dir — experiments can contain helper modules the adapter doesn't
        # declare, and scanning those could flip routing unexpectedly.
        candidate_files: list[Path] = []
        if self.adapter is not None:
            required = getattr(self.adapter.experiment, "required_files", None) or []
            entry = getattr(self.adapter.experiment, "entry_point", None)
            for name in list(required) + ([entry] if entry else []):
                p = exp_dir / name
                if p not in candidate_files:
                    candidate_files.append(p)
        if not candidate_files:
            for name in ("strategy.py", "run_experiment.py"):
                candidate_files.append(exp_dir / name)
        any_unreadable = False
        any_scanned = False
        for src_path in candidate_files:
            if src_path.exists():
                try:
                    source = src_path.read_text(encoding="utf-8")
                    any_scanned = True
                    if any(marker in source for marker in gpu_markers):
                        return False  # Code references GPU — run on GPU
                except (OSError, UnicodeDecodeError):
                    any_unreadable = True

        if any_unreadable or not any_scanned:
            # Unknown — default to GPU to avoid false-negative CPU routing.
            # `not any_scanned` covers the case where every declared source
            # file is missing (experiment not materialized, adapter declares
            # non-existent files, wrong exp_dir, etc.); routing those to CPU
            # would contradict the "safe default = GPU" invariant used
            # elsewhere in the dispatcher.
            return False

        # No GPU references found in any scanned source — run on CPU
        return True

    def _submit_checked(self) -> None:
        """Submit checked experiments to GPU or CPU executor."""
        if self._kill_signal_active():
            return
        d = deps.get()
        checked = self.db.list_by_status("checked")
        for exp in checked:
            if self._stop_requested:
                break

            # When no GPUs are configured, bypass the source scan and
            # send everything to the CPU executor. Experiments that truly
            # need a GPU will fail at runtime instead of stalling forever.
            has_gpus = bool(getattr(d.gpu_executor, "gpu_ids", True))

            if not has_gpus and d.cpu_executor is not None:
                use_cpu = True
            elif d.cpu_executor is not None and self._is_cpu_experiment(exp):
                use_cpu = True
            else:
                use_cpu = False

            if use_cpu:
                if not d.cpu_executor.can_submit(exp):
                    logger.debug(f"CPU slots full, skipping {exp.name}")
                    continue
                executor = d.cpu_executor
                executor_name = "CPU"
            else:
                if not d.gpu_executor.can_submit(exp):
                    logger.debug("GPU budget exhausted, waiting")
                    continue
                executor = d.gpu_executor
                executor_name = "GPU"

            try:
                with self._tracer.start_as_current_span(
                    f"submit_experiment {exp.name}",
                    kind=trace.SpanKind.INTERNAL,
                    attributes={
                        "alpha_lab.experiment.id": exp.id,
                        "alpha_lab.experiment.name": exp.name,
                        "alpha_lab.experiment.executor": executor_name,
                    },
                ) as submit_span:
                    job_id = executor.submit_experiment(exp, self.workspace)
                    submit_span.set_attribute("alpha_lab.experiment.job_id", job_id)
                if use_cpu:
                    self._cpu_job_ids.add(job_id)
                self.db.set_slurm_job(exp.id, job_id)
                self.db.update_status(exp.id, "queued")
                self.emit(ExperimentEvent(
                    experiment_id=exp.id,
                    name=exp.name,
                    status="queued",
                    prev_status="checked",
                    slurm_job_id=job_id,
                    detail=f"Submitted {executor_name} job {job_id}",
                ))
                logger.info(f"Experiment #{exp.id} {exp.name}: submitted as {executor_name} job {job_id}")
            except Exception as e:
                logger.error(f"Failed to submit experiment #{exp.id}: {e}")
                self.db.set_error_and_finish(exp.id, f"Submit failed: {e}")

    def _assign_workers(self) -> None:
        """Assign idle workers to pending tasks."""
        # First: release experiments assigned to workers that are no longer busy
        self._release_dead_assignments()

        idle = [w for w in self.workers if not w.busy]
        if not idle:
            return

        kill = self._kill_signal_active()

        # Close out analyzed experiments first when handoff is enabled.
        # KILL_SIGNAL drains existing evaluator work without starting handoff.
        handoff_enabled = not deps.get().config.pipeline.phase3.no_handoff
        analyzed_rows = (
            [] if kill or not handoff_enabled
            else self.db.list_by_status("analyzed")
        )
        unassigned_handoff = [exp for exp in analyzed_rows if exp.worker_id is None]

        for exp in unassigned_handoff:
            if not idle:
                break
            worker = idle.pop(0)
            logger.info(
                f"Assigning {worker.worker_id} to handoff #{exp.id} {exp.name}"
            )
            self._log("assign_worker", worker=worker.worker_id, task="handoff",
                       experiment_id=exp.id, experiment_name=exp.name)
            worker.handoff(exp, otel_context=copy_context())

        # Fix failed experiments (finished with error, no results, < max attempts).
        # Skipped while draining under KILL_SIGNAL: fixes recycle work back through
        # checked -> queued -> running, which the sentinel is meant to halt.
        MAX_FIX_ATTEMPTS = 2
        finished = self.db.list_by_status("finished")
        fixable = [] if kill else [
            exp for exp in finished
            if exp.worker_id is None
            and exp.error
            and not exp.results_json
            and exp.fix_attempts < MAX_FIX_ATTEMPTS
        ]

        for exp in fixable:
            if not idle:
                break
            worker = idle.pop(0)
            # Increment fix attempts before assigning
            attempts = self.db.increment_fix_attempts(exp.id)
            logger.info(
                f"Assigning {worker.worker_id} to fix #{exp.id} {exp.name} "
                f"(attempt {attempts}/{MAX_FIX_ATTEMPTS})"
            )
            self._log("assign_worker", worker=worker.worker_id, task="fix",
                       experiment_id=exp.id, experiment_name=exp.name,
                       fix_attempt=attempts)
            worker.fix(exp, otel_context=copy_context())

        # Analyze finished experiments (with results or max fix attempts reached).
        unassigned_finished = [
            exp for exp in finished
            if exp.worker_id is None and exp not in fixable
        ]

        for exp in unassigned_finished:
            if not idle:
                break
            worker = idle.pop(0)
            logger.info(
                f"Assigning {worker.worker_id} to analyze #{exp.id} {exp.name}"
            )
            self._log("assign_worker", worker=worker.worker_id, task="analyze",
                       experiment_id=exp.id, experiment_name=exp.name)
            worker.analyze(exp, otel_context=copy_context())

        # Implement new experiments (include "implemented" stuck experiments).
        # Skipped while draining under KILL_SIGNAL: no new evaluations are wanted.
        to_implement = [] if kill else self.db.list_by_status("to_implement", "implemented")
        unassigned_impl = [
            exp for exp in to_implement if exp.worker_id is None
        ]

        for exp in unassigned_impl:
            if not idle:
                break
            worker = idle.pop(0)
            logger.info(
                f"Assigning {worker.worker_id} to implement #{exp.id} {exp.name}"
            )
            self._log("assign_worker", worker=worker.worker_id, task="implement",
                       experiment_id=exp.id, experiment_name=exp.name)
            worker.implement(exp, otel_context=copy_context())

    def _release_dead_assignments(self) -> None:
        """Release experiments assigned to workers that are no longer busy.

        This catches the case where a worker thread died or finished but
        the experiment still has worker_id set (e.g. because the LLM agent
        didn't call update_experiment).

        To avoid racing with thread startup, only release if the assignment
        is at least 10 seconds old.
        """
        # Build set of currently-busy worker IDs
        busy_worker_ids = {w.worker_id for w in self.workers if w.busy}
        now = time.time()

        # Check experiments in states where workers should be active
        for status in ("to_implement", "implemented", "finished", "analyzed"):
            assigned = self.db.list_by_status(status)
            for exp in assigned:
                if exp.worker_id and exp.worker_id not in busy_worker_ids:
                    # Grace period: don't release if assignment is very recent
                    # (thread may still be starting up)
                    if now - exp.updated_at < 10:
                        continue
                    logger.warning(
                        f"Experiment #{exp.id} {exp.name} assigned to "
                        f"{exp.worker_id} but worker is idle — releasing"
                    )
                    self._log("release_dead_assignment",
                              experiment_id=exp.id, worker=exp.worker_id,
                              status=exp.status)
                    self.db.release_worker(exp.id)

    def _check_stale(self) -> None:
        """Detect stale workers (>5min) and release their assignments.

        Only releases if the assigned worker thread is no longer alive,
        to avoid prematurely releasing experiments from workers that are
        just running long shell commands.
        """
        busy_worker_ids = {w.worker_id for w in self.workers if w.busy}
        stale = self.db.stale_workers(timeout_s=300)
        for exp in stale:
            if exp.worker_id in busy_worker_ids:
                continue  # Worker thread still alive, don't release
            logger.warning(
                f"Stale worker on experiment #{exp.id} {exp.name} — releasing"
            )
            self.db.release_worker(exp.id)

    def _check_stuck_workers(self) -> None:
        """Detect workers with no events for >10 minutes (observability only)."""
        now = time.time()
        for w in self.workers:
            if w.busy and w.last_event_at > 0 and now - w.last_event_at > 600:
                logger.warning(
                    f"Worker {w.worker_id} stuck: no events for "
                    f"{int(now - w.last_event_at)}s"
                )
                self._log(
                    "stuck_worker",
                    worker=w.worker_id,
                    seconds_since_event=int(now - w.last_event_at),
                )

    def _maybe_supervisor_check(self) -> None:
        """Run supervisor health check if error rate exceeds 40%.

        Checked every 10 analyzed experiments. If the supervisor patches
        the adapter, reload it.
        """
        if self.supervisor is None:
            return

        analyzed = self.db.list_by_status("analyzed", "done")
        if len(analyzed) < 10 or len(analyzed) % 10 != 0:
            return

        # Calculate error rate
        all_finished = self.db.list_by_status("analyzed", "done")
        errors = [e for e in all_finished if e.error]
        error_rate = len(errors) / max(len(all_finished), 1)

        if error_rate <= 0.4:
            return

        logger.warning(
            f"Error rate {error_rate:.0%} exceeds 40% — running supervisor health check"
        )
        self._log("supervisor_health_check", error_rate=error_rate)

        try:
            self.supervisor.phase3_health_check()
            new_adapter = resolve_adapter(self.workspace)
            self.adapter = new_adapter
            # Update workers and strategist
            for w in self.workers:
                w.adapter = new_adapter
            self.strategist.adapter = new_adapter
        except Exception as e:
            logger.error(f"Supervisor health check failed: {e}")
            self._log("supervisor_error", error=str(e))

    def _emit_board_summary(self) -> None:
        """Emit a board summary event."""
        summary = self.db.board_summary()
        recent = self.db.list_all()[-10:]
        _metric = self._convergence_metric
        leaders = self.db.leaderboard(_metric, 5, self._metric_direction)

        experiments = []
        for exp in recent:
            experiments.append({
                "id": exp.id,
                "name": exp.name,
                "status": exp.status,
                "worker_id": exp.worker_id,
                "slurm_job_id": exp.slurm_job_id,
            })

        leaderboard = []
        for exp in leaders:
            metrics = {}
            if exp.results_json:
                try:
                    parsed = json.loads(exp.results_json)
                    if isinstance(parsed, dict):
                        metrics = parsed
                except (json.JSONDecodeError, TypeError):
                    pass
            leaderboard.append({
                "id": exp.id,
                "name": exp.name,
                "metrics": metrics,
            })

        self.emit(BoardSummaryEvent(
            counts=summary,
            experiments=experiments,
            leaderboard=leaderboard,
        ))

    def _maybe_generate_report(self) -> None:
        """Trigger a milestone report if enough experiments are done."""
        if self._report_in_progress:
            if self._report_worker is not None:
                if self._report_worker.busy:
                    return  # still running
                # Reporter thread finished (completed or died) — either way, reset state
            else:
                # _report_worker is None but flag is stuck — reset
                logger.warning("Report flag stuck with no worker — resetting")
                self._report_in_progress = False
                return
            # Reporter just finished — copy to output/
            try:
                from alpha_lab.output_generator import OutputGenerator
                gen = OutputGenerator(self.workspace, adapter=self.adapter)
                gen.copy_milestone_report(self._current_report_number)
                gen.generate_index()
            except Exception as e:
                logger.error(f"Output copy failed: {e}")
            self._report_in_progress = False
            self._report_worker = None
            return

        done_count = len(self.db.list_by_status("done", "analyzed"))
        experiments_since_report = done_count - self._last_report_at_done_count

        if experiments_since_report >= self._report_interval and done_count > 0:
            # Find an idle worker for reporting
            idle = [w for w in self.workers if not w.busy]
            if not idle:
                return  # all busy, try next cycle

            self._report_number += 1
            worker = idle[0]
            logger.info(
                f"Triggering milestone report #{self._report_number} "
                f"({done_count} experiments done)"
            )
            self.emit(ExperimentEvent(
                name="reporter",
                status="running",
                detail=f"Generating milestone report #{self._report_number} ({done_count} done)",
            ))

            # Ensure reports directory exists
            Path(self.workspace, "reports").mkdir(parents=True, exist_ok=True)

            worker.generate_report(self._report_number, done_count, otel_context=copy_context())
            self._last_report_at_done_count = done_count
            self._report_in_progress = True
            self._report_worker = worker
            self._current_report_number = self._report_number

    def _track_analyzed(self) -> None:
        """Track how many experiments have been analyzed since last strategist turn.

        Also tracks convergence: if no improvement in top metric for N experiments,
        sets a flag for early stopping.
        """
        current_analyzed = len(self.db.list_by_status("analyzed", "done"))
        new_analyzed = current_analyzed - self._last_analyzed_count
        if new_analyzed > 0:
            with self._state_lock:
                self._analyzed_since_strategist += new_analyzed
            self._last_analyzed_count = current_analyzed

            # Check for improvement in best metric
            leaders = self.db.leaderboard(self._convergence_metric, 1, self._metric_direction)
            if leaders:
                try:
                    metrics = json.loads(leaders[0].results_json or "{}")
                    if not isinstance(metrics, dict):
                        metrics = {}
                    default_val = float("inf") if self._metric_direction == "minimize" else float("-inf")
                    current_best = float(metrics.get(self._convergence_metric, default_val))
                    # For maximize: improvement = current > best
                    # For minimize: improvement = current < best
                    if self._metric_direction == "minimize":
                        improved = current_best < self._best_metric_value
                    else:
                        improved = current_best > self._best_metric_value
                    if improved:
                        improvement = abs(current_best - self._best_metric_value)
                        self._best_metric_value = current_best
                        self._experiments_since_improvement = 0
                        logger.info(
                            f"New best {self._convergence_metric}: {current_best:.4f} "
                            f"(improvement: {improvement:.4f})"
                        )
                        self._log("new_best", metric=self._convergence_metric,
                                  value=current_best, improvement=improvement)
                    else:
                        self._experiments_since_improvement += new_analyzed
                except (json.JSONDecodeError, ValueError, TypeError):
                    self._experiments_since_improvement += new_analyzed

    def _check_convergence(self) -> bool:
        """Check if we've converged (no improvement for N experiments)."""
        if self._convergence_threshold <= 0:
            return False
        if self._experiments_since_improvement >= self._convergence_threshold:
            logger.info(
                f"Convergence detected: no improvement in {self._convergence_metric} "
                f"for {self._experiments_since_improvement} experiments"
            )
            self._log("convergence",
                      experiments_since_improvement=self._experiments_since_improvement,
                      best_value=self._best_metric_value)
            return True
        return False

    def _should_terminate(self) -> bool:
        """Check if max_experiments is reached and no in-flight work remains.

        Under KILL_SIGNAL the run drains as soon as nothing remains in
        ``queued`` / ``running`` / ``finished``; ``to_implement``,
        ``implemented``, and ``checked`` work is intentionally stranded
        because the sentinel blocked further progress on those columns.
        """
        done = self.db.list_by_status("analyzed", "done")

        # Convergence is logged for observability but does not terminate the run.
        # We always run until max_experiments are fully completed so that cross-run
        # comparisons use the same number of real experiments.
        self._check_convergence()  # logs convergence state; result intentionally ignored

        if self._kill_signal_active():
            # Queued/running have already been cancelled in-place.
            # Only ``finished`` work (already produced metrics) may still
            # be analyzed; terminate once that bucket drains.
            draining = self.db.list_by_status("finished")
            return not draining

        # Stop only when max_experiments have reached a terminal-or-near-terminal
        # state. With handoff enabled, ``analyzed`` remains in-flight below.
        if len(done) < self._max_experiments:
            return False
        # Don't terminate while experiments are still in progress.
        # ``analyzed`` is in-flight only when the handoff stage is enabled
        # (otherwise it's a terminal state from the dispatcher's POV).
        in_flight_states = [
            "to_implement", "implemented", "checked", "queued", "running", "finished",
        ]
        if not deps.get().config.pipeline.phase3.no_handoff:
            in_flight_states.append("analyzed")
        in_flight = self.db.list_by_status(*in_flight_states)
        if in_flight:
            return False
        return True
