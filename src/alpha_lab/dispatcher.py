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
from typing import Protocol

from alpha_lab.conductor import Conductor
from alpha_lab.config import Phase3Config, TaskConfig
from alpha_lab.events import (
    AgentEvent,
    BoardSummaryEvent,
    ExperimentEvent,
    PhaseEvent,
)
from alpha_lab.experiment_db import Experiment, ExperimentDB, is_execution_failure
from alpha_lab.provider import Provider
from alpha_lab.strategist import Strategist
from alpha_lab.worker import Worker


# Prefix workers use to advertise an unsatisfied external precondition
# (missing upstream data, dependency artifact, etc.) on an
# ``implemented`` row. The dispatcher refuses to re-assign implement
# workers to such rows — they'd just spin re-running the same gate
# check. Convention is generic across domains: any error message
# starting with ``"blocked:"`` is treated as a structured "do not
# retry" signal, not a transient failure.
_BLOCKED_ERROR_PREFIX = "blocked:"
CANCELLATION_TIMEOUT_SECONDS = 300


def _is_externally_blocked(exp: Experiment) -> bool:
    """Return True if the experiment row has a worker-set ``error`` field
    declaring an unsatisfied precondition or invalidated hypothesis.

    Two recognized cases, both signalled by the ``"blocked:"`` error prefix:

    * ``implemented`` row whose canonical-run precondition failed (source
      data missing, dependency artifact unavailable, etc.) — the worker
      did the implementation work but found the run couldn't proceed.
    * ``to_implement`` row whose hypothesis was invalidated by a more
      recent experiment before the implementer started coding (the
      "is this proposal still warranted" check in the implement prompt).
      The implementer hasn't done any code work yet but writes a blocked
      error to signal "skip this row" to the dispatcher.

    Other status values are unaffected. The dispatcher's ``_assign_workers``
    filters both ``to_implement`` and ``implemented`` rows through this
    function so neither kind of blocked row gets reassigned to another
    implementer in a re-trigger loop.
    """
    if exp.status not in ("to_implement", "implemented"):
        return False
    err = (exp.error or "").lstrip()
    return err.startswith(_BLOCKED_ERROR_PREFIX)


class JobExecutor(Protocol):
    """Protocol for job executors (SLURM or local)."""

    def submit_experiment(self, exp: Experiment, workspace: str) -> str:
        """Submit experiment, return job ID."""
        ...

    def poll_jobs(self, job_ids: list[str]) -> dict[str, str]:
        """Poll job statuses. Returns {job_id: status}."""
        ...

    def cancel(self, job_id: str) -> None:
        """Cancel a job."""
        ...

    def can_submit(self, exp: Experiment | None = None) -> bool:
        """Check if capacity available for the given experiment."""
        ...

    def running_gpu_count(self) -> int:
        """Count running jobs."""
        ...

    def is_alive(self, job_id: str) -> bool:
        """Return True iff the job is currently tracked and still running."""
        ...


class CPUExecutor(Protocol):
    """Protocol for CPU executor (optional)."""

    def submit_experiment(self, exp: Experiment, workspace: str) -> str:
        ...

    def poll_jobs(self, job_ids: list[str]) -> dict[str, str]:
        ...

    def cancel(self, job_id: str) -> None:
        ...

    def can_submit(self, exp: Experiment | None = None) -> bool:
        ...

    def running_count(self) -> int:
        ...

    def is_alive(self, job_id: str) -> bool:
        ...

logger = logging.getLogger("alpha_lab.dispatcher")

POLL_INTERVAL = 10  # seconds


class Dispatcher:
    """Main orchestration loop for Phase 3 experiment system."""

    def __init__(
        self,
        provider: Provider,
        config: TaskConfig,
        workspace: str,
        db: ExperimentDB,
        executor: JobExecutor,
        event_callback: Callable[[AgentEvent], None],
        worker_count: int = 4,
        metrics: object | None = None,
        cpu_executor: CPUExecutor | None = None,
        adapter: object | None = None,
        supervisor: object | None = None,
        conductor: Conductor | None = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.workspace = workspace
        self.db = db
        self.executor = executor  # GPU executor
        self.cpu_executor = cpu_executor  # Optional CPU executor
        self.event_callback = event_callback
        self.metrics = metrics
        self.adapter = adapter
        self.supervisor = supervisor
        self._stop_requested = False
        self._crashed = False
        # Track which jobs are CPU vs GPU for polling
        self._cpu_job_ids: set[str] = set()
        # Drain mode is flipped on by ``_consume_run_end_marker`` when the
        # Conductor has called ``request_run_end``. Once on:
        #   * ``_submit_checked`` refuses new submissions,
        #   * ``_should_run_strategist`` returns False so no new proposals,
        #   * the main loop continues polling in-flight jobs until they
        #     finish, then ``_consume_run_end_marker`` flips
        #     ``_stop_requested`` so the loop exits cleanly.
        self._drain_mode = False
        self._run_end_marker_seen = False
        # Zero-GPU guard (introduced after d5_rfq_sol_cond deadlocked 13h on
        # 2026-08-02): the router's no-marker default is GPU, and a GPU
        # executor with an empty pool can never accept a submission — the
        # skip was a DEBUG line and _should_terminate counted the row as
        # in-flight forever. An impossible pool must fail at construction,
        # not idle at runtime.
        gpu_ids = getattr(executor, "gpu_ids", None)
        self._gpu_pool_empty = gpu_ids is not None and len(gpu_ids) == 0
        if self._gpu_pool_empty and cpu_executor is None:
            raise ValueError(
                "GPU executor has no GPUs (gpu_ids=[]) and no CPU executor "
                "is enabled — no experiment could ever be scheduled"
            )
        if self._gpu_pool_empty:
            logger.info(
                "GPU executor has no GPUs configured (gpu_ids=[]); "
                "GPU-routed experiments will fall back to the CPU pool"
            )
        # Consecutive submission-skip counts per experiment id — the
        # can-this-ever-complete detector. Reset on successful submit.
        self._submit_skip_counts: dict[int, int] = {}

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
            for i in range(worker_count)
        ]

        # Create strategist
        self.strategist = Strategist(
            provider=provider,
            config=config,
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
        # Saved at each trigger so a crashed reporter can roll the baseline
        # back (see _maybe_generate_report) rather than burning the interval.
        self._prev_report_at_done_count = 0
        self._report_number = 0
        self._current_report_number = 0
        self._report_in_progress = False
        self._report_worker: Worker | None = None

        # Conductor scheduling. The Conductor mirrors the strategist's
        # scheduling pattern (single in-flight, no self-queueing). It is
        # triggered by edge events (right after a milestone report finishes,
        # right after Phase 0/1/2 completes) and by a slow timer fallback
        # at ``conductor_interval`` seconds. ``no_conductor=True`` skips
        # construction entirely and the dispatcher behaves as it did before
        # the Conductor existed — no meta/ writes happen, _maybe_run_conductor
        # short-circuits.
        # Conductor is constructed once near the top of the run (in
        # run.py via build_conductor) and passed in so the same instance
        # spans all phases. If the caller didn't pass one, build it
        # here for backward compatibility — but the phase 0/1/2 steer
        # calls live in run.py, so a Conductor built only here will not
        # see the earlier phase boundaries.
        self._conductor_running: bool = False
        self._conductor_thread: threading.Thread | None = None
        self._conductor_interval = p3.conductor_interval
        # Start the timer clock at "now" when the conductor was passed
        # in from run.py — that means run.py already fired steer_phase2
        # at the phase 2→3 boundary, so the immediate "first turn fires
        # without delay" logic in _should_run_conductor would just
        # duplicate work. Starting the timer fresh defers the next turn
        # to either a milestone trigger or ``conductor_interval`` later.
        # When the dispatcher built its own conductor (no run.py
        # involvement), keep the legacy first-turn-immediate behavior.
        self._last_conductor_time: float = time.time() if conductor is not None else 0.0
        self._milestone_just_finished: bool = False
        if conductor is not None:
            self.conductor: Conductor | None = conductor
        elif not p3.no_conductor and adapter is not None:
            from alpha_lab.conductor import build_conductor
            self.conductor = build_conductor(
                main_provider=provider,
                config=config,
                workspace=workspace,
                db=db,
                adapter=adapter,
                event_callback=event_callback,
            )
        else:
            self.conductor = None

        # Verifier scheduling — mirrors the conductor/strategist (one in flight, daemon thread).
        # Fires on a Conductor `request_verification` marker OR auto once
        # conductor_verify_after_n_strategies experiments are analyzed (0 = never auto). NOOP when
        # no conductor is configured (there is no meta/ channel to commission a verification from).
        self._verifier_running = False
        self._verifier_thread: threading.Thread | None = None
        self._verifier_obj = None  # live Verifier, for the stop() drain
        self._verify_after_n = int(getattr(config, "conductor_verify_after_n_strategies", 0) or 0)
        self._verifier_ran_at_analyzed = -1

        # Limits
        self._max_experiments = p3.max_experiments

        # Convergence tracking
        self._convergence_threshold = p3.convergence_threshold
        # Resolve metric: config override > adapter primary > fallback "sharpe"
        if p3.convergence_metric:
            self._convergence_metric = p3.convergence_metric
        elif adapter is not None:
            self._convergence_metric = adapter.metric.primary_metric
        else:
            self._convergence_metric = "sharpe"
        # Determine metric direction for convergence comparison
        self._metric_direction = "maximize"
        if adapter is not None:
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
        # Persist the dispatcher's first-boot timestamp so the Conductor's
        # ``request_run_end`` tool can enforce the ``min_runtime_hours``
        # floor across process restarts. Idempotent: if the file already
        # has a valid ts (prior run), the helper preserves it.
        try:
            from alpha_lab.meta_layout import write_dispatcher_start_ts
            write_dispatcher_start_ts(self.workspace, time.time())
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("Failed to write dispatcher start ts: %s", e)

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

        # Join conductor thread (best-effort; the conductor catches its own
        # exceptions and is daemon-threaded so a hung join just times out
        # and the process exits cleanly).
        if self._conductor_thread is not None and self._conductor_thread.is_alive():
            self._conductor_thread.join(timeout=join_timeout)

        # Drain an in-flight verification: the run is over and nobody is
        # left to act on the verdict, but the decision itself must still be
        # recorded (user order 2026-08-07). Measured failure: three d5
        # verifications froze 0-2.6 minutes short of their verdicts when
        # process exit killed this daemon thread mid-round. The verifier is
        # told to finish the candidate in flight and start no new ones; the
        # wait is bounded by verifier_drain_seconds (default 7200 — two
        # rounds of 1800s notebook executions plus agent turns fit; 0
        # disables draining and restores the old cut-at-exit behavior).
        with self._state_lock:
            vt = self._verifier_thread
            vobj = self._verifier_obj
        if vt is not None and vt.is_alive():
            drain = float(getattr(self.config, "verifier_drain_seconds",
                                  7200) or 0)
            if drain > 0:
                if vobj is not None:
                    vobj.finish_current_and_stop()
                self._log("verifier_drain_start", seconds=drain)
                logger.warning(
                    "run is over; draining the in-flight verification for "
                    "up to %.0fs so its verdict is recorded", drain)
                vt.join(timeout=drain)
                if vt.is_alive():
                    self._log("verifier_drain_timeout", seconds=drain)
                    logger.error(
                        "verification still unfinished after %.0fs drain — "
                        "its STATE.json marks the step it was cut at", drain)
                else:
                    self._log("verifier_drain_done")

        # Kill all running executor jobs to prevent orphans
        try:
            self.executor.cleanup_all()
        except Exception as e:
            logger.warning("Failed to cleanup GPU executor: %s", e)
        if self.cpu_executor is not None:
            try:
                self.cpu_executor.cleanup_all()
            except Exception as e:
                logger.warning("Failed to cleanup CPU executor: %s", e)

        self._cleanup()

    def _cleanup(self) -> None:
        """Release all remaining worker assignments and close log file."""
        for status in ("to_implement", "implemented", "finished"):
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
        summary: dict = {
            "released_workers": 0,
            "slurm_reconciled": 0,
            "report_counter": 0,
            "reattached_jobs": 0,
        }

        # 1. Release orphaned worker assignments (no workers exist yet at startup)
        for status in ("to_implement", "implemented", "finished"):
            assigned = self.db.list_by_status(status)
            for exp in assigned:
                if exp.worker_id is not None:
                    logger.info(
                        f"Recovery: releasing orphaned worker {exp.worker_id} "
                        f"from experiment #{exp.id} {exp.name}"
                    )
                    self.db.release_worker(exp.id)
                    summary["released_workers"] += 1

        # 1b. Reattach live experiment subprocesses that outlived the
        # previous orchestrator. Without this step, the next poll_jobs
        # call returns UNKNOWN for every job (because the freshly-built
        # executors have empty _jobs dicts), and the DB stamps "SLURM
        # job lost" on rows whose python is in fact still training.
        # Local executors implement recover_running_jobs() by scanning
        # /proc; SLURM doesn't need it because squeue/sacct already
        # provide cluster-side reconciliation.
        slurm_exps_for_reattach = self.db.list_by_status(
            "queued", "running", "cancelling", include_parked=True
        )
        if slurm_exps_for_reattach:
            job_id_map = {
                exp.slurm_job_id: exp.name
                for exp in slurm_exps_for_reattach
                if exp.slurm_job_id
            }
            if job_id_map:
                # GPU executor (always present)
                recover_fn = getattr(self.executor, "recover_running_jobs", None)
                if callable(recover_fn):
                    try:
                        gpu_recovered = recover_fn(self.workspace, job_id_map)
                        summary["reattached_jobs"] += gpu_recovered
                        if gpu_recovered:
                            logger.info(
                                "Recovery: reattached %d GPU jobs from /proc scan",
                                gpu_recovered,
                            )
                    except Exception as e:
                        logger.warning("GPU recover_running_jobs failed: %s", e)
                # CPU executor (optional)
                if self.cpu_executor is not None:
                    cpu_recover = getattr(self.cpu_executor, "recover_running_jobs", None)
                    if callable(cpu_recover):
                        try:
                            cpu_recovered = cpu_recover(self.workspace, job_id_map)
                            summary["reattached_jobs"] += cpu_recovered
                            if cpu_recovered:
                                # Mark the reattached CPU job ids so subsequent
                                # poll_slurm dispatches them to the right executor.
                                for jid in job_id_map:
                                    if self.cpu_executor.is_alive(jid):
                                        self._cpu_job_ids.add(jid)
                                logger.info(
                                    "Recovery: reattached %d CPU jobs from /proc scan",
                                    cpu_recovered,
                                )
                        except Exception as e:
                            logger.warning("CPU recover_running_jobs failed: %s", e)

        # 2. Reconcile jobs. Route to the right executor's poll_jobs based
        # on whether the job_id is in _cpu_job_ids (populated by step 1b
        # for reattached CPU jobs). Without this split, a reattached CPU
        # job would be polled by the GPU executor, return UNKNOWN, and
        # get stamped "SLURM job lost" despite being alive.
        slurm_exps = self.db.list_by_status("queued", "running")
        job_ids = [exp.slurm_job_id for exp in slurm_exps if exp.slurm_job_id]
        if job_ids:
            gpu_job_ids = [j for j in job_ids if j not in self._cpu_job_ids]
            cpu_job_ids = [j for j in job_ids if j in self._cpu_job_ids]
            statuses: dict[str, str] = {}
            if gpu_job_ids:
                statuses.update(self.executor.poll_jobs(gpu_job_ids))
            if cpu_job_ids and self.cpu_executor is not None:
                statuses.update(self.cpu_executor.poll_jobs(cpu_job_ids))
            for exp in slurm_exps:
                if not exp.slurm_job_id or exp.slurm_job_id not in statuses:
                    continue
                slurm_status = statuses[exp.slurm_job_id]

                # Pick a label that's truthful for the executor that actually
                # produced this status. The interface is shared between SLURM
                # and the local executor; saying "SLURM" for a local job
                # confuses post-mortem (and the user who reads the DB).
                executor_label = (
                    "job" if exp.slurm_job_id in self._cpu_job_ids
                    or self.config.pipeline.phase3.executor == "local"
                    else "SLURM"
                )

                if slurm_status == "RUNNING" and exp.status == "queued":
                    self.db.update_status(exp.id, "running", started_at=time.time())
                    logger.info(f"Recovery: #{exp.id} queued -> running ({executor_label} RUNNING)")
                    summary["slurm_reconciled"] += 1

                elif slurm_status == "COMPLETED":
                    self.db.update_status(exp.id, "finished", finished_at=time.time())
                    logger.info(f"Recovery: #{exp.id} {exp.status} -> finished ({executor_label} COMPLETED)")
                    summary["slurm_reconciled"] += 1

                elif slurm_status in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"):
                    self.db.set_error_and_finish(exp.id, f"{executor_label} {slurm_status}")
                    logger.info(f"Recovery: #{exp.id} {exp.status} -> finished ({executor_label} {slurm_status})")
                    summary["slurm_reconciled"] += 1

                elif slurm_status == "PENDING":
                    pass  # Leave as queued

                elif slurm_status == "UNKNOWN":
                    # Three sub-cases (in priority order):
                    # 1. Orphaned-completion: the experiment actually
                    #    finished on disk while no dispatcher was watching
                    #    (results/metrics.json with run_scope="full"). The
                    #    bash wrapper can exit cleanly between a kill of
                    #    the old orchestrator and the start of the new
                    #    one; without this check a real success gets
                    #    falsely tagged "job lost".
                    # 2. Local executor + no on-disk completion: the
                    #    child was killed by the parent's
                    #    PR_SET_PDEATHSIG. Interrupted, not failed —
                    #    finish cleanly with no error so the GUI doesn't
                    #    show a red row for every in-flight experiment
                    #    when ``run.py`` restarts.
                    # 3. SLURM + no on-disk completion: a genuinely
                    #    suspicious state (job purged or never reached
                    #    the scheduler). Keep the error.
                    if self._orphaned_completion_detected(exp):
                        self.db.update_status(
                            exp.id, "finished", finished_at=time.time(),
                        )
                        logger.info(
                            f"Recovery: #{exp.id} {exp.status} -> finished "
                            f"(orphaned-completion: results/metrics.json with "
                            f"run_scope=full found on disk)"
                        )
                    elif executor_label == "job":
                        self.db.update_status(
                            exp.id, "finished", finished_at=time.time(),
                        )
                        logger.info(
                            f"Recovery: #{exp.id} {exp.status} -> finished "
                            "(local job interrupted by parent restart; no error stamped)"
                        )
                    else:
                        self.db.set_error_and_finish(exp.id, f"{executor_label} job lost")
                        logger.info(
                            f"Recovery: #{exp.id} {exp.status} -> finished "
                            f"({executor_label} job lost)"
                        )
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

        # Reconcile rows that the dispatcher left mid-kanban at last shutdown:
        # (a) status=checked with a slurm_job_id set — the run was queued or
        #     started before shutdown but the dispatcher never observed the
        #     terminal status. Treat like queued/running and reattach +
        #     reconcile via the same poll_jobs path.
        # (b) Any row that's parked AND in a non-terminal state — the
        #     Conductor killed it (or pre-killed it) but the dispatcher
        #     didn't transition it. Sweep these to ``cancelled`` so the
        #     board reflects reality. Without this, etfflow_g55 ended up
        #     with 110 mid-kanban zombies (14 checked + 32 running + 64
        #     implemented, all parked).
        checked_with_job = [
            exp for exp in self.db.list_by_status("checked", include_parked=True)
            if exp.slurm_job_id
        ]
        if checked_with_job:
            job_id_map = {exp.slurm_job_id: exp.name for exp in checked_with_job}
            for executor, label in ((self.executor, "gpu"),
                                    (self.cpu_executor, "cpu")):
                if executor is None:
                    continue
                fn = getattr(executor, "recover_running_jobs", None)
                if not callable(fn):
                    continue
                try:
                    n = fn(self.workspace, job_id_map)
                    if n:
                        summary["reattached_jobs"] += n
                        logger.info(
                            "Recovery: reattached %d %s jobs from checked rows", n, label
                        )
                except Exception as e:
                    logger.warning("%s recover_running_jobs (checked rows) failed: %s", label, e)
            # Poll their status and apply transitions
            statuses: dict[str, str] = {}
            try:
                statuses.update(self.executor.poll_jobs(list(job_id_map)))
            except Exception as e:
                logger.warning("Recovery: poll_jobs for checked rows failed (gpu): %s", e)
            if self.cpu_executor is not None:
                try:
                    statuses.update(self.cpu_executor.poll_jobs(list(job_id_map)))
                except Exception as e:
                    logger.warning("Recovery: poll_jobs for checked rows failed (cpu): %s", e)
            for exp in checked_with_job:
                ss = statuses.get(exp.slurm_job_id, "UNKNOWN")
                if ss == "RUNNING":
                    self.db.update_status(exp.id, "running", started_at=time.time())
                    logger.info(
                        f"Recovery: #{exp.id} checked -> running (live job {exp.slurm_job_id})"
                    )
                elif ss == "COMPLETED":
                    self.db.update_status(exp.id, "finished", finished_at=time.time())
                    logger.info(f"Recovery: #{exp.id} checked -> finished")
                elif ss in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "UNKNOWN"):
                    self.db.set_error_and_finish(exp.id, f"job {ss} (recovered from checked)")
                    logger.info(f"Recovery: #{exp.id} checked -> finished ({ss})")
        # Parked + non-terminal sweep: these are rows the Conductor parked
        # but that never reached a terminal state because the dispatcher
        # shut down (or because of the kill-without-status-transition bug
        # this commit also fixes). Move them to cancelled with a clear
        # marker so they don't sit forever in the board summary.
        parked_zombies = 0
        for st in ("to_implement", "implemented", "checked", "queued", "running"):
            try:
                rows = self.db.list_by_status(st, include_parked=True)
            except Exception:
                continue
            for exp in rows:
                if exp.parked_at is None:
                    continue
                # Skip ones we just handled via the executor path above.
                if st in ("queued", "running", "checked") and exp.slurm_job_id:
                    continue
                self.db.update_status(exp.id, "cancelled")
                parked_zombies += 1
        if parked_zombies:
            summary["parked_zombies_swept"] = parked_zombies
            logger.info(
                f"Recovery: swept {parked_zombies} parked non-terminal rows to cancelled"
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

        # Worker/strategist prompts reference research_state.md and
        # playbook.md from the very first session, but both files are only
        # written later (dispatcher snapshot / strategist update_playbook).
        # Real runs recorded dozens of failed read_file calls per run on
        # exactly these two paths before first write — create stubs so the
        # path contract holds from turn one.
        for stub_name, stub_body in (
            ("research_state.md",
             "# Research State\n\n(No snapshot yet — the dispatcher writes "
             "the first one after experiments start finishing.)\n"),
            ("playbook.md",
             "# Playbook\n\n(Empty — the strategist appends validated "
             "learnings here as the run progresses.)\n"),
        ):
            stub = Path(self.workspace, stub_name)
            if not stub.exists():
                try:
                    stub.write_text(stub_body)
                except OSError as e:
                    logger.warning("could not create %s stub: %s", stub_name, e)

        # Crash recovery before entering main loop
        try:
            self.recover()
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            self._log("recovery_error", error=str(e))

        try:
            while not self._stop_requested:
                # 1. Strategist turn (if due)
                if self._should_run_strategist():
                    self._run_strategist()

                # 2. Complete coordinated parking before ordinary job polls.
                # This gives a parking request exclusive ownership of the
                # terminal executor update.
                self._process_parkings()

                # 2a. Poll SLURM jobs
                self._poll_slurm()

                # 2b. Reclaim zombie rows: status=implemented/checked
                # but a canonical metrics.json is already on disk. Without
                # this, worker LLMs spin on dead-end transitions
                # (implemented->finished is not a legal kanban edge, so
                # they get rejected every cycle and the dispatcher just
                # keeps reassigning implement). Generic: relies on the
                # same canonical/smoke classifier set_results already uses.
                self._auto_promote_zombie_rows()

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

                # 7d. Conductor steering — non-blocking, runs in own thread.
                # Edge trigger fires when a milestone report just finished;
                # otherwise the slow timer (conductor_interval) fires. NOOP
                # when no_conductor=True. Phase 0/1/2 trigger paths are
                # handled by run.py / pipeline.py invoking
                # _maybe_run_conductor("phaseN_done") directly when those
                # phases complete; this loop handles Phase 3 only.
                self._maybe_run_conductor(
                    "milestone" if self._milestone_just_finished else "timer"
                )

                # 7e. Process any pending phase rewind request the Conductor
                # may have written. Also non-blocking — if the marker says
                # rewind, this just emits an event and clears the marker;
                # the actual phase replay is the responsibility of the
                # outer pipeline driver (run.py).
                self._consume_phase_rewind_marker()

                # 7f. Process any pending kill requests. Each marker line is
                # an experiment id the Conductor told us to kill; cancel
                # the executor job if one exists.
                self._consume_kill_requests()

                # 7g. Process any pending run-end request. When the Conductor
                # has Python-verified evidence the run is exhausted (and the
                # config floors are met), it writes meta/run_end_pending.json
                # and this hook flips _stop_requested so the loop exits
                # gracefully after this iteration's bookkeeping.
                self._consume_run_end_marker()

                # 7h. Verifier — Conductor-commissioned (request marker) or auto after N analyzed.
                # Runs in its own daemon thread, concurrent with Phase 3 (mirrors the conductor hook).
                self._maybe_run_verifier()

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

    def _should_run_strategist(self) -> bool:
        """Determine if it's time for a strategist turn."""
        # Drain mode (Conductor requested run end) — stop proposing.
        if getattr(self, "_drain_mode", False):
            return False
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

            # Edge-trigger on milestone completion. The Conductor and the
            # strategist BOTH read milestones; firing the strategist here
            # gives it a turn that's freshly informed by the milestone
            # report — important after the dispatcher.recover()
            # crash-recovery path or any long stretch where the interval
            # alone has been the only refill signal. The flag is cleared
            # by the conductor-milestone handler; we sample it without
            # claiming it so the conductor still sees it as a fresh edge.
            if self._milestone_just_finished:
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

        if self.config.pipeline.phase3.no_strategist:
            self._log("random_proposer_start")
            self.emit(ExperimentEvent(
                name="random_proposer",
                status="running",
                detail="Random proposer generating experiments (no strategist)",
            ))

            def _random_proposer_thread() -> None:
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

        def _strategist_thread() -> None:
            try:
                self.strategist.run_turn()
                with self._state_lock:
                    self._last_strategist_time = time.time()
                    self._analyzed_since_strategist = 0
                self._log("strategist_done")
            except Exception as e:
                logger.error(f"Strategist error: {e}")
                self._log("strategist_error", error=str(e), traceback=traceback.format_exc())
            finally:
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
        response = self.provider.complete(
            model=self.config.model,
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
        # Get all queued/running experiments with job IDs. INCLUDE parked rows:
        # when the Conductor's kill_experiment parks a running row and writes a
        # kill marker, the dispatcher cancels the executor job. Without
        # polling parked rows, the executor's CANCELLED status never reaches
        # the DB and the row stays at ``running`` forever — 32 such zombies
        # accumulated in workspace_etfflow_g55. Re-include them here so the
        # normal CANCELLED → finished transition runs.
        active = self.db.list_by_status("queued", "running", include_parked=True)
        job_ids = [exp.slurm_job_id for exp in active if exp.slurm_job_id]
        if not job_ids:
            return

        # Split into CPU and GPU jobs
        cpu_job_ids = [jid for jid in job_ids if jid in self._cpu_job_ids]
        gpu_job_ids = [jid for jid in job_ids if jid not in self._cpu_job_ids]

        # Poll both executors
        statuses: dict[str, str] = {}
        if gpu_job_ids:
            statuses.update(self.executor.poll_jobs(gpu_job_ids))
        if cpu_job_ids and self.cpu_executor:
            statuses.update(self.cpu_executor.poll_jobs(cpu_job_ids))

        for exp in active:
            if not exp.slurm_job_id or exp.slurm_job_id not in statuses:
                continue

            slurm_status = statuses[exp.slurm_job_id]
            # Truthful executor label: "job" for local (GPU or CPU),
            # "SLURM" only on actual SLURM. Avoids "SLURM FAILED" appearing
            # in the DB for a local-executor failure.
            executor_label = (
                "job" if exp.slurm_job_id in cpu_job_ids
                or self.config.pipeline.phase3.executor == "local"
                else "SLURM"
            )

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
                    detail=f"{executor_label} {exp.slurm_job_id} running",
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
                    detail=f"{executor_label} {exp.slurm_job_id} completed",
                ))
                logger.info(f"Experiment #{exp.id} {exp.name}: {exp.status} -> finished")

            elif exp.status in ("queued", "running") and slurm_status in (
                "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "UNKNOWN"
            ):
                applied = self.db.set_error_and_finish(
                    exp.id, f"{executor_label} {slurm_status}"
                )
                if not applied:
                    logger.info(
                        "Ignored late %s update for experiment #%d after its "
                        "lifecycle state changed",
                        slurm_status,
                        exp.id,
                    )
                    continue
                self.emit(ExperimentEvent(
                    experiment_id=exp.id,
                    name=exp.name,
                    status="finished",
                    prev_status=exp.status,
                    slurm_job_id=exp.slurm_job_id,
                    detail=f"{executor_label} {exp.slurm_job_id} {slurm_status}",
                ))
                logger.warning(
                    f"Experiment #{exp.id} {exp.name}: {executor_label} {slurm_status}"
                )

    def _process_parkings(self) -> int:
        """Cancel active parked jobs and finalize only after executor exit."""
        finalized = 0
        terminal_states = {
            "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
            "OUT_OF_MEMORY", "UNKNOWN",
        }
        for exp in self.db.list_by_status("cancelling", include_parked=True):
            job_id = exp.slurm_job_id
            if not job_id:
                if self.db.finalize_park(exp.id):
                    finalized += 1
                continue

            is_cpu = job_id in self._cpu_job_ids or job_id.startswith("cpu-")
            executor = self.cpu_executor if is_cpu and self.cpu_executor else self.executor
            try:
                executor.cancel(job_id)
            except Exception as exc:
                logger.warning(
                    "Parking cancellation for job %s (experiment #%d) failed: %s",
                    job_id,
                    exp.id,
                    exc,
                )
            try:
                executor_status = executor.poll_jobs([job_id]).get(job_id)
            except Exception as exc:
                logger.warning(
                    "Parking status check for job %s (experiment #%d) failed: %s",
                    job_id,
                    exp.id,
                    exc,
                )
                continue

            if executor_status not in terminal_states:
                if (
                    exp.parked_at is not None
                    and time.time() - exp.parked_at >= CANCELLATION_TIMEOUT_SECONDS
                ):
                    raise RuntimeError(
                        f"Parking experiment #{exp.id} {exp.name} did not "
                        f"complete within {CANCELLATION_TIMEOUT_SECONDS} seconds "
                        f"(job {job_id}, executor status={executor_status or 'missing'})"
                    )
                continue

            if self.db.finalize_park(exp.id):
                finalized += 1
                self.emit(ExperimentEvent(
                    experiment_id=exp.id,
                    name=exp.name,
                    status="checked",
                    prev_status="cancelling",
                    slurm_job_id=job_id,
                    detail=(
                        "Parked after executor confirmed terminal status "
                        f"{executor_status}"
                    ),
                ))
                logger.info(
                    "Parked experiment #%d (%s); executor status=%s",
                    exp.id,
                    exp.name,
                    executor_status,
                )
        return finalized

    def _is_cpu_experiment(self, exp: Experiment) -> bool:
        """Check if an experiment should run on CPU.

        Decision order:

        1. **Explicit ``resource`` tag** in config_json (``"cpu"``/``"gpu"``)
           — always honored.
        2. **CPU-library evidence wins over GPU markers.** Tree, linear,
           and statistical libraries (LightGBM, XGBoost, CatBoost,
           HistGradientBoosting, RandomForest, sklearn linear models,
           statsmodels) never benefit from GPU allocation. A defensive
           ``import torch``, ``PYTORCH_CUDA_ALLOC_CONF`` env-set,
           ``argparse(choices=["cuda","cpu"])``, or
           ``torch.cuda.is_available()`` check in the same script must not
           pin a tree-model experiment to a GPU slot.
        3. **GPU markers require actual GPU usage** — ``.cuda()`` calls,
           ``.to("cuda")``/``torch.device("cuda")`` placement, variable
           ``.to(device)`` placement, ``DataParallel`` /
           ``DistributedDataParallel``, Lightning ``accelerator="gpu"``.
           Weak mentions (env vars, comments, argparse choice strings,
           availability checks) are intentionally NOT markers because they
           are common defensive boilerplate in CPU scripts.
        4. **No markers either way** → GPU. CPU routing requires positive
           evidence (rule 1 or 2): the costs are asymmetric — a CPU script
           wrongly given a GPU slot merely wastes the slot, while a GPU
           training wrongly sent to the CUDA-disabled CPU pool is killed.
           Observed live (opus d2-cond): torch training code placing the
           model via an indirection the scanner cannot see
           (``model.to(DEV)`` with ``DEV = cfg["device"]``) carries no
           literal marker, and the old assume-CPU default misrouted all
           14 such experiments to the CPU pool where they crashed.
        """
        try:
            config = json.loads(exp.config_json or "{}")
            resource = ""
            if isinstance(config, dict):
                raw_resource = config.get("resource", "")
                if isinstance(raw_resource, str):
                    resource = raw_resource.lower()
            if resource == "cpu":
                return True
            if resource == "gpu":
                return False
        except (json.JSONDecodeError, TypeError):
            pass

        # Tree / linear / statistical libraries that don't use a GPU.
        # Presence of any of these overrides GPU markers — common-sense
        # routing: a LightGBM experiment is CPU even if the script
        # defensively imports torch or sets ``PYTORCH_CUDA_ALLOC_CONF``.
        cpu_library_markers = (
            "import lightgbm", "from lightgbm",
            "import xgboost", "from xgboost",
            "import catboost", "from catboost",
            "HistGradientBoosting",
            "RandomForestRegressor", "RandomForestClassifier",
            "GradientBoostingRegressor", "GradientBoostingClassifier",
            "ExtraTreesRegressor", "ExtraTreesClassifier",
            "LinearRegression", "LogisticRegression",
            "Ridge(", "Lasso(", "ElasticNet(",
            "import statsmodels", "from statsmodels",
        )
        # Markers indicating ACTUAL GPU usage — compute calls or device
        # placement, not defensive boilerplate. The previous broad markers
        # (``"CUDA"``, ``"torch.cuda"``, ``"accelerator"``) were dropped
        # because they match argparse choice strings,
        # ``torch.cuda.is_available()`` checks, and
        # ``PYTORCH_CUDA_ALLOC_CONF`` env-sets respectively.
        gpu_compute_markers = (
            ".cuda()",                # explicit GPU move method call
            '.to("cuda',              # device placement to literal "cuda"
            ".to('cuda",
            'torch.device("cuda',     # explicit GPU device construction
            "torch.device('cuda",
            ".to(device)",            # variable device placement
            ".to(device,",
            ".to(self.device)",
            ".to(self.device,",
            "DataParallel",           # nn.DataParallel / DistributedDataParallel
            'accelerator="gpu"',      # Lightning Trainer
            "accelerator='gpu'",
            'device_type="cuda"',
            "device_type='cuda'",
        )

        exp_dir = Path(self.workspace) / "experiments" / exp.name
        # Adapters vary in required filenames; scan the adapter-declared
        # files only, with the legacy ``strategy.py`` + ``run_experiment.py``
        # fallback when the adapter declares neither.
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

        found_cpu_lib = False
        found_gpu_compute = False
        any_unreadable = False
        any_scanned = False
        for src_path in candidate_files:
            if not src_path.exists():
                continue
            try:
                source = src_path.read_text(encoding="utf-8")
                any_scanned = True
            except (OSError, UnicodeDecodeError):
                any_unreadable = True
                continue
            if any(m in source for m in cpu_library_markers):
                found_cpu_lib = True
            if any(m in source for m in gpu_compute_markers):
                found_gpu_compute = True

        # CPU-library evidence wins over GPU markers (rule 2).
        if found_cpu_lib:
            return True
        if found_gpu_compute:
            return False
        # No markers either way (or missing/unreadable source) — GPU
        # (rule 4). CPU routing needs positive evidence; an ambiguous
        # script sent to a GPU slot wastes the slot, but a GPU training
        # sent to the CUDA-disabled CPU pool dies.
        return False

    def _orphaned_completion_detected(self, exp: Experiment) -> bool:
        """Return True if the experiment wrote a valid full-run metrics
        file on disk (i.e. it actually completed) even though the
        executor has no record of the job.

        Called from ``recover()`` before stamping a job ``job lost`` /
        ``SLURM job lost``. The bash wrapper for an experiment can exit
        cleanly while no dispatcher is alive — between a kill of the old
        orchestrator and the Phase 3 start of the new one. That run
        produces ``results/metrics.json`` and the model artifact but no
        ``run_status.json`` (the dispatcher's poll loop writes the
        latter). Without this check the run gets falsely tagged as lost
        despite being a real success on disk.

        We deliberately only treat ``run_scope="full"`` as evidence;
        smoke and dry-run metrics live in ``smoke_results/`` /
        ``dry_results/`` and would never end up in ``results/`` from the
        canonical entry point.
        """
        metrics_path = (
            Path(self.workspace) / "experiments" / exp.name
            / "results" / "metrics.json"
        )
        if not metrics_path.exists():
            return False
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        return payload.get("run_scope") == "full"

    def _auto_promote_zombie_rows(self) -> None:
        """Promote rows stuck at ``implemented``/``checked`` when a canonical
        ``results/metrics.json`` is already on disk.

        Driven entirely off the experiment-results document: any payload that
        the smoke/canonical classifier accepts as canonical is treated as
        evidence of a completed run, regardless of domain. The actual
        promotion goes through ``ExperimentDB.set_results``, which itself
        flips the row to ``finished`` (and writes the payload) in a single
        commit. We just locate the candidates here, hand them off, and emit
        an event so observers see the transition.
        """
        from alpha_lab.experiment_db import _classify_non_canonical_results
        candidates: list[Experiment] = []
        for status in ("implemented", "checked"):
            try:
                candidates.extend(self.db.list_by_status(status))
            except Exception:  # pragma: no cover - defensive
                continue
        for exp in candidates:
            if self._stop_requested:
                return
            # An error on a checked/implemented row can be a deliberate
            # quarantine.  Do not let an older metrics file undo it.
            if exp.error:
                continue
            metrics_path = (
                Path(self.workspace) / "experiments" / exp.name
                / "results" / "metrics.json"
            )
            if not metrics_path.exists():
                continue
            try:
                payload_text = metrics_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if _classify_non_canonical_results(payload_text) is not None:
                continue
            outcome = self.db.set_results(
                exp.id, payload_text, refuse_if_error=True,
            )
            if outcome != "applied:promoted":
                # Either it was already applied (status moved elsewhere
                # between our scan and the write) or the guard rejected
                # the payload after all — nothing to log here.
                continue
            self._log(
                "auto_promote_zombie",
                experiment_id=exp.id,
                experiment_name=exp.name,
                prev_status=exp.status,
            )
            logger.info(
                "Auto-promoted zombie row #%d %s -> finished "
                "(canonical metrics.json on disk)",
                exp.id, exp.status,
            )
            self.emit(ExperimentEvent(
                experiment_id=exp.id,
                name=exp.name,
                status="finished",
                prev_status=exp.status,
                slurm_job_id=exp.slurm_job_id or "",
                detail="auto-promoted: canonical results on disk",
            ))

    def _submit_checked(self) -> None:
        """Submit checked experiments to GPU or CPU executor.

        Honors meta/throttle.json:
          - ``halt-new`` for the relevant resource: skip new submissions
            entirely (in-flight jobs continue).
          - ``slow``: per-loop submission cap halved (rounded down to >=1)
            so the dispatcher claims fewer fresh slots per cycle.
          - ``none``: current behavior (no throttling).

        Also honors drain mode (Conductor-requested run end): refuses all
        new submissions while existing in-flight jobs finish. This is
        equivalent to ``halt-new`` on both resources for the rest of the run.

        The throttle is read once per ``_submit_checked`` call so a
        Conductor turn that lands mid-iteration sees its own writes on the
        next loop, not this one.
        """
        if getattr(self, "_drain_mode", False):
            return
        from alpha_lab.meta_layout import read_throttle
        throttle = read_throttle(self.workspace)
        gpu_throttle = throttle.get("gpu", "none")
        cpu_throttle = throttle.get("cpu", "none")

        checked = self.db.list_by_status("checked")
        gpu_submitted_this_cycle = 0
        cpu_submitted_this_cycle = 0
        gpu_cap = self._throttled_cap(gpu_throttle, default_cap=len(checked))
        cpu_cap = self._throttled_cap(cpu_throttle, default_cap=len(checked))

        for exp in checked:
            if self._stop_requested:
                break

            # Idempotency: if this experiment has a prior job that the executor
            # still believes is alive, do not submit a duplicate. Pairs with the
            # forward-only status guard in experiment_db.update_status: a stuck
            # status transition cannot escalate into a racing second subprocess.
            if exp.slurm_job_id:
                prior_executor = (
                    self.cpu_executor
                    if exp.slurm_job_id in self._cpu_job_ids
                    else self.executor
                )
                try:
                    prior_alive = prior_executor.is_alive(exp.slurm_job_id)
                except Exception as e:
                    logger.warning(
                        "is_alive check failed for #%d (%s): %s — skipping submit",
                        exp.id, exp.slurm_job_id, e,
                    )
                    prior_alive = True  # conservative: don't double-submit
                if prior_alive:
                    # The row is at `checked` but the executor reports a live
                    # job for it — the canonical run is already in progress.
                    # The right logical state is `running`, not `checked`.
                    # Advance once so the dispatcher's polling loop takes over
                    # and the normal COMPLETED/CANCELLED/FAILED path can fire.
                    # Without this, the dispatcher tight-loops on the row
                    # forever (in workspace_etfflow_g55: 645 duplicate_submit_
                    # blocked events on #106 + 626 on #107 over 2.4h).
                    self._log(
                        "duplicate_submit_blocked",
                        experiment_id=exp.id,
                        experiment_name=exp.name,
                        prior_job_id=exp.slurm_job_id,
                    )
                    outcome = self.db.update_status(
                        exp.id, "running",
                        started_at=time.time(),
                    )
                    if outcome == "applied":
                        executor_label = (
                            "job" if exp.slurm_job_id in self._cpu_job_ids
                            or self.config.pipeline.phase3.executor == "local"
                            else "SLURM"
                        )
                        self.emit(ExperimentEvent(
                            experiment_id=exp.id,
                            name=exp.name,
                            status="running",
                            prev_status="checked",
                            slurm_job_id=exp.slurm_job_id,
                            detail=f"{executor_label} {exp.slurm_job_id} already running (auto-advanced from checked)",
                        ))
                        logger.info(
                            f"Experiment #{exp.id} {exp.name}: checked -> running "
                            f"(auto-advanced; executor reports live job {exp.slurm_job_id})"
                        )
                    continue

            # Determine which executor to use
            use_cpu = self.cpu_executor is not None and self._is_cpu_experiment(exp)
            if not use_cpu and self._gpu_pool_empty:
                # cpu_executor is not None here: the constructor rejects the
                # empty-GPU + no-CPU combination outright.
                logger.info(
                    f"Experiment #{exp.id} {exp.name}: GPU-routed but the "
                    f"GPU pool is empty (gpu_ids=[]) — falling back to CPU"
                )
                use_cpu = True

            # Throttle: halt-new blocks new launches; slow halves capacity.
            if use_cpu:
                if cpu_throttle == "halt-new":
                    continue
                if cpu_submitted_this_cycle >= cpu_cap:
                    continue
            else:
                if gpu_throttle == "halt-new":
                    continue
                if gpu_submitted_this_cycle >= gpu_cap:
                    continue

            if use_cpu:
                if not self.cpu_executor.can_submit(exp):
                    self._note_submit_skip(exp, "CPU")
                    continue
                executor = self.cpu_executor
                executor_name = "CPU"
            else:
                if not self.executor.can_submit(exp):
                    self._note_submit_skip(exp, "GPU")
                    continue
                executor = self.executor
                executor_name = "GPU"
            self._submit_skip_counts.pop(exp.id, None)

            try:
                job_id = executor.submit_experiment(exp, self.workspace)
                if use_cpu:
                    self._cpu_job_ids.add(job_id)
                    cpu_submitted_this_cycle += 1
                else:
                    gpu_submitted_this_cycle += 1
                self.db.increment_launch_attempts(exp.id)
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

    def _note_submit_skip(self, exp, pool: str) -> None:
        """Capacity skips must be visible: a starving row looked identical
        to a busy pool for 13h at DEBUG level (d5_rfq_sol_cond, 2026-08-02).
        Warn on the first skip and every 100th after."""
        n = self._submit_skip_counts.get(exp.id, 0) + 1
        self._submit_skip_counts[exp.id] = n
        if n == 1 or n % 100 == 0:
            logger.warning(
                f"Experiment #{exp.id} {exp.name}: waiting for {pool} "
                f"capacity (skipped {n} submission cycle{'s' if n > 1 else ''})"
            )

    @staticmethod
    def _throttled_cap(level: str, default_cap: int) -> int:
        """Translate a throttle level into a per-loop submission cap.

        ``halt-new`` is handled by the caller (skip entirely); ``slow``
        halves the default cap rounded down to >=1; ``none`` returns the
        unchanged default.
        """
        if level == "slow":
            return max(1, default_cap // 2)
        return default_cap

    def _assign_workers(self) -> None:
        """Assign idle workers to pending tasks. Priority: fix > analyze > implement."""
        # First: release experiments assigned to workers that are no longer busy
        self._release_dead_assignments()

        idle = [w for w in self.workers if not w.busy]
        if not idle:
            return

        # Priority 1: fix failed experiments (finished with error, no results, < max attempts)
        MAX_FIX_ATTEMPTS = 2
        finished = self.db.list_by_status("finished")
        fixable = [
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
            worker.fix(exp)

        # Priority 2: analyze finished experiments (with results or max fix attempts reached)
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
            worker.analyze(exp)

        # Priority 3: implement new experiments (include "implemented" stuck experiments)
        to_implement = self.db.list_by_status("to_implement", "implemented")
        unassigned_impl = [
            exp for exp in to_implement
            if exp.worker_id is None
            and not _is_externally_blocked(exp)
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
            worker.implement(exp)

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
        for status in ("to_implement", "implemented", "finished"):
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
        """Detect workers with no events for >10 minutes (observability only).

        One warning per stuck-event per worker. Without the
        ``_stuck_worker_warned`` set, the main loop would re-warn every
        POLL_INTERVAL seconds for the duration of the stuckness, spamming
        pipeline.jsonl + the operator's terminal. The set is cleared
        when the worker becomes idle or starts emitting events again,
        so a subsequent stuck event re-warns.
        """
        warned: set = getattr(self, "_stuck_worker_warned", set())
        now = time.time()
        # Threshold is config-driven; previous hardcoded 600s was below the
        # upper tail of slow LLM API calls (gpt-5.5 xhigh / opus max) and
        # triggered false alarms at a steady rate. Default 1200s in config.
        threshold = getattr(self.config, "stuck_worker_threshold_seconds", 1200)
        for w in self.workers:
            wid = w.worker_id
            if w.busy and w.last_event_at > 0 and now - w.last_event_at > threshold:
                if wid not in warned:
                    logger.warning(
                        f"Worker {wid} stuck: no events for "
                        f"{int(now - w.last_event_at)}s"
                    )
                    self._log(
                        "stuck_worker",
                        worker=wid,
                        seconds_since_event=int(now - w.last_event_at),
                    )
                    warned.add(wid)
            else:
                # Worker is making progress (or idle); allow future stuck
                # events to re-warn.
                warned.discard(wid)
        self._stuck_worker_warned = warned

    # ----------------------------------------------------------------
    # Conductor scheduling. Mirrors _should_run_strategist /
    # _run_strategist: at most one Conductor turn in flight, no self-
    # queueing if a turn takes longer than the timer interval. NOOP if
    # ``no_conductor`` is set on the config.
    # ----------------------------------------------------------------

    def _should_run_conductor(self) -> bool:
        if self.conductor is None:
            return False
        with self._state_lock:
            if self._conductor_running:
                return False
            # Edge trigger: a milestone report just finished (set by
            # _maybe_generate_report after the reporter worker completes).
            if self._milestone_just_finished:
                return True
            # First turn fires immediately (mirrors the strategist) so the
            # Conductor can read any pre-existing instructions/from_user.md
            # before anything else runs. After the first turn the slow timer
            # governs.
            if self._last_conductor_time == 0:
                return True
            now = time.time()
            return (now - self._last_conductor_time) >= self._conductor_interval

    def _maybe_run_conductor(self, trigger: str = "timer") -> None:
        """Spawn a Conductor turn in a background thread. Mirrors the
        strategist's pattern — non-blocking, single in-flight, never queues
        behind itself. ``trigger`` is the cause string passed through to
        the Conductor's initial-message lookup; the dispatcher's main loop
        passes ``"timer"`` while phase callers pass ``"phase0_done"`` etc.
        """
        if not self._should_run_conductor():
            return
        with self._state_lock:
            self._conductor_running = True
            self._milestone_just_finished = False  # consumed by this turn
            self._last_conductor_time = time.time()

        self._log("conductor_start", trigger=trigger)

        def _conductor_thread() -> None:
            try:
                if trigger == "phase0_done":
                    self.conductor.steer_phase0()
                elif trigger == "phase1_done":
                    self.conductor.steer_phase1()
                elif trigger == "phase2_done":
                    self.conductor.steer_phase2()
                elif trigger == "milestone":
                    self.conductor.steer_milestone()
                else:
                    self.conductor.steer_timer()
                self._log("conductor_done", trigger=trigger)
            except Exception as e:
                # Conductor crashes never halt the dispatcher. The Conductor
                # itself catches its own exceptions; this is belt-and-
                # suspenders for anything that escapes.
                logger.error(f"Conductor turn ({trigger}) crashed: {e}")
                self._log("conductor_error", trigger=trigger, error=str(e))
            finally:
                with self._state_lock:
                    self._conductor_running = False
                    self._conductor_thread = None

        t = threading.Thread(target=_conductor_thread, daemon=True)
        with self._state_lock:
            self._conductor_thread = t
        t.start()

    # -- Verifier scheduling (mirrors _maybe_run_conductor) ----------------------------
    def _build_verifier(self):
        """Construct a Verifier on the LIVE workspace from the dispatcher's own deps + the
        config's verifier_* knobs (3 role models, watchdog, caps). Lazy import avoids any import
        cycle (mirrors the lazy build_conductor import)."""
        import sys as _sys
        from alpha_lab.verifier import Verifier
        c = self.config
        nb = str(Path(__file__).resolve().parent.parent.parent / "scripts" / "nb_run.py")
        return Verifier(
            provider=self.provider, model=c.model,
            reasoning_effort=(getattr(c, "verifier_reasoning_effort", "") or c.reasoning_effort),
            config=c, workspace=self.workspace, data_path=c.data_path, adapter=self.adapter,
            event_callback=self.event_callback, nb_run_path=nb, python_exe=_sys.executable,
            max_candidates=int(getattr(c, "verifier_max_candidates", 4)),
            max_rounds=int(getattr(c, "verifier_max_rounds", 2)),
            notebook_timeout=int(getattr(c, "verifier_notebook_timeout", 1800)),
            watchdog_interval=int(getattr(c, "verifier_watchdog_interval", 0)),
            worker_model=getattr(c, "verifier_worker_model", "") or "",
            critic_model=getattr(c, "verifier_critic_model", "") or "",
            userrep_model=getattr(c, "verifier_userrep_model", "") or "",
        )

    def _should_run_verifier(self) -> bool:
        """One in flight. Fire on a Conductor verify-request marker, OR auto once per fresh batch of
        verify_after_n newly-analyzed strategies (re-armed +N after each run). NOOP without a conductor."""
        if self.conductor is None:
            return False
        with self._state_lock:
            if self._verifier_running:
                return False
        from alpha_lab.conductor_tools import VERIFY_REQUEST_MARKER
        from alpha_lab.meta_layout import meta_dir
        if (meta_dir(self.workspace) / VERIFY_REQUEST_MARKER).exists():
            return True
        if self._verify_after_n > 0:
            try:
                analyzed = len(self.db.list_by_status("analyzed", "done"))
            except Exception:
                return False
            # Auto-fire once per fresh batch of verify_after_n analyzed strategies — NOT on every
            # increment. The old `analyzed != ran_at` re-fired after EVERY completion once past the
            # threshold (experiments keep finishing while a verifier runs), so the expensive verifier
            # ran back-to-back indefinitely. ran_at = analyzed count at last spawn (-1 = never);
            # re-arm only after +verify_after_n more finish. A Conductor request (checked above) still
            # fires immediately and also updates ran_at, so auto won't pile on right after a manual run.
            if self._verifier_ran_at_analyzed < 0:
                return analyzed >= self._verify_after_n
            return analyzed >= self._verifier_ran_at_analyzed + self._verify_after_n
        return False

    def _maybe_run_verifier(self) -> None:
        """Spawn the verifier in a daemon thread, one in flight (mirrors _maybe_run_conductor).
        Consumes a Conductor request marker if present; otherwise this is the auto-trigger."""
        if not self._should_run_verifier():
            return
        from alpha_lab.conductor_tools import VERIFY_REQUEST_MARKER
        from alpha_lab.meta_layout import meta_dir
        marker = meta_dir(self.workspace) / VERIFY_REQUEST_MARKER
        trigger = "request" if marker.exists() else "auto"
        if trigger == "request":
            try:
                marker.unlink()
            except OSError:
                pass
        with self._state_lock:
            self._verifier_running = True
            try:
                self._verifier_ran_at_analyzed = len(self.db.list_by_status("analyzed", "done"))
            except Exception:
                pass
        self._log("verifier_start", trigger=trigger, at_analyzed=self._verifier_ran_at_analyzed)

        def _verifier_thread() -> None:
            try:
                v = self._build_verifier()
                with self._state_lock:
                    self._verifier_obj = v
                v.run()
                self._log("verifier_done")
            except Exception as e:
                logger.error("Verifier run crashed: %s", e)
                self._log("verifier_error", error=str(e))
            finally:
                with self._state_lock:
                    self._verifier_running = False
                    self._verifier_thread = None
                    self._verifier_obj = None

        t = threading.Thread(target=_verifier_thread, daemon=True)
        with self._state_lock:
            self._verifier_thread = t
        t.start()

    def _consume_phase_rewind_marker(self) -> None:
        """If the Conductor wrote a phase_rewind_pending.json, emit a phase
        event and request that the dispatcher loop exit so the outer
        pipeline driver (``run.py``) can execute the rewind.

        We DO NOT rotate the marker here — the outer driver needs to
        read it after we return. The dispatcher's own
        ``_stop_requested`` flag is set so the main loop unwinds on its
        next iteration. We also guard with ``_rewind_marker_seen`` so
        we only emit the event once per turn, even if the loop spins
        before the stop takes effect.
        """
        if self.conductor is None:
            return
        from alpha_lab.meta_layout import meta_dir
        from alpha_lab.conductor_tools import PHASE_REWIND_MARKER
        marker = meta_dir(self.workspace) / PHASE_REWIND_MARKER
        if not marker.exists():
            return
        # One-shot per marker: don't spam the log while waiting for the
        # main loop to honor stop_requested.
        if getattr(self, "_rewind_marker_seen", False):
            return
        try:
            payload = json.loads(marker.read_text())
        except (OSError, ValueError):
            return
        target_phase = payload.get("target_phase", "?")
        backup_dir = payload.get("backup_dir", "?")
        logger.warning(
            "Conductor requested phase rewind: target=%s backup=%s — "
            "requesting dispatcher stop so the outer driver can replay.",
            target_phase, backup_dir,
        )
        self._log(
            "phase_rewind_requested",
            target_phase=target_phase,
            backup_dir=backup_dir,
            reason=payload.get("reason", ""),
        )
        self._rewind_marker_seen = True
        # Tell the dispatcher's main loop to exit so run.py regains control.
        self._stop_requested = True

    def _consume_run_end_marker(self) -> None:
        """Honor a Conductor-requested graceful run end.

        Two-phase shutdown so in-flight experiments can finish cleanly:

          1. **On first sight of the marker** — flip ``self._drain_mode = True``
             (so ``_submit_checked`` refuses new submissions and the strategist
             stops being invoked for new proposals), emit a log event, and
             remember we've seen the marker (``_run_end_marker_seen``).

          2. **Once drain-mode is on, every iteration** — check whether any
             experiments remain in-flight (``to_implement``, ``implemented``,
             ``checked``, ``queued``, ``running``, ``finished``). If yes,
             keep polling and let the executors finish; if no, set
             ``self._stop_requested = True`` so the main loop exits cleanly
             on its next iteration's loop-head check.

        No-op when no Conductor is configured (the marker is never written),
        when the marker is missing on first call, or after the loop has
        already converged to ``_stop_requested``.
        """
        if self.conductor is None:
            return
        from alpha_lab.meta_layout import meta_dir
        from alpha_lab.conductor_tools import RUN_END_MARKER
        marker = meta_dir(self.workspace) / RUN_END_MARKER

        # Phase 1: first sight of marker → enter drain mode.
        if not getattr(self, "_run_end_marker_seen", False):
            if not marker.exists():
                return
            try:
                payload = json.loads(marker.read_text())
            except (OSError, ValueError):
                return
            reason = payload.get("reason", "")
            elapsed = payload.get("elapsed_hours_at_request", 0.0)
            analyzed = payload.get("analyzed_at_request", 0)
            logger.warning(
                "Conductor requested graceful run end: elapsed=%.1fh, "
                "analyzed=%d, reason=%s — entering drain mode (no new "
                "submissions; let in-flight experiments finish)",
                elapsed, analyzed, reason[:200],
            )
            self._log(
                "run_end_requested",
                elapsed_hours=elapsed,
                analyzed_count=analyzed,
                reason=reason,
                evidence=payload.get("evidence", ""),
            )
            self._run_end_marker_seen = True
            self._drain_mode = True
            return

        # Phase 2: drain-mode steady state. Pre-launch rows (proposed or
        # implemented but never queued) can NEVER progress in drain mode —
        # _submit_checked refuses new submissions — so counting them as
        # in-flight deadlocks the drain forever. Observed live: a granted
        # run-end idled 11h behind one never-launched 'checked' row while
        # the conductor noted the stall without acting. Park them instead:
        # they consumed budget and are deliberately retired by the drain.
        for exp in self.db.list_by_status("to_implement", "implemented",
                                          "checked"):
            if not exp.slurm_job_id:  # no executor job: purely pre-launch
                self.db.park(exp.id)
                logger.warning(
                    "Drain mode: parked pre-launch experiment #%d (%s) — "
                    "cannot run after run-end, must not block the drain",
                    exp.id, exp.name,
                )
                self._log("run_end_drain_parked", experiment_id=exp.id)
        in_flight = self.db.list_by_status(
            "to_implement", "implemented", "checked", "queued", "running",
            "cancelling", "finished",
        )
        if not in_flight:
            logger.warning(
                "Conductor-requested run end: drain complete (0 in-flight) — "
                "exiting dispatcher main loop"
            )
            self._log("run_end_drained")
            self._stop_requested = True

    def _consume_kill_requests(self) -> None:
        """Read meta/kill_requests.jsonl and cancel matching executor jobs.

        Each line is an experiment id the Conductor wants killed. The
        Conductor's tool already parked the row; here we additionally
        terminate any running subprocess via the executor.
        """
        if self.conductor is None:
            return
        from alpha_lab.meta_layout import meta_dir
        path = meta_dir(self.workspace) / "kill_requests.jsonl"
        if not path.exists():
            return
        try:
            lines = path.read_text().splitlines()
        except OSError:
            return
        if not lines:
            return
        unprocessed: list[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except ValueError:
                continue
            eid = req.get("experiment_id")
            if not isinstance(eid, int):
                continue
            exp = self.db.get(eid)
            if exp is None or not exp.slurm_job_id:
                # Nothing to cancel at the executor level — parking already
                # happened in the tool implementation.
                continue
            try:
                # Try GPU executor first, then CPU. Either may not have the
                # job; both raise on unknown ids and we swallow.
                if exp.slurm_job_id in getattr(self, "_cpu_job_ids", set()):
                    if self.cpu_executor is not None:
                        self.cpu_executor.cancel(exp.slurm_job_id)
                else:
                    self.executor.cancel(exp.slurm_job_id)
                self._log(
                    "conductor_killed_experiment",
                    experiment_id=eid,
                    job_id=exp.slurm_job_id,
                )
            except Exception as e:
                logger.warning("Failed to cancel kill-requested job: %s", e)
        # Truncate the file — all requests processed (the meta_log already
        # captured them via the tool dispatch).
        try:
            path.write_text("")
        except OSError as e:
            logger.warning("Failed to truncate kill_requests.jsonl: %s", e)

    def _maybe_supervisor_check(self) -> None:
        """Run supervisor health check if error rate exceeds 40%.

        Checked once per multiple-of-10 analyzed-count. Without the
        ``_last_supervisor_check_count`` guard, the count == 10 / 20 /
        ... condition fires on every dispatcher main-loop iteration
        until the next experiment finishes — spawning multiple
        supervisor turns for the same checkpoint and burning tokens.
        If the supervisor patches the adapter, reload it.
        """
        if self.supervisor is None:
            return
        # NOOP fallback: skip the entire health-check trigger when
        # no_supervisor=True. Mirrors no_conductor — avoids logging the
        # supervisor_health_check trigger event when the supervisor is
        # disabled and would have been a no-op anyway.
        if self.config.pipeline.phase3.no_supervisor:
            return

        analyzed = self.db.list_by_status("analyzed", "done")
        n = len(analyzed)
        if n < 10 or n % 10 != 0:
            return
        # One supervisor check per checkpoint. ``_last_supervisor_check_count``
        # is initialized lazily so legacy dispatcher instances (or tests
        # that don't go through the full __init__) still work.
        last = getattr(self, "_last_supervisor_check_count", -1)
        if n == last:
            return
        self._last_supervisor_check_count = n

        # A populated results document proves execution completed; an error on
        # that row is a scientific rejection note, not a runtime failure.
        errors = [e for e in analyzed if is_execution_failure(e.error, e.results_json)]
        error_rate = len(errors) / max(len(analyzed), 1)

        if error_rate <= 0.4:
            return

        logger.warning(
            f"Error rate {error_rate:.0%} exceeds 40% — running supervisor health check"
        )
        self._log("supervisor_health_check", error_rate=error_rate)

        try:
            self.supervisor.phase3_health_check()
            # Reload adapter in case supervisor patched it
            from alpha_lab.adapter_loader import resolve_adapter
            new_adapter = resolve_adapter(self.workspace)
            self.adapter = new_adapter
            # Update workers, strategist, AND conductor. Missing the
            # conductor here means the next Conductor turn would use
            # cached old-adapter prompts (its prompt_builder reads
            # ``self.conductor.adapter.prompts["phase3_conductor"]``).
            for w in self.workers:
                w.adapter = new_adapter
            self.strategist.adapter = new_adapter
            if self.conductor is not None:
                self.conductor.adapter = new_adapter
        except Exception as e:
            logger.error(f"Supervisor health check failed: {e}")
            self._log("supervisor_error", error=str(e))

    def _emit_board_summary(self) -> None:
        """Emit a board summary event."""
        summary = self.db.board_summary()
        recent = self.db.list_all()[-10:]
        _metric = self._convergence_metric
        leaders = self.db.leaderboard(_metric, 5, direction=self._metric_direction)

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
            # Reporter thread finished. Before treating the milestone as done,
            # verify it actually produced its report.md. The reporter is
            # fire-once: unlike workers (which the dispatcher relaunches), a
            # transient gateway error mid-flight — e.g. a Bedrock 500 storm —
            # kills it with nothing to retry. If we let the number advance
            # anyway it is permanently burned, leaving a gap in the milestone
            # sequence that _scan_existing_milestones cannot distinguish from a
            # real report. Instead, roll the counter and baseline back so the
            # next cycle re-fires the SAME number and re-attempts the report.
            report_md = (
                Path(self.workspace) / "reports"
                / f"milestone_{self._current_report_number:03d}" / "report.md"
            )
            if not report_md.exists():
                logger.warning(
                    "Milestone report #%d produced no report.md (reporter "
                    "likely crashed mid-flight); rolling the counter back so "
                    "the milestone re-fires with the same number rather than "
                    "burning it",
                    self._current_report_number,
                )
                self._report_number -= 1
                self._last_report_at_done_count = self._prev_report_at_done_count
                self._report_in_progress = False
                self._report_worker = None
                # Deliberately do NOT set _milestone_just_finished: no milestone
                # was produced, so there is nothing for the Conductor to react to.
                return

            # Reporter succeeded — copy to output/
            try:
                from alpha_lab.output_generator import OutputGenerator
                gen = OutputGenerator(self.workspace, adapter=self.adapter)
                gen.copy_milestone_report(self._current_report_number)
                gen.generate_index()
            except Exception as e:
                logger.error(f"Output copy failed: {e}")
            self._report_in_progress = False
            self._report_worker = None
            # Edge-trigger the Conductor: a milestone report has just been
            # written. The next iteration of the dispatcher loop calls
            # _maybe_run_conductor("milestone") which sees this flag and
            # spawns a turn.
            with self._state_lock:
                self._milestone_just_finished = True
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

            worker.generate_report(self._report_number, done_count)
            # Remember the baseline so a crashed reporter can restore it.
            self._prev_report_at_done_count = self._last_report_at_done_count
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
            # Refresh the live-snapshot section of research_state.md so other
            # agents reading it between milestones see fresh leaderboard /
            # board counts instead of stale Reporter consolidations. This is
            # the code-only feedback channel; the Reporter still does the
            # narrative consolidation at milestone time.
            try:
                self._refresh_research_state_snapshot()
            except Exception as e:  # pragma: no cover — defensive
                logger.warning("research_state snapshot refresh failed: %s", e)
            # Refresh the aggregated token-usage summary at the same cadence —
            # cheaper than re-aggregating on every API call, fresh enough that
            # the user can `cat meta/token_usage_summary.jsonl` at any time.
            try:
                from alpha_lab.meta_layout import refresh_token_usage_summary
                refresh_token_usage_summary(self.workspace)
            except Exception as e:  # pragma: no cover — defensive
                logger.warning("token_usage_summary refresh failed: %s", e)

            # Check for improvement in best metric
            leaders = self.db.leaderboard(
                self._convergence_metric, 1, direction=self._metric_direction
            )
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
                        # Allow convergence to re-warn after the next plateau.
                        self._convergence_logged = False
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

    # Sentinels marking the auto-managed live-snapshot block inside
    # research_state.md. The Reporter's milestone-time narrative lives
    # OUTSIDE this block and is untouched by the dispatcher's refresh.
    _RS_BEGIN = "<!-- LIVE-SNAPSHOT-BEGIN (auto-managed by dispatcher; do not edit by hand) -->"
    _RS_END = "<!-- LIVE-SNAPSHOT-END -->"

    def _refresh_research_state_snapshot(self) -> None:
        """Rewrite the live-snapshot block of ``research_state.md`` from
        current DB state.

        Code-only — no LLM, no narrative. The Reporter owns the rest of the
        file (mechanism coverage, dead ends, sense of progress) and writes
        it at milestone boundaries; this method updates a small structural
        block delimited by HTML comment sentinels so the two writers don't
        step on each other.

        Idempotent — running it on every analyzed transition is fine; the
        block content is fully derived from DB state.

        Atomic write: writes to a tmp file then renames over the original
        so concurrent readers either see the previous content or the new
        content, never a torn intermediate.
        """
        path = Path(self.workspace) / "research_state.md"
        # Build the live-snapshot block from DB state.
        summary = self.db.board_summary()
        try:
            leaders = self.db.leaderboard(
                self._convergence_metric, 10, direction=self._metric_direction
            )
        except Exception:
            leaders = []
        analyzed_count = summary.get("analyzed", 0) + summary.get("done", 0)
        metric_name = self._convergence_metric
        direction = self._metric_direction

        lines: list[str] = []
        lines.append(self._RS_BEGIN)
        lines.append("")
        lines.append("## Live snapshot (auto-updated after every analyzed experiment)")
        lines.append("")
        lines.append(
            f"This block is rewritten by the dispatcher each time an experiment "
            f"transitions to `analyzed`. Code-only, no narrative. The Reporter's "
            f"milestone synthesis lives below this block and is updated less "
            f"frequently."
        )
        lines.append("")
        lines.append(f"- Metric tracked: `{metric_name}` ({direction})")
        lines.append(f"- Experiments analyzed (analyzed + done): {analyzed_count}")
        lines.append("- Board state:")
        for col, cnt in sorted(summary.items()):
            lines.append(f"  - {col}: {cnt}")
        lines.append("")
        if leaders:
            lines.append(f"### Current leaderboard (top {len(leaders)} by {metric_name})")
            lines.append("")
            for i, exp in enumerate(leaders, 1):
                try:
                    m = json.loads(exp.results_json or "{}")
                    val = m.get(metric_name, "?") if isinstance(m, dict) else "?"
                except (json.JSONDecodeError, TypeError):
                    val = "?"
                lines.append(f"{i}. #{exp.id} `{exp.name[:60]}` — {metric_name}={val}")
            lines.append("")
        # Most-recent analyzed experiments — keep readers fresh between milestones.
        try:
            recent = sorted(
                self.db.list_by_status("analyzed", "done"),
                key=lambda e: e.updated_at,
                reverse=True,
            )[:10]
        except Exception:
            recent = []
        if recent:
            lines.append(f"### Most recently analyzed (last {len(recent)})")
            lines.append("")
            for exp in recent:
                pid = f" (variant of #{exp.parent_id})" if exp.parent_id else ""
                lines.append(f"- #{exp.id} `{exp.name[:60]}`{pid}")
            lines.append("")
        lines.append(self._RS_END)
        new_block = "\n".join(lines) + "\n"

        # Merge with existing file content. The Reporter-owned narrative
        # lives outside the sentinel block; preserve it verbatim.
        if path.exists():
            try:
                existing = path.read_text()
            except OSError:
                existing = ""
        else:
            existing = ""

        if self._RS_BEGIN in existing and self._RS_END in existing:
            # Replace the existing block in place.
            before, _, rest = existing.partition(self._RS_BEGIN)
            _, _, after = rest.partition(self._RS_END)
            merged = before.rstrip() + "\n\n" + new_block + after.lstrip()
        else:
            # No block yet — prepend the snapshot to whatever exists.
            # The Reporter's standard header (if present) survives below.
            merged = new_block + ("\n" + existing if existing else "")

        # Atomic write: tmp + rename. Concurrent readers see either the old
        # file or the new one, never a torn intermediate. Concurrent
        # writers (the Reporter, an analyzer) are still possible — they
        # use the same atomic pattern in conductor_tools._atomic_write_text,
        # but a sequence of writes can race on what each reads-then-writes.
        # That's acceptable here: this block is fully DB-derived and
        # idempotent, so any racing writer can recover by re-running.
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp.write_text(merged)
            tmp.replace(path)
        except OSError as e:
            logger.warning("research_state.md atomic write failed: %s", e)

    def _check_convergence(self) -> bool:
        """Check if we've converged (no improvement for N experiments)."""
        if self._convergence_threshold <= 0:
            return False
        if self._experiments_since_improvement >= self._convergence_threshold:
            # Log/emit the event ONCE per crossing of the threshold. Without
            # this guard, the dispatcher's main loop calls this every
            # POLL_INTERVAL seconds and spams the log — workspace_etfflow_g55
            # accumulated 14_762 identical convergence events over ~41 hours.
            # Reset the flag when a new best lands (see _update_best_metric).
            if getattr(self, "_convergence_logged", False):
                return True
            logger.info(
                f"Convergence detected: no improvement in {self._convergence_metric} "
                f"for {self._experiments_since_improvement} experiments"
            )
            self._log("convergence",
                      experiments_since_improvement=self._experiments_since_improvement,
                      best_value=self._best_metric_value)
            self._convergence_logged = True
            return True
        return False

    def _should_terminate(self) -> bool:
        """Check whether the lifetime experiment budget is exhausted."""
        completed = [
            exp for exp in self.db.list_all()
            if exp.status in ("analyzed", "done")
            or (exp.parked_at is not None and exp.status != "cancelling")
        ]

        # Convergence is logged for observability but does not terminate the run.
        # The lifetime cap covers completed and deliberately retired attempts.
        self._check_convergence()  # logs convergence state; result intentionally ignored

        # Parked experiments consumed budget but were deliberately retired.
        if len(completed) < self._max_experiments:
            return False
        # A milestone reporter may have started in this same loop iteration.
        # Let it finish before the normal max-experiments shutdown.
        if self._report_in_progress:
            return False
        if self.db.list_by_status("cancelling", include_parked=True):
            return False
        # Don't terminate while experiments are still in progress
        in_flight = self.db.list_by_status(
            "to_implement", "implemented", "checked", "queued", "running",
            "cancelling", "finished"
        )
        if in_flight:
            return False
        return True
