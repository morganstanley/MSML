"""Integration tests for the Dispatcher with real DB and fake SLURM."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alpha_lab.config import Phase3Config, PipelineConfig, TaskConfig
from alpha_lab.dispatcher import Dispatcher
from alpha_lab.events import AgentEvent
from alpha_lab.experiment_db import ExperimentDB
from alpha_lab.slurm import SlurmManager


class FakeSlurmManager(SlurmManager):
    """In-memory SLURM simulator — no subprocess calls."""

    def __init__(self) -> None:
        super().__init__(partitions=["fake"], gpu_per_job=1, max_gpus=8)
        self._jobs: dict[str, str] = {}  # job_id -> status
        self._next_id = 1000

    def submit(self, script_path: str) -> str:
        job_id = str(self._next_id)
        self._next_id += 1
        self._jobs[job_id] = "PENDING"
        return job_id

    def submit_experiment(self, exp, workspace: str) -> str:
        return self.submit("")

    def poll_jobs(self, job_ids: list[str]) -> dict[str, str]:
        return {jid: self._jobs.get(jid, "UNKNOWN") for jid in job_ids}

    def running_gpu_count(self) -> int:
        return sum(1 for s in self._jobs.values() if s in ("RUNNING", "PENDING"))

    def can_submit(self) -> bool:
        return self.running_gpu_count() + self.gpu_per_job <= self.max_gpus

    def advance_job(self, job_id: str, status: str) -> None:
        self._jobs[job_id] = status

    def cancel(self, job_id: str) -> None:
        self._jobs[job_id] = "CANCELLED"


def _make_config() -> TaskConfig:
    return TaskConfig(
        data_path="/tmp/fake_data",
        description="Test task",
        pipeline=PipelineConfig(
            phases=["phase3"],
            phase3=Phase3Config(
                max_experiments=100,
                worker_count=2,
                strategist_interval=9999,  # don't auto-trigger
                report_interval=9999,
            ),
        ),
    )


def _make_dispatcher(
    db: ExperimentDB,
    slurm: FakeSlurmManager,
    workspace: str,
    worker_count: int = 2,
) -> Dispatcher:
    events: list[AgentEvent] = []
    config = _make_config()
    provider = MagicMock()
    d = Dispatcher(
        provider=provider,
        config=config,
        workspace=workspace,
        db=db,
        executor=slurm,
        event_callback=lambda e: events.append(e),
        worker_count=worker_count,
    )
    d._events = events  # stash for assertions
    return d


class TestExperimentLifecycle:
    """Full kanban flow with patched worker functions and real DB."""

    def test_experiment_lifecycle(self, tmp_workspace: str, db: ExperimentDB) -> None:
        slurm = FakeSlurmManager()
        d = _make_dispatcher(db, slurm, tmp_workspace)
        d._init_log()

        # Create experiment
        eid = db.create("lifecycle_exp", "Test lifecycle", "It works", "{}")
        assert db.get(eid).status == "to_implement"

        # Simulate implement: manually transition to checked
        db.update_status(eid, "implemented")
        db.update_status(eid, "checked")

        # Submit to SLURM
        d._submit_checked()
        exp = db.get(eid)
        assert exp.status == "queued"
        assert exp.slurm_job_id is not None
        job_id = exp.slurm_job_id

        # Advance SLURM to RUNNING
        slurm.advance_job(job_id, "RUNNING")
        d._poll_slurm()
        assert db.get(eid).status == "running"

        # Advance SLURM to COMPLETED
        slurm.advance_job(job_id, "COMPLETED")
        d._poll_slurm()
        assert db.get(eid).status == "finished"


class TestRecoverStaleWorkerAssignments:
    def test_recover_releases_orphaned(self, tmp_workspace: str, db: ExperimentDB) -> None:
        slurm = FakeSlurmManager()
        d = _make_dispatcher(db, slurm, tmp_workspace)

        # Create experiments with orphaned worker assignments
        eid1 = db.create("orphan_a", "D", "H", "{}")
        db.assign_worker(eid1, "dead_worker_0")

        eid2 = db.create("orphan_b", "D", "H", "{}")
        db.update_status(eid2, "implemented")
        db.assign_worker(eid2, "dead_worker_1")

        eid3 = db.create("orphan_c", "D", "H", "{}")
        db.update_status(eid3, "implemented")
        db.update_status(eid3, "checked")
        db.update_status(eid3, "queued")
        db.update_status(eid3, "running", started_at=time.time())
        db.update_status(eid3, "finished", finished_at=time.time())
        db.assign_worker(eid3, "dead_worker_2")

        summary = d.recover()
        assert summary["released_workers"] == 3

        # Verify all workers released
        assert db.get(eid1).worker_id is None
        assert db.get(eid2).worker_id is None
        assert db.get(eid3).worker_id is None


class TestRecoverSlurmCompletedJobs:
    def test_slurm_completed_during_downtime(self, tmp_workspace: str, db: ExperimentDB) -> None:
        slurm = FakeSlurmManager()
        d = _make_dispatcher(db, slurm, tmp_workspace)

        eid = db.create("slurm_done", "D", "H", "{}")
        db.update_status(eid, "implemented")
        db.update_status(eid, "checked")
        db.update_status(eid, "queued")
        db.set_slurm_job(eid, "9001")

        # SLURM completed during downtime
        slurm._jobs["9001"] = "COMPLETED"

        summary = d.recover()
        assert summary["slurm_reconciled"] >= 1
        assert db.get(eid).status == "finished"


class TestRecoverSlurmFailedJobs:
    def test_slurm_failed_during_downtime(self, tmp_workspace: str, db: ExperimentDB) -> None:
        slurm = FakeSlurmManager()
        d = _make_dispatcher(db, slurm, tmp_workspace)

        eid = db.create("slurm_fail", "D", "H", "{}")
        db.update_status(eid, "implemented")
        db.update_status(eid, "checked")
        db.update_status(eid, "queued")
        db.set_slurm_job(eid, "9002")

        slurm._jobs["9002"] = "FAILED"

        summary = d.recover()
        assert summary["slurm_reconciled"] >= 1
        exp = db.get(eid)
        assert exp.status == "finished"
        assert "SLURM FAILED" in exp.error


class TestRecoverSlurmRunningJobs:
    def test_slurm_running_reconciles(self, tmp_workspace: str, db: ExperimentDB) -> None:
        slurm = FakeSlurmManager()
        d = _make_dispatcher(db, slurm, tmp_workspace)

        eid = db.create("slurm_run", "D", "H", "{}")
        db.update_status(eid, "implemented")
        db.update_status(eid, "checked")
        db.update_status(eid, "queued")
        db.set_slurm_job(eid, "9003")

        slurm._jobs["9003"] = "RUNNING"

        summary = d.recover()
        assert summary["slurm_reconciled"] >= 1
        assert db.get(eid).status == "running"


class TestRecoverSlurmPendingJobs:
    def test_slurm_pending_stays_queued(self, tmp_workspace: str, db: ExperimentDB) -> None:
        slurm = FakeSlurmManager()
        d = _make_dispatcher(db, slurm, tmp_workspace)

        eid = db.create("slurm_pend", "D", "H", "{}")
        db.update_status(eid, "implemented")
        db.update_status(eid, "checked")
        db.update_status(eid, "queued")
        db.set_slurm_job(eid, "9004")

        slurm._jobs["9004"] = "PENDING"

        d.recover()
        assert db.get(eid).status == "queued"


class TestGracefulShutdown:
    def test_stop_joins_worker_threads(self, tmp_workspace: str, db: ExperimentDB) -> None:
        slurm = FakeSlurmManager()
        d = _make_dispatcher(db, slurm, tmp_workspace)

        # Simulate a busy worker with a short-lived thread
        done = threading.Event()

        def fake_task() -> None:
            done.wait(timeout=5)

        w = d.workers[0]
        w._thread = threading.Thread(target=fake_task, daemon=False)
        w._thread.start()

        # Signal the thread to finish, then stop dispatcher
        done.set()
        d.stop(join_timeout=5)

        # Worker thread should have been joined
        assert not w._thread.is_alive()


class TestCleanupReleasesAssignments:
    def test_cleanup_releases_all(self, tmp_workspace: str, db: ExperimentDB) -> None:
        slurm = FakeSlurmManager()
        d = _make_dispatcher(db, slurm, tmp_workspace)

        # Create experiments with worker assignments
        eid1 = db.create("cleanup_a", "D", "H", "{}")
        db.assign_worker(eid1, "worker_0")

        eid2 = db.create("cleanup_b", "D", "H", "{}")
        db.update_status(eid2, "implemented")
        db.assign_worker(eid2, "worker_1")

        d._cleanup()

        assert db.get(eid1).worker_id is None
        assert db.get(eid2).worker_id is None
