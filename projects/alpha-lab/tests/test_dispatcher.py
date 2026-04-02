"""Tests for the Dispatcher orchestration logic.

All LLM calls and SLURM interactions are mocked.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alpha_lab.config import Phase3Config, PipelineConfig, TaskConfig
from alpha_lab.dispatcher import Dispatcher
from alpha_lab.events import AgentEvent
from alpha_lab.experiment_db import ExperimentDB
from alpha_lab.slurm import SlurmManager


@pytest.fixture()
def config() -> TaskConfig:
    return TaskConfig(
        data_path="/data/test.csv",
        description="Test task",
        pipeline=PipelineConfig(
            phases=["phase3"],
            phase3=Phase3Config(
                max_concurrent_gpus=4,
                max_experiments=10,
                strategist_interval=300,
                worker_count=2,
                slurm_partitions=["h100"],
                report_interval=100,  # effectively disabled for unit tests
            ),
        ),
    )


@pytest.fixture()
def mock_slurm() -> MagicMock:
    slurm = MagicMock(spec=SlurmManager)
    slurm.can_submit.return_value = True
    slurm.submit_experiment.return_value = "12345"
    slurm.poll_jobs.return_value = {}
    return slurm


@pytest.fixture()
def events() -> list[AgentEvent]:
    return []


@pytest.fixture()
def dispatcher(
    config: TaskConfig,
    db: ExperimentDB,
    mock_slurm: MagicMock,
    events: list[AgentEvent],
    tmp_workspace: str,
) -> Dispatcher:
    provider = MagicMock()
    return Dispatcher(
        provider=provider,
        config=config,
        workspace=tmp_workspace,
        db=db,
        executor=mock_slurm,
        event_callback=lambda e: events.append(e),
        worker_count=2,
    )


class TestDispatcherStrategistTrigger:
    def test_first_turn_immediately(self, dispatcher: Dispatcher) -> None:
        assert dispatcher._should_run_strategist() is True

    def test_not_while_running(self, dispatcher: Dispatcher) -> None:
        dispatcher._strategist_running = True
        assert dispatcher._should_run_strategist() is False

    def test_after_enough_analyzed(self, dispatcher: Dispatcher) -> None:
        dispatcher._last_strategist_time = time.time()  # Not the first turn
        dispatcher._analyzed_since_strategist = 3
        assert dispatcher._should_run_strategist() is True

    def test_periodic_trigger(self, dispatcher: Dispatcher) -> None:
        dispatcher._last_strategist_time = time.time() - 600  # 10 min ago
        dispatcher._analyzed_since_strategist = 0
        assert dispatcher._should_run_strategist() is True

    def test_empty_queue_with_idle_workers(self, dispatcher: Dispatcher, db: ExperimentDB) -> None:
        """If to_implement is empty and workers are idle, trigger after 60s."""
        dispatcher._last_strategist_time = time.time() - 120  # 2 min ago
        dispatcher._analyzed_since_strategist = 0
        # No to_implement experiments in DB
        assert dispatcher._should_run_strategist() is True


class TestDispatcherSubmitChecked:
    def test_submits_checked_experiments(
        self, dispatcher: Dispatcher, db: ExperimentDB, mock_slurm: MagicMock,
    ) -> None:
        exp_id = db.create("sub_exp", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        db.update_status(exp_id, "checked")

        dispatcher._submit_checked()

        mock_slurm.submit_experiment.assert_called_once()
        exp = db.get(exp_id)
        assert exp.status == "queued"
        assert exp.slurm_job_id == "12345"

    def test_respects_gpu_budget(
        self, dispatcher: Dispatcher, db: ExperimentDB, mock_slurm: MagicMock,
    ) -> None:
        # Create 2 checked experiments
        for i in range(2):
            eid = db.create(f"budget_exp_{i}", "D", "H", "{}")
            db.update_status(eid, "implemented")
            db.update_status(eid, "checked")

        # First call: can_submit=True, second: False
        mock_slurm.can_submit.side_effect = [True, False]
        dispatcher._submit_checked()

        assert mock_slurm.submit_experiment.call_count == 1

    def test_handles_submit_failure(
        self, dispatcher: Dispatcher, db: ExperimentDB, mock_slurm: MagicMock,
    ) -> None:
        exp_id = db.create("fail_sub", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        db.update_status(exp_id, "checked")

        mock_slurm.submit_experiment.side_effect = RuntimeError("sbatch failed")
        dispatcher._submit_checked()

        exp = db.get(exp_id)
        assert exp.status == "finished"  # moved to finished (with error)
        assert exp.error is not None
        assert "Submit failed" in exp.error


class TestDispatcherPollSlurm:
    def test_queued_to_running(
        self, dispatcher: Dispatcher, db: ExperimentDB, mock_slurm: MagicMock,
    ) -> None:
        exp_id = db.create("poll_exp", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        db.update_status(exp_id, "checked")
        db.update_status(exp_id, "queued")
        db.set_slurm_job(exp_id, "11111")

        mock_slurm.poll_jobs.return_value = {"11111": "RUNNING"}
        dispatcher._poll_slurm()

        exp = db.get(exp_id)
        assert exp.status == "running"
        assert exp.started_at is not None

    def test_running_to_finished(
        self, dispatcher: Dispatcher, db: ExperimentDB, mock_slurm: MagicMock,
    ) -> None:
        exp_id = db.create("done_exp", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        db.update_status(exp_id, "checked")
        db.update_status(exp_id, "queued")
        db.set_slurm_job(exp_id, "22222")
        db.update_status(exp_id, "running", started_at=1000.0)

        mock_slurm.poll_jobs.return_value = {"22222": "COMPLETED"}
        dispatcher._poll_slurm()

        exp = db.get(exp_id)
        assert exp.status == "finished"
        assert exp.finished_at is not None

    def test_slurm_failure_sets_error(
        self, dispatcher: Dispatcher, db: ExperimentDB, mock_slurm: MagicMock,
    ) -> None:
        exp_id = db.create("fail_exp", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        db.update_status(exp_id, "checked")
        db.update_status(exp_id, "queued")
        db.set_slurm_job(exp_id, "33333")

        mock_slurm.poll_jobs.return_value = {"33333": "FAILED"}
        dispatcher._poll_slurm()

        exp = db.get(exp_id)
        assert exp.status == "finished"
        assert "SLURM FAILED" in exp.error


class TestDispatcherAssignWorkers:
    def test_prioritizes_analyze_over_implement(
        self, dispatcher: Dispatcher, db: ExperimentDB,
    ) -> None:
        # One finished, one to_implement
        fin_id = db.create("fin_exp", "D", "H", "{}")
        db.update_status(fin_id, "implemented")
        db.update_status(fin_id, "checked")
        db.update_status(fin_id, "queued")
        db.update_status(fin_id, "running", started_at=1000.0)
        db.update_status(fin_id, "finished", finished_at=2000.0)

        impl_id = db.create("impl_exp", "D", "H", "{}")

        # Mock workers: both idle
        for w in dispatcher.workers:
            w._thread = None

        # Patch worker methods to track calls
        analyze_calls = []
        implement_calls = []
        for w in dispatcher.workers:
            w.analyze = lambda exp, w=w: analyze_calls.append(exp.id)
            w.implement = lambda exp, w=w: implement_calls.append(exp.id)

        dispatcher._assign_workers()

        # Analyze should be first priority
        assert fin_id in analyze_calls
        assert impl_id in implement_calls

    def test_picks_up_implemented_experiments(
        self, dispatcher: Dispatcher, db: ExperimentDB,
    ) -> None:
        """Experiments stuck at 'implemented' should also be picked up."""
        exp_id = db.create("stuck_exp", "D", "H", "{}")
        db.update_status(exp_id, "implemented")

        for w in dispatcher.workers:
            w._thread = None

        implement_calls = []
        for w in dispatcher.workers:
            w.implement = lambda exp, w=w: implement_calls.append(exp.id)
            w.analyze = lambda exp, w=w: None

        dispatcher._assign_workers()

        assert exp_id in implement_calls


class TestDispatcherTermination:
    def test_terminates_when_all_done(
        self, dispatcher: Dispatcher, db: ExperimentDB,
    ) -> None:
        # Create max_experiments worth of "done" experiments
        for i in range(10):
            eid = db.create(f"term_exp_{i}", "D", "H", "{}")
            db.update_status(eid, "implemented")
            db.update_status(eid, "checked")
            db.update_status(eid, "queued")
            db.update_status(eid, "running", started_at=1000.0)
            db.update_status(eid, "finished", finished_at=2000.0)
            db.update_status(eid, "analyzed")
            db.update_status(eid, "done")

        assert dispatcher._should_terminate() is True

    def test_does_not_terminate_with_in_flight(
        self, dispatcher: Dispatcher, db: ExperimentDB,
    ) -> None:
        # 10 done + 1 still running
        for i in range(10):
            eid = db.create(f"term2_exp_{i}", "D", "H", "{}")
            db.update_status(eid, "implemented")
            db.update_status(eid, "checked")
            db.update_status(eid, "queued")
            db.update_status(eid, "running", started_at=1000.0)
            db.update_status(eid, "finished", finished_at=2000.0)
            db.update_status(eid, "analyzed")
            db.update_status(eid, "done")

        eid = db.create("inflight_exp", "D", "H", "{}")
        db.update_status(eid, "implemented")
        db.update_status(eid, "checked")
        db.update_status(eid, "queued")
        db.update_status(eid, "running", started_at=3000.0)

        assert dispatcher._should_terminate() is False

    def test_does_not_terminate_under_cap(
        self, dispatcher: Dispatcher, db: ExperimentDB,
    ) -> None:
        # Only 5 done, cap is 10
        for i in range(5):
            eid = db.create(f"cap_exp_{i}", "D", "H", "{}")
            db.update_status(eid, "implemented")
            db.update_status(eid, "checked")
            db.update_status(eid, "queued")
            db.update_status(eid, "running", started_at=1000.0)
            db.update_status(eid, "finished", finished_at=2000.0)
            db.update_status(eid, "analyzed")

        assert dispatcher._should_terminate() is False


class TestDispatcherStaleWorkers:
    def test_stale_detection(
        self, dispatcher: Dispatcher, db: ExperimentDB,
    ) -> None:
        import sqlite3

        exp_id = db.create("stale_exp", "D", "H", "{}")
        db.assign_worker(exp_id, "worker_0")
        # Force old timestamp
        conn = sqlite3.connect(db.db_path, timeout=10)
        conn.execute(
            "UPDATE experiments SET updated_at = ? WHERE id = ?",
            (time.time() - 600, exp_id),
        )
        conn.commit()
        conn.close()

        dispatcher._check_stale()

        exp = db.get(exp_id)
        assert exp.worker_id is None  # Released


class TestDispatcherBoardSummary:
    def test_emits_board_summary(
        self, dispatcher: Dispatcher, db: ExperimentDB, events: list[AgentEvent],
    ) -> None:
        db.create("board_exp", "D", "H", "{}")
        dispatcher._emit_board_summary()

        from alpha_lab.events import BoardSummaryEvent
        board_events = [e for e in events if isinstance(e, BoardSummaryEvent)]
        assert len(board_events) == 1
        assert "to_implement" in board_events[0].counts
