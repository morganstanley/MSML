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


class TestDispatcherZombiePromotion:
    def test_promotes_checked_row_without_error(
        self, dispatcher: Dispatcher, db: ExperimentDB, tmp_workspace: str,
    ) -> None:
        exp_id = db.create("recoverable", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        db.update_status(exp_id, "checked")
        results_dir = Path(tmp_workspace) / "experiments" / "recoverable" / "results"
        results_dir.mkdir(parents=True)
        (results_dir / "metrics.json").write_text('{"sharpe": 1.5}')

        dispatcher._auto_promote_zombie_rows()

        assert db.get(exp_id).status == "finished"

    def test_does_not_override_quarantined_checked_row(
        self, dispatcher: Dispatcher, db: ExperimentDB, tmp_workspace: str,
    ) -> None:
        exp_id = db.create("quarantined", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        db.update_status(exp_id, "checked")
        db.set_error(exp_id, "quarantined_invalid_split")
        results_dir = Path(tmp_workspace) / "experiments" / "quarantined" / "results"
        results_dir.mkdir(parents=True)
        (results_dir / "metrics.json").write_text('{"sharpe": 1.5}')

        dispatcher._auto_promote_zombie_rows()

        assert db.get(exp_id).status == "checked"


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
        # Error label reflects the actual executor: SLURM when slurm_partitions
        # are configured, "job" for local. Default config in this test is
        # executor=local, so we get "job FAILED".
        assert "FAILED" in exp.error


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

    def test_skips_blocked_implemented_experiments(
        self, dispatcher: Dispatcher, db: ExperimentDB,
    ) -> None:
        """Rows at 'implemented' with error='blocked: ...' must NOT be
        re-assigned to implement workers — that's the loop pattern the
        prefix is designed to break.
        """
        ok_id = db.create("ok_exp", "D", "H", "{}")
        db.update_status(ok_id, "implemented")

        blocked_id = db.create("blocked_exp", "D", "H", "{}")
        db.update_status(blocked_id, "implemented")
        db.set_error(blocked_id, "blocked: upstream source predictions missing")

        for w in dispatcher.workers:
            w._thread = None

        implement_calls = []
        for w in dispatcher.workers:
            w.implement = lambda exp, w=w: implement_calls.append(exp.id)
            w.analyze = lambda exp, w=w: None

        dispatcher._assign_workers()

        assert ok_id in implement_calls
        assert blocked_id not in implement_calls

    def test_blocked_prefix_is_exact(
        self, dispatcher: Dispatcher, db: ExperimentDB,
    ) -> None:
        """Only error messages starting with 'blocked:' suppress assignment.
        Generic transient errors must still be retried.
        """
        flaky_id = db.create("flaky_exp", "D", "H", "{}")
        db.update_status(flaky_id, "implemented")
        db.set_error(flaky_id, "Traceback: ImportError on flaky package")

        for w in dispatcher.workers:
            w._thread = None
        implement_calls = []
        for w in dispatcher.workers:
            w.implement = lambda exp, w=w: implement_calls.append(exp.id)
            w.analyze = lambda exp, w=w: None

        dispatcher._assign_workers()

        # Non-"blocked:" errors do NOT suppress assignment.
        assert flaky_id in implement_calls

    def test_blocked_to_implement_row_is_skipped(
        self, dispatcher: Dispatcher, db: ExperimentDB,
    ) -> None:
        """A 'blocked:' error on a `to_implement` row also suppresses
        assignment. This is the "is the proposal still warranted" path
        from the implementer prompt: the implementer reads recent
        debriefs at turn start, finds the proposal's hypothesis has been
        invalidated by a newer experiment, and writes
        `update_experiment(error="blocked: superseded by #N")` without
        ever transitioning the row. The dispatcher must not re-assign,
        otherwise every implementer that picks up the row will do the
        same dance (cascading failure).
        """
        blocked_id = db.create("blocked_to_impl", "D", "H", "{}")
        # Still at to_implement; no status transition. Just an error.
        db.set_error(blocked_id, "blocked: superseded by #42")

        for w in dispatcher.workers:
            w._thread = None
        implement_calls = []
        for w in dispatcher.workers:
            w.implement = lambda exp, w=w: implement_calls.append(exp.id)
            w.analyze = lambda exp, w=w: None

        dispatcher._assign_workers()

        assert blocked_id not in implement_calls

    def test_blocked_other_status_is_not_filtered(
        self, dispatcher: Dispatcher, db: ExperimentDB,
    ) -> None:
        """`_is_externally_blocked` only matches to_implement / implemented.
        A `checked` row with a blocked error in its field (e.g. inherited
        from a prior precondition that's now satisfied) is not the
        function's concern — the dispatcher's submit path handles those.
        """
        from alpha_lab.dispatcher import _is_externally_blocked
        from alpha_lab.experiment_db import Experiment
        exp = Experiment(
            id=1, name="x", description="d", hypothesis="h",
            status="checked", config_json="{}", worker_id=None,
            slurm_job_id=None, results_json=None,
            error="blocked: stale", debrief_path=None,
            created_at=0.0, updated_at=0.0,
            started_at=None, finished_at=None,
        )
        assert _is_externally_blocked(exp) is False


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

    def test_waits_for_final_milestone_report(
        self, dispatcher: Dispatcher, db: ExperimentDB,
    ) -> None:
        for i in range(10):
            eid = db.create(f"report_exp_{i}", "D", "H", "{}")
            for status in (
                "implemented", "checked", "queued", "running", "finished",
                "analyzed", "done",
            ):
                db.update_status(eid, status)
        dispatcher._report_in_progress = True

        assert dispatcher._should_terminate() is False

    def test_parked_rows_complete_lifetime_budget(
        self, dispatcher: Dispatcher, db: ExperimentDB,
    ) -> None:
        for i in range(8):
            eid = db.create(f"done_{i}", "D", "H", "{}")
            for status in (
                "implemented", "checked", "queued", "running", "finished",
                "analyzed", "done",
            ):
                db.update_status(eid, status)
        for i in range(2):
            eid = db.create(f"parked_{i}", "D", "H", "{}")
            db.update_status(eid, "implemented")
            db.update_status(eid, "checked")
            db.park(eid)

        assert dispatcher._should_terminate() is True

        cancelling = db.create("still_cancelling", "D", "H", "{}")
        db.update_status(cancelling, "implemented")
        db.update_status(cancelling, "checked")
        db.set_slurm_job(cancelling, "job-1")
        db.update_status(cancelling, "queued")
        db.park(cancelling)
        assert dispatcher._should_terminate() is False

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


class TestDispatcherCrashed:
    def test_crashed_initially_false(self, dispatcher: Dispatcher) -> None:
        assert dispatcher.crashed is False

    def test_crashed_true_after_unhandled_exception(
        self, dispatcher: Dispatcher, events: list[AgentEvent],
    ) -> None:
        """run() swallows unhandled exceptions and exposes them via .crashed."""
        from alpha_lab.events import PhaseEvent

        with patch.object(dispatcher, "_poll_slurm", side_effect=RuntimeError("boom")):
            dispatcher.run()

        assert dispatcher.crashed is True
        phase_events = [e for e in events if isinstance(e, PhaseEvent) and e.phase == "phase3"]
        assert any(e.status == "error" for e in phase_events)


class TestIsCpuExperiment:
    """Unit tests for Dispatcher._is_cpu_experiment source-scan routing."""

    def _make_exp(
        self, dispatcher: Dispatcher, config_json: str = "{}",
        files: dict[str, str] | None = None,
    ):
        from alpha_lab.experiment_db import Experiment
        exp_id = dispatcher.db.create("t_cpu", "D", "H", config_json)
        exp = dispatcher.db.get(exp_id)
        if files is not None:
            exp_dir = Path(dispatcher.workspace) / "experiments" / exp.name
            exp_dir.mkdir(parents=True, exist_ok=True)
            for name, contents in files.items():
                (exp_dir / name).write_text(contents)
        return exp

    def test_explicit_resource_cpu_wins(self, dispatcher: Dispatcher) -> None:
        """config.resource='cpu' forces CPU even if source has GPU markers."""
        exp = self._make_exp(
            dispatcher,
            config_json='{"resource": "cpu"}',
            files={"strategy.py": "import torch; torch.cuda.empty_cache()"},
        )
        assert dispatcher._is_cpu_experiment(exp) is True

    def test_explicit_resource_gpu_wins(self, dispatcher: Dispatcher) -> None:
        """config.resource='gpu' forces GPU even if source has no markers."""
        exp = self._make_exp(
            dispatcher,
            config_json='{"resource": "gpu"}',
            files={"strategy.py": "x = 1"},
        )
        assert dispatcher._is_cpu_experiment(exp) is False

    def test_resource_non_string_falls_through_to_scan(self, dispatcher: Dispatcher) -> None:
        """Non-string `resource` (null/number) falls through to source scan, not crash."""
        exp = self._make_exp(
            dispatcher,
            config_json='{"resource": null}',
            files={"strategy.py": "x = 1", "run_experiment.py": "y = 2"},
        )
        assert dispatcher._is_cpu_experiment(exp) is False

    def test_scan_detects_torch_device(self, dispatcher: Dispatcher) -> None:
        exp = self._make_exp(
            dispatcher,
            files={
                "strategy.py": "device = torch.device('cuda')",
                "run_experiment.py": "",
            },
        )
        assert dispatcher._is_cpu_experiment(exp) is False

    def test_scan_cpu_library_means_cpu(self, dispatcher: Dispatcher) -> None:
        exp = self._make_exp(
            dispatcher,
            files={
                "strategy.py": "from sklearn.linear_model import Ridge\nmodel = Ridge()",
                "run_experiment.py": "print('hello')",
            },
        )
        assert dispatcher._is_cpu_experiment(exp) is True

    def test_scan_device_indirection_defaults_to_gpu(self, dispatcher: Dispatcher) -> None:
        """Regression (opus d2-cond CPU misroute): torch training placing the
        model through an indirection the scanner cannot see carries no literal
        GPU marker; ambiguity must route GPU, since the CPU pool disables CUDA."""
        exp = self._make_exp(
            dispatcher,
            files={
                "strategy.py": 'DEV = tr_cfg["device"]\nmodel = Model().to(DEV)\n',
                "run_experiment.py": "run()",
            },
        )
        assert dispatcher._is_cpu_experiment(exp) is False

    def test_unreadable_source_defaults_to_gpu(self, dispatcher: Dispatcher) -> None:
        """If declared source file exists but can't be decoded, assume GPU (safe default)."""
        exp = self._make_exp(
            dispatcher,
            files={},
        )
        exp_dir = Path(dispatcher.workspace) / "experiments" / exp.name
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "strategy.py").write_bytes(b"\xff\xfe\x00\x00invalid-utf8")
        (exp_dir / "run_experiment.py").write_bytes(b"\x00\x01\x02")
        assert dispatcher._is_cpu_experiment(exp) is False

    def test_no_source_files_defaults_to_gpu(self, dispatcher: Dispatcher) -> None:
        """If no declared source files exist on disk, assume GPU (safe default)."""
        exp = self._make_exp(dispatcher, files={})
        assert dispatcher._is_cpu_experiment(exp) is False


class TestOrphanedCompletionDetected:
    """Tests for Dispatcher._orphaned_completion_detected.

    Covers the case where an experiment subprocess exits cleanly between
    a kill of the old dispatcher and the start of the new one: the run
    wrote ``results/metrics.json`` and the model artifact, but nobody
    wrote ``run_status.json`` (that's the dispatcher's job). Without
    this detection, ``recover()`` would stamp the row as ``"job lost"``
    despite a real successful full run on disk.
    """

    def _make_exp(
        self, dispatcher: Dispatcher, name: str = "t_orphan",
        files: dict[str, str] | None = None,
    ):
        exp_id = dispatcher.db.create(name, "D", "H", "{}")
        exp = dispatcher.db.get(exp_id)
        if files is not None:
            exp_dir = Path(dispatcher.workspace) / "experiments" / exp.name
            (exp_dir / "results").mkdir(parents=True, exist_ok=True)
            for relpath, contents in files.items():
                p = exp_dir / relpath
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(contents)
        return exp

    def test_no_metrics_file_returns_false(self, dispatcher: Dispatcher) -> None:
        exp = self._make_exp(dispatcher, files={})
        assert dispatcher._orphaned_completion_detected(exp) is False

    def test_metrics_with_run_scope_full_returns_true(
        self, dispatcher: Dispatcher,
    ) -> None:
        exp = self._make_exp(
            dispatcher,
            files={
                "results/metrics.json": (
                    '{"run_scope": "full", "primary_sharpe": 6.52, '
                    '"max_drawdown": -1.5}'
                ),
            },
        )
        assert dispatcher._orphaned_completion_detected(exp) is True

    def test_metrics_with_run_scope_smoke_returns_false(
        self, dispatcher: Dispatcher,
    ) -> None:
        """A smoke-scope metrics.json is NOT evidence of a successful full
        run. We only treat ``run_scope="full"`` as orphaned-completion."""
        exp = self._make_exp(
            dispatcher,
            files={"results/metrics.json": '{"run_scope": "smoke", "sharpe": 0.1}'},
        )
        assert dispatcher._orphaned_completion_detected(exp) is False

    def test_metrics_with_run_scope_dry_run_returns_false(
        self, dispatcher: Dispatcher,
    ) -> None:
        exp = self._make_exp(
            dispatcher,
            files={"results/metrics.json": '{"run_scope": "dry_run"}'},
        )
        assert dispatcher._orphaned_completion_detected(exp) is False

    def test_metrics_without_run_scope_returns_false(
        self, dispatcher: Dispatcher,
    ) -> None:
        """Old metrics.json without a ``run_scope`` field is ambiguous —
        treat it as not-evidence rather than guess."""
        exp = self._make_exp(
            dispatcher,
            files={"results/metrics.json": '{"sharpe": 0.5}'},
        )
        assert dispatcher._orphaned_completion_detected(exp) is False

    def test_invalid_json_returns_false(self, dispatcher: Dispatcher) -> None:
        """Corrupt metrics.json is not evidence of completion."""
        exp = self._make_exp(
            dispatcher,
            files={"results/metrics.json": '{not valid json'},
        )
        assert dispatcher._orphaned_completion_detected(exp) is False

    def test_metrics_is_list_not_dict_returns_false(
        self, dispatcher: Dispatcher,
    ) -> None:
        """JSON that parses but isn't an object — guard against weird shapes."""
        exp = self._make_exp(
            dispatcher,
            files={"results/metrics.json": '["run_scope", "full"]'},
        )
        assert dispatcher._orphaned_completion_detected(exp) is False


class TestDispatcherRunEndDrainMode:
    """The Conductor-requested graceful run-end path: when meta/run_end_pending.json
    appears, the dispatcher enters drain mode (no new submissions, no new
    strategist proposals) and stops cleanly once in-flight goes to 0."""

    def _write_marker(self, workspace: str) -> None:
        from alpha_lab.meta_layout import meta_dir
        from alpha_lab.conductor_tools import RUN_END_MARKER
        meta_dir(workspace).mkdir(parents=True, exist_ok=True)
        marker = meta_dir(workspace) / RUN_END_MARKER
        marker.write_text(
            '{"ts": 1, "reason": "exhausted", "evidence": "...", '
            '"elapsed_hours_at_request": 8.0, "analyzed_at_request": 150}'
        )

    def test_drain_mode_off_without_marker(self, dispatcher: Dispatcher) -> None:
        # Drain mode requires a conductor; in this fixture conductor is None
        # so the consume call is a no-op regardless. Verify the flag stays False.
        dispatcher._consume_run_end_marker()
        assert dispatcher._drain_mode is False

    def test_drain_mode_set_on_marker(
        self, dispatcher: Dispatcher, tmp_workspace: str
    ) -> None:
        # Provide a stub conductor so _consume_run_end_marker doesn't short-circuit.
        dispatcher.conductor = MagicMock()
        self._write_marker(tmp_workspace)
        dispatcher._consume_run_end_marker()
        assert dispatcher._drain_mode is True
        # _run_end_marker_seen suppresses a second log+set on the same marker.
        assert dispatcher._run_end_marker_seen is True

    def test_drain_mode_blocks_submit(
        self, dispatcher: Dispatcher, db: ExperimentDB, mock_slurm: MagicMock,
    ) -> None:
        # Seed a `checked` row that would otherwise be submitted.
        exp_id = db.create("drain_target", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        db.update_status(exp_id, "checked")
        # Without drain mode, submit fires.
        dispatcher._drain_mode = False
        dispatcher._submit_checked()
        # Either submitted to executor or skipped on capacity — we don't care
        # for this test. Reset the row, set drain mode, and verify submit is
        # a no-op.
        mock_slurm.submit_experiment.reset_mock()
        dispatcher._drain_mode = True
        dispatcher._submit_checked()
        mock_slurm.submit_experiment.assert_not_called()

    def test_drain_mode_blocks_strategist_trigger(
        self, dispatcher: Dispatcher,
    ) -> None:
        # Force the conditions that would normally fire the strategist.
        dispatcher._last_strategist_time = 0.0  # first-turn-immediate
        dispatcher._drain_mode = False
        assert dispatcher._should_run_strategist() is True
        dispatcher._drain_mode = True
        assert dispatcher._should_run_strategist() is False

    def test_drain_parks_prelaunch_rows_instead_of_deadlocking(
        self, dispatcher: Dispatcher, db: ExperimentDB, tmp_workspace: str,
    ) -> None:
        # Regression: a 'checked' row with no executor job can never progress
        # in drain mode (submissions are refused), so counting it as in-flight
        # deadlocked a granted run-end for 11h in a live run. The drain must
        # park it and then complete.
        dispatcher.conductor = MagicMock()
        exp_id = db.create("never_launched", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        db.update_status(exp_id, "checked")
        self._write_marker(tmp_workspace)
        dispatcher._consume_run_end_marker()  # phase 1: drain mode on
        assert dispatcher._drain_mode is True
        dispatcher._consume_run_end_marker()  # phase 2: park + drain complete
        row = db.get(exp_id)
        assert row.parked_at is not None, "pre-launch row must be parked"
        assert dispatcher._stop_requested is True

    def test_drain_complete_sets_stop_requested(
        self, dispatcher: Dispatcher, tmp_workspace: str,
    ) -> None:
        # After drain mode is on AND no in-flight rows remain, the next
        # consume call flips _stop_requested.
        dispatcher.conductor = MagicMock()
        self._write_marker(tmp_workspace)
        dispatcher._consume_run_end_marker()
        assert dispatcher._drain_mode is True
        # No experiments at all → no in-flight → consume should flip stop.
        dispatcher._consume_run_end_marker()
        assert dispatcher._stop_requested is True


class TestDispatcherResearchStateSnapshot:
    """The dispatcher writes a sentinel-delimited LIVE-SNAPSHOT block to
    research_state.md after every analyzed transition. The Reporter's
    narrative lives outside the sentinels; the dispatcher's block is
    code-only and idempotent."""

    def test_refresh_creates_file_with_sentinels(
        self, dispatcher: Dispatcher, tmp_workspace: str,
    ) -> None:
        path = Path(tmp_workspace) / "research_state.md"
        assert not path.exists()
        dispatcher._refresh_research_state_snapshot()
        assert path.exists()
        content = path.read_text()
        assert dispatcher._RS_BEGIN in content
        assert dispatcher._RS_END in content
        # Header signals the block is dispatcher-owned.
        assert "Live snapshot" in content

    def test_refresh_replaces_block_in_place(
        self, dispatcher: Dispatcher, tmp_workspace: str, db: ExperimentDB,
    ) -> None:
        path = Path(tmp_workspace) / "research_state.md"
        # Seed with a file that has the dispatcher's block sandwiched between
        # narrative content the Reporter would have written.
        path.write_text(
            "# Research state\n\n"
            "Reporter's narrative paragraph above the block.\n\n"
            f"{dispatcher._RS_BEGIN}\n"
            "old snapshot content\n"
            f"{dispatcher._RS_END}\n\n"
            "Reporter's narrative below the block — coverage notes.\n"
        )
        dispatcher._refresh_research_state_snapshot()
        new_content = path.read_text()
        # The narrative outside the sentinels is preserved.
        assert "Reporter's narrative paragraph above the block" in new_content
        assert "Reporter's narrative below the block" in new_content
        # The old snapshot is gone; the fresh one is in place.
        assert "old snapshot content" not in new_content
        assert "Live snapshot" in new_content

    def test_refresh_atomic_write_no_torn_state(
        self, dispatcher: Dispatcher, tmp_workspace: str,
    ) -> None:
        # The implementation uses tmp + rename. The tmp file should not
        # linger after a successful write.
        path = Path(tmp_workspace) / "research_state.md"
        dispatcher._refresh_research_state_snapshot()
        tmp = path.with_suffix(path.suffix + ".tmp")
        assert not tmp.exists()


class TestDispatcherMilestoneCrashRollback:
    """A fire-once reporter that dies mid-flight (e.g. a transient gateway
    500) must not permanently burn its milestone number. The dispatcher
    detects the missing report.md when the reporter thread finishes and rolls
    the counter + baseline back so the milestone re-fires with the same
    number on a later cycle, instead of leaving a gap in the sequence.
    """

    def _arm_finished_reporter(self, dispatcher: Dispatcher) -> None:
        """Put the dispatcher in the 'reporter just finished' state for
        milestone #4 (number advanced to 4, baseline advanced 18 -> 22)."""
        worker = MagicMock()
        worker.busy = False  # reporter thread has finished (or died)
        dispatcher._report_in_progress = True
        dispatcher._report_worker = worker
        dispatcher._report_number = 4
        dispatcher._current_report_number = 4
        dispatcher._last_report_at_done_count = 22
        dispatcher._prev_report_at_done_count = 18
        dispatcher._milestone_just_finished = False

    def test_crash_without_report_rolls_back_counter(
        self, dispatcher: Dispatcher, tmp_workspace: str,
    ) -> None:
        self._arm_finished_reporter(dispatcher)
        # No reports/milestone_004/report.md exists -> reporter crashed.
        dispatcher._maybe_generate_report()

        # Number rolled back so it is reused, not burned.
        assert dispatcher._report_number == 3
        # Baseline restored so the milestone re-fires at the same threshold.
        assert dispatcher._last_report_at_done_count == 18
        assert dispatcher._report_in_progress is False
        assert dispatcher._report_worker is None
        # No milestone was produced -> nothing for the Conductor to react to.
        assert dispatcher._milestone_just_finished is False

    def test_success_with_report_keeps_counter_and_edge_triggers(
        self, dispatcher: Dispatcher, tmp_workspace: str,
    ) -> None:
        self._arm_finished_reporter(dispatcher)
        # The reporter produced its artifact.
        report_dir = Path(tmp_workspace) / "reports" / "milestone_004"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "report.md").write_text("# Milestone 4\nreal content")

        dispatcher._maybe_generate_report()

        # Counter stays advanced; baseline stays at the trigger value.
        assert dispatcher._report_number == 4
        assert dispatcher._last_report_at_done_count == 22
        assert dispatcher._report_in_progress is False
        assert dispatcher._report_worker is None
        # A real milestone landed -> Conductor edge-trigger is armed.
        assert dispatcher._milestone_just_finished is True

    def test_rolled_back_number_is_reused_on_next_trigger(
        self, dispatcher: Dispatcher, tmp_workspace: str,
    ) -> None:
        # After a crash rolls back to #3 / baseline 18, the next trigger (with
        # enough done experiments) must re-issue milestone #4, not skip to #5.
        self._arm_finished_reporter(dispatcher)
        dispatcher._maybe_generate_report()  # crash path -> rollback to 3 / 18
        assert dispatcher._report_number == 3

        # Now simulate the next cycle: report_interval reached again. Use a
        # tiny interval and an idle worker so the trigger path runs.
        dispatcher._report_interval = 4
        gen_worker = MagicMock()
        gen_worker.busy = False
        dispatcher.workers = [gen_worker]
        with patch.object(
            dispatcher.db, "list_by_status", return_value=list(range(27))
        ):
            dispatcher._maybe_generate_report()

        # The re-fired report reuses number 4 (gap filled), not 5.
        assert dispatcher._report_number == 4
        gen_worker.generate_report.assert_called_once()
        assert gen_worker.generate_report.call_args[0][0] == 4


class TestDispatcherZeroGpuPool:
    """Regression tests for the 2026-08-02 zero-GPU deadlock: a no-marker
    experiment was GPU-routed in a run configured with gpu_ids=[], the empty
    pool never accepted it, the skip logged at DEBUG, and the run idled 13h
    (d5_rfq_sol_cond). An empty GPU pool must fall back to CPU, an
    impossible pool combination must refuse to construct, and capacity
    starvation must be loud.
    """

    def _zero_gpu_dispatcher(self, config, db, events, tmp_workspace, cpu):
        gpu = MagicMock()
        gpu.gpu_ids = []
        dispatcher = Dispatcher(
            provider=MagicMock(),
            config=config,
            workspace=tmp_workspace,
            db=db,
            executor=gpu,
            cpu_executor=cpu,
            event_callback=lambda e: events.append(e),
            worker_count=2,
        )
        return dispatcher, gpu

    def test_no_marker_experiment_falls_back_to_cpu(
        self, config: TaskConfig, db: ExperimentDB, events, tmp_workspace: str,
    ) -> None:
        cpu = MagicMock()
        cpu.can_submit.return_value = True
        cpu.submit_experiment.return_value = "cpu-1"
        dispatcher, gpu = self._zero_gpu_dispatcher(
            config, db, events, tmp_workspace, cpu)

        exp_id = db.create("no_marker_exp", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        db.update_status(exp_id, "checked")

        dispatcher._submit_checked()

        cpu.submit_experiment.assert_called_once()
        gpu.submit_experiment.assert_not_called()
        assert db.get(exp_id).status == "queued"

    def test_zero_gpu_without_cpu_pool_refuses_to_construct(
        self, config: TaskConfig, db: ExperimentDB, tmp_workspace: str,
    ) -> None:
        gpu = MagicMock()
        gpu.gpu_ids = []
        with pytest.raises(ValueError, match="could ever be scheduled"):
            Dispatcher(
                provider=MagicMock(),
                config=config,
                workspace=tmp_workspace,
                db=db,
                executor=gpu,
                cpu_executor=None,
                event_callback=lambda e: None,
                worker_count=2,
            )

    def test_capacity_skip_is_loud(
        self, dispatcher: Dispatcher, db: ExperimentDB, mock_slurm: MagicMock,
        caplog,
    ) -> None:
        import logging

        exp_id = db.create("starved_exp", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        db.update_status(exp_id, "checked")
        mock_slurm.can_submit.return_value = False

        with caplog.at_level(logging.WARNING):
            dispatcher._submit_checked()

        assert any("waiting for GPU capacity" in r.message
                   for r in caplog.records)


class TestVerifierDrain:
    """Run-end drain (2026-08-07): an in-flight verification is signaled to
    finish its current candidate and is waited on before process exit, so
    its verdict is recorded — three real d5 verifications froze 0-2.6
    minutes short of their verdicts when exit killed the daemon thread."""

    def test_stop_signals_and_drains_inflight_verification(
            self, dispatcher: Dispatcher) -> None:
        import threading

        release = threading.Event()
        signaled = threading.Event()

        class _FakeVerifier:
            def finish_current_and_stop(self) -> None:
                signaled.set()
                release.set()  # "current candidate" completes on signal

        t = threading.Thread(target=lambda: release.wait(timeout=30),
                             daemon=True)
        with dispatcher._state_lock:
            dispatcher._verifier_thread = t
            dispatcher._verifier_obj = _FakeVerifier()
        t.start()
        dispatcher.stop(join_timeout=1)
        assert signaled.is_set()          # verifier was told to wind down
        assert not t.is_alive()           # and was drained to completion

    def test_drain_disabled_restores_cut_at_exit(
            self, dispatcher: Dispatcher) -> None:
        import threading

        hang = threading.Event()
        t = threading.Thread(target=lambda: hang.wait(timeout=30),
                             daemon=True)
        with dispatcher._state_lock:
            dispatcher._verifier_thread = t
        t.start()
        dispatcher.config.verifier_drain_seconds = 0
        dispatcher.stop(join_timeout=1)   # must NOT wait on the verifier
        assert t.is_alive()
        hang.set()
