"""Tests for the Dispatcher orchestration logic.

All LLM calls and SLURM interactions are mocked.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alpha_lab import deps
from alpha_lab.adapter import DomainAdapter
from alpha_lab.config import Phase3Config, PipelineConfig, TaskConfig
from alpha_lab.constants import Phase
from alpha_lab.dispatcher import Dispatcher
from alpha_lab.events import AgentEvent, ExperimentEvent, PhaseEvent
from alpha_lab.experiment_db import ExperimentDB
from alpha_lab.slurm import SlurmManager


@pytest.fixture()
def config() -> TaskConfig:
    return TaskConfig(
        data_path="/data/test.csv",
        description="Test task",
        pipeline=PipelineConfig(
            phases=[Phase.PHASE3],
            phase3=Phase3Config(
                max_concurrent_gpus=4,
                max_experiments=10,
                strategist_interval=300,
                worker_count=2,
                slurm_partitions=["h100"],
                report_interval=100,  # effectively disabled for unit tests
                # These tests exercise the batch strategist-trigger and
                # handoff-off paths; pin the flags rather than depend on defaults.
                jit=False,
                handoff=False,
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
    adapter: DomainAdapter,
):
    provider = MagicMock()
    # The dispatcher reads config + executors off the run deps; inject the mock GPU
    # executor and keep the scope active for the test's duration (no CPU pool).
    config.pipeline.phase3.cpu_enabled = False
    with deps.RunDeps(config, run_id="test", workspace=Path(tmp_workspace), _gpu_executor=mock_slurm):
        yield Dispatcher(
            provider=provider,
            db=db,
            event_callback=lambda e: events.append(e),
            adapter=adapter,
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
    def test_prioritizes_handoff_when_enabled(
        self, dispatcher: Dispatcher, db: ExperimentDB, config: TaskConfig,
    ) -> None:
        # Same object RunDeps holds (function-scoped fixture); the dispatcher reads
        # the flag off the run deps, not off itself.
        config.pipeline.phase3.handoff = True
        analyzed_id = db.create("analyzed_exp", "D", "H", "{}")
        for status in ("implemented", "checked", "queued"):
            db.update_status(analyzed_id, status)
        db.update_status(analyzed_id, "running", started_at=1000.0)
        db.update_status(analyzed_id, "finished", finished_at=2000.0)
        db.update_status(analyzed_id, "analyzed")

        finished_id = db.create("finished_exp", "D", "H", "{}")
        for status in ("implemented", "checked", "queued"):
            db.update_status(finished_id, status)
        db.update_status(finished_id, "running", started_at=1000.0)
        db.update_status(finished_id, "finished", finished_at=2000.0)

        for w in dispatcher.workers:
            w._thread = None

        handoff_calls = []
        analyze_calls = []
        implement_calls = []
        for w in dispatcher.workers:
            w.handoff = lambda exp, w=w, **kw: handoff_calls.append(exp.id)
            w.analyze = lambda exp, w=w, **kw: analyze_calls.append(exp.id)
            w.implement = lambda exp, w=w, **kw: implement_calls.append(exp.id)

        dispatcher._assign_workers()

        assert handoff_calls == [analyzed_id]
        assert analyze_calls == [finished_id]
        assert implement_calls == []

    def test_skips_handoff_by_default(
        self, dispatcher: Dispatcher, db: ExperimentDB,
    ) -> None:
        analyzed_id = db.create("analyzed_exp", "D", "H", "{}")
        for status in ("implemented", "checked", "queued"):
            db.update_status(analyzed_id, status)
        db.update_status(analyzed_id, "running", started_at=1000.0)
        db.update_status(analyzed_id, "finished", finished_at=2000.0)
        db.update_status(analyzed_id, "analyzed")

        for w in dispatcher.workers:
            w._thread = None
            w.handoff = MagicMock()

        dispatcher._assign_workers()

        for w in dispatcher.workers:
            w.handoff.assert_not_called()

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
            w.analyze = lambda exp, w=w, **kw: analyze_calls.append(exp.id)
            w.implement = lambda exp, w=w, **kw: implement_calls.append(exp.id)

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
            w.implement = lambda exp, w=w, **kw: implement_calls.append(exp.id)
            w.analyze = lambda exp, w=w, **kw: None

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


class TestDispatcherResearchCompletion:
    def test_completion_stops_new_strategist_turns(
        self, dispatcher: Dispatcher,
    ) -> None:
        dispatcher._research_complete = True
        assert dispatcher._should_run_strategist() is False

    def test_completion_terminates_below_cap_after_analysis_drains(
        self, dispatcher: Dispatcher, db: ExperimentDB,
    ) -> None:
        experiment_id = db.create("final_result", "D", "H", "{}")
        db.set_results(experiment_id, '{"metric": 1.0}')
        db.update_status(experiment_id, "finished")
        dispatcher._research_complete = True
        assert dispatcher._should_terminate() is False
        db.update_status(experiment_id, "analyzed")
        assert dispatcher._should_terminate() is True

    def test_strategist_completion_is_recorded_without_a_crash(
        self, dispatcher: Dispatcher,
    ) -> None:
        dispatcher.strategist._completion_reason = "No admissible experiment remains."
        dispatcher.strategist._completion_evidence = ("The final holdout is open.",)
        dispatcher.strategist.run_turn = MagicMock()
        dispatcher._run_strategist()
        assert dispatcher._strategist_thread is not None
        dispatcher._strategist_thread.join(timeout=2)
        assert dispatcher.research_complete is True
        assert dispatcher.research_completion_reason == "No admissible experiment remains."
        assert dispatcher.crashed is False

    def test_completion_waits_for_handoff_when_enabled(
        self, dispatcher: Dispatcher, db: ExperimentDB, config: TaskConfig,
    ) -> None:
        config.pipeline.phase3.handoff = True
        experiment_id = db.create("handoff_result", "D", "H", "{}")
        db.set_results(experiment_id, '{"metric": 1.0}')
        db.update_status(experiment_id, "analyzed")
        dispatcher._research_complete = True
        assert dispatcher._should_terminate() is False
        db.update_status(experiment_id, "done")
        assert dispatcher._should_terminate() is True

    def test_normal_completion_run_records_drained_success(
        self, dispatcher: Dispatcher, events: list[AgentEvent],
    ) -> None:
        dispatcher._research_complete = True
        dispatcher._research_completion_reason = "No admissible experiment remains."
        dispatcher.recover = MagicMock(return_value={})
        dispatcher.run()
        final_event = events[-1]
        assert isinstance(final_event, PhaseEvent)
        assert dispatcher._research_completion_drained is True
        assert final_event.status == "completed"
        assert "research completed" in final_event.detail

    def test_external_stop_before_drain_does_not_claim_research_completed(
        self, dispatcher: Dispatcher, events: list[AgentEvent],
    ) -> None:
        dispatcher._research_complete = True
        dispatcher._research_completion_reason = "No admissible experiment remains."
        dispatcher._stop_requested = True
        dispatcher.recover = MagicMock(return_value={})
        dispatcher.run()
        final_event = events[-1]
        assert isinstance(final_event, PhaseEvent)
        assert dispatcher._research_completion_drained is False
        assert "research completed" not in final_event.detail


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
        phase_events = [e for e in events if isinstance(e, PhaseEvent) and e.phase == Phase.PHASE3]
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
            exp_dir = Path(deps.workspace) / "experiments" / exp.name
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
        assert dispatcher._is_cpu_experiment(exp) is True

    def test_scan_detects_torch_device(self, dispatcher: Dispatcher) -> None:
        exp = self._make_exp(
            dispatcher,
            files={
                "strategy.py": "device = torch.device('cuda')",
                "run_experiment.py": "",
            },
        )
        assert dispatcher._is_cpu_experiment(exp) is False

    def test_scan_no_markers_means_cpu(self, dispatcher: Dispatcher) -> None:
        exp = self._make_exp(
            dispatcher,
            files={
                "strategy.py": "from sklearn.linear_model import Ridge\nmodel = Ridge()",
                "run_experiment.py": "print('hello')",
            },
        )
        assert dispatcher._is_cpu_experiment(exp) is True

    def test_unreadable_source_defaults_to_gpu(self, dispatcher: Dispatcher) -> None:
        """If declared source file exists but can't be decoded, assume GPU (safe default)."""
        exp = self._make_exp(
            dispatcher,
            files={},
        )
        exp_dir = Path(deps.workspace) / "experiments" / exp.name
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "strategy.py").write_bytes(b"\xff\xfe\x00\x00invalid-utf8")
        (exp_dir / "run_experiment.py").write_bytes(b"\x00\x01\x02")
        assert dispatcher._is_cpu_experiment(exp) is False

    def test_no_source_files_defaults_to_gpu(self, dispatcher: Dispatcher) -> None:
        """If no declared source files exist on disk, assume GPU (safe default)."""
        exp = self._make_exp(dispatcher, files={})
        assert dispatcher._is_cpu_experiment(exp) is False


class TestFinalReport:
    """Completion-time final-report regeneration (termination branch)."""

    def test_pending_when_new_done_since_last_report(self, dispatcher: Dispatcher) -> None:
        dispatcher.db = MagicMock()
        dispatcher.db.list_by_status.return_value = [MagicMock()] * 3
        dispatcher._last_report_at_done_count = 2
        assert dispatcher._final_report_pending() is True

    def test_not_pending_when_board_already_reported(self, dispatcher: Dispatcher) -> None:
        dispatcher.db = MagicMock()
        dispatcher.db.list_by_status.return_value = [MagicMock()] * 2
        dispatcher._last_report_at_done_count = 2
        assert dispatcher._final_report_pending() is False

    def test_not_pending_after_final_report_done(self, dispatcher: Dispatcher) -> None:
        dispatcher.db = MagicMock()
        dispatcher.db.list_by_status.return_value = [MagicMock()] * 5
        dispatcher._last_report_at_done_count = 0
        dispatcher._final_report_done = True
        assert dispatcher._final_report_pending() is False

    def test_wait_budget_arms_then_expires(self, dispatcher: Dispatcher) -> None:
        # First check arms the timer rather than expiring immediately.
        assert dispatcher._final_report_wait_expired() is False
        assert dispatcher._final_report_wait_expired() is False  # within budget
        dispatcher._final_report_wait_started = (
            time.time() - Dispatcher.FINAL_REPORT_WAIT_SECONDS - 1
        )
        assert dispatcher._final_report_wait_expired() is True

    def test_generate_waits_when_no_worker_idle(self, dispatcher: Dispatcher) -> None:
        dispatcher.workers = [MagicMock(busy=True), MagicMock(busy=True)]
        dispatcher._generate_final_report()
        # No reporter kicked; pending stays true so the loop retries next cycle.
        assert dispatcher._final_report_done is False
        assert dispatcher._report_in_progress is False
        for w in dispatcher.workers:
            w.generate_report.assert_not_called()

    def test_generate_kicks_idle_worker_over_final_board(self, dispatcher: Dispatcher) -> None:
        dispatcher.db = MagicMock()
        dispatcher.db.list_by_status.return_value = [MagicMock()] * 4
        idle = MagicMock(busy=False)
        dispatcher.workers = [MagicMock(busy=True), idle]
        dispatcher._generate_final_report()
        idle.generate_report.assert_called_once()
        assert idle.generate_report.call_args.args[1] == 4  # covers the final done count
        assert dispatcher._final_report_done is True
        assert dispatcher._report_in_progress is True
        assert dispatcher._report_worker is idle
        assert dispatcher._last_report_at_done_count == 4


class TestExplicitCompletionTermination:
    def test_completion_terminates_after_drain(self, dispatcher: Dispatcher) -> None:
        # An accepted explicit completion decision ends the run once nothing
        # is in flight; without it the same idle board keeps running (the
        # experiment cap is nowhere near reached).
        assert dispatcher._should_terminate() is False
        with dispatcher._state_lock:
            dispatcher._research_complete = True
            dispatcher._research_completion_reason = "protocol locked"
        assert dispatcher._should_terminate() is True

    def test_completion_waits_for_in_flight_work(
        self, dispatcher: Dispatcher, db: ExperimentDB
    ) -> None:
        with dispatcher._state_lock:
            dispatcher._research_complete = True
        eid = db.create("live", "d", "h", "{}")
        db.update_status(eid, "running")
        assert dispatcher._should_terminate() is False


def _touch_sentinel(db: ExperimentDB, exp_name: str) -> None:
    """Create the canonical-run sentinel a row needs to appear on the leaderboard."""
    from alpha_lab.experiment_db import CANONICAL_RUN_COMPLETE_SENTINEL
    workspace = Path(db.db_path).parent.parent
    results = workspace / "experiments" / exp_name / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / CANONICAL_RUN_COMPLETE_SENTINEL).touch()


class TestTrackAnalyzedConvergence:
    """PROJ-765: _track_analyzed re-derives the leaderboard's top value with
    its own metric lookup rather than trusting leaderboard()'s own ranking --
    a variant-spelled metric must still register as a real improvement, not
    silently fall back to "no value" (the direction's worst-value sentinel),
    which would wrongly count toward convergence/early-stopping."""

    def test_detects_improvement_from_variant_key_spelling(
        self, dispatcher: Dispatcher, db: ExperimentDB,
    ) -> None:
        # Default adapter fixture: primary_metric="sharpe", direction="maximize".
        exp_id = db.create("gp_run", "D", "H", "{}")
        db.set_results(exp_id, '{"val/sharpe": 2.5}')
        db.update_status(exp_id, "analyzed")
        _touch_sentinel(db, "gp_run")

        dispatcher._track_analyzed()

        assert dispatcher._best_metric_value == 2.5
        assert dispatcher._experiments_since_improvement == 0

    def test_exact_key_still_wins_over_flattering_variant(
        self, dispatcher: Dispatcher, db: ExperimentDB,
    ) -> None:
        """A same-tokened neighbor field (e.g. a champion comparison) must
        never be mistaken for the experiment's own declared score here
        either -- same rule as leaderboard()/summarize_workspace.py."""
        exp_id = db.create("gp_run", "D", "H", "{}")
        db.set_results(exp_id, '{"sharpe": 1.0, "champion_sharpe": 99.0}')
        db.update_status(exp_id, "analyzed")
        _touch_sentinel(db, "gp_run")

        dispatcher._track_analyzed()

        assert dispatcher._best_metric_value == 1.0

    def test_missing_metric_does_not_register_as_improvement(
        self, dispatcher: Dispatcher, db: ExperimentDB,
    ) -> None:
        exp_id = db.create("no_metric", "D", "H", "{}")
        db.set_results(exp_id, '{"unrelated_field": 1.0}')
        db.update_status(exp_id, "analyzed")
        _touch_sentinel(db, "no_metric")

        dispatcher._track_analyzed()

        # -inf is the initial value for a maximize metric; nothing beat it.
        assert dispatcher._best_metric_value == float("-inf")
        assert dispatcher._experiments_since_improvement == 1
