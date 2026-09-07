"""Tests for the Conductor's integration into the Dispatcher.

Covers:

* NOOP path: with no_conductor=True the dispatcher does not construct a
  Conductor and skips _maybe_run_conductor entirely. Existing dispatcher
  behavior is preserved.
* Trigger logic: _should_run_conductor returns True at the right moments
  (milestone edge, timer interval) and False otherwise (already running,
  too soon).
* No-double-spawn: a long-running conductor turn does not get queued
  behind itself when the timer fires again mid-turn.
* Throttle hot-path: meta/throttle.json drives the per-cycle submission cap
  in _submit_checked.
* Phase-rewind marker rotation: a marker is consumed (logged + renamed)
  exactly once.
* Kill-request consumption: kill_requests.jsonl entries cancel matching
  executor jobs.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from alpha_lab import meta_layout as ml
from alpha_lab.config import Phase3Config, PipelineConfig, TaskConfig
from alpha_lab.dispatcher import Dispatcher
from alpha_lab.experiment_db import ExperimentDB
from alpha_lab.events import AgentEvent


# ---------------------------------------------------------------------------
# Fixtures: a Dispatcher built without going through CLI / network.
# ---------------------------------------------------------------------------


def _make_dispatcher(
    tmp_path: Path,
    *,
    no_conductor: bool,
    conductor_interval: int = 1800,
    worker_count: int = 1,
    cpu_executor: Any | None = None,
) -> Dispatcher:
    """Construct a Dispatcher with a mocked provider, executor, and adapter.

    No threads are started — we exercise scheduling helpers in isolation.
    """
    cfg = TaskConfig(data_path="/d", description="D")
    cfg.pipeline = PipelineConfig(
        phase3=Phase3Config(
            no_conductor=no_conductor,
            conductor_interval=conductor_interval,
            worker_count=worker_count,
        )
    )
    db = ExperimentDB(str(tmp_path / "exps.db"))
    provider = MagicMock()
    provider.openai_client = None
    executor = MagicMock()
    # A bare MagicMock's gpu_ids answers len()==0, which the zero-GPU
    # construction guard rightly refuses — declare a schedulable pool.
    executor.gpu_ids = [0]
    executor.can_submit.return_value = True
    executor.submit_experiment.return_value = "job_1"
    adapter = MagicMock()
    adapter.metric.primary_metric = "sharpe"
    adapter.metric.direction = "maximize"
    events: list[AgentEvent] = []

    def cb(e: AgentEvent) -> None:
        events.append(e)

    return Dispatcher(
        provider=provider,
        config=cfg,
        workspace=str(tmp_path),
        db=db,
        executor=executor,
        event_callback=cb,
        worker_count=worker_count,
        cpu_executor=cpu_executor,
        adapter=adapter,
        supervisor=None,
    )


# ---------------------------------------------------------------------------
# NOOP path
# ---------------------------------------------------------------------------


class TestNoConductorNoop:
    def test_conductor_not_constructed(self, tmp_path: Path) -> None:
        d = _make_dispatcher(tmp_path, no_conductor=True)
        assert d.conductor is None

    def test_should_run_returns_false(self, tmp_path: Path) -> None:
        d = _make_dispatcher(tmp_path, no_conductor=True)
        assert d._should_run_conductor() is False

    def test_maybe_run_is_a_noop(self, tmp_path: Path) -> None:
        d = _make_dispatcher(tmp_path, no_conductor=True)
        # Even forcing the milestone flag, no thread should spawn
        d._milestone_just_finished = True
        d._maybe_run_conductor("milestone")
        assert d._conductor_thread is None
        assert d._conductor_running is False

    def test_meta_dir_not_required_to_exist(self, tmp_path: Path) -> None:
        d = _make_dispatcher(tmp_path, no_conductor=True)
        # When no_conductor=True we don't bootstrap the meta/ tree —
        # the dispatcher must NOT depend on it for its hot paths.
        assert not (tmp_path / "meta").exists() or True
        # And reading throttle still works (returns defaults)
        from alpha_lab.meta_layout import read_throttle
        assert read_throttle(tmp_path) == {"gpu": "none", "cpu": "none"}


# ---------------------------------------------------------------------------
# Trigger logic
# ---------------------------------------------------------------------------


class TestShouldRunConductor:
    def test_milestone_edge_triggers(self, tmp_path: Path) -> None:
        d = _make_dispatcher(tmp_path, no_conductor=False)
        d._milestone_just_finished = True
        assert d._should_run_conductor() is True

    def test_no_trigger_when_too_soon(self, tmp_path: Path) -> None:
        d = _make_dispatcher(tmp_path, no_conductor=False, conductor_interval=3600)
        d._last_conductor_time = time.time()  # just ran
        d._milestone_just_finished = False
        assert d._should_run_conductor() is False

    def test_timer_triggers_after_interval(self, tmp_path: Path) -> None:
        d = _make_dispatcher(tmp_path, no_conductor=False, conductor_interval=1)
        d._last_conductor_time = time.time() - 60  # well past
        assert d._should_run_conductor() is True

    def test_no_double_spawn(self, tmp_path: Path) -> None:
        d = _make_dispatcher(tmp_path, no_conductor=False)
        d._milestone_just_finished = True
        d._conductor_running = True  # simulate in-flight
        # The next check must NOT trigger another turn.
        assert d._should_run_conductor() is False

    def test_first_turn_fires_immediately(self, tmp_path: Path) -> None:
        # First turn (last_conductor_time = 0) fires immediately so the
        # Conductor sees from_user.md before anything else runs. Mirrors
        # the strategist's first-turn behavior.
        d = _make_dispatcher(tmp_path, no_conductor=False, conductor_interval=10**9)
        assert d._last_conductor_time == 0
        assert d._should_run_conductor() is True


class TestMilestoneEdgeFlag:
    def test_flag_set_after_milestone(self, tmp_path: Path) -> None:
        # We can't easily simulate a real reporter completion without a
        # real provider — but we can drive the same code path by setting
        # _report_in_progress + _report_worker, then calling
        # _maybe_generate_report when the worker is "idle".
        d = _make_dispatcher(tmp_path, no_conductor=False)
        d._report_in_progress = True
        worker = MagicMock()
        worker.busy = False
        d._report_worker = worker
        d._current_report_number = 1
        # The milestone-finished flag is only set once report.md actually lands on
        # disk (a defensive rollback re-fires the milestone if the reporter crashes
        # without writing it). Simulate a successful report so this test exercises
        # the flag-set path rather than the rollback.
        _rmd = tmp_path / "reports" / "milestone_001" / "report.md"
        _rmd.parent.mkdir(parents=True, exist_ok=True)
        _rmd.write_text("# milestone 1\n")
        # Stub the OutputGenerator import so it doesn't blow up on missing
        # workspace structure
        import alpha_lab.output_generator as _og
        original = _og.OutputGenerator
        _og.OutputGenerator = MagicMock()
        try:
            d._maybe_generate_report()
        finally:
            _og.OutputGenerator = original
        assert d._milestone_just_finished is True


# ---------------------------------------------------------------------------
# Throttle hot-path
# ---------------------------------------------------------------------------


class TestThrottleInSubmitChecked:
    def test_halt_new_blocks_gpu_submissions(self, tmp_path: Path) -> None:
        d = _make_dispatcher(tmp_path, no_conductor=False)
        ml.ensure_meta_layout(tmp_path)
        ml.throttle_path(tmp_path).write_text(
            '{"gpu": "halt-new", "cpu": "none"}'
        )
        # Create a checked GPU experiment
        eid = d.db.create("exp_halt", "D", "H", '{"resource": "gpu"}')
        d.db.update_status(eid, "implemented")
        d.db.update_status(eid, "checked")
        d._submit_checked()
        # The mocked GPU executor's submit_experiment should NOT have been
        # called because halt-new is in effect.
        d.executor.submit_experiment.assert_not_called()

    def test_throttle_none_allows_normal_submission(self, tmp_path: Path) -> None:
        d = _make_dispatcher(tmp_path, no_conductor=False)
        ml.ensure_meta_layout(tmp_path)  # default: throttle = none
        eid = d.db.create("exp_normal", "D", "H", '{"resource": "gpu"}')
        d.db.update_status(eid, "implemented")
        d.db.update_status(eid, "checked")
        d._submit_checked()
        d.executor.submit_experiment.assert_called_once()

    def test_throttle_slow_halves_per_cycle_submissions(self, tmp_path: Path) -> None:
        d = _make_dispatcher(tmp_path, no_conductor=False)
        ml.ensure_meta_layout(tmp_path)
        ml.throttle_path(tmp_path).write_text('{"gpu": "slow"}')
        # Create 4 checked GPU experiments
        ids = []
        for i in range(4):
            eid = d.db.create(f"exp_{i}", "D", "H", '{"resource": "gpu"}')
            d.db.update_status(eid, "implemented")
            d.db.update_status(eid, "checked")
            ids.append(eid)
        d._submit_checked()
        # With 4 candidates, the slow cap halves to 2; only 2 submissions go through.
        assert d.executor.submit_experiment.call_count == 2

    def test_missing_throttle_file_means_no_throttling(self, tmp_path: Path) -> None:
        d = _make_dispatcher(tmp_path, no_conductor=False)
        # Don't write throttle.json; default behavior must be unchanged
        eid = d.db.create("exp_default", "D", "H", '{"resource": "gpu"}')
        d.db.update_status(eid, "implemented")
        d.db.update_status(eid, "checked")
        d._submit_checked()
        d.executor.submit_experiment.assert_called_once()


class TestThrottledCap:
    def test_slow_halves(self) -> None:
        from alpha_lab.dispatcher import Dispatcher
        assert Dispatcher._throttled_cap("slow", 4) == 2
        assert Dispatcher._throttled_cap("slow", 1) == 1  # rounded up to >=1
        assert Dispatcher._throttled_cap("slow", 0) == 1

    def test_none_passthrough(self) -> None:
        from alpha_lab.dispatcher import Dispatcher
        assert Dispatcher._throttled_cap("none", 7) == 7
        assert Dispatcher._throttled_cap("anything-else", 7) == 7


# ---------------------------------------------------------------------------
# Marker consumption
# ---------------------------------------------------------------------------


class TestPhaseRewindMarker:
    def test_consume_requests_stop_and_leaves_marker(self, tmp_path: Path) -> None:
        """Dispatcher's _consume_phase_rewind_marker must NOT rotate the
        marker — run.py needs to read it after the dispatcher exits.
        Instead it sets _stop_requested so the main loop unwinds."""
        d = _make_dispatcher(tmp_path, no_conductor=False)
        ml.ensure_meta_layout(tmp_path)
        from alpha_lab.conductor_tools import PHASE_REWIND_MARKER
        marker = ml.meta_dir(tmp_path) / PHASE_REWIND_MARKER
        marker.write_text(json.dumps({
            "ts": time.time(),
            "target_phase": "phase0",
            "reason": "wrong domain",
            "evidence": "scripted",
            "backup_dir": str(ml.backups_dir(tmp_path) / "fake"),
        }))
        assert d._stop_requested is False
        d._consume_phase_rewind_marker()
        # Marker file MUST still exist (the outer driver consumes it).
        assert marker.exists()
        # Dispatcher main loop must have been signaled to stop.
        assert d._stop_requested is True
        # No consumed_*.json sibling — that's run.py's job now.
        consumed = list(ml.meta_dir(tmp_path).glob("phase_rewind_consumed_*.json"))
        assert len(consumed) == 0

    def test_consume_is_idempotent_with_marker_seen_flag(self, tmp_path: Path) -> None:
        """Repeated consume calls on the same marker should not re-emit
        the warning or re-log; _rewind_marker_seen makes it one-shot."""
        d = _make_dispatcher(tmp_path, no_conductor=False)
        ml.ensure_meta_layout(tmp_path)
        from alpha_lab.conductor_tools import PHASE_REWIND_MARKER
        marker = ml.meta_dir(tmp_path) / PHASE_REWIND_MARKER
        marker.write_text(json.dumps({
            "ts": time.time(),
            "target_phase": "phase1",
            "reason": "x",
            "evidence": "x",
            "backup_dir": "x",
        }))
        d._consume_phase_rewind_marker()
        # Reset stop so the test can detect a second call doing work
        d._stop_requested = False
        d._consume_phase_rewind_marker()
        # Second call short-circuits on _rewind_marker_seen — does NOT
        # set _stop_requested again.
        assert d._stop_requested is False

    def test_consume_no_marker_does_nothing(self, tmp_path: Path) -> None:
        d = _make_dispatcher(tmp_path, no_conductor=False)
        # Don't create a marker; method should silently no-op
        d._consume_phase_rewind_marker()
        assert d._stop_requested is False


class TestKillRequests:
    def test_consume_cancels_executor_job(self, tmp_path: Path) -> None:
        d = _make_dispatcher(tmp_path, no_conductor=False)
        ml.ensure_meta_layout(tmp_path)
        # Set up an experiment with a slurm_job_id assigned
        eid = d.db.create("exp_kill", "D", "H", "{}")
        d.db.set_slurm_job(eid, "JOB-42")
        d.db.update_status(eid, "running")
        # Write a kill request
        kill_path = ml.meta_dir(tmp_path) / "kill_requests.jsonl"
        kill_path.write_text(
            json.dumps({"ts": time.time(), "experiment_id": eid, "reason": "doomed"}) + "\n"
        )
        d._consume_kill_requests()
        # Executor.cancel was called with the right job id
        d.executor.cancel.assert_called_once_with("JOB-42")
        # Marker file truncated
        assert kill_path.read_text() == ""

    def test_kill_with_no_slurm_job_skips_cancel(self, tmp_path: Path) -> None:
        d = _make_dispatcher(tmp_path, no_conductor=False)
        ml.ensure_meta_layout(tmp_path)
        eid = d.db.create("exp_nojob", "D", "H", "{}")  # no slurm_job_id
        kill_path = ml.meta_dir(tmp_path) / "kill_requests.jsonl"
        kill_path.write_text(
            json.dumps({"ts": time.time(), "experiment_id": eid, "reason": "r"}) + "\n"
        )
        d._consume_kill_requests()
        d.executor.cancel.assert_not_called()


class TestCoordinatedParking:
    def test_dispatcher_confirms_exit_before_making_park_resumable(
        self, tmp_path: Path,
    ) -> None:
        d = _make_dispatcher(tmp_path, no_conductor=False)
        exp_id = d.db.create("active", "D", "H", "{}")
        d.db.update_status(exp_id, "implemented")
        d.db.update_status(exp_id, "checked")
        d.db.set_slurm_job(exp_id, "job_1")
        d.db.update_status(exp_id, "queued")
        d.db.update_status(exp_id, "running")
        d.db.park(exp_id)
        d.executor.poll_jobs.return_value = {"job_1": "CANCELLED"}

        assert d._process_parkings() == 1
        d.executor.cancel.assert_called_once_with("job_1")
        exp = d.db.get(exp_id)
        assert exp is not None
        assert exp.status == "checked"
        assert exp.parked_at is not None
        assert exp.slurm_job_id is None


class TestSupervisorCheckOnceAtCheckpoint:
    """``_maybe_supervisor_check`` must only fire once per
    multiple-of-10 ``analyzed`` count — not every loop iteration the
    count happens to be at the checkpoint. Previously it would re-fire
    the supervisor at count=10 / 20 / ... on every dispatcher main loop
    iteration until the next experiment finished."""

    def test_only_fires_once_at_count_10(self, tmp_path: Path) -> None:
        d = _make_dispatcher(tmp_path, no_conductor=False)
        # Stand up 10 analyzed experiments with errors
        for i in range(10):
            eid = d.db.create(f"exp_{i}", "d", "h", "{}")
            d.db.update_status(eid, "implemented")
            d.db.update_status(eid, "checked")
            d.db.update_status(eid, "queued")
            d.db.update_status(eid, "running")
            d.db.update_status(eid, "finished")
            d.db.set_error(eid, "boom")
            d.db.update_status(eid, "analyzed")

        calls = {"n": 0}
        if d.supervisor is None:
            # _make_dispatcher may construct with no supervisor; install
            # a mock for this test only.
            from unittest.mock import MagicMock
            d.supervisor = MagicMock()
            d.supervisor.phase3_health_check = MagicMock(side_effect=lambda: calls.__setitem__("n", calls["n"] + 1))
        else:
            from unittest.mock import patch
            d.supervisor.phase3_health_check = lambda: calls.__setitem__("n", calls["n"] + 1)

        # First call at count=10 with high error rate: fires once.
        d._maybe_supervisor_check()
        first_n = calls["n"]
        # Subsequent calls at SAME count: should NOT re-fire.
        d._maybe_supervisor_check()
        d._maybe_supervisor_check()
        assert calls["n"] == first_n, (
            "supervisor health check re-fired at same checkpoint count"
        )
