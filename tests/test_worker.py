"""Tests for Worker agent: status transitions, error handling, filesystem checks."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alpha_lab.config import TaskConfig
from alpha_lab.events import AgentEvent
from alpha_lab.experiment_db import Experiment, ExperimentDB
from alpha_lab.worker import Worker


@pytest.fixture()
def config() -> TaskConfig:
    return TaskConfig(data_path="/data/test.csv", description="Test")


@pytest.fixture()
def worker(
    config: TaskConfig, db: ExperimentDB, tmp_workspace: str,
) -> Worker:
    return Worker(
        worker_id="worker_test",
        provider=MagicMock(),
        config=config,
        workspace=tmp_workspace,
        db=db,
        event_callback=lambda e: None,
    )


@pytest.fixture()
def sample_experiment(db: ExperimentDB) -> Experiment:
    exp_id = db.create("test_exp", "Description", "Hypothesis", '{"model": "test"}')
    return db.get(exp_id)


class TestWorkerBusy:
    def test_not_busy_initially(self, worker: Worker) -> None:
        assert worker.busy is False

    def test_busy_during_work(self, worker: Worker, sample_experiment: Experiment) -> None:
        """Worker should be busy while thread is alive."""
        import threading

        event = threading.Event()

        # Patch _run_implement to block on event
        original = worker._run_implement

        def blocking_impl(exp):
            event.wait(timeout=5)

        worker._run_implement = blocking_impl
        worker.implement(sample_experiment)

        assert worker.busy is True
        event.set()
        worker._thread.join(timeout=2)


class TestWorkerImplementStatusTransition:
    """Test _ensure_implement_status fallback logic."""

    def test_auto_transition_to_checked(
        self, worker: Worker, db: ExperimentDB, sample_experiment: Experiment,
        tmp_workspace: str,
    ) -> None:
        """If files exist but status is still to_implement, auto-transition to checked."""
        exp_dir = Path(tmp_workspace) / "experiments" / sample_experiment.name
        exp_dir.mkdir(parents=True)
        (exp_dir / "strategy.py").write_text("class MyStrategy: pass")
        (exp_dir / "run_experiment.py").write_text("print('run')")

        worker._ensure_implement_status(sample_experiment)

        exp = db.get(sample_experiment.id)
        assert exp.status == "checked"

    def test_sets_error_when_files_missing(
        self, worker: Worker, db: ExperimentDB, sample_experiment: Experiment,
        tmp_workspace: str,
    ) -> None:
        """If files are missing, set error."""
        exp_dir = Path(tmp_workspace) / "experiments" / sample_experiment.name
        exp_dir.mkdir(parents=True)
        # Only strategy.py, missing run_experiment.py
        (exp_dir / "strategy.py").write_text("class MyStrategy: pass")

        worker._ensure_implement_status(sample_experiment)

        exp = db.get(sample_experiment.id)
        assert exp.error is not None
        assert "run_experiment.py" in exp.error

    def test_skips_if_already_past_to_implement(
        self, worker: Worker, db: ExperimentDB, sample_experiment: Experiment,
    ) -> None:
        """If status is already past to_implement, don't overwrite."""
        db.update_status(sample_experiment.id, "checked")
        worker._ensure_implement_status(sample_experiment)
        # Status should still be checked
        assert db.get(sample_experiment.id).status == "checked"

    def test_skips_if_error_already_set(
        self, worker: Worker, db: ExperimentDB, sample_experiment: Experiment,
    ) -> None:
        """If error is already set (agent crashed), don't overwrite."""
        db.set_error(sample_experiment.id, "Agent crashed with OOM")
        worker._ensure_implement_status(sample_experiment)
        exp = db.get(sample_experiment.id)
        assert exp.error == "Agent crashed with OOM"


class TestWorkerAnalyzeStatusTransition:
    """Test _ensure_analyze_status fallback logic."""

    @pytest.fixture()
    def finished_experiment(self, db: ExperimentDB) -> Experiment:
        exp_id = db.create("analyze_exp", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        db.update_status(exp_id, "checked")
        db.update_status(exp_id, "queued")
        db.update_status(exp_id, "running", started_at=1000.0)
        db.update_status(exp_id, "finished", finished_at=2000.0)
        return db.get(exp_id)

    def test_auto_transition_with_debrief(
        self, worker: Worker, db: ExperimentDB,
        finished_experiment: Experiment, tmp_workspace: str,
    ) -> None:
        exp_dir = Path(tmp_workspace) / "experiments" / finished_experiment.name
        exp_dir.mkdir(parents=True)
        (exp_dir / "debrief.md").write_text("# Debrief\nGood results.")

        worker._ensure_analyze_status(finished_experiment)
        assert db.get(finished_experiment.id).status == "done"

    def test_auto_transition_without_debrief(
        self, worker: Worker, db: ExperimentDB,
        finished_experiment: Experiment,
    ) -> None:
        """A missing debrief keeps the experiment available for analysis retry."""
        worker._ensure_analyze_status(finished_experiment)
        assert db.get(finished_experiment.id).status == "finished"

    def test_skips_if_error_and_no_results(
        self, worker: Worker, db: ExperimentDB,
        finished_experiment: Experiment,
    ) -> None:
        """If analysis crashed (error set, no results), keep at finished for retry."""
        db.set_error(finished_experiment.id, "LLM timeout")
        worker._ensure_analyze_status(finished_experiment)
        assert db.get(finished_experiment.id).status == "finished"

    def test_promotes_with_error_and_results(
        self, worker: Worker, db: ExperimentDB,
        finished_experiment: Experiment, tmp_workspace: str,
    ) -> None:
        """If there are results despite error, still finalize to done."""
        db.set_error(finished_experiment.id, "Partial results")
        db.set_results(finished_experiment.id, '{"sharpe": 0.5}')
        exp_dir = Path(tmp_workspace) / "experiments" / finished_experiment.name
        exp_dir.mkdir(parents=True)
        (exp_dir / "debrief.md").write_text("# Debrief")

        worker._ensure_analyze_status(finished_experiment)
        assert db.get(finished_experiment.id).status == "done"

    def test_finalizes_already_analyzed_to_done(
        self, worker: Worker, db: ExperimentDB,
        finished_experiment: Experiment, tmp_workspace: str,
    ) -> None:
        """An experiment the analyze agent already moved to 'analyzed' is
        finalized to the terminal 'done'."""
        db.update_status(finished_experiment.id, "analyzed")
        exp_dir = Path(tmp_workspace) / "experiments" / finished_experiment.name
        exp_dir.mkdir(parents=True)
        (exp_dir / "debrief.md").write_text("# Debrief")
        worker._ensure_analyze_status(finished_experiment)
        assert db.get(finished_experiment.id).status == "done"

    def test_does_not_finalize_analyzed_without_debrief(
        self, worker: Worker, db: ExperimentDB,
        finished_experiment: Experiment,
    ) -> None:
        db.update_status(finished_experiment.id, "analyzed")

        worker._ensure_analyze_status(finished_experiment)

        assert db.get(finished_experiment.id).status == "analyzed"


class TestWorkerExperimentContext:
    def test_build_experiment_context(self, worker: Worker) -> None:
        exp = Experiment(
            id=42, name="ctx_exp", description="Test experiment",
            hypothesis="It works", status="to_implement",
            config_json='{"model": "lstm", "lr": 0.001}',
            worker_id=None, slurm_job_id=None, results_json=None,
            error="Previous error", debrief_path=None,
            created_at=0, updated_at=0, started_at=None, finished_at=None,
        )
        ctx = worker._build_experiment_context(exp)
        assert "ctx_exp" in ctx
        assert "Test experiment" in ctx
        assert "It works" in ctx
        assert "lstm" in ctx
        assert "Previous error" in ctx
        assert "**Variant:** no" in ctx
        assert "**Available artifacts:** none" in ctx

    def test_build_variant_context_names_parent(self, worker: Worker) -> None:
        exp = Experiment(
            id=43, name="variant_exp", description="Test variant",
            hypothesis="It improves", status="to_implement", config_json="{}",
            worker_id=None, slurm_job_id=None, results_json=None, error=None,
            debrief_path=None, created_at=0, updated_at=0, started_at=None,
            finished_at=None, parent_id=7,
        )

        ctx = worker._build_experiment_context(exp)

        assert "Variant of:** experiment #7" in ctx


class TestWorkerReportContext:
    """Regression tests for _build_report_context against malformed results_json.

    The db.leaderboard sort_key is hardened separately (PR #21); these tests
    target the worker's own ``isinstance(m, dict)`` guards in the two
    ``metrics_str`` loops by stubbing leaderboard to return the malformed
    experiment directly — the test fails if worker removes its guards.
    """

    @pytest.mark.parametrize("payload", ["null", "[1, 2, 3]", "0", '"scalar"'])
    def test_non_dict_results_json_does_not_crash(
        self, worker: Worker, db: ExperimentDB, payload: str,
    ) -> None:
        exp_id = db.create("non_dict_results", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        db.update_status(exp_id, "checked")
        db.update_status(exp_id, "queued")
        db.update_status(exp_id, "running", started_at=1.0)
        db.update_status(exp_id, "finished", finished_at=2.0)
        db.set_results(exp_id, payload)
        db.update_status(exp_id, "analyzed")
        db.update_status(exp_id, "done")

        exp = db.get(exp_id)
        with patch.object(worker.db, "leaderboard", return_value=[exp]):
            ctx = worker._build_report_context(milestone_number=1, done_count=1)
        assert "non_dict_results" in ctx
        assert "no metrics" in ctx

    def test_invalid_json_does_not_crash(
        self, worker: Worker, db: ExperimentDB,
    ) -> None:
        exp_id = db.create("bad_json", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        db.update_status(exp_id, "checked")
        db.update_status(exp_id, "queued")
        db.update_status(exp_id, "running", started_at=1.0)
        db.update_status(exp_id, "finished", finished_at=2.0)
        db.set_results(exp_id, "this is not json at all")
        db.update_status(exp_id, "analyzed")
        db.update_status(exp_id, "done")

        exp = db.get(exp_id)
        with patch.object(worker.db, "leaderboard", return_value=[exp]):
            ctx = worker._build_report_context(milestone_number=1, done_count=1)
        assert "bad_json" in ctx
        assert "no metrics" in ctx

    def test_completed_experiment_context_lists_exact_debrief(
        self, worker: Worker, db: ExperimentDB, tmp_workspace: str,
    ) -> None:
        exp_id = db.create("reported", "D", "H", "{}")
        for status in (
            "implemented", "checked", "queued", "running", "finished",
            "analyzed", "done",
        ):
            db.update_status(exp_id, status)
        debrief = Path(tmp_workspace) / "experiments" / "reported" / "debrief.md"
        debrief.parent.mkdir(parents=True)
        debrief.write_text("# Debrief")

        context = worker._build_report_context(milestone_number=1, done_count=1)

        assert "debrief: experiments/reported/debrief.md" in context
