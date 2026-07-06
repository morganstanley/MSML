"""Tests for Worker agent: status transitions, error handling, filesystem checks."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alpha_lab.adapter import DomainAdapter
from alpha_lab.config import TaskConfig
from alpha_lab.events import AgentEvent
from alpha_lab.experiment_db import (
    Experiment,
    ExperimentDB,
)
from alpha_lab.memory import MemoryStore
from alpha_lab.worker import Worker


@pytest.fixture()
def config() -> TaskConfig:
    return TaskConfig(data_path="/data/test.csv", description="Test")


@pytest.fixture()
def worker(
    config: TaskConfig, db: ExperimentDB, tmp_workspace: str, adapter: DomainAdapter,
) -> Worker:
    return Worker(
        worker_id="worker_test",
        provider=MagicMock(),
        config=config,
        workspace=tmp_workspace,
        db=db,
        event_callback=lambda e: None,
        adapter=adapter,
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

        def blocking_impl(exp, otel_context=None):
            event.wait(timeout=5)

        worker._run_implement = blocking_impl
        worker.implement(sample_experiment)

        assert worker.busy is True
        event.set()
        worker._thread.join(timeout=2)


class TestWorkerImplementStatusTransition:
    """Test _ensure_implement_status fallback logic."""

    def test_auto_transition_no_adapter(
        self, worker: Worker, db: ExperimentDB, sample_experiment: Experiment,
        tmp_workspace: str,
    ) -> None:
        """No adapter means empty required_files -- always transitions to checked."""
        worker.adapter = None
        exp_dir = Path(tmp_workspace) / "experiments" / sample_experiment.name
        exp_dir.mkdir(parents=True)

        worker._ensure_implement_status(sample_experiment)

        exp = db.get(sample_experiment.id)
        assert exp.status == "checked"

    def test_auto_transition_with_adapter(
        self, worker: Worker, db: ExperimentDB, sample_experiment: Experiment,
        tmp_workspace: str,
    ) -> None:
        """With adapter required_files, transitions to checked when all present."""
        from alpha_lab.adapter import DomainAdapter, ExperimentStructure, MetricConfig
        worker.adapter = DomainAdapter(
            domain_name="test", domain_description="test",
            metric=MetricConfig(primary_metric="acc", direction="maximize"),
            experiment=ExperimentStructure(
                required_files=["model.py", "run_experiment.py"],
            ),
            prompts={},
        )
        exp_dir = Path(tmp_workspace) / "experiments" / sample_experiment.name
        exp_dir.mkdir(parents=True)
        (exp_dir / "model.py").write_text("class M: pass")
        (exp_dir / "run_experiment.py").write_text("print('run')")

        worker._ensure_implement_status(sample_experiment)

        exp = db.get(sample_experiment.id)
        assert exp.status == "checked"

    def test_sets_error_when_adapter_files_missing(
        self, worker: Worker, db: ExperimentDB, sample_experiment: Experiment,
        tmp_workspace: str,
    ) -> None:
        """With adapter required_files, sets error when files are missing."""
        from alpha_lab.adapter import DomainAdapter, ExperimentStructure, MetricConfig
        worker.adapter = DomainAdapter(
            domain_name="test", domain_description="test",
            metric=MetricConfig(primary_metric="acc", direction="maximize"),
            experiment=ExperimentStructure(
                required_files=["model.py", "run_experiment.py"],
            ),
            prompts={},
        )
        exp_dir = Path(tmp_workspace) / "experiments" / sample_experiment.name
        exp_dir.mkdir(parents=True)
        (exp_dir / "model.py").write_text("class M: pass")

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
        assert db.get(finished_experiment.id).status == "analyzed"

        store = MemoryStore(tmp_workspace)
        matches = store.search("good results", phase="phase3", kind="result")
        assert len(matches) == 1
        assert matches[0].summary == f"Experiment {finished_experiment.name} debrief"
        assert matches[0].agent == "worker"
        assert matches[0].source_path == f"experiments/{finished_experiment.name}/debrief.md"

    def test_auto_transition_without_debrief(
        self, worker: Worker, db: ExperimentDB,
        finished_experiment: Experiment,
    ) -> None:
        """finished → analyzed even without a debrief."""
        worker._ensure_analyze_status(finished_experiment)
        assert db.get(finished_experiment.id).status == "analyzed"

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
        """If there are results despite error, still promote to analyzed."""
        db.set_error(finished_experiment.id, "Partial results")
        db.set_results(finished_experiment.id, '{"sharpe": 0.5}')
        exp_dir = Path(tmp_workspace) / "experiments" / finished_experiment.name
        exp_dir.mkdir(parents=True)
        (exp_dir / "debrief.md").write_text("# Debrief")

        worker._ensure_analyze_status(finished_experiment)
        assert db.get(finished_experiment.id).status == "analyzed"

    def test_skips_if_already_past_finished(
        self, worker: Worker, db: ExperimentDB,
        finished_experiment: Experiment,
    ) -> None:
        db.update_status(finished_experiment.id, "analyzed")
        worker._ensure_analyze_status(finished_experiment)
        assert db.get(finished_experiment.id).status == "analyzed"


class TestWorkerHandoff:
    """Test _run_handoff deterministic status transition (no LLM-driven update)."""

    @pytest.fixture()
    def analyzed_experiment(self, db: ExperimentDB) -> Experiment:
        exp_id = db.create("handoff_exp", "D", "H", "{}")
        for s in ("implemented", "checked", "queued"):
            db.update_status(exp_id, s)
        db.update_status(exp_id, "running", started_at=1000.0)
        db.update_status(exp_id, "finished", finished_at=2000.0)
        db.update_status(exp_id, "analyzed")
        return db.get(exp_id)

    def test_success_promotes_to_done(
        self, worker: Worker, db: ExperimentDB, analyzed_experiment: Experiment,
    ) -> None:
        """When the agent loop returns cleanly, the experiment moves analyzed → done."""
        with patch("alpha_lab.worker.sandbox.run_agent"):
            worker._run_handoff(analyzed_experiment)
        current = db.get(analyzed_experiment.id)
        assert current.status == "done"
        assert current.error is None

    def test_success_promotes_to_done_with_existing_error(
        self, worker: Worker, db: ExperimentDB, analyzed_experiment: Experiment,
    ) -> None:
        """A stale pre-handoff error must not block a successful handoff."""
        db.set_error(analyzed_experiment.id, "previous analyzer warning")
        with patch("alpha_lab.worker.sandbox.run_agent"):
            worker._run_handoff(analyzed_experiment)
        current = db.get(analyzed_experiment.id)
        assert current.status == "done"
        assert current.error == "previous analyzer warning"

    def test_exception_leaves_at_analyzed_and_records_error(
        self, worker: Worker, db: ExperimentDB, analyzed_experiment: Experiment,
    ) -> None:
        """When the agent loop raises, status stays at analyzed and error is set."""
        with patch("alpha_lab.worker.sandbox.run_agent", side_effect=RuntimeError("boom")):
            worker._run_handoff(analyzed_experiment)
        current = db.get(analyzed_experiment.id)
        assert current.status == "analyzed"
        assert current.error is not None
        assert "boom" in current.error


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
