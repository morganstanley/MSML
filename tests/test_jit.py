"""Unit tests for the JIT-proposals feature: capacity readers, the propose-time gate,
the strategist trigger, and executor slot totals."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from alpha_lab import deps, utils
from alpha_lab.config import Phase3Config, PipelineConfig, TaskConfig
from alpha_lab.constants import Phase
from alpha_lab.dispatcher import Dispatcher
from alpha_lab.events import ToolResultEvent
from alpha_lab.strategist import StallError, Strategist
from alpha_lab.slurm import SlurmManager
from alpha_lab.tools import execute_tool


class _FakeExec:
    def __init__(self, slots: int) -> None:
        self._slots = slots

    def total_slots(self) -> int:
        return self._slots


def _config(**phase3) -> TaskConfig:
    return TaskConfig(
        data_path="d",
        description="x",
        pipeline=PipelineConfig(phases=[Phase.PHASE3], phase3=Phase3Config(**phase3)),
    )


def _run_deps(*, gpu: int = 0, cpu: int = 4, **phase3) -> deps.RunDeps:
    cfg = _config(gpu_ids=[], **phase3)
    return deps.RunDeps(
        cfg,
        run_id="test",
        workspace=Path(tempfile.mkdtemp()),
        _gpu_executor=_FakeExec(gpu),
        _cpu_executor=_FakeExec(cpu),
    )


class TestExperimentResource:
    def test_tagged_cpu(self) -> None:
        assert utils.experiment_resource(SimpleNamespace(config_json='{"resource":"cpu"}')) == "cpu"

    def test_untagged_defaults_gpu(self) -> None:
        assert utils.experiment_resource(SimpleNamespace(config_json="{}")) == "gpu"

    def test_invalid_json_defaults_gpu(self) -> None:
        assert utils.experiment_resource(SimpleNamespace(config_json="nope")) == "gpu"


class TestSlotStates:
    @pytest.mark.no_run_deps
    def test_raises_without_deps(self, db) -> None:
        with pytest.raises(LookupError):
            utils.slot_states(db)

    def test_counts_busy_per_type_excludes_finished(self, db) -> None:
        db.create("a", "d", "h", '{"resource":"cpu"}')                  # to_implement
        b = db.create("b", "d", "h", '{"resource":"cpu"}'); db.update_status(b, "running")
        f = db.create("c", "d", "h", '{"resource":"cpu"}'); db.update_status(f, "finished")
        with _run_deps(gpu=0, cpu=4):
            assert utils.slot_states(db)["cpu"] == {"total": 4, "busy": 2, "free": 2}

    def test_untagged_counts_as_gpu(self, db) -> None:
        db.create("a", "d", "h", "{}")   # untagged -> gpu
        with _run_deps(gpu=2, cpu=4):
            st = utils.slot_states(db)
            assert st["gpu"]["busy"] == 1
            assert st["cpu"]["busy"] == 0


class TestWorkerStates:
    def test_counts_assigned_against_config_count(self, db) -> None:
        i = db.create("a", "d", "h", "{}"); db.assign_worker(i, "w0")
        with _run_deps(worker_count=3):
            assert utils.worker_states(db) == {"busy": 1, "free": 2}

    @pytest.mark.no_run_deps
    def test_raises_without_deps(self, db) -> None:
        with pytest.raises(LookupError):
            utils.worker_states(db)


class TestProposeGate:
    def _propose(self, db, workspace, resource="cpu"):
        return execute_tool(
            "propose_experiment",
            {"name": "n", "description": "d", "hypothesis": "h",
             "config": '{"resource":"%s"}' % resource},
            workspace=workspace, db=db,
        )["output"]

    def test_rejects_when_full(self, db, tmp_workspace) -> None:
        b = db.create("b", "d", "h", '{"resource":"cpu"}'); db.update_status(b, "running")
        with _run_deps(cpu=1, jit=True):
            assert "no free" in self._propose(db, tmp_workspace).lower()

    def test_allows_when_free(self, db, tmp_workspace) -> None:
        with _run_deps(cpu=2, jit=True):
            assert "[ERROR]" not in self._propose(db, tmp_workspace)

    def test_no_gate_when_flag_off(self, db, tmp_workspace) -> None:
        # Config differs from _propose's (beyond "resource", which dedup ignores)
        # so this stays a pure capacity-gate test, not a dedup collision.
        b = db.create("b", "d", "h", '{"resource":"cpu","tag":"other"}'); db.update_status(b, "running")
        with _run_deps(cpu=1, jit=False):
            assert "[ERROR]" not in self._propose(db, tmp_workspace)


class TestJitTrigger:
    def _config(self, **phase3) -> TaskConfig:
        return _config(jit=True, worker_count=2, report_interval=100, **phase3)

    def _dispatcher(self, db, adapter) -> Dispatcher:
        # Constructed inside an active RunDeps scope — the dispatcher reads its config there.
        return Dispatcher(
            provider=MagicMock(), db=db,
            event_callback=lambda e: None, adapter=adapter,
        )

    def test_fires_when_slot_and_worker_free(self, db, tmp_workspace, adapter) -> None:
        with deps.RunDeps(self._config(), run_id="test", workspace=Path(tmp_workspace), _gpu_executor=_FakeExec(4), _cpu_executor=_FakeExec(0)):
            d = self._dispatcher(db, adapter)
            assert d._should_run_strategist() is True   # first turn, capacity free

    def test_blocked_when_no_free_worker(self, db, tmp_workspace, adapter) -> None:
        for i, name in enumerate(("a", "b")):
            eid = db.create(name, "d", "h", "{}"); db.assign_worker(eid, "w%d" % i)
        with deps.RunDeps(self._config(), run_id="test", workspace=Path(tmp_workspace), _gpu_executor=_FakeExec(4), _cpu_executor=_FakeExec(0)):
            d = self._dispatcher(db, adapter)
            assert d._should_run_strategist() is False

    def test_blocked_when_no_free_slot(self, db, tmp_workspace, adapter) -> None:
        b = db.create("b", "d", "h", "{}"); db.update_status(b, "running")  # the one gpu slot busy
        with deps.RunDeps(self._config(cpu_enabled=False), run_id="test", workspace=Path(tmp_workspace), _gpu_executor=_FakeExec(1)):
            d = self._dispatcher(db, adapter)
            d._last_strategist_time = time.time()  # not first turn; interval not elapsed
            assert d._should_run_strategist() is False


class TestStrategistCompletion:
    @staticmethod
    def _strategist(db, adapter) -> Strategist:
        return Strategist(
            provider=MagicMock(), db=db,
            event_callback=lambda event: None, adapter=adapter,
        )

    @staticmethod
    def _completed_experiment(db) -> None:
        experiment_id = db.create("completed", "d", "h", '{"resource":"gpu"}')
        db.set_results(experiment_id, '{"metric": 1.0}')
        db.update_status(experiment_id, "finished")

    def test_explicit_completion_avoids_false_stall(
        self, db, tmp_workspace, adapter,
    ) -> None:
        self._completed_experiment(db)
        strategist = self._strategist(db, adapter)
        decision = {
            "decision": "research_complete",
            "reason": "The final holdout is open; another trial would be test-informed.",
            "evidence": ["The completed experiment used the locked final holdout."],
            "completed_experiment_ids": [1],
        }

        def run_agent(_agent_id, event_callback, **_kwargs) -> None:
            event_callback(ToolResultEvent(
                call_id="complete-1", name="complete_research",
                output=json.dumps(decision),
            ))

        with _run_deps(gpu=1, cpu=0, jit=True, worker_count=1):
            with patch("alpha_lab.strategist.sandbox.run_agent", side_effect=run_agent):
                strategist.run_turn()

        assert strategist.research_complete is True
        assert strategist.completion_reason == decision["reason"]
        assert strategist.completion_evidence == tuple(decision["evidence"])

    def test_zero_proposal_without_completion_still_fails(
        self, db, tmp_workspace, adapter,
    ) -> None:
        self._completed_experiment(db)
        strategist = self._strategist(db, adapter)
        with _run_deps(gpu=1, cpu=0, jit=True, worker_count=1):
            with patch("alpha_lab.strategist.sandbox.run_agent", return_value=None):
                with pytest.raises(StallError, match="proposed nothing"):
                    strategist.run_turn()
        assert strategist.research_complete is False

    def test_no_strategist_completion_removes_tool_and_prompt(
        self, db, tmp_workspace, adapter,
    ) -> None:
        # Ablation flag: the strategist must not see the tool nor be told to
        # call it — Phase 3 then runs to max_experiments.
        self._completed_experiment(db)
        strategist = self._strategist(db, adapter)
        captured: dict = {}

        def run_agent(_agent_id, _event_callback, **kwargs) -> None:
            captured.update(kwargs)

        with _run_deps(gpu=1, cpu=0, jit=True, worker_count=1,
                       no_strategist_completion=True):
            with patch("alpha_lab.strategist.sandbox.run_agent", side_effect=run_agent):
                with pytest.raises(StallError, match="proposed nothing"):
                    strategist.run_turn()

        assert captured["tools_include"] is not None
        assert "complete_research" not in captured["tools_include"]
        assert "complete_research" not in captured["extra_context"]
        assert strategist.research_complete is False

    def test_completion_tool_present_by_default(
        self, db, tmp_workspace, adapter,
    ) -> None:
        self._completed_experiment(db)
        strategist = self._strategist(db, adapter)
        captured: dict = {}

        def run_agent(_agent_id, _event_callback, **kwargs) -> None:
            captured.update(kwargs)

        with _run_deps(gpu=1, cpu=0, jit=True, worker_count=1):
            with patch("alpha_lab.strategist.sandbox.run_agent", side_effect=run_agent):
                with pytest.raises(StallError, match="proposed nothing"):
                    strategist.run_turn()

        assert captured["tools_include"] is None
        assert "complete_research" in captured["extra_context"]

    def test_later_active_work_invalidates_completion(
        self, db, tmp_workspace, adapter,
    ) -> None:
        self._completed_experiment(db)
        strategist = self._strategist(db, adapter)
        decision = {
            "decision": "research_complete",
            "reason": "No admissible follow-up remains.",
            "evidence": ["The protocol is locked."],
            "completed_experiment_ids": [1],
        }

        def run_agent(_agent_id, event_callback, **_kwargs) -> None:
            event_callback(ToolResultEvent(
                call_id="complete-1", name="complete_research",
                output=json.dumps(decision),
            ))
            db.create("late_proposal", "d", "h", '{"resource":"gpu"}')

        with _run_deps(gpu=1, cpu=0, jit=True, worker_count=1):
            with patch("alpha_lab.strategist.sandbox.run_agent", side_effect=run_agent):
                strategist.run_turn()

        assert strategist.research_complete is False
        assert db.list_by_status("to_implement")[0].name == "late_proposal"

    def test_later_cancelled_proposal_still_invalidates_completion(
        self, db, tmp_workspace, adapter,
    ) -> None:
        self._completed_experiment(db)
        strategist = self._strategist(db, adapter)
        decision = {
            "decision": "research_complete",
            "reason": "No admissible follow-up remains.",
            "evidence": ["The protocol is locked."],
            "completed_experiment_ids": [1],
        }

        def run_agent(_agent_id, event_callback, **_kwargs) -> None:
            event_callback(ToolResultEvent(
                call_id="complete-1", name="complete_research",
                output=json.dumps(decision),
            ))
            proposal_id = db.create(
                "late_cancelled_proposal", "d", "h", '{"resource":"gpu"}'
            )
            event_callback(ToolResultEvent(
                call_id="propose-1", name="propose_experiment",
                output=f"Experiment #{proposal_id} created.",
            ))
            db.update_status(proposal_id, "cancelled")

        with _run_deps(gpu=1, cpu=0, jit=True, worker_count=1):
            with patch("alpha_lab.strategist.sandbox.run_agent", side_effect=run_agent):
                with pytest.raises(StallError, match="proposed nothing"):
                    strategist.run_turn()

        assert strategist.research_complete is False
        assert db.list_by_status("cancelled")[0].name == "late_cancelled_proposal"


class TestTotalSlots:
    def test_slurm_guards_zero_gpu_per_job(self) -> None:
        mgr = SlurmManager(partitions=["p"], gpu_per_job=0, max_gpus=8)
        assert mgr.total_slots() == 8   # no ZeroDivisionError
