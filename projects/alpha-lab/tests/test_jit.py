"""Unit tests for the JIT-proposals feature: capacity readers, the propose-time gate,
the strategist trigger, and executor slot totals."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from alpha_lab import deps, utils
from alpha_lab.config import Phase3Config, PipelineConfig, TaskConfig
from alpha_lab.dispatcher import Dispatcher
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
        pipeline=PipelineConfig(phases=["phase3"], phase3=Phase3Config(**phase3)),
    )


def _run_deps(*, gpu: int = 0, cpu: int = 4, **phase3) -> deps.RunDeps:
    cfg = _config(gpu_ids=[], **phase3)
    return deps.RunDeps(cfg, gpu_executor=_FakeExec(gpu), cpu_executor=_FakeExec(cpu))


class TestExperimentResource:
    def test_tagged_cpu(self) -> None:
        assert utils.experiment_resource(SimpleNamespace(config_json='{"resource":"cpu"}')) == "cpu"

    def test_untagged_defaults_gpu(self) -> None:
        assert utils.experiment_resource(SimpleNamespace(config_json="{}")) == "gpu"

    def test_invalid_json_defaults_gpu(self) -> None:
        assert utils.experiment_resource(SimpleNamespace(config_json="nope")) == "gpu"


class TestSlotStates:
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
        b = db.create("b", "d", "h", '{"resource":"cpu"}'); db.update_status(b, "running")
        with _run_deps(cpu=1, jit=False):
            assert "[ERROR]" not in self._propose(db, tmp_workspace)


class TestJitTrigger:
    def _config(self, **phase3) -> TaskConfig:
        return _config(jit=True, worker_count=2, report_interval=100, **phase3)

    def _dispatcher(self, db, workspace, adapter) -> Dispatcher:
        # Constructed inside an active RunDeps scope — the dispatcher reads its config there.
        return Dispatcher(
            provider=MagicMock(), workspace=workspace, db=db,
            event_callback=lambda e: None, adapter=adapter,
        )

    def test_fires_when_slot_and_worker_free(self, db, tmp_workspace, adapter) -> None:
        with deps.RunDeps(self._config(), gpu_executor=_FakeExec(4), cpu_executor=_FakeExec(0)):
            d = self._dispatcher(db, tmp_workspace, adapter)
            assert d._should_run_strategist() is True   # first turn, capacity free

    def test_blocked_when_no_free_worker(self, db, tmp_workspace, adapter) -> None:
        for i, name in enumerate(("a", "b")):
            eid = db.create(name, "d", "h", "{}"); db.assign_worker(eid, "w%d" % i)
        with deps.RunDeps(self._config(), gpu_executor=_FakeExec(4), cpu_executor=_FakeExec(0)):
            d = self._dispatcher(db, tmp_workspace, adapter)
            assert d._should_run_strategist() is False

    def test_blocked_when_no_free_slot(self, db, tmp_workspace, adapter) -> None:
        b = db.create("b", "d", "h", "{}"); db.update_status(b, "running")  # the one gpu slot busy
        with deps.RunDeps(self._config(cpu_enabled=False), gpu_executor=_FakeExec(1)):
            d = self._dispatcher(db, tmp_workspace, adapter)
            d._last_strategist_time = time.time()  # not first turn; interval not elapsed
            assert d._should_run_strategist() is False


class TestTotalSlots:
    def test_slurm_guards_zero_gpu_per_job(self) -> None:
        mgr = SlurmManager(partitions=["p"], gpu_per_job=0, max_gpus=8)
        assert mgr.total_slots() == 8   # no ZeroDivisionError
