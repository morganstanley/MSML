"""Tests for alpha_lab.deps — RunDeps construction and its context-manager lifecycle."""

from __future__ import annotations

import threading

import pytest

from alpha_lab import deps
from alpha_lab.config import Phase3Config, PipelineConfig, TaskConfig
from alpha_lab.local_cpu import LocalCPUManager
from alpha_lab.local_gpu import LocalGPUManager
from alpha_lab.slurm import SlurmManager


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


class TestRunDepsConstruction:
    def test_builds_local_gpu_and_cpu_from_config(self) -> None:
        rd = deps.RunDeps(_config(executor="local", gpu_ids=[0, 1], cpu_enabled=True))
        assert isinstance(rd.gpu_executor, LocalGPUManager)
        assert isinstance(rd.cpu_executor, LocalCPUManager)

    def test_cpu_is_none_when_disabled(self) -> None:
        rd = deps.RunDeps(_config(executor="local", gpu_ids=[], cpu_enabled=False))
        assert rd.cpu_executor is None

    def test_builds_slurm_executor(self) -> None:
        rd = deps.RunDeps(_config(executor="slurm", cpu_enabled=False))
        assert isinstance(rd.gpu_executor, SlurmManager)

    def test_injected_executors_skip_construction(self) -> None:
        gpu, cpu = _FakeExec(2), _FakeExec(3)
        rd = deps.RunDeps(
            _config(gpu_ids=[], cpu_enabled=True), gpu_executor=gpu, cpu_executor=cpu
        )
        assert rd.gpu_executor is gpu
        assert rd.cpu_executor is cpu


class TestLifecycle:
    def test_get_strict_raises_outside_scope(self) -> None:
        with pytest.raises(LookupError):
            deps.get()

    def test_get_non_strict_is_none_outside_scope(self) -> None:
        assert deps.get(strict=False) is None

    def test_with_publishes_and_auto_resets(self) -> None:
        rd = deps.RunDeps(_config(gpu_ids=[], cpu_enabled=False), gpu_executor=_FakeExec(0))
        with rd:
            assert deps.get() is rd
        assert deps.get(strict=False) is None

    def test_nested_restores_outer(self) -> None:
        a = deps.RunDeps(_config(gpu_ids=[], cpu_enabled=False), gpu_executor=_FakeExec(0))
        b = deps.RunDeps(_config(gpu_ids=[], cpu_enabled=False), gpu_executor=_FakeExec(0))
        with a:
            with b:
                assert deps.get() is b
            assert deps.get() is a
        assert deps.get(strict=False) is None

    def test_visible_in_child_thread(self) -> None:
        # The deps live in a module global, so a spawned thread sees them without any
        # propagation — this is the strategist/worker case that a context var would break.
        rd = deps.RunDeps(_config(gpu_ids=[], cpu_enabled=False), gpu_executor=_FakeExec(0))
        seen: dict[str, object] = {}
        with rd:
            t = threading.Thread(target=lambda: seen.update(deps=deps.get(strict=False)))
            t.start()
            t.join()
        assert seen["deps"] is rd

    def test_close_tears_down_executors(self) -> None:
        class _Cleanable(_FakeExec):
            def __init__(self) -> None:
                super().__init__(0)
                self.cleaned = False

            def cleanup_all(self) -> None:
                self.cleaned = True

        gpu = _Cleanable()
        rd = deps.RunDeps(_config(gpu_ids=[], cpu_enabled=False), gpu_executor=gpu)
        with rd:
            pass
        assert gpu.cleaned is True
