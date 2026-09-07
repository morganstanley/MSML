"""Tests for alpha_lab.deps — RunDeps construction and its context-manager lifecycle."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from alpha_lab import deps
from alpha_lab.config import Phase3Config, PipelineConfig, TaskConfig
from alpha_lab.constants import DEFAULT, Phase
from alpha_lab.local_cpu import LocalCPUManager
from alpha_lab.local_gpu import LocalGPUManager
from alpha_lab.slurm import SlurmManager

# This module tests the RunDeps publish/get/reset machinery directly, so it must run
# without the conftest autouse default deps published.
pytestmark = pytest.mark.no_run_deps


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


class TestRunDepsConstruction:
    def test_builds_local_gpu_and_cpu_from_config(self, tmp_path: Path) -> None:
        rd = deps.RunDeps(
            _config(executor="local", gpu_ids=[0, 1], cpu_enabled=True),
            run_id="test", workspace=tmp_path,
        )
        assert isinstance(rd.gpu_executor, LocalGPUManager)
        assert isinstance(rd.cpu_executor, LocalCPUManager)

    def test_cpu_is_none_when_disabled(self, tmp_path: Path) -> None:
        rd = deps.RunDeps(
            _config(executor="local", gpu_ids=[], cpu_enabled=False),
            run_id="test", workspace=tmp_path,
        )
        assert rd.cpu_executor is None

    def test_builds_slurm_executor(self, tmp_path: Path) -> None:
        rd = deps.RunDeps(
            _config(executor="slurm", cpu_enabled=False),
            run_id="test", workspace=tmp_path,
        )
        assert isinstance(rd.gpu_executor, SlurmManager)

    def test_injected_executors_skip_construction(self, tmp_path: Path) -> None:
        gpu, cpu = _FakeExec(2), _FakeExec(3)
        rd = deps.RunDeps(
            _config(gpu_ids=[], cpu_enabled=True), run_id="test", workspace=tmp_path,
            _gpu_executor=gpu, _cpu_executor=cpu,
        )
        assert rd.gpu_executor is gpu
        assert rd.cpu_executor is cpu


class TestLifecycle:
    def test_get_strict_raises_outside_scope(self) -> None:
        with pytest.raises(LookupError):
            deps.get()

    def test_get_non_strict_is_none_outside_scope(self) -> None:
        assert deps.get(strict=False) is None

    def test_with_publishes_and_auto_resets(self, tmp_path: Path) -> None:
        rd = deps.RunDeps(_config(gpu_ids=[], cpu_enabled=False), run_id="test",
                          workspace=tmp_path, _gpu_executor=_FakeExec(0))
        with rd:
            assert deps.get() is rd
        assert deps.get(strict=False) is None

    def test_nested_restores_outer(self, tmp_path: Path) -> None:
        a = deps.RunDeps(_config(gpu_ids=[], cpu_enabled=False), run_id="test",
                         workspace=tmp_path, _gpu_executor=_FakeExec(0))
        b = deps.RunDeps(_config(gpu_ids=[], cpu_enabled=False), run_id="test",
                         workspace=tmp_path, _gpu_executor=_FakeExec(0))
        with a:
            with b:
                assert deps.get() is b
            assert deps.get() is a
        assert deps.get(strict=False) is None

    def test_visible_in_child_thread(self, tmp_path: Path) -> None:
        # The deps live in a module global, so a spawned thread sees them without any
        # propagation — this is the strategist/worker case that a context var would break.
        rd = deps.RunDeps(_config(gpu_ids=[], cpu_enabled=False), run_id="test",
                          workspace=tmp_path, _gpu_executor=_FakeExec(0))
        seen: dict[str, object] = {}
        with rd:
            t = threading.Thread(target=lambda: seen.update(deps=deps.get(strict=False)))
            t.start()
            t.join()
        assert seen["deps"] is rd

    def test_close_tears_down_executors(self, tmp_path: Path) -> None:
        class _Cleanable(_FakeExec):
            def __init__(self) -> None:
                super().__init__(0)
                self.cleaned = False

            def cleanup_all(self) -> None:
                self.cleaned = True

        gpu = _Cleanable()
        rd = deps.RunDeps(_config(gpu_ids=[], cpu_enabled=False), run_id="test",
                          workspace=tmp_path, _gpu_executor=gpu)
        with rd:
            pass
        assert gpu.cleaned is True

    def test_close_releases_memory_embedding_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rd = deps.RunDeps(
            _config(gpu_ids=[], cpu_enabled=False),
            run_id="test",
            workspace=tmp_path,
            _gpu_executor=_FakeExec(0),
        )
        closed = False

        def close() -> None:
            nonlocal closed
            closed = True

        monkeypatch.setattr(rd.memory_store, "close", close)
        with rd:
            pass

        assert closed is True

    def test_close_removes_owned_temp_dir(self, tmp_path: Path) -> None:
        rd = deps.RunDeps(
            _config(gpu_ids=[], cpu_enabled=False),
            run_id="test",
            workspace=str(tmp_path),
            _gpu_executor=_FakeExec(0),
            owns_temp=True,
        )
        temp_dir = rd.temp_dir
        (temp_dir / "memory").mkdir(parents=True)
        with rd:
            assert temp_dir.exists()
        assert not temp_dir.exists()


class TestLazyInit:
    def test_construction_builds_nothing(self, tmp_path: Path) -> None:
        rd = deps.RunDeps(
            _config(executor="local", gpu_ids=[0], cpu_enabled=True),
            run_id="test", workspace=tmp_path,
        )
        assert rd._gpu_executor is DEFAULT
        assert rd._cpu_executor is DEFAULT
        assert rd._memory_store is DEFAULT

    def test_first_access_builds_and_caches(self, tmp_path: Path) -> None:
        rd = deps.RunDeps(
            _config(executor="local", gpu_ids=[0], cpu_enabled=True),
            run_id="test", workspace=tmp_path,
        )
        gpu = rd.gpu_executor
        assert isinstance(gpu, LocalGPUManager)
        assert rd.gpu_executor is gpu  # cached, not rebuilt
        assert rd._gpu_executor is gpu

    def test_close_does_not_build_unrealized(self, tmp_path: Path) -> None:
        rd = deps.RunDeps(
            _config(executor="local", gpu_ids=[0], cpu_enabled=True),
            run_id="test", workspace=tmp_path,
        )
        with rd:  # never touch the executors
            pass
        assert rd._gpu_executor is DEFAULT
        assert rd._cpu_executor is DEFAULT


class TestUpdateConfig:
    def _deps(self, tmp_path: Path) -> deps.RunDeps:
        return deps.RunDeps(
            TaskConfig(data_path="d.csv", description="orig"),
            run_id="test", workspace=tmp_path,
            _gpu_executor=_FakeExec(0), _cpu_executor=None,
        )

    def test_merges_and_mutates_in_place(self, tmp_path: Path) -> None:
        rd = self._deps(tmp_path)
        cfg = rd.config
        out = rd.update_config({"description": "new", "reasoning_effort": "high"})
        assert out is cfg  # same object, mutated in place
        assert cfg.description == "new"
        assert cfg.reasoning_effort == "high"

    def test_nested_pipeline_merge(self, tmp_path: Path) -> None:
        rd = self._deps(tmp_path)
        rd.update_config({"pipeline": {"phase3": {"max_per_gpu": 5}}})
        assert rd.config.pipeline.phase3.max_per_gpu == 5

    def test_rejects_unknown_top_level_field(self, tmp_path: Path) -> None:
        # extra="forbid" on the pydantic TaskConfig rejects unknowns on rebuild.
        rd = self._deps(tmp_path)
        with pytest.raises(ValueError):
            rd.update_config({"bogus": 1})

    def test_rejects_unknown_phase3_field(self, tmp_path: Path) -> None:
        rd = self._deps(tmp_path)
        with pytest.raises(ValueError):
            rd.update_config({"pipeline": {"phase3": {"nope": 1}}})

    def test_no_filesystem_side_effects(self, tmp_path: Path) -> None:
        rd = self._deps(tmp_path)
        rd.update_config({"description": "new"})
        assert not (tmp_path / ".alpha_lab" / "config.json").exists()


class TestMemoryStoreClose:
    def test_close_closes_present_store(self, tmp_path: Path) -> None:
        # close() always releases a present store; safety for the sandboxed
        # child lives in BackendProxy.close() being a no-op, not in
        # ownership bookkeeping here.
        from unittest.mock import MagicMock

        injected = MagicMock(name="memory_store")
        rd = deps.RunDeps(
            _config(executor="local", gpu_ids=[0]),
            run_id="test", workspace=tmp_path,
            _gpu_executor=MagicMock(), _memory_store=injected,
        )
        with rd:
            pass
        injected.close.assert_called_once_with()

    def test_close_does_not_build_store_just_to_close_it(self, tmp_path: Path) -> None:
        # Same rule the executor teardown documents: close() reads the
        # private field and must not trigger the lazy build.
        from unittest.mock import MagicMock

        rd = deps.RunDeps(
            _config(executor="local", gpu_ids=[0]),
            run_id="test", workspace=tmp_path,
            _gpu_executor=MagicMock(),
        )
        with rd:
            pass
        assert rd._memory_store is DEFAULT

    def test_child_proxy_close_is_a_no_op(self) -> None:
        # The sandboxed child's stand-in store must swallow close() locally:
        # forwarding it would close the parent's shared embedding client
        # under every other agent still using it.
        from unittest.mock import MagicMock

        from alpha_lab.sandboxing.db_proxy import BackendProxy

        channel = MagicMock(name="channel")
        proxy = BackendProxy(channel, "memory")
        proxy.close()
        channel.call.assert_not_called()
