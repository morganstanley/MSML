"""Tests for the dispatcher/executor fixes applied after the etfflows-g55
duplicate-submission cascade and orchestrator-self-kill incident.

Covers:
- F2: dispatcher refuses to resubmit when the prior job is still alive.
- F3: _cleanup_job kills the process group if the process is still alive.
- F4: each submission writes ``local_job.<job_id>.out`` plus a stable
      ``local_job.out`` symlink.
- F9: run_status.json is written on every terminal transition.
- Memory estimator: realistic, conservative numbers across the
      architectures that triggered OOMs in the prior run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from alpha_lab.experiment_db import Experiment, ExperimentDB
from alpha_lab.local_cpu import LocalCPUManager
from alpha_lab.local_gpu import LocalGPUManager


def _make_exp_row(name: str, exp_id: int = 1, config_json: str = "{}") -> Experiment:
    """Hand-build an Experiment dataclass for unit tests."""
    now = time.time()
    return Experiment(
        id=exp_id, name=name, description="d", hypothesis="h",
        status="checked", config_json=config_json,
        worker_id=None, slurm_job_id=None, results_json=None,
        error=None, debrief_path=None,
        created_at=now, updated_at=now,
        started_at=None, finished_at=None, fix_attempts=0,
    )


# ---------------------------------------------------------------------------
# F2 — dispatcher idempotent submit
# ---------------------------------------------------------------------------
class TestF2DuplicateSubmitBlocked:
    """Belt-and-suspenders on top of F1: even if a backwards status move
    sneaks through, the dispatcher must not spawn a second subprocess for
    a job that the executor still considers alive.
    """

    def test_blocks_when_prior_job_alive(self, monkeypatch, tmp_path) -> None:
        from alpha_lab.config import Phase3Config, PipelineConfig, TaskConfig
        from alpha_lab.dispatcher import Dispatcher

        db_path = tmp_path / "db.sqlite"
        db = ExperimentDB(str(db_path))
        eid = db.create("dup_exp", "D", "H", "{}")
        # Walk forward to ``checked`` and pretend a prior submission set a job id.
        for s in ("implemented", "checked"):
            db.update_status(eid, s)
        db.set_slurm_job(eid, "alive-job-1")

        exec_mock = MagicMock()
        exec_mock.is_alive.return_value = True
        exec_mock.can_submit.return_value = True
        # If the dispatcher does try to submit, the mock would record it —
        # which is exactly the regression we're guarding against.
        exec_mock.submit_experiment.return_value = "should-never-be-used"

        cfg = TaskConfig(
            data_path="/x", description="d",
            pipeline=PipelineConfig(
                phases=["phase3"],
                phase3=Phase3Config(
                    max_concurrent_gpus=1, max_experiments=10,
                    strategist_interval=300, worker_count=1,
                    slurm_partitions=["h100"], report_interval=100,
                ),
            ),
        )
        d = Dispatcher(
            provider=MagicMock(), config=cfg, workspace=str(tmp_path),
            db=db, executor=exec_mock, event_callback=lambda e: None,
            worker_count=1,
        )
        d._init_log()
        try:
            d._submit_checked()
        finally:
            d._cleanup()

        exec_mock.is_alive.assert_called_with("alive-job-1")
        exec_mock.submit_experiment.assert_not_called()
        # Status remains "checked"; the row stays in the queue for the
        # next poll cycle to discover that the prior job has terminated.
        assert db.get(eid).status == "checked"

    def test_proceeds_when_prior_job_dead(self, tmp_path) -> None:
        from alpha_lab.config import Phase3Config, PipelineConfig, TaskConfig
        from alpha_lab.dispatcher import Dispatcher

        db = ExperimentDB(str(tmp_path / "db.sqlite"))
        eid = db.create("dead_exp", "D", "H", "{}")
        for s in ("implemented", "checked"):
            db.update_status(eid, s)
        db.set_slurm_job(eid, "dead-job-1")

        exec_mock = MagicMock()
        exec_mock.is_alive.return_value = False  # prior job already done
        exec_mock.can_submit.return_value = True
        exec_mock.submit_experiment.return_value = "new-job"

        cfg = TaskConfig(
            data_path="/x", description="d",
            pipeline=PipelineConfig(
                phases=["phase3"],
                phase3=Phase3Config(
                    max_concurrent_gpus=1, max_experiments=10,
                    strategist_interval=300, worker_count=1,
                    slurm_partitions=["h100"], report_interval=100,
                ),
            ),
        )
        d = Dispatcher(
            provider=MagicMock(), config=cfg, workspace=str(tmp_path),
            db=db, executor=exec_mock, event_callback=lambda e: None,
            worker_count=1,
        )
        d._init_log()
        try:
            d._submit_checked()
        finally:
            d._cleanup()

        exec_mock.submit_experiment.assert_called_once()
        assert db.get(eid).status == "queued"


# ---------------------------------------------------------------------------
# F4 — per-job log filename + stable symlink
# ---------------------------------------------------------------------------
class TestF4PerJobLogFile:
    """Each submission writes to local_job.<job_id>.out and updates a
    stable local_job.out symlink; the prior run's log is never truncated.
    """

    # The LocalGPU wrapper imports torch AND lightning.pytorch before
    # running anything (to install the deterministic-mode interception
    # patches). Measured on this host: torch alone 6-9s, with lightning
    # 40-50s. The 60s budget the test used to set was barely above that
    # on a quiet machine and below it on a busy one — flaky. 180s gives
    # the slow-import path enough headroom to land on either.
    _SUB_TIMEOUT = 180

    def test_log_filename_uses_job_id_and_symlink_points_to_latest(
        self, tmp_path
    ) -> None:
        mgr = LocalGPUManager(
            gpu_ids=[0], max_per_gpu=4,
            time_limit_seconds=120, python_executable=sys.executable,
        )
        exp = _make_exp_row("log_fname")
        exp_dir = tmp_path / "experiments" / exp.name
        exp_dir.mkdir(parents=True)

        # Cheap python that exits quickly so we don't have to wait.
        # ``run_experiment.py`` is what the runner imports via runpy.
        (exp_dir / "run_experiment.py").write_text(
            "import sys; print('hello', flush=True); sys.exit(0)\n"
        )

        # Disable nvidia-smi probe so we don't depend on a real GPU.
        mgr._get_gpu_memory_free = lambda: {}
        mgr._estimate_memory_requirement = lambda exp_, ws: 100

        job_id = mgr.submit_experiment(exp, str(tmp_path))
        mgr._jobs[job_id].proc.wait(timeout=self._SUB_TIMEOUT)
        mgr.poll_jobs([job_id])  # transitions through COMPLETED + cleanup

        per_job_log = exp_dir / f"local_job.{job_id}.out"
        symlink = exp_dir / "local_job.out"

        assert per_job_log.exists(), "per-job log file must be created"
        assert symlink.is_symlink(), "local_job.out must be a symlink"
        assert os.readlink(symlink) == per_job_log.name
        # Sanity: the symlink resolves to the per-job file.
        assert symlink.resolve() == per_job_log.resolve()

    def test_resubmit_preserves_prior_log(self, tmp_path) -> None:
        mgr = LocalGPUManager(
            gpu_ids=[0], max_per_gpu=4,
            time_limit_seconds=120, python_executable=sys.executable,
        )
        exp = _make_exp_row("preserve")
        exp_dir = tmp_path / "experiments" / exp.name
        exp_dir.mkdir(parents=True)
        (exp_dir / "run_experiment.py").write_text(
            "import sys\n"
            "print('first', flush=True)\n"
            "sys.exit(0)\n"
        )
        mgr._get_gpu_memory_free = lambda: {}
        mgr._estimate_memory_requirement = lambda exp_, ws: 100

        job1 = mgr.submit_experiment(exp, str(tmp_path))
        mgr._jobs[job1].proc.wait(timeout=self._SUB_TIMEOUT)
        mgr.poll_jobs([job1])
        first_log = (exp_dir / f"local_job.{job1}.out").read_text()
        assert "first" in first_log

        # Second submission with different output.
        (exp_dir / "run_experiment.py").write_text(
            "import sys\n"
            "print('second', flush=True)\n"
            "sys.exit(0)\n"
        )
        job2 = mgr.submit_experiment(exp, str(tmp_path))
        assert job2 != job1
        mgr._jobs[job2].proc.wait(timeout=self._SUB_TIMEOUT)
        mgr.poll_jobs([job2])

        # The first log file still exists and still says "first" — was not
        # truncated by the second open(..., "w").
        assert (exp_dir / f"local_job.{job1}.out").exists()
        assert "first" in (exp_dir / f"local_job.{job1}.out").read_text()
        # The symlink now points to job2's log.
        assert os.readlink(exp_dir / "local_job.out") == f"local_job.{job2}.out"


# ---------------------------------------------------------------------------
# F3 — pgid kill on cleanup / supersede on resubmit
# ---------------------------------------------------------------------------
class TestF3PgidKill:
    """A live subprocess at cleanup time must be SIGTERM'd, not just
    have its captured output_file closed.
    """

    def test_cleanup_kills_live_process(self, tmp_path) -> None:
        mgr = LocalGPUManager(
            gpu_ids=[0], max_per_gpu=4,
            time_limit_seconds=300, python_executable=sys.executable,
        )
        # A run_experiment.py that sleeps long enough that we explicitly
        # call _cleanup_job while it's still running.
        exp = _make_exp_row("kill_me")
        exp_dir = tmp_path / "experiments" / exp.name
        exp_dir.mkdir(parents=True)
        (exp_dir / "run_experiment.py").write_text(
            "import time, sys; sys.stdout.flush(); time.sleep(60)\n"
        )
        mgr._get_gpu_memory_free = lambda: {}
        mgr._estimate_memory_requirement = lambda exp_, ws: 100

        job_id = mgr.submit_experiment(exp, str(tmp_path))
        # Verify the process is up.
        time.sleep(0.5)
        assert mgr.is_alive(job_id), "subprocess should be alive before cleanup"

        # Cleanup must terminate the pgid.
        mgr._cleanup_job(job_id)

        # Give the OS a moment to reap.
        deadline = time.time() + 5
        while mgr.is_alive(job_id) and time.time() < deadline:
            time.sleep(0.1)
        assert not mgr.is_alive(job_id), "subprocess should be killed by _cleanup_job"

    def test_supersede_kills_prior_job_on_resubmit(self, tmp_path) -> None:
        mgr = LocalGPUManager(
            gpu_ids=[0], max_per_gpu=4,
            time_limit_seconds=300, python_executable=sys.executable,
        )
        exp = _make_exp_row("supersede")
        exp_dir = tmp_path / "experiments" / exp.name
        exp_dir.mkdir(parents=True)
        (exp_dir / "run_experiment.py").write_text(
            "import time, sys; sys.stdout.flush(); time.sleep(60)\n"
        )
        mgr._get_gpu_memory_free = lambda: {}
        mgr._estimate_memory_requirement = lambda exp_, ws: 100

        job1 = mgr.submit_experiment(exp, str(tmp_path))
        time.sleep(0.5)
        assert mgr.is_alive(job1)

        # Resubmission with the same exp.name must supersede job1.
        job2 = mgr.submit_experiment(exp, str(tmp_path))
        deadline = time.time() + 5
        while mgr.is_alive(job1) and time.time() < deadline:
            time.sleep(0.1)
        assert not mgr.is_alive(job1), "job1 must be killed by supersede-on-submit"
        assert mgr.is_alive(job2)

        # Clean up job2 too.
        mgr._cleanup_job(job2)


# ---------------------------------------------------------------------------
# F9 — run_status.json structured exit summary
# ---------------------------------------------------------------------------
class TestF9RunStatusJson:
    """poll_jobs must drop a run_status.json next to each experiment on
    every terminal transition (COMPLETED, FAILED, TIMEOUT). Analyzer
    prompts read this file in preference to grepping raw stdout.
    """

    # See TestF4PerJobLogFile._SUB_TIMEOUT for the rationale — torch +
    # lightning imports take 40-50s on this host, well past 60s on a
    # busy machine.
    _SUB_TIMEOUT = 180

    def test_completed_writes_run_status(self, tmp_path) -> None:
        mgr = LocalGPUManager(
            gpu_ids=[0], max_per_gpu=4,
            time_limit_seconds=120, python_executable=sys.executable,
        )
        exp = _make_exp_row("completed")
        exp_dir = tmp_path / "experiments" / exp.name
        exp_dir.mkdir(parents=True)
        (exp_dir / "run_experiment.py").write_text(
            "import sys; print('done', flush=True); sys.exit(0)\n"
        )
        mgr._get_gpu_memory_free = lambda: {}
        mgr._estimate_memory_requirement = lambda exp_, ws: 100

        job_id = mgr.submit_experiment(exp, str(tmp_path))
        mgr._jobs[job_id].proc.wait(timeout=self._SUB_TIMEOUT)
        statuses = mgr.poll_jobs([job_id])
        assert statuses[job_id] == "COMPLETED"

        rs = exp_dir / "run_status.json"
        assert rs.exists()
        payload = json.loads(rs.read_text())
        assert payload["status"] == "COMPLETED"
        assert payload["returncode"] == 0
        assert payload["error_signature"] is None
        assert payload["exp_name"] == "completed"
        assert payload["executor"] == "local_gpu"
        assert payload["job_id"] == job_id

    def test_failed_writes_run_status_with_signature(self, tmp_path) -> None:
        mgr = LocalGPUManager(
            gpu_ids=[0], max_per_gpu=4,
            time_limit_seconds=120, python_executable=sys.executable,
        )
        exp = _make_exp_row("failed")
        exp_dir = tmp_path / "experiments" / exp.name
        exp_dir.mkdir(parents=True)
        # Trigger a TypeError that matches the framework_api_mismatch
        # signature so we exercise the classifier.
        (exp_dir / "run_experiment.py").write_text(
            "def f(): pass\n"
            "f(unexpected='kwarg')  # raises TypeError: got an unexpected keyword argument\n"
        )
        mgr._get_gpu_memory_free = lambda: {}
        mgr._estimate_memory_requirement = lambda exp_, ws: 100

        job_id = mgr.submit_experiment(exp, str(tmp_path))
        mgr._jobs[job_id].proc.wait(timeout=self._SUB_TIMEOUT)
        statuses = mgr.poll_jobs([job_id])
        assert statuses[job_id] == "FAILED"

        rs = exp_dir / "run_status.json"
        payload = json.loads(rs.read_text())
        assert payload["status"] == "FAILED"
        assert payload["returncode"] != 0
        assert payload["error_signature"] == "framework_api_mismatch"
        assert "got an unexpected keyword argument" in payload["last_lines"]


# ---------------------------------------------------------------------------
# Memory estimator (replaces the broken ~1.3 GB heuristic)
# ---------------------------------------------------------------------------
class TestMemoryEstimator:
    """Sanity-check the rewritten heuristic against the three architectures
    that OOM'd in workspace-etfflows-g55 and against a tree model.
    """

    @pytest.fixture()
    def mgr(self) -> LocalGPUManager:
        return LocalGPUManager(
            gpu_ids=[0], max_per_gpu=4,
            time_limit_seconds=60, python_executable=sys.executable,
        )

    def _write_config(self, tmp_path: Path, name: str, cfg: dict) -> Experiment:
        exp_dir = tmp_path / "experiments" / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        import yaml
        (exp_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
        return _make_exp_row(name)

    def test_tft_at_prompt_caps_is_not_tiny(self, mgr, tmp_path) -> None:
        exp = self._write_config(tmp_path, "tft_cap", {
            "model": {"model_type": "tft"},
            "hyperparams": {
                "hidden_dim": 128, "num_layers": 3, "attention_heads": 4,
            },
            "training": {"batch_size": 32, "context_length": 128},
        })
        mem = mgr._estimate_memory_requirement(exp, str(tmp_path))
        # Prior heuristic returned ~1.3 GB for this; the actual job ate
        # 62 GB. The new estimate must be at least an order of magnitude
        # larger than 1.3 GB, capped at our 70 GB ceiling.
        assert mem >= 20_000, f"TFT estimate too small: {mem} MB"
        assert mem <= 70_000

    def test_mamba_uses_high_base(self, mgr, tmp_path) -> None:
        exp = self._write_config(tmp_path, "mamba_cap", {
            "model": {"model_type": "mamba"},
            "hyperparams": {"hidden_dim": 128, "num_layers": 3},
            "training": {"batch_size": 32, "context_length": 128},
        })
        mem = mgr._estimate_memory_requirement(exp, str(tmp_path))
        assert mem >= 20_000, f"Mamba estimate too small: {mem} MB"

    def test_tabtransformer_uses_high_base(self, mgr, tmp_path) -> None:
        exp = self._write_config(tmp_path, "tab", {
            "model": {"model_type": "tabtransformer"},
            "hyperparams": {"hidden_dim": 128, "num_layers": 3},
            "training": {"batch_size": 32, "context_length": 128},
        })
        mem = mgr._estimate_memory_requirement(exp, str(tmp_path))
        assert mem >= 20_000, f"TabTransformer estimate too small: {mem} MB"

    def test_lightgbm_is_tiny(self, mgr, tmp_path) -> None:
        exp = self._write_config(tmp_path, "lgbm", {
            "model": {"model_type": "lightgbm"},
            "training": {"batch_size": 4096},
        })
        mem = mgr._estimate_memory_requirement(exp, str(tmp_path))
        assert mem == 2_000

    def test_missing_config_uses_conservative_default(self, mgr, tmp_path) -> None:
        # No config.yaml on disk.
        exp = _make_exp_row("nocfg")
        mem = mgr._estimate_memory_requirement(exp, str(tmp_path))
        assert mem == 20_000

    def test_pick_gpu_refuses_when_estimate_exceeds_free(
        self, mgr, tmp_path
    ) -> None:
        """If every GPU has less free memory than the estimate, _pick_gpu
        must refuse rather than co-tenant a heavy job onto a full GPU.
        """
        exp = self._write_config(tmp_path, "heavy", {
            "model": {"model_type": "tft"},
            "hyperparams": {"hidden_dim": 256, "num_layers": 4},
            "training": {"batch_size": 64, "context_length": 256},
        })
        mgr._workspace = str(tmp_path)
        # Pretend every GPU has only 5 GB free.
        mgr._get_gpu_memory_free = lambda: {0: 5_000}
        assert mgr._pick_gpu(exp, str(tmp_path)) is None


# ---------------------------------------------------------------------------
# CPU manager sanity
# ---------------------------------------------------------------------------
class TestF13RecoverRunningJobsWired:
    """Dispatcher.recover() must call executor.recover_running_jobs() so
    live subprocesses that outlived a previous orchestrator get re-attached
    to the new executor's tracking before poll_jobs would otherwise return
    UNKNOWN and stamp ``SLURM job lost`` on rows whose python is in fact
    still training.
    """

    def test_gpu_recover_called_before_poll(self, tmp_path) -> None:
        from alpha_lab.config import Phase3Config, PipelineConfig, TaskConfig
        from alpha_lab.dispatcher import Dispatcher

        db = ExperimentDB(str(tmp_path / "db.sqlite"))
        eid = db.create("recover_target", "D", "H", "{}")
        for s in ("implemented", "checked", "queued", "running"):
            db.update_status(eid, s)
        db.set_slurm_job(eid, "live-gpu-1")

        # Order matters: recover_running_jobs MUST be called before
        # poll_jobs so the reattached job shows up as RUNNING, not UNKNOWN.
        calls: list[str] = []
        exec_mock = MagicMock()
        exec_mock.recover_running_jobs = MagicMock(
            side_effect=lambda ws, jmap: (calls.append("recover"), 1)[1]
        )
        # After reattach, poll_jobs should be able to see the job — we
        # simulate that by returning RUNNING.
        exec_mock.poll_jobs.side_effect = (
            lambda jids: (calls.append("poll"), {jid: "RUNNING" for jid in jids})[1]
        )

        cfg = TaskConfig(
            data_path="/x", description="d",
            pipeline=PipelineConfig(
                phases=["phase3"],
                phase3=Phase3Config(
                    max_concurrent_gpus=1, max_experiments=10,
                    strategist_interval=300, worker_count=1,
                    slurm_partitions=["h100"], report_interval=100,
                ),
            ),
        )
        d = Dispatcher(
            provider=MagicMock(), config=cfg, workspace=str(tmp_path),
            db=db, executor=exec_mock, event_callback=lambda e: None,
            worker_count=1,
        )
        d._init_log()
        try:
            summary = d.recover()
        finally:
            d._cleanup()

        # recover_running_jobs called before poll_jobs:
        assert calls == ["recover", "poll"]
        exec_mock.recover_running_jobs.assert_called_once()
        args, _ = exec_mock.recover_running_jobs.call_args
        assert args[0] == str(tmp_path)
        assert args[1] == {"live-gpu-1": "recover_target"}
        assert summary["reattached_jobs"] == 1
        # The row stays as ``running`` because the reattached job
        # responded RUNNING to the poll instead of UNKNOWN.
        assert db.get(eid).status == "running"

    def test_cpu_recover_routes_poll_to_cpu_executor(self, tmp_path) -> None:
        """A reattached CPU job must be polled by cpu_executor, not the GPU
        executor — otherwise UNKNOWN comes back and the row is mislabeled.
        """
        from alpha_lab.config import Phase3Config, PipelineConfig, TaskConfig
        from alpha_lab.dispatcher import Dispatcher

        db = ExperimentDB(str(tmp_path / "db.sqlite"))
        eid = db.create("cpu_alive", "D", "H", "{}")
        for s in ("implemented", "checked", "queued", "running"):
            db.update_status(eid, s)
        db.set_slurm_job(eid, "cpu-live-1")

        gpu_mock = MagicMock()
        gpu_mock.recover_running_jobs = MagicMock(return_value=0)
        gpu_mock.poll_jobs = MagicMock(return_value={})  # GPU should NOT see this id

        cpu_mock = MagicMock()
        # CPU executor "recovers" the job and marks it alive.
        cpu_mock.recover_running_jobs = MagicMock(return_value=1)
        cpu_mock.is_alive = MagicMock(return_value=True)
        cpu_mock.poll_jobs = MagicMock(return_value={"cpu-live-1": "RUNNING"})

        cfg = TaskConfig(
            data_path="/x", description="d",
            pipeline=PipelineConfig(
                phases=["phase3"],
                phase3=Phase3Config(
                    max_concurrent_gpus=1, max_experiments=10,
                    strategist_interval=300, worker_count=1,
                    slurm_partitions=["h100"], report_interval=100,
                ),
            ),
        )
        d = Dispatcher(
            provider=MagicMock(), config=cfg, workspace=str(tmp_path),
            db=db, executor=gpu_mock, event_callback=lambda e: None,
            worker_count=1, cpu_executor=cpu_mock,
        )
        d._init_log()
        try:
            summary = d.recover()
        finally:
            d._cleanup()

        # Both recover_running_jobs were called.
        gpu_mock.recover_running_jobs.assert_called_once()
        cpu_mock.recover_running_jobs.assert_called_once()
        # The cpu job id was added to _cpu_job_ids so subsequent
        # _poll_slurm calls route it to the CPU executor.
        assert "cpu-live-1" in d._cpu_job_ids
        # poll_jobs went to cpu_executor (since the id is now in _cpu_job_ids)
        # and NOT to the GPU executor.
        cpu_mock.poll_jobs.assert_called_once_with(["cpu-live-1"])
        gpu_mock.poll_jobs.assert_not_called()
        assert summary["reattached_jobs"] == 1
        assert db.get(eid).status == "running"

    def test_recover_handles_executor_without_recover_method(self, tmp_path) -> None:
        """SLURM executor doesn't implement recover_running_jobs and that's
        fine — recovery must fall through to its existing poll-based path."""
        from alpha_lab.config import Phase3Config, PipelineConfig, TaskConfig
        from alpha_lab.dispatcher import Dispatcher

        db = ExperimentDB(str(tmp_path / "db.sqlite"))
        eid = db.create("slurm_path", "D", "H", "{}")
        for s in ("implemented", "checked", "queued", "running"):
            db.update_status(eid, s)
        db.set_slurm_job(eid, "12345")

        slurm_mock = MagicMock(spec=["poll_jobs", "submit_experiment", "cancel",
                                     "can_submit", "running_gpu_count", "is_alive"])
        slurm_mock.poll_jobs.return_value = {"12345": "RUNNING"}

        cfg = TaskConfig(
            data_path="/x", description="d",
            pipeline=PipelineConfig(
                phases=["phase3"],
                phase3=Phase3Config(
                    max_concurrent_gpus=1, max_experiments=10,
                    strategist_interval=300, worker_count=1,
                    slurm_partitions=["h100"], report_interval=100,
                ),
            ),
        )
        d = Dispatcher(
            provider=MagicMock(), config=cfg, workspace=str(tmp_path),
            db=db, executor=slurm_mock, event_callback=lambda e: None,
            worker_count=1,
        )
        d._init_log()
        try:
            summary = d.recover()
        finally:
            d._cleanup()

        # No reattach happened (SLURM doesn't need it).
        assert summary["reattached_jobs"] == 0
        # Existing poll-based reconciliation still fires.
        slurm_mock.poll_jobs.assert_called_once()
        # Row left as running because SLURM reported RUNNING.
        assert db.get(eid).status == "running"


class TestCPUManagerParity:
    """LocalCPUManager mirrors LocalGPUManager's idempotency + log layout."""

    def test_is_alive_reflects_proc_state(self, tmp_path) -> None:
        mgr = LocalCPUManager(
            max_parallel=2, time_limit_seconds=60,
            python_executable=sys.executable,
        )
        exp = _make_exp_row("cpu_alive")
        exp_dir = tmp_path / "experiments" / exp.name
        exp_dir.mkdir(parents=True)
        (exp_dir / "run_experiment.py").write_text(
            "import time; time.sleep(0.5)\n"
        )

        job_id = mgr.submit_experiment(exp, str(tmp_path))
        assert mgr.is_alive(job_id)
        # Wait for completion via the bash parent's proc handle.
        mgr._jobs[job_id].proc.wait(timeout=30)
        mgr.poll_jobs([job_id])
        assert not mgr.is_alive(job_id)
