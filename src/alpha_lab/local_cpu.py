"""Local CPU job management for Phase 3 experiment dispatch.

Runs tree-based models and data preprocessing jobs on CPU while GPU
experiments run on LocalGPUManager. Same interface as LocalGPUManager.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from alpha_lab.experiment_db import Experiment
# reuse _read_log_tail/_classify_error and RecoveredProcess from local_gpu
from alpha_lab.local_gpu import LocalGPUManager, RecoveredProcess

logger = logging.getLogger("alpha_lab.local_cpu")

RUN_SCRIPT = """\
#!/bin/bash
cd {exp_dir}
export PYTHONPATH={workspace}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=""  # Disable GPU
{data_env}
{python_exe} -c "
import runpy, sys
sys.argv = ['run_experiment.py']
runpy.run_path('run_experiment.py', run_name='__main__')
"
"""


@dataclass
class LocalCPUJob:
    """Tracks a running local CPU subprocess."""
    proc: subprocess.Popen
    exp_name: str
    output_file: TextIO
    workspace: str
    start_time: float = 0.0


class LocalCPUManager:
    """Manages local CPU job spawning for tree-based models.

    Same interface as LocalGPUManager:
    - submit_experiment(exp, workspace) -> job_id
    - poll_jobs(job_ids) -> {job_id: status}
    - cancel(job_id)
    - can_submit() -> bool
    - running_count() -> int
    """

    def __init__(
        self,
        max_parallel: int = 4,
        time_limit_seconds: int = 3600,
        python_executable: str = "",
        data_path: str = "",
    ) -> None:
        """
        Parameters
        ----------
        max_parallel : int
            Max concurrent CPU experiments.
        time_limit_seconds : int
            Subprocess timeout.
        python_executable : str
            Path to the Python interpreter for experiment subprocesses.
            Empty string (default) uses sys.executable.
        """
        self.max_parallel = max_parallel
        self.time_limit = time_limit_seconds
        self.python_executable = python_executable or sys.executable
        # See LocalGPUManager: exported to jobs as ALPHALAB_DATA_PATH/ROOT.
        self.data_path = data_path
        self._jobs: dict[str, LocalCPUJob] = {}

    def running_count(self) -> int:
        """Count number of jobs currently running."""
        count = 0
        for job in self._jobs.values():
            if job.proc.poll() is None:
                count += 1
        return count

    def can_submit(self, exp: Experiment | None = None) -> bool:
        """Check if we have capacity to submit another job.

        Two gates:

        1. Local slot count: ``running_count() < max_parallel``. This is
           the executor's own accounting — only counts jobs *we*
           submitted.
        2. Host-level resource gate: ``host_has_capacity()`` reads
           ``/proc/meminfo`` and ``/proc/loadavg`` so the dispatcher
           refuses to submit when system RAM or loadavg is already
           saturated by other users' jobs, orphaned alphalab processes
           the recovery scan missed, or system services. Without this
           gate the CPU executor would happily submit 40 LightGBM jobs
           onto a box where another user has pinned 250 GB of RAM,
           leading to Bus-error core dumps when mmap'd parquet pages
           can't be reread fast enough under memory pressure.
        """
        from alpha_lab.host import host_has_capacity
        if self.running_count() >= self.max_parallel:
            return False
        return host_has_capacity()

    def is_alive(self, job_id: str) -> bool:
        """Return True iff job_id is tracked and its process is still running."""
        job = self._jobs.get(job_id)
        return job is not None and job.proc.poll() is None

    def recover_running_jobs(self, workspace: str, job_id_map: dict[str, str]) -> int:
        """Discover already-running CPU experiment subprocesses and resume
        tracking them — analogue of LocalGPUManager.recover_running_jobs.

        Without this, a dispatcher restart loses track of CPU jobs that
        outlived the previous orchestrator: poll_jobs has no entry for
        them, returns UNKNOWN, and the DB row is stamped "SLURM job lost"
        even though the python subprocess is still happily training.
        """
        recovered = 0
        try:
            # See local_gpu.recover_running_jobs for the rationale —
            # ``ps aux`` can exceed 5s under heavy host load, causing
            # silent recovery failures.
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True, text=True, timeout=30,
            )
            for line in result.stdout.split("\n"):
                # CPU run script is run_cpu.sh (mirror of run_local.sh for GPU)
                if "run_cpu.sh" not in line or "bash" not in line or "grep" in line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                pid = int(parts[1])
                try:
                    # See local_gpu.recover_running_jobs for rationale —
                    # 1s readlink timeout was the silent cause of 0
                    # recoveries on a heavily-loaded box.
                    cwd_result = subprocess.run(
                        ["readlink", f"/proc/{pid}/cwd"],
                        capture_output=True, text=True, timeout=10,
                    )
                    cwd = cwd_result.stdout.strip()
                except Exception:
                    continue
                if "/experiments/" not in cwd:
                    continue
                exp_name = Path(cwd).name
                job_id = None
                for jid, ename in job_id_map.items():
                    if ename == exp_name:
                        job_id = jid
                        break
                if not job_id:
                    continue
                try:
                    proc = RecoveredProcess(pid)
                    # Append to whatever the prior submission wrote; F4 has
                    # the canonical per-job file but we can't always tell
                    # which one the recovered pid belongs to, so log into
                    # the symlinked latest if present, else /dev/null.
                    output_path = Path(cwd) / "cpu_job.out"
                    try:
                        output_file = open(output_path, "a")
                    except Exception:
                        output_file = open(os.devnull, "w")
                    job = LocalCPUJob(
                        proc=proc, exp_name=exp_name,
                        output_file=output_file, workspace=workspace,
                        start_time=0.0,  # Unknown — won't enforce timeout for recovered jobs
                    )
                    self._jobs[job_id] = job
                    recovered += 1
                    logger.info(f"Recovered CPU job {job_id} ({exp_name}), PID {pid}")
                except Exception as e:
                    logger.warning(f"Failed to recover CPU job for {exp_name}: {e}")
        except Exception as e:
            logger.error(f"CPU job recovery failed: {e}")
        return recovered

    def submit_experiment(self, exp: Experiment, workspace: str) -> str:
        """Spawn experiment as subprocess. Returns job ID (UUID)."""
        # Supersede any live prior job for the same experiment name so we
        # don't accumulate orphans (see local_gpu.submit_experiment).
        for prior_id, prior_job in list(self._jobs.items()):
            if prior_job.exp_name == exp.name and prior_job.proc.poll() is None:
                logger.warning(
                    "Superseding live CPU job %s for %s before resubmit",
                    prior_id, exp.name,
                )
                self._terminate_pgid(prior_job, "superseded by new submission")
                self._cleanup_job(prior_id)

        if not self.can_submit():
            raise RuntimeError(f"No CPU slot available (all {self.max_parallel} in use)")

        exp_dir = Path(workspace) / "experiments" / exp.name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Generate job id before opening files so we can name the log file
        # with it and avoid clobbering a prior submission's output.
        job_id = f"cpu-{str(uuid.uuid4())[:8]}"

        # Write run script
        data_env = ""
        if self.data_path:
            data_env = (
                f'export ALPHALAB_DATA_PATH="{self.data_path}"\n'
                f'export ALPHALAB_DATA_ROOT="{self.data_path}"'
            )
        script_content = RUN_SCRIPT.format(
            exp_dir=str(exp_dir),
            workspace=workspace,
            python_exe=self.python_executable,
            data_env=data_env,
        )
        script_path = exp_dir / "run_cpu.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        # Per-job log + stable symlink (see local_gpu for rationale).
        output_path = exp_dir / f"cpu_job.{job_id}.out"
        output_file = open(output_path, "w")
        latest_link = exp_dir / "cpu_job.out"
        try:
            if latest_link.is_symlink() or latest_link.exists():
                try:
                    latest_link.unlink()
                except OSError as unlink_err:
                    logger.warning("could not unlink old cpu_job.out: %s", unlink_err)
            latest_link.symlink_to(output_path.name)
        except OSError as link_err:
            logger.warning("cpu_job.out symlink failed: %s", link_err)

        try:
            from alpha_lab.tools import _preexec_setup, _register_subprocess
            proc = subprocess.Popen(
                ["bash", str(script_path)],
                stdout=output_file,
                stderr=subprocess.STDOUT,
                cwd=str(exp_dir),
                # New process group + PR_SET_PDEATHSIG SIGKILL (Linux):
                # CPU experiment dies with the parent on restart instead
                # of orphaning. See local_gpu.py for the full rationale.
                preexec_fn=_preexec_setup,
            )
            _register_subprocess(proc)
        except Exception:
            output_file.close()
            raise

        self._jobs[job_id] = LocalCPUJob(
            proc=proc,
            exp_name=exp.name,
            output_file=output_file,
            workspace=workspace,
            start_time=time.time(),
        )

        logger.info(f"Submitted CPU job {job_id} for {exp.name} (PID {proc.pid})")
        return job_id

    def poll_jobs(self, job_ids: list[str]) -> dict[str, str]:
        """Poll for job statuses."""
        result: dict[str, str] = {}
        now = time.time()

        for job_id in job_ids:
            if job_id not in self._jobs:
                result[job_id] = "UNKNOWN"
                continue

            job = self._jobs[job_id]
            retcode = job.proc.poll()

            if retcode is None:
                # ``start_time == 0.0`` marks a recovered orphan whose
                # actual start time is unknown — skip the time-limit
                # check or we'd kill every recovered job immediately
                # (elapsed = now - 0 is ~10^9 s).
                if job.start_time > 0:
                    elapsed = now - job.start_time
                    over_limit = self.time_limit > 0 and elapsed > self.time_limit
                else:
                    elapsed = 0.0
                    over_limit = False
                if over_limit:
                    logger.warning(
                        f"CPU job {job_id} ({job.exp_name}) exceeded time limit "
                        f"({elapsed:.0f}s > {self.time_limit}s), killing"
                    )
                    self._write_run_status(job_id, "TIMEOUT", -signal.SIGKILL)
                    self._kill_job(job_id)
                    result[job_id] = "TIMEOUT"
                else:
                    result[job_id] = "RUNNING"
            elif retcode == 0:
                result[job_id] = "COMPLETED"
                self._write_run_status(job_id, "COMPLETED", 0)
                self._cleanup_job(job_id)
            else:
                result[job_id] = "FAILED"
                logger.warning(f"CPU job {job_id} ({job.exp_name}) failed with exit code {retcode}")
                self._write_run_status(job_id, "FAILED", retcode)
                self._cleanup_job(job_id)

        return result

    def _write_run_status(self, job_id: str, status: str, returncode: int) -> None:
        """Write run_status.json for a CPU job (analogue of LocalGPUManager._write_run_status)."""
        job = self._jobs.get(job_id)
        if job is None:
            return
        exp_dir = Path(job.workspace) / "experiments" / job.exp_name
        per_job_log = exp_dir / f"cpu_job.{job_id}.out"
        log_path = per_job_log if per_job_log.exists() else (exp_dir / "cpu_job.out")
        tail = LocalGPUManager._read_log_tail(log_path) if log_path.exists() else ""
        finished_at = time.time()
        wall = (finished_at - job.start_time) if job.start_time else None
        payload = {
            "job_id": job_id,
            "executor": "local_cpu",
            "exp_name": job.exp_name,
            "status": status,
            "returncode": returncode,
            "started_at": job.start_time or None,
            "finished_at": finished_at,
            "wall_seconds": wall,
            "log_path": str(log_path) if log_path.exists() else None,
            "last_lines": tail,
            "error_signature": (
                LocalGPUManager._classify_error(tail) if status != "COMPLETED" else None
            ),
        }
        try:
            (exp_dir / "run_status.json").write_text(
                json.dumps(payload, indent=2, default=str)
            )
        except OSError as e:
            logger.warning("write run_status.json for CPU %s failed: %s", job.exp_name, e)

    def _terminate_pgid(self, job: LocalCPUJob, why: str) -> None:
        """SIGTERM the job's process group, wait, escalate to SIGKILL, reap."""
        if job.proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(job.proc.pid), signal.SIGTERM)
            job.proc.wait(timeout=5)
        except (ProcessLookupError, ChildProcessError):
            return
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(job.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, ChildProcessError):
                return
            try:
                job.proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                pass
        except OSError as e:
            logger.warning("Failed to kill pgid for CPU job %s: %s", job.exp_name, e)
            return
        logger.info("Killed pgid for CPU job (%s, %s)", job.exp_name, why)

    def _kill_job(self, job_id: str) -> None:
        """Kill a job's process."""
        if job_id not in self._jobs:
            return
        self._terminate_pgid(self._jobs[job_id], "explicit kill")
        self._cleanup_job(job_id)

    def cancel(self, job_id: str) -> None:
        """Kill a running job."""
        if job_id not in self._jobs:
            logger.warning(f"Cannot cancel unknown CPU job {job_id}")
            return
        job = self._jobs[job_id]
        logger.info(f"Cancelling CPU job {job_id} ({job.exp_name})")
        self._kill_job(job_id)

    def _cleanup_job(self, job_id: str) -> None:
        """Close output file. If the process is still alive (children
        outlived the parent), kill the pgid first."""
        if job_id in self._jobs:
            job = self._jobs[job_id]
            if job.proc.poll() is None:
                self._terminate_pgid(job, "still alive at cleanup")
            try:
                job.output_file.close()
            except OSError as e:
                logger.warning("Failed to close output file for CPU job %s: %s", job_id, e)
            # Unregister from parent-death cleanup set (see local_gpu.py).
            try:
                from alpha_lab.tools import _unregister_subprocess
                if isinstance(job.proc, subprocess.Popen):
                    _unregister_subprocess(job.proc)
            except Exception:
                pass

    def cleanup_all(self) -> None:
        """Kill all running jobs. Call on shutdown."""
        for job_id in list(self._jobs.keys()):
            job = self._jobs[job_id]
            if job.proc.poll() is None:
                self.cancel(job_id)
        self._jobs.clear()


def is_cpu_experiment(exp: Experiment, workspace: str = "") -> bool:
    """Check if an experiment should run on CPU.

    Priority:
    1. Explicit resource tag in config ("cpu"/"gpu") — always honored.
    2. Source code scan — if neither strategy.py nor run_experiment.py
       references CUDA, the experiment cannot use the GPU.
    """
    import json
    try:
        config = json.loads(exp.config_json or "{}")
        resource = config.get("resource", "").lower() if isinstance(config, dict) else ""

        if resource == "cpu":
            return True
        if resource == "gpu":
            return False
    except (json.JSONDecodeError, TypeError):
        pass

    # Scan experiment source files for GPU usage indicators. Only classify
    # as CPU if we actually read a file and found no GPU markers; a missing
    # or unreadable source tree falls through to GPU so we don't silently
    # route GPU workloads onto the CPU manager.
    if workspace:
        exp_dir = Path(workspace) / "experiments" / exp.name
        gpu_markers = {"torch.device", ".cuda()", "CUDA", "torch.cuda", ".to(device",
                       ".to(self.device", "gpu_id", "accelerator"}
        scanned_source = False
        for filename in ("strategy.py", "run_experiment.py"):
            src_path = exp_dir / filename
            if src_path.exists():
                try:
                    source = src_path.read_text()
                    scanned_source = True
                    if any(marker in source for marker in gpu_markers):
                        return False
                except OSError:
                    pass

        return scanned_source

    # No workspace provided, can't scan — fall back to False (assume GPU)
    return False
