"""Local CPU job management for Phase 3 experiment dispatch.

Runs tree-based models and data preprocessing jobs on CPU while GPU
experiments run on LocalGPUManager. Same interface as LocalGPUManager.
"""

from __future__ import annotations

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

logger = logging.getLogger("alpha_lab.local_cpu")

RUN_SCRIPT = """\
#!/bin/bash
cd {exp_dir}
export PYTHONPATH={workspace}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=""  # Disable GPU
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
        self._jobs: dict[str, LocalCPUJob] = {}

    def running_count(self) -> int:
        """Count number of jobs currently running."""
        count = 0
        for job in self._jobs.values():
            if job.proc.poll() is None:
                count += 1
        return count

    def can_submit(self) -> bool:
        """Check if we have capacity to submit another job."""
        return self.running_count() < self.max_parallel

    def submit_experiment(self, exp: Experiment, workspace: str) -> str:
        """Spawn experiment as subprocess. Returns job ID (UUID)."""
        if not self.can_submit():
            raise RuntimeError(f"No CPU slot available (all {self.max_parallel} in use)")

        exp_dir = Path(workspace) / "experiments" / exp.name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Write run script
        script_content = RUN_SCRIPT.format(
            exp_dir=str(exp_dir),
            workspace=workspace,
            python_exe=self.python_executable,
        )
        script_path = exp_dir / "run_cpu.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        # Open output file and spawn subprocess (try/finally to avoid fd leak)
        output_path = exp_dir / "cpu_job.out"
        output_file = open(output_path, "w")
        try:
            proc = subprocess.Popen(
                ["bash", str(script_path)],
                stdout=output_file,
                stderr=subprocess.STDOUT,
                cwd=str(exp_dir),
                start_new_session=True,
            )
        except Exception:
            output_file.close()
            raise

        job_id = f"cpu-{str(uuid.uuid4())[:8]}"
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
                elapsed = now - job.start_time
                if self.time_limit > 0 and elapsed > self.time_limit:
                    logger.warning(
                        f"CPU job {job_id} ({job.exp_name}) exceeded time limit "
                        f"({elapsed:.0f}s > {self.time_limit}s), killing"
                    )
                    self._kill_job(job_id)
                    result[job_id] = "TIMEOUT"
                else:
                    result[job_id] = "RUNNING"
            elif retcode == 0:
                result[job_id] = "COMPLETED"
                self._cleanup_job(job_id)
            else:
                result[job_id] = "FAILED"
                logger.warning(f"CPU job {job_id} ({job.exp_name}) failed with exit code {retcode}")
                self._cleanup_job(job_id)

        return result

    def _kill_job(self, job_id: str) -> None:
        """Kill a job's process."""
        if job_id not in self._jobs:
            return
        job = self._jobs[job_id]
        if job.proc.poll() is None:
            try:
                os.killpg(os.getpgid(job.proc.pid), signal.SIGTERM)
                job.proc.wait(timeout=5)
            except (ProcessLookupError, ChildProcessError):
                pass  # Process already exited
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(job.proc.pid), signal.SIGKILL)
                except (ProcessLookupError, ChildProcessError):
                    pass  # Process already exited
                # Reap the child to prevent zombie
                try:
                    job.proc.wait(timeout=5)
                except (subprocess.TimeoutExpired, OSError):
                    pass
            except OSError as e:
                logger.warning("Failed to kill CPU job %s: %s", job_id, e)
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
        """Close output file."""
        if job_id in self._jobs:
            job = self._jobs[job_id]
            try:
                job.output_file.close()
            except OSError as e:
                logger.warning("Failed to close output file for CPU job %s: %s", job_id, e)

    def cleanup_all(self) -> None:
        """Kill all running jobs. Call on shutdown."""
        for job_id in list(self._jobs.keys()):
            job = self._jobs[job_id]
            if job.proc.poll() is None:
                self.cancel(job_id)
        self._jobs.clear()


def is_cpu_experiment(exp: Experiment) -> bool:
    """Check if an experiment should run on CPU based on its config.

    Experiments with model_type containing these keywords run on CPU:
    - xgboost, lightgbm, gbdt, gradient_boost
    - random_forest, decision_tree
    - linear, lasso, ridge, elastic
    - catboost
    """
    import json
    try:
        config = json.loads(exp.config_json or "{}")
        model_type = config.get("model_type", "").lower()
        resource = config.get("resource", "").lower()

        # Explicit resource tag takes precedence
        if resource == "cpu":
            return True
        if resource == "gpu":
            return False

        # Infer from model type
        cpu_keywords = [
            "xgboost", "lightgbm", "gbdt", "gradient_boost",
            "random_forest", "decision_tree", "tree",
            "linear", "lasso", "ridge", "elastic",
            "catboost", "sklearn",
        ]
        return any(kw in model_type for kw in cpu_keywords)
    except (json.JSONDecodeError, TypeError):
        return False
