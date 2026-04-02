"""SLURM job management for Phase 3 experiment dispatch."""

from __future__ import annotations

import itertools
import logging
import os
import subprocess
import sys
from pathlib import Path

from alpha_lab.experiment_db import Experiment

logger = logging.getLogger("alpha_lab.slurm")

SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=alpha_{name}
#SBATCH --partition={partition}
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time={time_limit}
#SBATCH --output={output_path}
cd {workspace}/experiments/{name}
export PYTHONPATH={workspace}:$PYTHONPATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
{python_exe} -c "
import torch, runpy, sys
# Override: disable deterministic algorithms regardless of what the experiment sets.
# Many CUDA ops (upsample, scatter, median) don't support it and crash on H100s.
_orig = torch.use_deterministic_algorithms
def _patched(mode=True, **kw):
    if mode:
        print('[SLURM wrapper] Intercepted torch.use_deterministic_algorithms(True) -> skipped')
        return
    _orig(mode, **kw)
torch.use_deterministic_algorithms = _patched
# Also patch Lightning's deterministic flag
try:
    import lightning.pytorch as pl
    _orig_init = pl.Trainer.__init__
    def _trainer_init(self, *a, deterministic=None, **kw):
        if deterministic:
            print('[SLURM wrapper] Intercepted Trainer(deterministic=True) -> False')
            deterministic = False
        _orig_init(self, *a, deterministic=deterministic, **kw)
    pl.Trainer.__init__ = _trainer_init
except Exception:
    pass
sys.argv = ['run_experiment.py']
runpy.run_path('run_experiment.py', run_name='__main__')
"
"""


class SlurmManager:
    """Manages SLURM job submission, polling, and GPU budget."""

    def __init__(
        self,
        partitions: list[str],
        gpu_per_job: int = 1,
        max_gpus: int = 8,
        time_limit: str = "02:00:00",
        python_executable: str = "",
    ) -> None:
        self.partitions = partitions
        self.gpu_per_job = gpu_per_job
        self.max_gpus = max_gpus
        self.time_limit = time_limit
        self.python_executable = python_executable or sys.executable
        self._partition_cycle = itertools.cycle(partitions)

    def generate_sbatch_script(self, exp: Experiment, workspace: str) -> str:
        partition = next(self._partition_cycle)
        exp_dir = os.path.join(workspace, "experiments", exp.name)
        output_path = os.path.join(exp_dir, "slurm_%j.out")
        return SBATCH_TEMPLATE.format(
            name=exp.name,
            partition=partition,
            time_limit=self.time_limit,
            output_path=output_path,
            workspace=workspace,
            python_exe=self.python_executable,
        )

    def submit(self, script_path: str) -> str:
        """Submit an sbatch script. Returns the SLURM job ID."""
        try:
            result = subprocess.run(
                ["sbatch", script_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")
            # Output: "Submitted batch job 12345"
            parts = result.stdout.strip().split()
            job_id = parts[-1]
            logger.info(f"Submitted SLURM job {job_id} from {script_path}")
            return job_id
        except subprocess.TimeoutExpired:
            raise RuntimeError("sbatch timed out")

    def cancel(self, job_id: str) -> None:
        """Cancel a SLURM job."""
        try:
            subprocess.run(
                ["scancel", job_id],
                capture_output=True,
                text=True,
                timeout=15,
            )
            logger.info(f"Cancelled SLURM job {job_id}")
        except Exception as e:
            logger.warning(f"Failed to cancel job {job_id}: {e}")

    def poll_jobs(self, job_ids: list[str]) -> dict[str, str]:
        """Poll SLURM for job statuses.

        Returns {job_id: status} where status is one of:
        PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, TIMEOUT, UNKNOWN.
        """
        if not job_ids:
            return {}

        result_map: dict[str, str] = {}

        # Try squeue first for active jobs
        try:
            result = subprocess.run(
                ["squeue", "--me", "--format=%i %T", "--noheader"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().splitlines():
                    parts = line.split()
                    if len(parts) >= 2:
                        jid, state = parts[0], parts[1]
                        if jid in job_ids:
                            result_map[jid] = state
        except Exception as e:
            logger.warning(f"squeue failed: {e}")

        # Use sacct for jobs not found in squeue (completed/failed)
        missing = [jid for jid in job_ids if jid not in result_map]
        if missing:
            try:
                job_list = ",".join(missing)
                result = subprocess.run(
                    [
                        "sacct",
                        "-j", job_list,
                        "--format=JobID,State",
                        "--noheader",
                        "--parsable2",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().splitlines():
                        parts = line.split("|")
                        if len(parts) >= 2:
                            jid = parts[0].split(".")[0]  # strip step suffix
                            state = parts[1].strip().rstrip("+")  # strip sacct + suffix
                            if jid in missing and jid not in result_map:
                                result_map[jid] = state
            except Exception as e:
                logger.warning(f"sacct failed: {e}")

        # Mark truly missing as UNKNOWN
        for jid in job_ids:
            if jid not in result_map:
                result_map[jid] = "UNKNOWN"

        return result_map

    def running_gpu_count(self) -> int:
        """Count GPUs currently used by our running/pending jobs."""
        try:
            result = subprocess.run(
                ["squeue", "--me", "--format=%i %T %b", "--noheader"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                return 0

            count = 0
            for line in result.stdout.strip().splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    state = parts[1]
                    if state in ("RUNNING", "PENDING"):
                        count += self.gpu_per_job
            return count
        except (subprocess.SubprocessError, OSError, ValueError):
            return 0

    def can_submit(self) -> bool:
        """Check if we have GPU budget to submit another job."""
        return self.running_gpu_count() + self.gpu_per_job <= self.max_gpus

    def submit_experiment(self, exp: Experiment, workspace: str) -> str:
        """Generate sbatch script, write it, submit it. Returns job ID."""
        exp_dir = Path(workspace) / "experiments" / exp.name
        exp_dir.mkdir(parents=True, exist_ok=True)

        script_content = self.generate_sbatch_script(exp, workspace)
        script_path = exp_dir / "submit.sh"
        script_path.write_text(script_content)

        return self.submit(str(script_path))
