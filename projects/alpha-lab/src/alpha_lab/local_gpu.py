"""Local GPU job management for Phase 3 experiment dispatch.

Replaces SLURM with direct subprocess spawning on a multi-GPU box.
Each experiment runs as a subprocess with CUDA_VISIBLE_DEVICES pinned
to a specific GPU.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from alpha_lab.experiment_db import Experiment

logger = logging.getLogger("alpha_lab.local_gpu")


class RecoveredProcess:
    """Minimal wrapper for recovered process that implements poll() interface."""
    def __init__(self, pid: int):
        self.pid = pid
        self.returncode = None

    def poll(self):
        """Check if process is still running."""
        if self.returncode is not None:
            return self.returncode
        try:
            os.kill(self.pid, 0)
            return None  # Still running
        except OSError:
            self.returncode = 0  # Assume successful completion
            return 0


RUN_SCRIPT = """\
#!/bin/bash
cd {exp_dir}
export PYTHONPATH={workspace}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES={gpu_id}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
{python_exe} -c "
import torch, runpy, sys
# Override: disable deterministic algorithms regardless of what the experiment sets.
# Many CUDA ops (upsample, scatter, median) don't support it and crash on H100s.
_orig = torch.use_deterministic_algorithms
def _patched(mode=True, **kw):
    if mode:
        print('[Local GPU wrapper] Intercepted torch.use_deterministic_algorithms(True) -> skipped')
        return
    _orig(mode, **kw)
torch.use_deterministic_algorithms = _patched
# Also patch Lightning's deterministic flag
try:
    import lightning.pytorch as pl
    _orig_init = pl.Trainer.__init__
    def _trainer_init(self, *a, deterministic=None, **kw):
        if deterministic:
            print('[Local GPU wrapper] Intercepted Trainer(deterministic=True) -> False')
            deterministic = False
        _orig_init(self, *a, deterministic=deterministic, **kw)
    pl.Trainer.__init__ = _trainer_init
except Exception:
    pass
sys.argv = ['run_experiment.py']
runpy.run_path('run_experiment.py', run_name='__main__')
"
"""


@dataclass
class LocalJob:
    """Tracks a running local subprocess."""
    proc: subprocess.Popen | RecoveredProcess
    gpu_id: int
    exp_name: str
    output_file: TextIO
    workspace: str
    start_time: float = 0.0  # time.time() when started


class LocalGPUManager:
    """Manages local GPU job spawning, polling, and GPU budget.

    Drop-in replacement for SlurmManager. Same 5-method interface:
    - submit_experiment(exp, workspace) -> job_id
    - poll_jobs(job_ids) -> {job_id: status}
    - cancel(job_id)
    - can_submit() -> bool
    - running_gpu_count() -> int
    """

    def __init__(
        self,
        gpu_ids: list[int] | None = None,
        max_per_gpu: int = 1,
        time_limit_seconds: int = 7200,
        python_executable: str = "",
    ) -> None:
        """
        Parameters
        ----------
        gpu_ids : list[int], optional
            Which GPU indices to use. Defaults to [0,1,2,3] (auto-detect would
            be better but this is simple).
        max_per_gpu : int
            Max concurrent experiments per GPU. Start with 1, increase for
            packing if models fit.
        time_limit_seconds : int
            Subprocess timeout. Killed after this (like SLURM --time).
        python_executable : str
            Path to the Python interpreter for experiment subprocesses.
            Empty string (default) uses sys.executable.
        """
        self.gpu_ids = gpu_ids if gpu_ids is not None else [0, 1, 2, 3]
        self.max_per_gpu = max_per_gpu
        self.time_limit = time_limit_seconds
        self.python_executable = python_executable or sys.executable
        self._jobs: dict[str, LocalJob] = {}
        self._workspace: str | None = None  # Set on first submit

    def recover_running_jobs(self, workspace: str, job_id_map: dict[str, str]) -> int:
        """Discover already-running GPU processes and resume tracking them.

        Called on startup to reconcile database state with actual running processes.

        Parameters
        ----------
        workspace : str
            Workspace path to match against process working directories
        job_id_map : dict[str, str]
            Map of {job_id: experiment_name} from database for validation

        Returns
        -------
        int
            Number of jobs recovered
        """
        self._workspace = workspace
        recovered = 0

        try:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True, text=True, timeout=5,
            )

            for line in result.stdout.split('\n'):
                if 'run_local.sh' not in line or 'bash' not in line or 'grep' in line:
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                pid = int(parts[1])

                try:
                    cwd_result = subprocess.run(
                        ["readlink", f"/proc/{pid}/cwd"],
                        capture_output=True, text=True, timeout=1,
                    )
                    cwd = cwd_result.stdout.strip()
                except Exception:
                    continue

                if '/experiments/' not in cwd:
                    continue

                exp_name = Path(cwd).name

                job_id = None
                for jid, ename in job_id_map.items():
                    if ename == exp_name:
                        job_id = jid
                        break

                if not job_id:
                    continue

                gpu_id = -1
                try:
                    env_result = subprocess.run(
                        ["cat", f"/proc/{pid}/environ"],
                        capture_output=True, text=True, timeout=1,
                    )
                    for env_var in env_result.stdout.split('\x00'):
                        if env_var.startswith('CUDA_VISIBLE_DEVICES='):
                            gpu_str = env_var.split('=', 1)[1]
                            if gpu_str:
                                gpu_id = int(gpu_str.split(',')[0])
                            break
                except Exception:
                    pass

                if gpu_id < 0 or gpu_id not in self.gpu_ids:
                    logger.warning(f"Could not determine GPU for process {pid} ({exp_name}), skipping")
                    continue

                try:
                    proc = RecoveredProcess(pid)
                    output_path = Path(cwd) / "local_job.out"
                    try:
                        output_file = open(output_path, "a")
                    except Exception:
                        output_file = open(os.devnull, "w")

                    job = LocalJob(
                        proc=proc, gpu_id=gpu_id, exp_name=exp_name,
                        output_file=output_file, workspace=workspace,
                        start_time=0.0,  # Unknown — won't enforce timeout for recovered jobs
                    )
                    self._jobs[job_id] = job
                    recovered += 1
                    logger.info(f"Recovered job {job_id} ({exp_name}) on GPU {gpu_id}, PID {pid}")

                except Exception as e:
                    logger.warning(f"Failed to recover job for {exp_name}: {e}")

        except Exception as e:
            logger.error(f"Job recovery failed: {e}")

        return recovered

    def _gpu_load(self) -> dict[int, int]:
        """Return {gpu_id: num_running_jobs}."""
        load = {g: 0 for g in self.gpu_ids}
        for job in self._jobs.values():
            if job.proc.poll() is None:  # still running
                load[job.gpu_id] += 1
        return load

    def _get_gpu_memory_free(self) -> dict[int, int]:
        """Query nvidia-smi for free memory (in MB) per GPU.

        Returns {gpu_id: free_memory_mb}. Returns empty dict on error.
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode != 0:
                logger.warning("nvidia-smi query failed, falling back to job-count heuristic")
                return {}

            memory = {}
            for line in result.stdout.strip().split('\n'):
                parts = line.strip().split(',')
                if len(parts) == 2:
                    gpu_idx = int(parts[0].strip())
                    free_mb = int(parts[1].strip())
                    if gpu_idx in self.gpu_ids:
                        memory[gpu_idx] = free_mb
            return memory
        except Exception as e:
            logger.warning(f"Failed to query GPU memory: {e}")
            return {}

    def _estimate_memory_requirement(self, exp: Experiment, workspace: str) -> int:
        """Estimate GPU memory requirement (in MB) from experiment config.

        Heuristic based on model type, batch size, sequence length.
        Conservative estimates to avoid OOM.
        """
        try:
            import yaml

            config_path = Path(workspace) / "experiments" / exp.name / "config.yaml"
            if not config_path.exists():
                return 8000

            with open(config_path) as f:
                config = yaml.safe_load(f)

            model_cfg = config.get('model', {}) if isinstance(config.get('model'), dict) else {}
            training_cfg = config.get('training', {}) if isinstance(config.get('training'), dict) else {}

            model_type = (
                model_cfg.get('model_type')
                or model_cfg.get('type')
                or model_cfg.get('library')
                or config.get('model_type')
                or ''
            ).lower()
            batch_size = (
                training_cfg.get('batch_size')
                or model_cfg.get('batch_size')
                or config.get('batch_size')
                or 512
            )
            input_size = (
                model_cfg.get('input_size')
                or model_cfg.get('context_length')
                or config.get('input_size')
                or 120
            )

            base_memory = {
                'tft': 6000, 'transformer': 8000, 'patchtst': 4000,
                'nhits': 3000, 'timesnet': 5000, 'deepar': 4000,
                'tcn': 2000, 'lstm': 3000, 'gru': 3000,
                'mlp': 2000, 'linear': 1000, 'lightgbm': 1000, 'xgboost': 1000,
            }

            mem_base = 5000  # default
            for family, mem in base_memory.items():
                if family in model_type:
                    mem_base = mem
                    break

            mem_batch_scale = (batch_size / 512) ** 0.7
            mem_seq_scale = (input_size / 120) ** 0.5
            estimated = int(mem_base * mem_batch_scale * mem_seq_scale)
            estimated = int(estimated * 1.2)  # 20% safety margin
            estimated = max(2000, min(estimated, 40000))  # clamp

            logger.debug(f"Estimated memory for {exp.name}: {estimated} MB "
                        f"(model={model_type}, batch={batch_size}, seq={input_size})")
            return estimated

        except Exception as e:
            logger.warning(f"Failed to estimate memory for {exp.name}: {e}, using default 8GB")
            return 8000

    def _pick_gpu(self, exp: Experiment | None = None, workspace: str | None = None) -> int | None:
        """Return GPU ID with sufficient free memory, or None if none available.

        Strategy:
        1. Query actual free memory per GPU
        2. Estimate memory requirement for experiment
        3. Pick GPU with most free memory that can fit the job
        4. Fall back to job-count heuristic if nvidia-smi unavailable
        5. Respect max_per_gpu hard limit
        """
        load = self._gpu_load()

        available_gpus = [gpu for gpu, count in load.items() if count < self.max_per_gpu]
        if not available_gpus:
            return None

        # Try memory-aware allocation
        gpu_memory = self._get_gpu_memory_free()

        if gpu_memory and exp is not None and workspace is not None:
            mem_required = self._estimate_memory_requirement(exp, workspace)

            candidates = []
            for gpu in available_gpus:
                free_mem = gpu_memory.get(gpu, 0)
                if free_mem >= mem_required:
                    candidates.append((free_mem, gpu))

            if candidates:
                return max(candidates)[1]  # most free memory
            else:
                logger.info(f"No GPU has sufficient memory ({mem_required} MB required). "
                           f"Available: {[(g, gpu_memory.get(g)) for g in available_gpus]}. "
                           f"Job will wait for memory to free up.")
                return None

        # Fallback: pick GPU with fewest jobs
        return min((load[gpu], gpu) for gpu in available_gpus)[1]

    def running_gpu_count(self) -> int:
        """Count number of jobs currently running (one GPU per job)."""
        count = 0
        for job in self._jobs.values():
            if job.proc.poll() is None:
                count += 1
        return count

    def can_submit(self, exp: Experiment | None = None) -> bool:
        """Check if we have capacity to submit a job.

        If exp is provided, checks if there's a GPU with sufficient memory.
        Otherwise, checks if any GPU has capacity.
        """
        return self._pick_gpu(exp, self._workspace) is not None

    def submit_experiment(self, exp: Experiment, workspace: str) -> str:
        """Spawn experiment as subprocess. Returns job ID (UUID)."""
        if self._workspace is None:
            self._workspace = workspace

        gpu_id = self._pick_gpu(exp, workspace)
        if gpu_id is None:
            raise RuntimeError(
                "No GPU available with sufficient memory. "
                "Job will wait until memory frees up."
            )

        exp_dir = Path(workspace) / "experiments" / exp.name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Write run script
        script_content = RUN_SCRIPT.format(
            exp_dir=str(exp_dir),
            workspace=workspace,
            gpu_id=gpu_id,
            python_exe=self.python_executable,
        )
        script_path = exp_dir / "run_local.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        # Open output file and spawn subprocess (try/finally to avoid fd leak)
        output_path = exp_dir / "local_job.out"
        output_file = open(output_path, "w")
        try:
            proc = subprocess.Popen(
                ["bash", str(script_path)],
                stdout=output_file,
                stderr=subprocess.STDOUT,
                cwd=str(exp_dir),
                start_new_session=True,  # detach from terminal
            )
        except Exception:
            output_file.close()
            raise

        job_id = str(uuid.uuid4())[:8]
        self._jobs[job_id] = LocalJob(
            proc=proc,
            gpu_id=gpu_id,
            exp_name=exp.name,
            output_file=output_file,
            workspace=workspace,
            start_time=time.time(),
        )

        logger.info(f"Submitted local job {job_id} for {exp.name} on GPU {gpu_id} (PID {proc.pid})")
        return job_id

    def poll_jobs(self, job_ids: list[str]) -> dict[str, str]:
        """Poll for job statuses.

        Returns {job_id: status} where status is one of:
        RUNNING, COMPLETED, FAILED, TIMEOUT, UNKNOWN.
        """
        result: dict[str, str] = {}
        now = time.time()

        for job_id in job_ids:
            if job_id not in self._jobs:
                result[job_id] = "UNKNOWN"
                continue

            job = self._jobs[job_id]
            retcode = job.proc.poll()

            if retcode is None:
                # Still running - check for timeout
                elapsed = now - job.start_time
                if self.time_limit > 0 and elapsed > self.time_limit:
                    logger.warning(
                        f"Job {job_id} ({job.exp_name}) exceeded time limit "
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
                logger.warning(f"Job {job_id} ({job.exp_name}) failed with exit code {retcode}")
                self._cleanup_job(job_id)

        return result

    def _kill_job(self, job_id: str) -> None:
        """Kill a job's process (internal helper)."""
        if job_id not in self._jobs:
            return
        job = self._jobs[job_id]
        if job.proc.poll() is None:
            try:
                # Kill the whole process group
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
                logger.warning("Failed to kill job %s: %s", job_id, e)
        self._cleanup_job(job_id)

    def cancel(self, job_id: str) -> None:
        """Kill a running job."""
        if job_id not in self._jobs:
            logger.warning(f"Cannot cancel unknown job {job_id}")
            return
        job = self._jobs[job_id]
        logger.info(f"Cancelling job {job_id} ({job.exp_name})")
        self._kill_job(job_id)

    def _cleanup_job(self, job_id: str) -> None:
        """Close output file and remove from tracking."""
        if job_id in self._jobs:
            job = self._jobs[job_id]
            try:
                job.output_file.close()
            except OSError as e:
                logger.warning("Failed to close output file for job %s: %s", job_id, e)
            # Don't delete from _jobs - keep for status queries
            # The dispatcher will stop polling completed jobs anyway

    def cleanup_all(self) -> None:
        """Kill all running jobs and clean up. Call on shutdown."""
        for job_id in list(self._jobs.keys()):
            job = self._jobs[job_id]
            if job.proc.poll() is None:
                self.cancel(job_id)
        self._jobs.clear()
