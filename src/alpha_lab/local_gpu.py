"""Local GPU job management for Phase 3 experiment dispatch.

Replaces SLURM with direct subprocess spawning on a multi-GPU box.
Each experiment runs as a subprocess with CUDA_VISIBLE_DEVICES pinned
to a specific GPU.
"""

from __future__ import annotations

import json
import logging
import os
import re
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

# (regex, label) pairs scanned against the tail of a failed job's log to
# stamp a canonical error category into run_status.json. First match wins.
_ERROR_SIGNATURES: tuple[tuple[str, str], ...] = (
    (r"torch\.OutOfMemoryError|CUDA out of memory",        "cuda_oom"),
    (r"got an unexpected keyword argument",                "framework_api_mismatch"),
    (r"Expected b and A to have the same dtype",           "dtype_mismatch"),
    (r"Expected all tensors to be on the same device",     "device_mismatch"),
    (r"ImportError|ModuleNotFoundError",                   "import_error"),
    (r"\bNaN\b.*loss|loss.*\bNaN\b",                       "nan_loss"),
    (r"\bSIGKILL\b|\bSIGTERM\b|\bTerminated\b",            "signal_terminated"),
    (r"RuntimeError",                                       "runtime_error"),
    (r"TypeError",                                          "type_error"),
    (r"ValueError",                                         "value_error"),
)


class RecoveredProcess:
    """Minimal wrapper for a recovered orphan that mimics enough of
    ``subprocess.Popen`` for the poll-and-kill path to work.

    A recovered process is NOT our child (the previous dispatcher's
    Popen object was lost when the dispatcher died), so we cannot
    ``os.waitpid`` on it. Instead, ``poll()`` and ``wait()`` are
    implemented via existence checks (``os.kill(pid, 0)``).
    """
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

    def wait(self, timeout: float | None = None):
        """Block until the process exits, then return its exit code.

        Since recovered processes aren't our children we can't use
        ``os.waitpid``; we poll ``os.kill(pid, 0)`` until it fails. The
        caller in ``_terminate_pgid`` uses a 5s timeout to detect
        whether SIGTERM landed before escalating to SIGKILL — that
        contract is preserved by raising ``subprocess.TimeoutExpired``
        on timeout, matching ``subprocess.Popen.wait``.
        """
        import time as _time
        start = _time.time()
        while True:
            rc = self.poll()
            if rc is not None:
                return rc
            if timeout is not None and (_time.time() - start) >= timeout:
                raise subprocess.TimeoutExpired(cmd=f"pid={self.pid}", timeout=timeout)
            _time.sleep(0.1)


RUN_SCRIPT = """\
#!/bin/bash
cd {exp_dir}
export PYTHONPATH={workspace}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES={gpu_id}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
{data_env}
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
        data_path: str = "",
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
        # Exported to every experiment job as ALPHALAB_DATA_PATH /
        # ALPHALAB_DATA_ROOT. Harness scripts that need the dataset root
        # get a guaranteed env var instead of relying on ad-hoc names the
        # dispatcher never sets (real runs burned dozens of launch retries
        # on jobs that died at startup for want of a data-root env var).
        self.data_path = data_path
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
            # ``ps aux`` can be slow on heavily-loaded shared hosts
            # (load avg in the hundreds). The previous 5s budget timed
            # out every recovery on this cluster, causing
            # ``recover_running_jobs`` to silently return 0 even when
            # orphans were alive. 30s leaves headroom for /proc scans
            # while still bounding pathological hangs.
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True, text=True, timeout=30,
            )

            for line in result.stdout.split('\n'):
                if 'run_local.sh' not in line or 'bash' not in line or 'grep' in line:
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                pid = int(parts[1])

                try:
                    # 1s was too tight: under heavy host load (loadavg
                    # in the hundreds) ``readlink`` over /proc routinely
                    # exceeds it, the TimeoutExpired is caught silently
                    # below, and every orphan iteration ``continue``s —
                    # producing zero recoveries with no log evidence of
                    # the cause.
                    cwd_result = subprocess.run(
                        ["readlink", f"/proc/{pid}/cwd"],
                        capture_output=True, text=True, timeout=10,
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

                # Determine which GPU this orphan is pinned to. Reading
                # ``/proc/<bash_pid>/environ`` does NOT work: the wrapper
                # script sets ``CUDA_VISIBLE_DEVICES`` via ``export``
                # *inside* the running shell (line 30 of run_local.sh),
                # which never updates the kernel-snapshotted environ of
                # the bash process itself. Only the bash's python child
                # inherits the var. So we look for it in two places, in
                # order of reliability:
                #   1. The wrapper script on disk — deterministic, the
                #      dispatcher wrote it, and it survives even if the
                #      python child has already exited.
                #   2. The python child's ``/proc/<child_pid>/environ``
                #      as a fallback if the script is missing or doesn't
                #      contain a literal ``CUDA_VISIBLE_DEVICES=N`` line
                #      (e.g. dynamically-generated wrappers).
                gpu_id = -1
                script_path = Path(cwd) / "run_local.sh"
                if script_path.exists():
                    try:
                        for line in script_path.read_text(
                            encoding="utf-8"
                        ).splitlines():
                            stripped = line.strip()
                            if stripped.startswith("export CUDA_VISIBLE_DEVICES="):
                                gpu_str = stripped.split("=", 1)[1].strip()
                                # Trim potential trailing comments
                                gpu_str = gpu_str.split("#", 1)[0].strip()
                                if gpu_str:
                                    gpu_id = int(gpu_str.split(",")[0])
                                break
                    except (OSError, ValueError, UnicodeDecodeError):
                        pass

                if gpu_id < 0:
                    # Fallback: walk to the bash's python child and read
                    # its environ. ``/proc/<pid>/task/<pid>/children``
                    # gives a space-separated PID list of direct children.
                    try:
                        children_path = f"/proc/{pid}/task/{pid}/children"
                        with open(children_path, encoding="ascii") as f:
                            child_pids = f.read().split()
                        for child_pid in child_pids:
                            try:
                                with open(
                                    f"/proc/{child_pid}/environ", "rb"
                                ) as ef:
                                    raw = ef.read()
                            except OSError:
                                continue
                            for env_var in raw.split(b"\x00"):
                                if env_var.startswith(b"CUDA_VISIBLE_DEVICES="):
                                    gpu_str = env_var.split(b"=", 1)[1].decode(
                                        "ascii", errors="ignore"
                                    )
                                    if gpu_str:
                                        try:
                                            gpu_id = int(gpu_str.split(",")[0])
                                        except ValueError:
                                            pass
                                    break
                            if gpu_id >= 0:
                                break
                    except OSError:
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

        Calibrated against observed peaks from the etfflows-g55 run where the
        previous heuristic returned ~1.3 GB for a TFT that actually used
        62 GiB. Per-architecture base costs come from those observations; the
        scaling exponents are deliberately modest because the family base
        already captures most of the variance. No new config knobs — reads
        only fields the implementer LLM already writes into config.yaml.
        """
        try:
            import yaml

            config_path = Path(workspace) / "experiments" / exp.name / "config.yaml"
            if not config_path.exists():
                # Conservative default for unknown models: pack at most ~3
                # such jobs onto an 80 GiB GPU.
                return 20000

            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

            def _pick(keys: tuple[str, ...], default):
                for source in (
                    config.get("training", {}),
                    config.get("model", {}),
                    config.get("hyperparams", {}),
                    config,
                ):
                    if not isinstance(source, dict):
                        continue
                    for k in keys:
                        v = source.get(k)
                        if v is not None:
                            return v
                return default

            model_type = str(_pick(
                ("model_type", "type", "library", "name", "architecture"),
                "",
            )).lower()
            batch_size = int(_pick(
                ("batch_size", "train_batch_size", "batch_sequences", "global_batch_size"),
                64,
            ))
            pred_batch = int(_pick(
                ("prediction_batch_rows", "predict_batch_size", "batch_queries",
                 "eval_batch_size"),
                batch_size * 4,
            ))
            seq_len = int(_pick(
                ("context_length", "input_size", "max_sequence_events",
                 "encoder_length", "encoder_length_bins", "max_sequence_days",
                 "seq_len", "lookback"),
                128,
            ))
            hidden = int(_pick(
                ("hidden_dim", "hidden_size", "d_model", "embedding_dim",
                 "model_dim", "n_embd"),
                128,
            ))
            n_layers = int(_pick(
                ("num_layers", "n_layers", "depth", "encoder_layers"),
                3,
            ))
            n_heads = int(_pick(
                ("attention_heads", "num_heads", "n_heads", "nhead"),
                4,
            ))
            amp = bool(_pick(
                ("mixed_precision", "use_amp", "bf16", "fp16", "amp"),
                False,
            ))
            dtype_bytes = 2 if amp else 4

            # Tree / linear models route to CPU executor anyway, but a tiny
            # GPU budget here is the right answer if the dispatcher ever
            # routes them to GPU (e.g., LightGBM with GPU support).
            tree_or_linear_markers = (
                "lightgbm", "xgboost", "catboost", "randomforest",
                "sklearn", "ridge", "lasso", "elasticnet", "gbm", "gbdt",
                "lambdarank", "lambdamart",
            )
            if any(t in model_type for t in tree_or_linear_markers):
                return 2000

            # Per-family base costs calibrated to observed peaks (in MiB).
            # Long-tailed deliberately: TFT/Mamba/TabTransformer all hit
            # 60+ GiB at default caps in the prior run.
            family_base = {
                "tft":             30000,
                "tabtransformer":  25000,
                "fttransformer":   20000,
                "transformer":     18000,
                "nonstationary":   18000,
                "crossformer":     20000,
                "itransformer":    18000,
                "patchtst":        15000,
                "timexer":         15000,
                "timemixer":       12000,
                "timesnet":        15000,
                "nhits":            8000,
                "deepar":          10000,
                "deepvar":         18000,
                "tcn":              8000,
                "lstm":             6000,
                "gru":              6000,
                "retnet":          20000,
                "xlstm":           18000,
                "mamba2":          18000,
                "mamba":           25000,  # pscan materializes large tensors
                "ssm":             18000,
                "csdi":            25000,  # diffusion
                "sasrec":          15000,
                "bert4rec":        15000,
                "bert":            18000,
                "hawkes":           8000,
                "hgt":             15000,
                "tgn":             15000,
                "graph":           12000,
                "gnn":             12000,
                "wavenet":         10000,
                "time_moe":        30000,  # foundation
                "moment":          30000,  # foundation
                "chronos":         20000,
                "lag_llama":       20000,
                "moirai":          25000,
                "ttm":             20000,
                "neural_cde":      15000,
                "perceiver":       20000,
                "tide":            10000,
                "tabm":            15000,
                "deepfm":           8000,
                "duet":            12000,
                "cyclenet":         8000,
                "deeplob":         12000,
                "set_transformer": 15000,
                "two_tower":       10000,
                "mlp":              5000,
                "linear":           3000,
            }
            base_mb = 8000  # default for an unknown neural model
            for family, mb in family_base.items():
                if family in model_type:
                    base_mb = mb
                    break

            # Attention term: B * heads * L^2 * dtype, only for attention-y models.
            attention_families = (
                "transformer", "tft", "tabtransformer", "fttransformer",
                "patchtst", "sasrec", "bert", "bert4rec", "itransformer",
                "crossformer", "timexer", "mamba", "ssm", "time_moe",
                "moment", "csdi", "perceiver", "set_transformer", "nonstationary",
                "retnet", "xlstm",
            )
            is_attention = any(t in model_type for t in attention_families)
            attn_mb = (
                (batch_size * max(n_heads, 1) * seq_len * seq_len * dtype_bytes) / 1e6
                if is_attention else 0.0
            )

            # Sub-linear scaling on each axis: the family base already
            # captures most of the variance, this nudges the estimate up
            # or down within reason.
            scale = (
                (batch_size / 32) ** 0.6 *
                (seq_len / 128) ** 0.5 *
                (hidden / 128) ** 0.8 *
                (max(n_layers, 1) / 3) ** 0.7
            )

            # Inference-time peak (hurdle/quantile models often materialize
            # bigger tensors at predict time than at train time).
            pred_mb = (pred_batch * seq_len * hidden * dtype_bytes * 2) / 1e6

            total = base_mb * scale + attn_mb + pred_mb
            total = int(total * 1.3)  # 30% safety margin
            # Floor 2 GiB so the memory-aware allocator never treats neural
            # jobs as "free"; ceiling 70 GiB because anything beyond is
            # effectively single-GPU-only on an 80 GiB H100.
            total = max(2000, min(total, 70000))

            logger.debug(
                "Estimated memory for %s: %d MB "
                "(model=%s, B=%d, L=%d, H=%d, layers=%d, heads=%d, amp=%s)",
                exp.name, total, model_type, batch_size, seq_len,
                hidden, n_layers, n_heads, amp,
            )
            return total

        except Exception as e:
            logger.warning(
                "Failed to estimate memory for %s: %s, using conservative 20GB default",
                exp.name, e,
            )
            return 20000

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

        Two gates:

        1. ``_pick_gpu`` returns a GPU id with enough free memory.
           ``_get_gpu_memory_free`` reads ``nvidia-smi`` so it sees
           memory used by *all* GPU processes on the box, including
           other users' jobs and our own orphans the recovery scan
           failed to reattach.
        2. ``host_has_capacity()`` checks system RAM and loadavg via
           ``/proc/meminfo`` and ``/proc/loadavg``. GPU jobs also use
           system RAM (DataLoader workers, in-memory feature panels,
           model state on the way to the device); a GPU with free
           memory is not enough if the host's RAM is exhausted.
        """
        from alpha_lab.host import host_has_capacity
        if self._pick_gpu(exp, self._workspace) is None:
            return False
        return host_has_capacity()

    def is_alive(self, job_id: str) -> bool:
        """Return True iff job_id is tracked and its process is still running."""
        job = self._jobs.get(job_id)
        return job is not None and job.proc.poll() is None

    def submit_experiment(self, exp: Experiment, workspace: str) -> str:
        """Spawn experiment as subprocess. Returns job ID (UUID)."""
        if self._workspace is None:
            self._workspace = workspace

        # Supersede: if a prior tracked job for the same experiment name
        # is still running (status guard / idempotency check raced, or the
        # caller is intentionally restarting), kill its pgid before we
        # launch a replacement. Prevents the dispatcher from accumulating
        # orphaned subprocesses that hold GPU memory after the row has
        # been advanced.
        for prior_id, prior_job in list(self._jobs.items()):
            if prior_job.exp_name == exp.name and prior_job.proc.poll() is None:
                logger.warning(
                    "Superseding live job %s for %s before resubmit",
                    prior_id, exp.name,
                )
                self._terminate_pgid(prior_job, "superseded by new submission")
                self._cleanup_job(prior_id)

        gpu_id = self._pick_gpu(exp, workspace)
        if gpu_id is None:
            raise RuntimeError(
                "No GPU available with sufficient memory. "
                "Job will wait until memory frees up."
            )

        exp_dir = Path(workspace) / "experiments" / exp.name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Generate job id before opening files so we can name the log file
        # with it and avoid clobbering a prior submission's output.
        job_id = str(uuid.uuid4())[:8]

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
            gpu_id=gpu_id,
            python_exe=self.python_executable,
            data_env=data_env,
        )
        script_path = exp_dir / "run_local.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        # Each submission gets its own log file so a resubmission can't
        # truncate the prior run's output (and so a worker's `rm -f
        # local_job.out` only removes the stable symlink, not the live
        # file the executor is writing — eliminating the NFS silly-rename
        # path that led to a worker LLM SIGKILLing the orchestrator).
        output_path = exp_dir / f"local_job.{job_id}.out"
        output_file = open(output_path, "w")
        # Stable "latest" symlink for analyzer/human readers.
        latest_link = exp_dir / "local_job.out"
        try:
            if latest_link.is_symlink() or latest_link.exists():
                try:
                    latest_link.unlink()
                except OSError as unlink_err:
                    logger.warning("could not unlink old local_job.out: %s", unlink_err)
            latest_link.symlink_to(output_path.name)
        except OSError as link_err:
            logger.warning("local_job.out symlink failed: %s", link_err)

        try:
            from alpha_lab.tools import _preexec_setup, _register_subprocess
            proc = subprocess.Popen(
                ["bash", str(script_path)],
                stdout=output_file,
                stderr=subprocess.STDOUT,
                cwd=str(exp_dir),
                # New process group (so cancel() can ``killpg``) +
                # PR_SET_PDEATHSIG SIGKILL on Linux (so the GPU job dies if
                # ``run.py`` is killed, instead of orphaning and fighting
                # the next run for the same GPU). Belt-and-suspenders via
                # the parent's SIGTERM handler in ``tools.py``.
                preexec_fn=_preexec_setup,
            )
            _register_subprocess(proc)
        except Exception:
            output_file.close()
            raise

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
                # Still running - check for timeout. ``start_time == 0.0``
                # marks a recovered orphan whose actual start time is
                # unknown (the previous dispatcher's bookkeeping is gone).
                # Without this guard, ``elapsed = now - 0`` is ~10^9 s,
                # the time-limit check fires on the first poll, and we
                # kill every recovered job immediately.
                if job.start_time > 0:
                    elapsed = now - job.start_time
                    over_limit = self.time_limit > 0 and elapsed > self.time_limit
                else:
                    elapsed = 0.0
                    over_limit = False
                if over_limit:
                    logger.warning(
                        f"Job {job_id} ({job.exp_name}) exceeded time limit "
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
                logger.warning(f"Job {job_id} ({job.exp_name}) failed with exit code {retcode}")
                self._write_run_status(job_id, "FAILED", retcode)
                self._cleanup_job(job_id)

        return result

    @staticmethod
    def _read_log_tail(log_path: Path, max_lines: int = 50, max_bytes: int = 65536) -> str:
        """Return up to ``max_lines`` last lines of ``log_path``, bounded by bytes.

        Reads only the tail of the file so a huge log (the TFT run was 36 MB)
        doesn't blow up the dispatcher's working set.
        """
        try:
            with open(log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                read = min(size, max_bytes)
                f.seek(size - read)
                data = f.read(read)
            text = data.decode("utf-8", errors="replace")
            lines = text.splitlines()
            return "\n".join(lines[-max_lines:])
        except OSError:
            return ""

    @staticmethod
    def _classify_error(tail: str) -> str | None:
        if not tail:
            return None
        for pat, label in _ERROR_SIGNATURES:
            if re.search(pat, tail):
                return label
        return "unknown"

    def _write_run_status(self, job_id: str, status: str, returncode: int) -> None:
        """Write a structured run_status.json next to the experiment dir.

        Single canonical record per terminal transition: status, returncode,
        wall time, log path, last 50 lines, and an error_signature when the
        job failed. The analyzer prompt reads this in preference to grepping
        the raw stdout, which fixes the long-standing "no slurm_*.out found"
        debrief class.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return
        exp_dir = Path(job.workspace) / "experiments" / job.exp_name
        # The per-job log lives at local_job.<job_id>.out; fall back to the
        # symlink for robustness if naming changed mid-run.
        per_job_log = exp_dir / f"local_job.{job_id}.out"
        log_path = per_job_log if per_job_log.exists() else (exp_dir / "local_job.out")
        tail = self._read_log_tail(log_path) if log_path.exists() else ""
        finished_at = time.time()
        wall = (finished_at - job.start_time) if job.start_time else None
        payload = {
            "job_id": job_id,
            "executor": "local_gpu",
            "exp_name": job.exp_name,
            "gpu_id": job.gpu_id,
            "status": status,
            "returncode": returncode,
            "started_at": job.start_time or None,
            "finished_at": finished_at,
            "wall_seconds": wall,
            "log_path": str(log_path) if log_path.exists() else None,
            "last_lines": tail,
            "error_signature": (
                self._classify_error(tail) if status != "COMPLETED" else None
            ),
        }
        try:
            (exp_dir / "run_status.json").write_text(
                json.dumps(payload, indent=2, default=str)
            )
        except OSError as e:
            logger.warning("write run_status.json for %s failed: %s", job.exp_name, e)

    def _terminate_pgid(self, job: LocalJob, why: str) -> None:
        """SIGTERM the job's process group, wait, escalate to SIGKILL, reap.

        Single shared helper for kill paths (timeout, explicit cancel,
        and cleanup of jobs whose children outlived the parent). Idempotent
        and tolerant of a process that's already gone.
        """
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
            logger.warning("Failed to kill pgid for %s: %s", job.exp_name, e)
            return
        logger.info("Killed pgid for job (%s, %s)", job.exp_name, why)

    def _kill_job(self, job_id: str) -> None:
        """Kill a job's process (internal helper)."""
        if job_id not in self._jobs:
            return
        self._terminate_pgid(self._jobs[job_id], "explicit kill")
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
        """Close output file. If the process is still alive (children
        outlived the parent), kill the pgid first — otherwise DataLoader
        workers and Lightning subprocesses can hold GPU memory after the
        experiment row has been marked terminal in the DB.
        """
        if job_id in self._jobs:
            job = self._jobs[job_id]
            if job.proc.poll() is None:
                self._terminate_pgid(job, "still alive at cleanup")
            try:
                job.output_file.close()
            except OSError as e:
                logger.warning("Failed to close output file for job %s: %s", job_id, e)
            # Unregister from the parent-death cleanup set: process is
            # already reaped (or being killed here) so atexit shouldn't
            # try to killpg it again. Skip if it isn't a Popen (e.g. a
            # ``RecoveredProcess`` from a prior run).
            try:
                from alpha_lab.tools import _unregister_subprocess
                if isinstance(job.proc, subprocess.Popen):
                    _unregister_subprocess(job.proc)
            except Exception:
                pass
            # Don't delete from _jobs - keep for status queries
            # The dispatcher will stop polling completed jobs anyway

    def cleanup_all(self) -> None:
        """Kill all running jobs and clean up. Call on shutdown."""
        for job_id in list(self._jobs.keys()):
            job = self._jobs[job_id]
            if job.proc.poll() is None:
                self.cancel(job_id)
        self._jobs.clear()
