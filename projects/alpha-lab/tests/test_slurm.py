"""Tests for SLURM wrapper — all subprocess calls are mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alpha_lab.experiment_db import Experiment
from alpha_lab.slurm import SlurmManager


@pytest.fixture()
def slurm() -> SlurmManager:
    return SlurmManager(
        partitions=["h100", "hpc-mid"],
        gpu_per_job=1,
        max_gpus=8,
        time_limit="02:00:00",
    )


@pytest.fixture()
def sample_experiment() -> Experiment:
    return Experiment(
        id=1,
        name="exp_lstm_v1",
        description="LSTM baseline",
        hypothesis="RNNs work",
        status="checked",
        config_json='{"model": "lstm"}',
        worker_id=None,
        slurm_job_id=None,
        results_json=None,
        error=None,
        debrief_path=None,
        created_at=1000.0,
        updated_at=1000.0,
        started_at=None,
        finished_at=None,
    )


class TestSbatchGeneration:
    def test_script_contains_job_name(self, slurm: SlurmManager, sample_experiment: Experiment) -> None:
        script = slurm.generate_sbatch_script(sample_experiment, "/workspace")
        assert "alpha_exp_lstm_v1" in script

    def test_script_contains_partition(self, slurm: SlurmManager, sample_experiment: Experiment) -> None:
        script = slurm.generate_sbatch_script(sample_experiment, "/workspace")
        assert "--partition=h100" in script or "--partition=hpc-mid" in script

    def test_script_contains_gpu(self, slurm: SlurmManager, sample_experiment: Experiment) -> None:
        script = slurm.generate_sbatch_script(sample_experiment, "/workspace")
        assert "--gpus=1" in script

    def test_script_contains_time_limit(self, slurm: SlurmManager, sample_experiment: Experiment) -> None:
        script = slurm.generate_sbatch_script(sample_experiment, "/workspace")
        assert "02:00:00" in script

    def test_script_contains_run_command(self, slurm: SlurmManager, sample_experiment: Experiment) -> None:
        script = slurm.generate_sbatch_script(sample_experiment, "/workspace")
        assert "uv run python run_experiment.py" in script
        assert "cd /workspace/experiments/exp_lstm_v1" in script

    def test_partition_round_robin(self, slurm: SlurmManager, sample_experiment: Experiment) -> None:
        """Partitions should cycle through the list."""
        scripts = [slurm.generate_sbatch_script(sample_experiment, "/ws") for _ in range(4)]
        partitions_used = []
        for script in scripts:
            for line in script.splitlines():
                if "--partition=" in line:
                    partitions_used.append(line.split("=")[1])
        # Should cycle: h100, hpc-mid, h100, hpc-mid
        assert partitions_used[0] == "h100"
        assert partitions_used[1] == "hpc-mid"
        assert partitions_used[2] == "h100"
        assert partitions_used[3] == "hpc-mid"


class TestSubmit:
    @patch("alpha_lab.slurm.subprocess.run")
    def test_submit_success(self, mock_run: MagicMock, slurm: SlurmManager) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Submitted batch job 12345\n",
            stderr="",
        )
        job_id = slurm.submit("/path/to/script.sh")
        assert job_id == "12345"
        mock_run.assert_called_once()

    @patch("alpha_lab.slurm.subprocess.run")
    def test_submit_failure(self, mock_run: MagicMock, slurm: SlurmManager) -> None:
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="sbatch: error: Invalid partition",
        )
        with pytest.raises(RuntimeError, match="sbatch failed"):
            slurm.submit("/path/to/script.sh")

    @patch("alpha_lab.slurm.subprocess.run")
    def test_submit_timeout(self, mock_run: MagicMock, slurm: SlurmManager) -> None:
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="sbatch", timeout=30)
        with pytest.raises(RuntimeError, match="timed out"):
            slurm.submit("/path/to/script.sh")


class TestCancel:
    @patch("alpha_lab.slurm.subprocess.run")
    def test_cancel_success(self, mock_run: MagicMock, slurm: SlurmManager) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        slurm.cancel("12345")  # Should not raise
        mock_run.assert_called_once()

    @patch("alpha_lab.slurm.subprocess.run")
    def test_cancel_failure_no_crash(self, mock_run: MagicMock, slurm: SlurmManager) -> None:
        mock_run.side_effect = Exception("scancel failed")
        slurm.cancel("12345")  # Should not raise — just logs warning


class TestPollJobs:
    @patch("alpha_lab.slurm.subprocess.run")
    def test_poll_active_jobs(self, mock_run: MagicMock, slurm: SlurmManager) -> None:
        # squeue returns active jobs
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="12345 RUNNING\n12346 PENDING\n",
            stderr="",
        )
        result = slurm.poll_jobs(["12345", "12346"])
        assert result["12345"] == "RUNNING"
        assert result["12346"] == "PENDING"

    @patch("alpha_lab.slurm.subprocess.run")
    def test_poll_completed_jobs(self, mock_run: MagicMock, slurm: SlurmManager) -> None:
        # First call (squeue) returns nothing, second (sacct) returns completed
        def side_effect(*args, **kwargs):
            cmd = args[0]
            if "squeue" in cmd:
                return MagicMock(returncode=0, stdout="", stderr="")
            elif "sacct" in cmd:
                return MagicMock(returncode=0, stdout="12345|COMPLETED\n", stderr="")
            return MagicMock(returncode=1, stdout="", stderr="")

        mock_run.side_effect = side_effect
        result = slurm.poll_jobs(["12345"])
        assert result["12345"] == "COMPLETED"

    @patch("alpha_lab.slurm.subprocess.run")
    def test_poll_sacct_strips_plus_suffix(self, mock_run: MagicMock, slurm: SlurmManager) -> None:
        """sacct may return states like COMPLETED+ — strip the +."""
        def side_effect(*args, **kwargs):
            cmd = args[0]
            if "squeue" in cmd:
                return MagicMock(returncode=0, stdout="", stderr="")
            elif "sacct" in cmd:
                return MagicMock(returncode=0, stdout="12345|COMPLETED+\n", stderr="")
            return MagicMock(returncode=1, stdout="", stderr="")

        mock_run.side_effect = side_effect
        result = slurm.poll_jobs(["12345"])
        assert result["12345"] == "COMPLETED"

    @patch("alpha_lab.slurm.subprocess.run")
    def test_poll_unknown_job(self, mock_run: MagicMock, slurm: SlurmManager) -> None:
        """Jobs not found anywhere get UNKNOWN status."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = slurm.poll_jobs(["99999"])
        assert result["99999"] == "UNKNOWN"

    def test_poll_empty_list(self, slurm: SlurmManager) -> None:
        result = slurm.poll_jobs([])
        assert result == {}


class TestGPUCounting:
    @patch("alpha_lab.slurm.subprocess.run")
    def test_running_gpu_count(self, mock_run: MagicMock, slurm: SlurmManager) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="12345 RUNNING gpu:1\n12346 PENDING gpu:1\n12347 RUNNING gpu:1\n",
            stderr="",
        )
        count = slurm.running_gpu_count()
        assert count == 3  # 3 jobs × 1 GPU each

    @patch("alpha_lab.slurm.subprocess.run")
    def test_can_submit_under_budget(self, mock_run: MagicMock, slurm: SlurmManager) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="12345 RUNNING gpu:1\n",
            stderr="",
        )
        assert slurm.can_submit() is True  # 1 + 1 = 2 <= 8

    @patch("alpha_lab.slurm.subprocess.run")
    def test_can_submit_at_budget(self, mock_run: MagicMock, slurm: SlurmManager) -> None:
        lines = "\n".join(f"{i} RUNNING gpu:1" for i in range(8))
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=lines + "\n",
            stderr="",
        )
        assert slurm.can_submit() is False  # 8 + 1 = 9 > 8


class TestSubmitExperiment:
    @patch("alpha_lab.slurm.subprocess.run")
    def test_submit_experiment_creates_files(
        self, mock_run: MagicMock, slurm: SlurmManager,
        sample_experiment: Experiment, tmp_workspace: str,
    ) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Submitted batch job 99999\n",
            stderr="",
        )
        job_id = slurm.submit_experiment(sample_experiment, tmp_workspace)
        assert job_id == "99999"

        # Verify script was written
        exp_dir = Path(tmp_workspace) / "experiments" / "exp_lstm_v1"
        assert (exp_dir / "submit.sh").exists()
        content = (exp_dir / "submit.sh").read_text()
        assert "sbatch" not in content  # it's the script, not the command
        assert "exp_lstm_v1" in content
