"""Tests for config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from alpha_lab.config import (
    Phase3Config,
    PipelineConfig,
    TaskConfig,
    load_config,
)


class TestTaskConfig:
    def test_defaults(self) -> None:
        config = TaskConfig(data_path="/data", description="Test task")
        assert config.target == ""
        assert config.reasoning_effort == "low"
        assert config.model == "gpt-5.2"
        assert config.pipeline.phases == ["phase1"]

    def test_resolve_data_path_absolute(self) -> None:
        config = TaskConfig(data_path="/abs/path/data.csv", description="D")
        resolved = config.resolve_data_path("/base")
        assert resolved == "/abs/path/data.csv"

    def test_resolve_data_path_relative(self) -> None:
        config = TaskConfig(data_path="data/file.csv", description="D")
        resolved = config.resolve_data_path("/base")
        assert "data/file.csv" in resolved
        assert resolved.startswith("/")


class TestLoadConfig:
    def test_load_minimal(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "data_path: /data/test.csv\n"
            "description: Test analysis\n"
        )
        config = load_config(str(config_file))
        assert config.data_path == "/data/test.csv"
        assert config.description == "Test analysis"

    def test_load_with_target(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "data_path: /data/test.csv\n"
            "description: Test\n"
            "target: close\n"
        )
        config = load_config(str(config_file))
        assert config.target == "close"

    def test_load_with_pipeline(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "data_path: /data/test.csv\n"
            "description: Test\n"
            "pipeline:\n"
            "  phases: ['phase1', 'phase2']\n"
            "  max_fix_iterations: 5\n"
        )
        config = load_config(str(config_file))
        assert config.pipeline.phases == ["phase1", "phase2"]
        assert config.pipeline.max_fix_iterations == 5

    def test_load_with_phase3(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "data_path: /data/test.csv\n"
            "description: Test\n"
            "pipeline:\n"
            "  phases: ['phase1', 'phase2', 'phase3']\n"
            "  phase3:\n"
            "    max_concurrent_gpus: 4\n"
            "    max_experiments: 20\n"
            "    worker_count: 2\n"
            "    slurm_partitions: ['h100', 'hpc-mid']\n"
        )
        config = load_config(str(config_file))
        assert config.pipeline.phase3.max_concurrent_gpus == 4
        assert config.pipeline.phase3.max_experiments == 20
        assert config.pipeline.phase3.worker_count == 2
        assert config.pipeline.phase3.slurm_partitions == ["h100", "hpc-mid"]

    def test_load_missing_required_field(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("data_path: /data/test.csv\n")
        with pytest.raises(ValueError, match="Missing required"):
            load_config(str(config_file))

    def test_load_nonexistent_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_invalid_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("just a string")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_config(str(config_file))

    def test_load_strips_whitespace(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "data_path: '  /data/test.csv  '\n"
            "description: '  Test  '\n"
        )
        config = load_config(str(config_file))
        assert config.data_path == "/data/test.csv"
        assert config.description == "Test"

    def test_load_unknown_fields_ignored(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "data_path: /data/test.csv\n"
            "description: Test\n"
            "unknown_field: ignored\n"
        )
        config = load_config(str(config_file))
        assert config.data_path == "/data/test.csv"
        assert not hasattr(config, "unknown_field")


class TestPhase3Config:
    def test_defaults(self) -> None:
        config = Phase3Config()
        assert config.max_concurrent_gpus == 8
        assert config.max_experiments == 50
        assert config.worker_count == 4
        assert config.gpu_per_job == 1

    def test_custom_values(self) -> None:
        config = Phase3Config(max_concurrent_gpus=4, worker_count=2)
        assert config.max_concurrent_gpus == 4
        assert config.worker_count == 2
