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
        assert config.shell_timeout == 300
        assert config.tool_output_max_chars == 8000
        assert config.pipeline.phases == ["phase1"]

    def test_tool_output_max_chars_override(self, tmp_path: Path) -> None:
        import json

        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(
            json.dumps(
                {
                    "data_path": "/data",
                    "description": "D",
                    "tool_output_max_chars": 30000,
                }
            )
        )
        config = load_config(cfg_file)
        assert config.tool_output_max_chars == 30000

    def test_tool_output_max_chars_rejects_non_int(self) -> None:
        with pytest.raises(ValueError, match="must be an int"):
            TaskConfig(
                data_path="/d",
                description="d",
                tool_output_max_chars="8000",  # type: ignore[arg-type]
            )

    def test_tool_output_max_chars_rejects_bool(self) -> None:
        # bool subclasses int in Python; must be rejected explicitly.
        with pytest.raises(ValueError, match="must be an int"):
            TaskConfig(
                data_path="/d",
                description="d",
                tool_output_max_chars=True,  # type: ignore[arg-type]
            )

    def test_tool_output_max_chars_rejects_below_floor(self) -> None:
        with pytest.raises(ValueError, match=">= 100"):
            TaskConfig(
                data_path="/d",
                description="d",
                tool_output_max_chars=1,
            )

    def test_tool_output_max_chars_at_floor_allowed(self) -> None:
        config = TaskConfig(
            data_path="/d",
            description="d",
            tool_output_max_chars=100,
        )
        assert config.tool_output_max_chars == 100

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
        with pytest.raises(ValueError, match="must be a mapping"):
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


class TestConductorConfig:
    """Conductor-related config: NOOP flag, interval, reasoning effort."""

    def test_defaults_safe_for_existing_pipelines(self) -> None:
        # The defaults must be NOOP-compatible so existing pipelines that don't
        # mention the conductor at all keep working as before. no_conductor is
        # False (so the conductor will actually run), but other agents must
        # tolerate the absence of meta/ files when nothing has been written
        # there yet. User instructions live in meta/instructions/from_user.md
        # — the config has no user_directives field.
        config = TaskConfig(data_path="/d", description="D")
        assert not hasattr(config, "user_directives")
        assert config.pipeline.phase3.no_conductor is False
        assert config.pipeline.phase3.conductor_interval == 1800
        # Conductor reasoning effort is top-level (parallel to the main
        # `reasoning_effort` knob), not nested under phase3.
        assert config.conductor_reasoning_effort == "high"
        # Conductor defaults to Bedrock + opus regardless of main pipeline
        # provider — its job benefits from the strongest available model.
        assert config.conductor_provider == "bedrock"
        assert config.conductor_model == "claude-opus-4-7"

    def test_conductor_provider_and_model_round_trip(self, tmp_path: Path) -> None:
        import json
        cfg_file = tmp_path / "conf.json"
        cfg_file.write_text(json.dumps({
            "data_path": "/d",
            "description": "D",
            "conductor_provider": "openai",
            "conductor_model": "gpt-5.4",
        }))
        config = load_config(cfg_file)
        assert config.conductor_provider == "openai"
        assert config.conductor_model == "gpt-5.4"

    def test_conductor_provider_empty_string_means_inherit(
        self, tmp_path: Path
    ) -> None:
        # Explicit "" opts out of the bedrock default and tells the
        # dispatcher to reuse the main pipeline's provider.
        import json
        cfg_file = tmp_path / "conf.json"
        cfg_file.write_text(json.dumps({
            "data_path": "/d",
            "description": "D",
            "conductor_provider": "",
            "conductor_model": "",
        }))
        config = load_config(cfg_file)
        assert config.conductor_provider == ""
        assert config.conductor_model == ""

    def test_legacy_user_directives_field_ignored(self, tmp_path: Path) -> None:
        """If a stale config still carries a top-level user_directives field,
        load_config should silently drop it (filter unknown fields). The
        Conductor only reads from meta/instructions/from_user.md."""
        import json
        cfg_file = tmp_path / "conf.json"
        cfg_file.write_text(
            json.dumps({
                "data_path": "/d",
                "description": "D",
                "user_directives": "this should be ignored",
            })
        )
        config = load_config(cfg_file)
        assert not hasattr(config, "user_directives")

    def test_no_conductor_flag_round_trip(self, tmp_path: Path) -> None:
        import json
        cfg_file = tmp_path / "conf.json"
        cfg_file.write_text(
            json.dumps({
                "data_path": "/d",
                "description": "D",
                "pipeline": {
                    "phases": ["phase3"],
                    "phase3": {"no_conductor": True, "conductor_interval": 60},
                },
            })
        )
        config = load_config(cfg_file)
        assert config.pipeline.phase3.no_conductor is True
        assert config.pipeline.phase3.conductor_interval == 60
