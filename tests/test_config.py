"""Tests for config loading and validation."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from alpha_lab.agents import load_agent
from alpha_lab.agents.agent_definition import AgentDefinition
from alpha_lab.config import (
    MEMORY_REPO_GITIGNORE,
    Phase3Config,
    PipelineConfig,
    TaskConfig,
    config_to_jsonable,
    load_config,
    task_config_from_mapping,
)
from alpha_lab.constants import Phase
from alpha_lab.git import GitSpec


class TestTaskConfig:
    def test_defaults(self) -> None:
        config = TaskConfig(data_path="/data", description="Test task")
        assert config.target == ""
        assert config.reasoning_effort == "low"
        # Non-local (default openai) provider without a model falls back to the default.
        assert config.model == "gpt-5.2"
        assert config.shell_timeout == 300
        assert config.tool_output_max_chars == 8000
        assert config.pipeline.phases == [Phase.PHASE1]
        assert all(item in config.memory_spec.gitignore for item in MEMORY_REPO_GITIGNORE)

    def test_memory_spec_preserves_custom_ignores(self) -> None:
        config = TaskConfig(
            data_path="/data",
            description="Test task",
            memory_spec=GitSpec(gitignore=("custom/",)),
        )
        assert config.memory_spec.gitignore == ("custom/", *MEMORY_REPO_GITIGNORE)

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

    def test_tool_output_max_chars_rejects_non_numeric(self) -> None:
        # pydantic lax coercion accepts "8000"/8000.0/True, but a non-numeric
        # value has no int interpretation and must still be rejected.
        with pytest.raises(ValueError):
            TaskConfig(
                data_path="/d",
                description="d",
                tool_output_max_chars="not-a-number",  # type: ignore[arg-type]
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

    def test_model_tags_default_empty(self) -> None:
        config = TaskConfig(data_path="/d", description="d")
        assert config.model_tags == []

    def test_model_tags_allowed_with_local_provider(self, monkeypatch) -> None:
        # local + model_tags validates the tags against the proxy, which needs
        # LOCAL_BASE_URL + a live query — stub both so the check passes offline.
        monkeypatch.setenv("LOCAL_BASE_URL", "http://proxy:4001")
        monkeypatch.setattr("alpha_lab.config.get_available_model_tags", lambda *a, **k: {"local1"})
        config = TaskConfig(
            data_path="/d", description="d", provider="local", model_tags=["local1"]
        )
        assert config.model_tags == ["local1"]

    def test_model_tags_rejected_with_non_local_provider(self) -> None:
        with pytest.raises(ValueError, match="model_tags is only supported with provider 'local'"):
            TaskConfig(
                data_path="/d", description="d", provider="openai", model_tags=["local1"]
            )

    def test_non_local_provider_without_model_gets_default(self) -> None:
        cfg = TaskConfig(data_path="/d", description="d", provider="bedrock")
        assert cfg.model == "gpt-5.2"

    def test_explicit_model_preserved_for_non_local(self) -> None:
        cfg = TaskConfig(data_path="/d", description="d", model="claude-opus-4-7", provider="bedrock")
        assert cfg.model == "claude-opus-4-7"

    def test_local_with_tags_keeps_model_none(self, monkeypatch) -> None:
        monkeypatch.setenv("LOCAL_BASE_URL", "http://proxy:4001")
        monkeypatch.setattr("alpha_lab.config.get_available_model_tags", lambda *a, **k: {"local1"})
        cfg = TaskConfig(data_path="/d", description="d", provider="local", model_tags=["local1"])
        assert cfg.model is None

    def test_model_and_model_tags_together_rejected(self) -> None:
        with pytest.raises(ValueError, match="Exactly one of 'model' or 'model_tags' must be set"):
            TaskConfig(
                data_path="/d", description="d", provider="local",
                model="glm-5.2", model_tags=["local1"],
            )


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
        assert config.pipeline.phases == [Phase.PHASE1, Phase.PHASE2]
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

    def test_load_unknown_fields_rejected(self, tmp_path: Path) -> None:
        # extra="forbid": an unknown field fails loudly rather than being
        # silently dropped.
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "data_path: /data/test.csv\n"
            "description: Test\n"
            "unknown_field: ignored\n"
        )
        with pytest.raises(ValueError):
            load_config(str(config_file))


class TestPhase3Config:
    def test_defaults(self) -> None:
        config = Phase3Config()
        assert config.max_concurrent_gpus == 8
        assert config.max_experiments == 50
        assert config.worker_count == 4
        assert config.gpu_per_job == 1
        assert config.gpu_ids == "auto"
        assert config.handoff is True

    def test_custom_values(self) -> None:
        config = Phase3Config(max_concurrent_gpus=4, worker_count=2)
        assert config.max_concurrent_gpus == 4
        assert config.worker_count == 2


@pytest.fixture
def agents_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point load_agent at a temp directory containing test agent .md files."""
    monkeypatch.setattr("alpha_lab.agents.AGENTS_DIR", tmp_path)
    return tmp_path


def _write_agent(agents_dir: Path, agent_id: str, content: str) -> None:
    (agents_dir / f"{agent_id}.md").write_text(content)


class TestLoadAgent:
    def test_inline_prompt_populates_all_fields(self, agents_dir: Path) -> None:

        content = dedent(
            """\
            ---
            name: builder
            description: Builds the framework
            allowed-tools: [shell_exec, read_file]
            metadata:
              include_web_search: true
              reasoning_effort: medium
              log_name: builder.log
              min_report_attempts: 4
              prompt_source: inline
            ---
            You are the builder agent.
            Do the thing.
            """
        )
        _write_agent(agents_dir, "builder", content)

        result = load_agent("builder")

        assert tuple(tool.name for tool in result.tools) == ("shell_exec", "read_file")
        assert result.name == "builder"
        assert result.description == "Builds the framework"
        assert result.include_web_search is True
        assert result.reasoning_effort == "medium"
        assert result.log_name == "builder.log"
        assert result.min_report_attempts == 4
        assert result.prompt_source == "inline"
        assert result.prompt_body == "You are the builder agent.\nDo the thing.\n"

    def test_non_inline_prompt_source_produces_empty_prompt_body(
        self, agents_dir: Path
    ) -> None:

        content = dedent(
            """\
            ---
            name: phase1
            description: Phase 1 agent
            allowed-tools: [shell_exec]
            metadata:
              log_name: phase1.log
              prompt_source: adapter:phase1
            ---
            This body should be discarded because prompt_source is not "inline".
            """
        )
        _write_agent(agents_dir, "phase1", content)

        result = load_agent("phase1")

        assert result.prompt_source == "adapter:phase1"
        assert result.prompt_body == ""

    def test_omitted_optional_fields_use_documented_defaults(
        self, agents_dir: Path
    ) -> None:

        content = dedent(
            """\
            ---
            name: minimal
            description: Minimal agent
            allowed-tools: []
            metadata:
              log_name: minimal.log
              prompt_source: inline
            ---
            body
            """
        )
        _write_agent(agents_dir, "minimal", content)

        result = load_agent("minimal")

        assert result.include_web_search is False
        assert result.min_report_attempts == 2
        assert result.reasoning_effort is None

    def test_optional_field_overrides_propagate(self, agents_dir: Path) -> None:

        content = dedent(
            """\
            ---
            name: configured
            description: Fully configured agent
            allowed-tools: [shell_exec]
            metadata:
              include_web_search: true
              reasoning_effort: high
              min_report_attempts: 5
              log_name: configured.log
              prompt_source: inline
            ---
            body
            """
        )
        _write_agent(agents_dir, "configured", content)

        result = load_agent("configured")

        assert result.include_web_search is True
        assert result.reasoning_effort == "high"
        assert result.min_report_attempts == 5

    def test_coerces_string_min_report_attempts_to_int(
        self, agents_dir: Path
    ) -> None:

        content = dedent(
            """\
            ---
            name: stringy
            description: min_report_attempts given as string
            allowed-tools: []
            metadata:
              log_name: stringy.log
              min_report_attempts: "3"
              prompt_source: inline
            ---
            body
            """
        )
        _write_agent(agents_dir, "stringy", content)

        result = load_agent("stringy")

        assert result.min_report_attempts == 3
        assert isinstance(result.min_report_attempts, int)

    def test_strips_leading_newlines_from_prompt_body(
        self, agents_dir: Path
    ) -> None:
        content = (
            "---\n"
            "name: padded\n"
            "description: Leading newlines in body\n"
            "allowed-tools: []\n"
            "metadata:\n"
            "  log_name: padded.log\n"
            "  prompt_source: inline\n"
            "---\n"
            "\n\n\nfirst real line\nsecond line\n"
        )
        _write_agent(agents_dir, "padded", content)

        result = load_agent("padded")

        assert result.prompt_body == "first real line\nsecond line\n"

    def test_missing_file_raises_file_not_found(self, agents_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_agent("does_not_exist")

    def test_missing_opening_fence_raises_value_error(
        self, agents_dir: Path
    ) -> None:
        content = "name: broken\nprompt_source: inline\n---\nbody\n"
        _write_agent(agents_dir, "no_opening", content)

        with pytest.raises(ValueError, match="opening"):
            load_agent("no_opening")

    def test_missing_closing_fence_raises_value_error(
        self, agents_dir: Path
    ) -> None:
        content = (
            "---\nname: broken\nprompt_source: inline\nbody without closing fence\n"
        )
        _write_agent(agents_dir, "no_closing", content)

        with pytest.raises(ValueError, match="closing"):
            load_agent("no_closing")

    def test_missing_required_field_raises_key_error(
        self, agents_dir: Path
    ) -> None:

        content = dedent(
            """\
            ---
            name: incomplete
            description: Missing prompt_source
            allowed-tools: []
            metadata:
              log_name: incomplete.log
            ---
            body
            """
        )
        _write_agent(agents_dir, "incomplete", content)

        with pytest.raises(KeyError):
            load_agent("incomplete")


class TestConfigToJsonable:
    def test_roundtrips_local_memory_spec_path(self) -> None:
        # A local memory remote makes GitPointer.src a Path, which json.dumps can't
        # serialize directly; config_to_jsonable must normalize it to a string.
        import json

        config = task_config_from_mapping(
            {
                "data_path": "/d",
                "description": "x",
                "memory_spec": {"remote": {"src": "/local/repo"}},
            }
        )
        payload = config_to_jsonable(config)
        js = json.dumps(payload)  # must not raise
        assert json.loads(js)["memory_spec"]["remote"]["src"] == "/local/repo"
        # and it reloads to an equivalent config
        reloaded = task_config_from_mapping(payload)
        assert reloaded.memory_spec.remote.is_local()

    def test_plain_config_serializes(self) -> None:
        import json

        config = TaskConfig(data_path="/d", description="x")
        assert json.dumps(config_to_jsonable(config))  # no raise
