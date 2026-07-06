"""Tests for the alpha_lab.agents package.

Focuses on AgentDefinition.build_from_config (a pure dict+str -> dataclass
function) called directly with plain dicts. load_agent is already exercised by
tests/test_config.py::TestLoadAgent; here we only add integration behaviors not
covered there: nested agent_id resolution and a missing-file error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alpha_lab.agents import load_agent
from alpha_lab.agents.agent_definition import AgentDefinition
from alpha_lab.tools.tool_definition import ToolDefinition


def make_meta(
    *,
    name: str = "builder",
    description: str = "Builds the framework",
    allowed_tools: object = ("shell_exec", "read_file"),
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a meta dict with overridable fields and sensible required defaults."""
    if metadata is None:
        metadata = {"log_name": "builder.log", "prompt_source": "inline"}
    return {
        "name": name,
        "description": description,
        "allowed-tools": allowed_tools,
        "metadata": metadata,
    }


def test_build_from_config_populates_all_fields() -> None:
    meta = make_meta(
        allowed_tools=["shell_exec", "read_file"],
        metadata={
            "include_web_search": True,
            "reasoning_effort": "medium",
            "log_name": "builder.log",
            "min_report_attempts": 4,
            "prompt_source": "inline",
        },
    )

    result = AgentDefinition.build_from_config(meta, "You are the builder agent.\n")

    assert tuple(tool.name for tool in result.tools) == ("shell_exec", "read_file")
    assert all(isinstance(tool, ToolDefinition) for tool in result.tools)
    assert result.name == "builder"
    assert result.description == "Builds the framework"
    assert result.include_web_search is True
    assert result.reasoning_effort == "medium"
    assert result.log_name == "builder.log"
    assert result.min_report_attempts == 4
    assert result.prompt_source == "inline"
    assert result.prompt_body == "You are the builder agent.\n"


def test_build_from_config_strips_leading_newlines_from_body() -> None:
    meta = make_meta()

    result = AgentDefinition.build_from_config(meta, "\n\n\nfirst line\nsecond line\n")

    assert result.prompt_body == "first line\nsecond line\n"


def test_build_from_config_inline_source_keeps_body() -> None:
    meta = make_meta(metadata={"log_name": "a.log", "prompt_source": "inline"})

    result = AgentDefinition.build_from_config(meta, "the prompt body")

    assert result.prompt_body == "the prompt body"


def test_build_from_config_non_inline_source_discards_body() -> None:
    meta = make_meta(metadata={"log_name": "a.log", "prompt_source": "adapter:phase1"})

    result = AgentDefinition.build_from_config(meta, "this body should be dropped")

    assert result.prompt_source == "adapter:phase1"
    assert result.prompt_body == ""


def test_build_from_config_omitted_optionals_use_defaults() -> None:
    meta = make_meta(metadata={"log_name": "a.log", "prompt_source": "inline"})

    result = AgentDefinition.build_from_config(meta, "body")

    assert result.include_web_search is False
    assert result.min_report_attempts == 2
    assert result.reasoning_effort is None


def test_build_from_config_normalizes_tools_to_tuple() -> None:
    meta = make_meta(allowed_tools=["shell_exec", "read_file"])

    result = AgentDefinition.build_from_config(meta, "body")

    assert tuple(tool.name for tool in result.tools) == ("shell_exec", "read_file")
    assert isinstance(result.tools, tuple)


def test_build_from_config_explicit_integer_min_report_attempts_passes_through() -> None:
    meta = make_meta(
        metadata={
            "log_name": "a.log",
            "prompt_source": "inline",
            "min_report_attempts": 5,
        }
    )

    result = AgentDefinition.build_from_config(meta, "body")

    assert result.min_report_attempts == 5


def test_build_from_config_coerces_string_min_report_attempts_to_int() -> None:
    meta = make_meta(
        metadata={
            "log_name": "a.log",
            "prompt_source": "inline",
            "min_report_attempts": "3",
        }
    )

    result = AgentDefinition.build_from_config(meta, "body")

    assert result.min_report_attempts == 3
    assert isinstance(result.min_report_attempts, int)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [(1, True), (0, False), ("", False), ("yes", True)],
)
def test_build_from_config_coerces_include_web_search_via_bool(
    raw_value: object, expected: bool
) -> None:
    meta = make_meta(
        metadata={
            "log_name": "a.log",
            "prompt_source": "inline",
            "include_web_search": raw_value,
        }
    )

    result = AgentDefinition.build_from_config(meta, "body")

    assert result.include_web_search is expected


@pytest.mark.parametrize("missing_key", ["name", "description", "allowed-tools"])
def test_build_from_config_missing_top_level_key_raises_key_error(
    missing_key: str,
) -> None:
    meta = make_meta()
    del meta[missing_key]

    with pytest.raises(KeyError):
        AgentDefinition.build_from_config(meta, "body")


@pytest.mark.parametrize("missing_key", ["prompt_source", "log_name"])
def test_build_from_config_missing_metadata_key_raises_key_error(
    missing_key: str,
) -> None:
    metadata = {"log_name": "a.log", "prompt_source": "inline"}
    del metadata[missing_key]
    meta = make_meta(metadata=metadata)

    with pytest.raises(KeyError):
        AgentDefinition.build_from_config(meta, "body")


def test_build_from_config_missing_metadata_block_raises_key_error() -> None:
    meta = {
        "name": "builder",
        "description": "Builds the framework",
        "allowed-tools": [],
    }

    with pytest.raises(KeyError):
        AgentDefinition.build_from_config(meta, "body")


def make_agent_definition(prompt_source: str) -> AgentDefinition:
    return AgentDefinition.build_from_config(
        make_meta(metadata={"log_name": "a.log", "prompt_source": prompt_source}),
        "body",
    )


def test_adapter_prompt_key_returns_key_after_prefix() -> None:
    agent_definition = make_agent_definition("adapter:phase3_reporter")

    assert agent_definition.adapter_prompt_key == "phase3_reporter"


@pytest.mark.parametrize("prompt_source", ["inline", "adapter:", "phase3_reporter"])
def test_adapter_prompt_key_rejects_non_adapter_source(prompt_source: str) -> None:
    agent_definition = make_agent_definition(prompt_source)

    with pytest.raises(ValueError):
        agent_definition.adapter_prompt_key


def test_adapter_prompt_key_greedily_captures_colon_containing_key() -> None:
    agent_definition = make_agent_definition("adapter:phase3:extra")

    assert agent_definition.adapter_prompt_key == "phase3:extra"


def test_adapter_prompt_key_rejects_source_that_only_starts_with_adapter() -> None:
    agent_definition = make_agent_definition("adapterX:key")

    with pytest.raises(ValueError):
        agent_definition.adapter_prompt_key


def test_equal_agent_definitions_compare_equal() -> None:
    first = make_agent_definition("adapter:phase3_reporter")
    second = make_agent_definition("adapter:phase3_reporter")

    assert first == second


def test_agent_definition_is_hashable() -> None:
    agent_definition = make_agent_definition("adapter:phase3_reporter")

    assert hash(agent_definition) == hash(make_agent_definition("adapter:phase3_reporter"))
    assert {agent_definition, agent_definition} == {agent_definition}


@pytest.fixture
def agents_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point load_agent at a temp directory containing test agent .md files."""
    monkeypatch.setattr("alpha_lab.agents.AGENTS_DIR", tmp_path)
    return tmp_path


def test_load_agent_resolves_nested_agent_id_under_subdirectory(
    agents_dir: Path,
) -> None:
    nested_file = agents_dir / "phase2" / "builder.md"
    nested_file.parent.mkdir()
    nested_file.write_text(
        "---\n"
        "name: builder\n"
        "description: Nested builder agent\n"
        "allowed-tools: [shell_exec]\n"
        "metadata:\n"
        "  log_name: builder.log\n"
        "  prompt_source: inline\n"
        "---\n"
        "nested body\n"
    )

    result = load_agent("phase2/builder")

    assert result.name == "builder"
    assert result.prompt_body == "nested body\n"


def test_load_agent_missing_file_raises_file_not_found(agents_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_agent("does_not_exist")


def test_load_agent_adapter_source_round_trips_to_adapter_prompt_key(
    agents_dir: Path,
) -> None:
    agent_file = agents_dir / "reporter.md"
    agent_file.write_text(
        "---\n"
        "name: reporter\n"
        "description: Reports results\n"
        "allowed-tools: [read_file]\n"
        "metadata:\n"
        "  log_name: reporter.log\n"
        "  prompt_source: adapter:phase3_reporter\n"
        "---\n"
        "body that the adapter source discards\n"
    )

    result = load_agent("reporter")

    assert result.adapter_prompt_key == "phase3_reporter"
    assert result.prompt_body == ""
