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
            "max_turns": 50,
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
    assert result.max_turns == 50
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
    assert result.max_turns is None


@pytest.mark.parametrize("raw_value", [0, -1, "0", True, 50.0, 50.5, "50.5"])
def test_build_from_config_rejects_invalid_max_turns(raw_value: object) -> None:
    meta = make_meta(metadata={
        "log_name": "a.log", "prompt_source": "inline", "max_turns": raw_value,
    })

    with pytest.raises(ValueError, match="max_turns"):
        AgentDefinition.build_from_config(meta, "body")


def test_build_from_config_accepts_numeric_string_max_turns() -> None:
    meta = make_meta(metadata={
        "log_name": "a.log", "prompt_source": "inline", "max_turns": " 50 ",
    })

    assert AgentDefinition.build_from_config(meta, "body").max_turns == 50


def test_phase2_agents_have_turn_limits() -> None:
    for agent_id in (
        "phase2/builder",
        "phase2/critic",
        "phase2/tester",
        "supervisor/phase2_reviewer",
    ):
        assert load_agent(agent_id).max_turns == 50, agent_id


def test_build_from_config_blocked_paths_defaults_to_alpha_lab_and_private() -> None:
    result = AgentDefinition.build_from_config(make_meta(), "body")
    assert result.blocked_paths == (".alpha_lab", "private")


def test_build_from_config_blocked_paths_explicit_list_overrides_default() -> None:
    meta = make_meta(metadata={
        "log_name": "a.log", "prompt_source": "inline",
        "blocked_paths": ["private", "harness"],
    })
    result = AgentDefinition.build_from_config(meta, "body")
    # .alpha_lab is injected at the front since the override omits it.
    assert result.blocked_paths == (".alpha_lab", "private", "harness")


def test_build_from_config_normalizes_blocked_paths_whitespace() -> None:
    # Stored value is the stripped one used to derive the mount (no validated-vs-stored skew).
    meta = make_meta(metadata={
        "log_name": "a.log", "prompt_source": "inline", "blocked_paths": [" private "],
    })
    result = AgentDefinition.build_from_config(meta, "body")
    assert result.blocked_paths == (".alpha_lab", "private")


def test_post_init_injects_alpha_lab_when_omitted() -> None:
    # Even an explicit empty list still gets .alpha_lab — the invariant can't be opted out of.
    meta = make_meta(metadata={
        "log_name": "a.log", "prompt_source": "inline", "blocked_paths": [],
    })
    result = AgentDefinition.build_from_config(meta, "body")
    assert result.blocked_paths == (".alpha_lab",)


def test_build_from_config_rejects_scalar_blocked_paths() -> None:
    # A bare string would otherwise be iterated character-by-character.
    meta = make_meta(metadata={
        "log_name": "a.log", "prompt_source": "inline", "blocked_paths": "private",
    })
    with pytest.raises(ValueError, match="blocked_paths"):
        AgentDefinition.build_from_config(meta, "body")


def test_build_from_config_rejects_empty_blocked_path() -> None:
    meta = make_meta(metadata={
        "log_name": "a.log", "prompt_source": "inline", "blocked_paths": [""],
    })
    with pytest.raises(ValueError, match="blocked_paths"):
        AgentDefinition.build_from_config(meta, "body")


def test_build_from_config_rejects_non_string_blocked_path() -> None:
    meta = make_meta(metadata={
        "log_name": "a.log", "prompt_source": "inline", "blocked_paths": [123],
    })
    with pytest.raises(ValueError, match="blocked_paths"):
        AgentDefinition.build_from_config(meta, "body")


def test_build_from_config_rejects_blocked_path_traversal() -> None:
    meta = make_meta(metadata={
        "log_name": "a.log", "prompt_source": "inline", "blocked_paths": ["../escape"],
    })
    with pytest.raises(ValueError, match="relative"):
        AgentDefinition.build_from_config(meta, "body")


def test_only_proxy_agents_see_private_alpha_lab_always_blocked() -> None:
    """Security boundary: only the proxy agents see private/; every other agent blocks it too.
    .alpha_lab (board + memory, reached via the gateway) is blocked for every agent. The
    experiment executor reads private/ as an unsandboxed subprocess, not as an agent."""
    for agent_id in ("proxy/intake", "proxy/handoff"):
        assert load_agent(agent_id).blocked_paths == (".alpha_lab",), agent_id
    for agent_id in ("phase2/builder", "phase2/critic", "phase2/tester",
                     "phase1/explorer", "phase3/worker_implement", "phase3/strategist"):
        assert load_agent(agent_id).blocked_paths == (".alpha_lab", "private"), agent_id


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
