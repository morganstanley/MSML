"""Tool definitions and dispatch for alpha-lab."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from importlib import resources

import yaml

from alpha_lab.config import split_frontmatter_from_config_body
from alpha_lab.tools import registry
from alpha_lab.tools.access import WorkspaceAccess, build_minimal_workspace_access_schema_for_tools
from alpha_lab.tools.execution import DEFAULT_TIMEOUT, execute_tool, parse_tool_args
from alpha_lab.tools.tool_definition import ToolDefinition, ToolEffect

TOOLS_DIR = resources.files(registry)

WEB_SEARCH_TOOL: dict[str, object] = {"type": "web_search_preview"}

_DEFAULT_TOOL_NAMES = (
    "shell_exec",
    "view_image",
    "ask_user",
    "report_to_user",
    "memory_store",
    "memory_search",
    "memory_read",
)


def load_tool(name: str) -> ToolDefinition:
    """Load a tool definition from registry/<name>.md."""
    text = (TOOLS_DIR / f"{name}.md").read_text()
    frontmatter_text, _ = split_frontmatter_from_config_body(text)
    return ToolDefinition.build_from_config(yaml.safe_load(frontmatter_text))


def load_tools(names: Iterable[str]) -> tuple[ToolDefinition, ...]:
    """Load named tools, skipping names with no registry file (e.g. web_search)."""
    return tuple(
        load_tool(name) for name in names if (TOOLS_DIR / f"{name}.md").is_file()
    )


def get_tool_schemas(
    tools: Sequence[ToolDefinition],
    include_web_search: bool = False,
) -> list[dict]:
    """Build a tool schema list from tool definitions."""
    schemas = [tool.schema for tool in tools]
    if include_web_search:
        schemas.append(WEB_SEARCH_TOOL)
    return schemas


def default_tool_schemas() -> list[dict]:
    """Build the default tool schema list used by a bare AgentLoop."""
    return [load_tool(name).schema for name in _DEFAULT_TOOL_NAMES] + [WEB_SEARCH_TOOL]


def load_all_tools() -> tuple[ToolDefinition, ...]:
    """Load every registry tool. For introspection and validation only."""
    return tuple(
        load_tool(path.name[:-3])
        for path in TOOLS_DIR.iterdir()
        if path.name.endswith(".md")
    )


__all__ = [
    "TOOLS_DIR",
    "ToolDefinition",
    "ToolEffect",
    "WorkspaceAccess",
    "build_minimal_workspace_access_schema_for_tools",
    "load_tool",
    "load_tools",
    "get_tool_schemas",
    "default_tool_schemas",
    "load_all_tools",
    "WEB_SEARCH_TOOL",
    "execute_tool",
    "parse_tool_args",
    "DEFAULT_TIMEOUT",
]
