from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolEffect(str, Enum):
    """How a tool accesses a workspace path."""

    RO = "ro"
    RW = "rw"


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any] = field(hash=False)
    # Minimal workspace footprint: {workspace-relative path: effect}, derived into
    # per-agent mounts by tools.access. Empty when the tool touches no workspace path.
    workspace_access: dict[str, ToolEffect] = field(default_factory=dict, hash=False)

    @classmethod
    def build_from_config(cls, frontmatter: dict[str, Any]) -> ToolDefinition:
        if not isinstance(frontmatter, dict):
            raise ValueError("frontmatter must be a YAML mapping")

        name = frontmatter.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("'name' must be a non-empty string")

        description = frontmatter.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("'description' must be a non-empty string")

        metadata = frontmatter.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("'metadata' must be a mapping")

        parameters = metadata.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError("'metadata.parameters' must be a mapping")

        workspace_access = metadata.get("workspace_access")
        if not isinstance(workspace_access, (dict, type(None))):
            raise ValueError("'metadata.workspace_access' must be a mapping if provided")

        return cls(
            name=name,
            description=description,
            parameters=parameters,
            workspace_access=cls._parse_workspace_access(workspace_access or {}),
        )

    @staticmethod
    def _parse_workspace_access(access: dict) -> dict[str, ToolEffect]:
        """Parse the optional ``metadata.workspace_access`` {path: effect} mapping."""
        parsed: dict[str, ToolEffect] = {}
        for path, effect in access.items():
            if not isinstance(path, str) or not path.strip():
                raise ValueError(
                    "'metadata.workspace_access' keys must be non-empty relative paths"
                )
            try:
                parsed[path] = ToolEffect(effect)
            except ValueError:
                raise ValueError(
                    f"'metadata.workspace_access[{path!r}]' must be 'ro' or 'rw', "
                    f"got {effect!r}"
                ) from None
        return parsed

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
