from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any] = field(hash=False)

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

        return cls(
            name=name,
            description=description,
            parameters=parameters,
        )

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
