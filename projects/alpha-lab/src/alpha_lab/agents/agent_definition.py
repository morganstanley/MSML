from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from alpha_lab.tools import load_tools
from alpha_lab.tools.tool_definition import ToolDefinition

_ADAPTER_PROMPT_SOURCE_RE = re.compile(r"^adapter:(?P<key>.+)$")


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    description: str
    tools: tuple[ToolDefinition, ...]
    include_web_search: bool
    reasoning_effort: str | None
    log_name: str
    min_report_attempts: int
    prompt_source: str
    prompt_body: str
    needs_gpu: bool = False

    @classmethod
    def build_from_config(cls, frontmatter: dict[str, Any], body: str) -> AgentDefinition:
        body = body.lstrip("\n")
        metadata = frontmatter["metadata"]
        prompt_source = metadata["prompt_source"]
        return cls(
            name=frontmatter["name"],
            description=frontmatter["description"],
            tools=load_tools(frontmatter["allowed-tools"]),
            include_web_search=bool(metadata.get("include_web_search", False)),
            reasoning_effort=metadata.get("reasoning_effort"),
            log_name=metadata["log_name"],
            min_report_attempts=int(metadata.get("min_report_attempts", 2)),
            prompt_source=prompt_source,
            prompt_body=body if prompt_source == "inline" else "",
            needs_gpu=bool(metadata.get("needs_gpu", False)),
        )

    @property
    def adapter_prompt_key(self) -> str:
        match = _ADAPTER_PROMPT_SOURCE_RE.match(self.prompt_source)
        if match is None:
            raise ValueError(
                f"prompt_source {self.prompt_source!r} is not an adapter reference"
            )
        return match.group("key")
