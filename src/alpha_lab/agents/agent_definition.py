from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alpha_lab.tools import load_tools
from alpha_lab.tools.tool_definition import ToolDefinition

logger = logging.getLogger("alpha_lab.agents.agent_definition")

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
    max_turns: int | None = None
    needs_gpu: bool = False
    # Workspace-relative paths blocked (replaced by an empty, read-only overlay) in this
    # agent's sandbox, so it can neither read their real contents nor write them. Always
    # includes ``.alpha_lab`` (the parent-owned stores — board, memory — reached only via the
    # gateway; injected in __post_init__ if omitted). Defaults to also blocking the held-out
    # ``private`` set; an agent overrides this to see ``private`` (it cannot drop ``.alpha_lab``).
    blocked_paths: tuple[str, ...] = field(default_factory=lambda: (".alpha_lab", "private"))

    def __post_init__(self) -> None:
        # ``.alpha_lab`` must be blocked for every agent. Inject it (with a warning) rather
        # than raise: agents load lazily, so a raise would surface only at the agent's first
        # run, whereas injecting keeps the invariant fail-safe regardless of author omission.
        if ".alpha_lab" not in self.blocked_paths:
            logger.warning(
                "agent %r blocked_paths is missing '.alpha_lab'; injecting it", self.name
            )
            object.__setattr__(self, "blocked_paths", (".alpha_lab", *self.blocked_paths))

    @classmethod
    def build_from_config(cls, frontmatter: dict[str, Any], body: str) -> AgentDefinition:
        body = body.lstrip("\n")
        metadata = frontmatter["metadata"]
        prompt_source = metadata["prompt_source"]
        raw_max_turns = metadata.get("max_turns")
        max_turns: int | None
        if raw_max_turns is None:
            max_turns = None
        elif isinstance(raw_max_turns, int) and not isinstance(raw_max_turns, bool):
            max_turns = raw_max_turns
        elif isinstance(raw_max_turns, str) and raw_max_turns.strip().isdigit():
            max_turns = int(raw_max_turns.strip())
        else:
            raise ValueError("'metadata.max_turns' must be a positive integer")
        if max_turns is not None and max_turns <= 0:
            raise ValueError("'metadata.max_turns' must be a positive integer")
        fields: dict[str, Any] = dict(
            name=frontmatter["name"],
            description=frontmatter["description"],
            tools=load_tools(frontmatter["allowed-tools"]),
            include_web_search=bool(metadata.get("include_web_search", False)),
            reasoning_effort=metadata.get("reasoning_effort"),
            log_name=metadata["log_name"],
            min_report_attempts=int(metadata.get("min_report_attempts", 2)),
            prompt_source=prompt_source,
            prompt_body=body if prompt_source == "inline" else "",
            max_turns=max_turns,
            needs_gpu=bool(metadata.get("needs_gpu", False)),
        )
        # Present (even as []) overrides the default; absent keeps the default. ``.alpha_lab``
        # is enforced by __post_init__, so an override that omits it is still safe.
        if "blocked_paths" in metadata:
            paths = metadata["blocked_paths"]
            if not isinstance(paths, list):
                raise ValueError(
                    "'metadata.blocked_paths' must be a list of workspace-relative paths"
                )
            cleaned: list[str] = []
            for path in paths:
                if not isinstance(path, str) or not path.strip():
                    raise ValueError("'metadata.blocked_paths' entries must be non-empty strings")
                # Normalize on store (matches the config loader), so the validated value is
                # the one used to derive the mount at runtime.
                path = path.strip()
                if Path(path).is_absolute() or ".." in Path(path).parts:
                    raise ValueError(
                        f"'metadata.blocked_paths' entries must be plain relative paths "
                        f"(no absolute paths or '..'), got {path!r}"
                    )
                cleaned.append(path)
            fields["blocked_paths"] = tuple(cleaned)
        return cls(**fields)

    @property
    def adapter_prompt_key(self) -> str:
        match = _ADAPTER_PROMPT_SOURCE_RE.match(self.prompt_source)
        if match is None:
            raise ValueError(
                f"prompt_source {self.prompt_source!r} is not an adapter reference"
            )
        return match.group("key")
