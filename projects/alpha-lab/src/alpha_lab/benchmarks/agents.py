"""Typed override fragment merged into generated benchmark configs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    """Optional overrides applied on top of a benchmark's TaskConfig fields.

    Each field defaults to ``None``; only non-``None`` fields override the
    benchmark or generator-supplied value during workspace materialization.
    """

    provider: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    shell_timeout: int | None = None
    tool_output_max_chars: int | None = None
