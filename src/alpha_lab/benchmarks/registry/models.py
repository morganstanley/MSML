"""Typed records used by benchmark registry code."""

from pathlib import Path

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from alpha_lab.config import PipelineConfig


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class Benchmark:
    """One row of the benchmark registry.

    ``pipeline`` accepts a mapping, which pydantic coerces to a
    :class:`PipelineConfig`.
    """

    id: str
    name: str
    data_path: Path
    description: str
    target: str
    domain: str
    provider: str
    model: str
    reasoning_effort: str
    shell_timeout: int
    tool_output_max_chars: int
    pipeline: PipelineConfig
    adapter_path: Path | None
    seed_path: Path | None
    notes: str
    created_at: str | None = None
    updated_at: str | None = None
    creator: str | None = None
    owner: str | None = None
