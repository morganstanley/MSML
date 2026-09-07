"""Typed records used by benchmark registry code."""

from dataclasses import dataclass
from pathlib import Path

from alpha_lab.config import PipelineConfig


@dataclass(frozen=True)
class Benchmark:
    """One row of the benchmark registry."""

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

    def __post_init__(self) -> None:
        """Coerce ``pipeline`` from dict to :class:`PipelineConfig`."""
        if isinstance(self.pipeline, dict):
            object.__setattr__(self, "pipeline", PipelineConfig(**self.pipeline))
        elif not isinstance(self.pipeline, PipelineConfig):
            raise TypeError(
                f"pipeline must be a dict or PipelineConfig, got "
                f"{type(self.pipeline).__name__}"
            )
