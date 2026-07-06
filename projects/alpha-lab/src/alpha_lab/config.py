"""Task configuration for alpha-lab."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Try yaml, fall back to json-only
yaml = None
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class Phase3Config:
    """Configuration for Phase 3 experiment orchestration."""

    # Executor type: "slurm" or "local"
    executor: str = "local"

    max_experiments: int = 50
    strategist_interval: int = 300  # seconds between strategist turns
    worker_count: int = 4
    report_interval: int = 10  # generate milestone report every N done experiments

    # SLURM settings (used when executor="slurm")
    max_concurrent_gpus: int = 8
    slurm_partitions: list[str] = field(default_factory=lambda: ["hpc-mid"])
    gpu_per_job: int = 1
    slurm_time_limit: str = "02:00:00"

    # Local GPU settings (used when executor="local")
    gpu_ids: list[int] | str = "auto"  # "auto" = detect, [] = CPU-only
    max_per_gpu: int = 1  # experiments per GPU (increase for packing)
    time_limit_seconds: int = 7200  # 2 hours default

    # CPU executor settings (for tree-based models)
    cpu_enabled: bool = True  # Run CPU experiments in parallel with GPU
    cpu_max_parallel: int = 4  # Max concurrent CPU experiments
    cpu_time_limit_seconds: int = 3600  # 1 hour default for CPU jobs

    # Python executable for experiment subprocesses
    # Falls back to ALPHALAB_PYTHON env var, then sys.executable
    python_executable: str = ""

    # When False, the dispatcher assigns a user-proxy handoff turn to each
    # ``analyzed`` experiment (closes out the lifecycle with directional
    # feedback at ``{workspace}/agenda.md``). Default off pending A/B results.
    no_handoff: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.gpu_ids, str) and self.gpu_ids != "auto":
            raise ValueError(
                f"gpu_ids must be 'auto' or a list of int GPU indices, "
                f"got {self.gpu_ids!r}"
            )
        if isinstance(self.gpu_ids, list) and not all(
            isinstance(i, int) and not isinstance(i, bool) for i in self.gpu_ids
        ):
            raise ValueError(
                f"gpu_ids list must contain only int GPU indices, "
                f"got {self.gpu_ids!r}"
            )
        if not self.python_executable:
            self.python_executable = os.environ.get("ALPHALAB_PYTHON", "")

    # Convergence detection
    convergence_threshold: int = 20  # Stop if no improvement for N experiments
    convergence_metric: str = ""  # Metric to track (empty = use adapter's primary_metric)

    # Ablation flags
    no_strategist: bool = False  # Replace strategist with random experiment proposals
    no_playbook: bool = False  # Disable playbook accumulation

    # JIT (just-in-time) proposals: make the strategist resource-aware — gate proposals
    # against free slots, capacity-driven trigger, fail-loud when idle. Off = batch behavior.
    jit: bool = False


@dataclass
class PipelineConfig:
    """Configuration for the multi-phase pipeline."""

    phases: list[str] = field(default_factory=lambda: ["phase1"])
    max_fix_iterations: int = 3
    phase3: Phase3Config = field(default_factory=Phase3Config)

    def __post_init__(self) -> None:
        """Coerce ``phase3`` from dict to :class:`Phase3Config`."""
        if isinstance(self.phase3, dict):
            self.phase3 = Phase3Config(**self.phase3)
        elif not isinstance(self.phase3, Phase3Config):
            raise TypeError(
                f"phase3 must be a dict or Phase3Config, got "
                f"{type(self.phase3).__name__}"
            )


@dataclass
class TaskConfig:
    """Configuration for an analysis task."""

    data_path: str
    description: str
    target: str = ""
    reasoning_effort: str = "low"
    model: str = "gpt-5.2"
    provider: str = "openai"  # currently only "openai" is supported
    domain: str | None = None
    """Adapter name (e.g. ``"tabular_regression"``) or absolute path to an
    adapter dir. ``None`` triggers the Phase 0 generation agent. Empty
    string is rejected as ambiguous."""
    workspace_includes: list[str] = field(default_factory=list)
    """Workspace-relative directory or file names (e.g. ``["private"]``)
    that the generator's ``bootstrap`` method and ``copy_workspace``
    must carry over alongside ``data/``. Entries must be plain relative
    names (no absolute paths or ``..`` traversal). Each entry must exist
    in the source workspace at bootstrap time, else bootstrap raises."""
    shell_timeout: int = 300  # seconds for shell_exec commands (agent can request less)
    tool_output_max_chars: int = 8000  # per-tool-result char cap applied in the agent loop
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    def __post_init__(self) -> None:
        if isinstance(self.domain, str) and self.domain == "":
            raise ValueError(
                "domain must be a non-empty adapter name/path or None "
                "(empty string is ambiguous; use None to trigger generation)."
            )
        for entry in self.workspace_includes:
            if not isinstance(entry, str) or not entry:
                raise ValueError(
                    f"workspace_includes entries must be non-empty strings, "
                    f"got {entry!r}"
                )
            p = Path(entry)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError(
                    f"workspace_includes entries must be plain relative "
                    f"paths (no absolute paths or '..'), got {entry!r}"
                )
        # tool_output_max_chars is user-settable via top-level config.json; reject
        # values that would make compact_tool_output silently misbehave. bool is a
        # subclass of int in Python, so exclude it explicitly.
        v = self.tool_output_max_chars
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(
                f"tool_output_max_chars must be an int, got "
                f"{type(v).__name__}={v!r}"
            )
        if v < 100:
            raise ValueError(
                f"tool_output_max_chars must be >= 100 so head+tail slicing "
                f"leaves room for content, got {v}"
            )
        if isinstance(self.pipeline, dict):
            self.pipeline = PipelineConfig(**self.pipeline)
        elif not isinstance(self.pipeline, PipelineConfig):
            raise TypeError(
                f"pipeline must be a dict or PipelineConfig, got "
                f"{type(self.pipeline).__name__}"
            )

    def resolve_data_path(self, base_dir: str | Path) -> str:
        """Resolve data_path relative to base_dir if not absolute."""
        p = Path(self.data_path)
        if not p.is_absolute():
            p = Path(base_dir) / p
        return str(p.resolve())


def load_config(path: str | Path) -> TaskConfig:
    """Load a TaskConfig from a YAML or JSON file.

    Required fields: data_path, description.
    Optional: target, reasoning_effort, model, provider, domain,
    shell_timeout (seconds; max wall-clock for shell_exec commands),
    tool_output_max_chars (per-tool-result char cap in agent loop; default 8000),
    pipeline (nested PipelineConfig/Phase3Config).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        content = f.read()

    # Try JSON first, then YAML
    if path.suffix == ".json" or content.strip().startswith("{"):
        raw: dict[str, Any] = json.loads(content)
    elif YAML_AVAILABLE:
        raw = yaml.safe_load(content)
    else:
        raise ImportError(
            "YAML config requires pyyaml. Either install it or use a .json config file."
        )

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a mapping, got {type(raw).__name__}")

    return task_config_from_mapping(raw)


def task_config_from_mapping(raw: dict[str, Any]) -> TaskConfig:
    """Build a TaskConfig from an already-parsed mapping.

    Shared by :func:`load_config` (file path) and callers that carry a serialized
    config (e.g. ``dataclasses.asdict(config)`` sent to a sandboxed agent), so the
    nested-pipeline coercion and field filtering live in one place.
    """
    # Validate required fields
    for key in ("data_path", "description"):
        if key not in raw:
            raise ValueError(f"Missing required config field: {key}")

    # Strip whitespace from string values
    cleaned: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, str):
            cleaned[k] = v.strip()
        else:
            cleaned[k] = v

    # Handle nested pipeline config
    if "pipeline" in cleaned and isinstance(cleaned["pipeline"], dict):
        pipeline_raw = dict(cleaned["pipeline"])
        # Handle nested phase3 config inside pipeline
        if "phase3" in pipeline_raw and isinstance(pipeline_raw["phase3"], dict):
            p3_known = {f.name for f in Phase3Config.__dataclass_fields__.values()}
            p3_data = {k: v for k, v in pipeline_raw["phase3"].items() if k in p3_known}
            pipeline_raw["phase3"] = Phase3Config(**p3_data)
        pipeline_known = {f.name for f in PipelineConfig.__dataclass_fields__.values()}
        pipeline_data = {k: v for k, v in pipeline_raw.items() if k in pipeline_known}
        cleaned["pipeline"] = PipelineConfig(**pipeline_data)

    # Only pass known fields to TaskConfig
    known_fields = {f.name for f in TaskConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in cleaned.items() if k in known_fields}

    return TaskConfig(**filtered)


def split_frontmatter_from_config_body(text: str) -> tuple[str, str]:
    if not text.startswith("---\n"):
        raise ValueError("missing YAML frontmatter opening")
    if (end := text.find("\n---\n", 4)) == -1:
        raise ValueError("missing YAML frontmatter closing")
    return text[4:end], text[end + len("\n---\n") :]
