"""Task configuration for alpha-lab."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Try yaml, fall back to json-only
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
    slurm_partitions: list[str] = field(default_factory=lambda: ["gpu"])
    gpu_per_job: int = 1
    slurm_time_limit: str = "02:00:00"

    # Local GPU settings (used when executor="local")
    gpu_ids: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    max_per_gpu: int = 1  # experiments per GPU (increase for packing)
    time_limit_seconds: int = 7200  # 2 hours default

    # CPU executor settings (for tree-based models)
    cpu_enabled: bool = True  # Run CPU experiments in parallel with GPU
    cpu_max_parallel: int = 4  # Max concurrent CPU experiments
    cpu_time_limit_seconds: int = 3600  # 1 hour default for CPU jobs

    # Python executable for experiment subprocesses
    # Falls back to ALPHALAB_PYTHON env var, then sys.executable
    python_executable: str = ""

    def __post_init__(self) -> None:
        if not self.python_executable:
            self.python_executable = os.environ.get("ALPHALAB_PYTHON", "")

    # Convergence detection
    convergence_threshold: int = 20  # Stop if no improvement for N experiments
    convergence_metric: str = ""  # Metric to track (empty = use adapter's primary_metric)

    # Ablation flags
    no_strategist: bool = False  # Replace strategist with random experiment proposals
    no_playbook: bool = False  # Disable playbook accumulation


@dataclass
class PipelineConfig:
    """Configuration for the multi-phase pipeline."""

    phases: list[str] = field(default_factory=lambda: ["phase1"])
    max_fix_iterations: int = 3
    phase3: Phase3Config = field(default_factory=Phase3Config)


@dataclass
class TaskConfig:
    """Configuration for an analysis task."""

    data_path: str
    description: str
    target: str = ""
    reasoning_effort: str = "low"
    model: str = "gpt-5.2"
    provider: str = "openai"  # "openai" or "anthropic"
    domain: str = ""  # "time_series", "cuda_kernel", "nanogpt", or free-text for Phase 0
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    def resolve_data_path(self, base_dir: str | Path) -> str:
        """Resolve data_path relative to base_dir if not absolute."""
        p = Path(self.data_path)
        if not p.is_absolute():
            p = Path(base_dir) / p
        return str(p.resolve())


def load_config(path: str | Path) -> TaskConfig:
    """Load a TaskConfig from a YAML or JSON file.

    Required fields: data_path, description.
    Optional: target, reasoning_effort, model.
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
