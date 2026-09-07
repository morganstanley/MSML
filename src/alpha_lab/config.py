"""Task configuration for alpha-lab."""

from __future__ import annotations

import json
import os
from dataclasses import field, replace
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, TypeAdapter
from pydantic.dataclasses import dataclass

from alpha_lab.constants import Phase
from alpha_lab.git import GitSpec
from alpha_lab.providers.litellm_proxy import get_available_model_tags, validate_model_tags

MEMORY_REPO_GITIGNORE = ("index.db", "embeddings/")

# All config dataclasses reject unknown fields loudly rather than silently dropping them.
_STRICT = ConfigDict(extra="forbid")
DEFAULT_MODEL = "gpt-5.2"

# Try yaml, fall back to json-only
yaml = None
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass(config=_STRICT)
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

    # Handoff is on by default (``handoff=True``): the dispatcher gives each
    # ``analyzed`` experiment a user-proxy handoff turn, closing out the lifecycle
    # with directional feedback at ``{workspace}/agenda.md``. Set ``handoff=False``
    # to skip it.
    handoff: bool = True

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
    convergence_threshold: int = 20  # Log-only: N experiments without improvement (never ends the run)
    convergence_metric: str = ""  # Metric to track (empty = use adapter's primary_metric)

    # Ablation flags
    no_strategist: bool = False  # Replace strategist with random experiment proposals
    no_playbook: bool = False  # Disable playbook accumulation
    # Remove complete_research from the strategist: no explicit completion
    # decisions; Phase 3 then runs to max_experiments (convergence_threshold
    # logs only).
    # This restores the pre-completion failure mode (runs that never conclude
    # cleanly) — see docs/02_configuration.md before enabling.
    no_strategist_completion: bool = False

    # JIT (just-in-time) proposals: make the strategist resource-aware — gate proposals
    # against free slots, capacity-driven trigger, fail-loud when idle. Off = batch behavior.
    jit: bool = True


@dataclass(config=_STRICT)
class PipelineConfig:
    """Configuration for the multi-phase pipeline."""

    phases: list[Phase] = field(default_factory=lambda: [Phase.PHASE1])
    max_fix_iterations: int = 3
    phase3: Phase3Config = field(default_factory=Phase3Config)


@dataclass(config=_STRICT)
class TaskConfig:
    """Configuration for an analysis task."""

    data_path: str
    description: str
    target: str = ""
    reasoning_effort: str = "low"
    model: str | None = None
    provider: str = "openai"  # "openai", "anthropic", "grok", "bedrock", or "local"
    model_tags: list[str | list[str]] = field(default_factory=list)
    """LiteLLM-proxy tag clauses (``local`` provider only). When set, the proxy is
    queried lazily (first request; cached pool) and one matching deployment is
    selected per request instead of using ``model`` directly. A bare string is a
    one-tag clause; a nested list is an AND of tags; clauses are OR'd."""
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
    memory_spec: GitSpec = field(default_factory=GitSpec)
    """Optional local memory path or git clone source. A mapping is coerced to
    a :class:`GitSpec` by pydantic."""
    shell_timeout: int = 300  # seconds for shell_exec commands (agent can request less)
    tool_output_max_chars: int = 8000  # per-tool-result char cap applied in the agent loop
    web_search_model: str = "gpt-4.1-mini"  # model for the web_search proxy (non-OpenAI providers)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


    def validate_local_model_configs(self) -> None:
        if self.provider != "local":
            if not self.model_tags:
                return

            raise ValueError(
                f"model_tags is only supported with provider 'local', "
                f"got provider {self.provider!r}"
            )

        # TODO: Consider allowing model override when tags are available.
        if not (bool(self.model) ^ bool(self.model_tags)):
            raise ValueError(
                "Exactly one of 'model' or 'model_tags' must be set for provider 'local'."
            )

        local_base_url = os.getenv("LOCAL_BASE_URL")
        if not local_base_url:
            raise EnvironmentError(
                "LOCAL_BASE_URL environment variable is not set."
            )

        if self.model_tags:
            available_model_tags: set = get_available_model_tags(local_base_url)
            validate_model_tags(available_model_tags, self.model_tags)


    def __post_init__(self) -> None:
        if isinstance(self.domain, str) and self.domain == "":
            raise ValueError(
                "domain must be a non-empty adapter name/path or None "
                "(empty string is ambiguous; use None to trigger generation)."
            )

        self.validate_local_model_configs()

        # Non-local providers need a concrete model for the SDK call; fall back to
        # the default when unset. 'local' intentionally keeps model=None (model_tags
        # selects the model per request; a null model there drives tag-based
        # observability).
        if not self.model and self.provider != "local":
            self.model = DEFAULT_MODEL
        for entry in self.workspace_includes:
            if not entry:
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
        # pydantic guarantees an int here; enforce the lower bound that keeps
        # compact_tool_output's head+tail slicing meaningful.
        if self.tool_output_max_chars < 100:
            raise ValueError(
                f"tool_output_max_chars must be >= 100 so head+tail slicing "
                f"leaves room for content, got {self.tool_output_max_chars}"
            )

        existing_ignores = {str(item) for item in self.memory_spec.gitignore}
        missing_ignores = tuple(
            item for item in MEMORY_REPO_GITIGNORE if item not in existing_ignores
        )
        if missing_ignores:
            self.memory_spec = replace(
                self.memory_spec,
                gitignore=(*self.memory_spec.gitignore, *missing_ignores),
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
    config (e.g. ``dataclasses.asdict(config)`` sent to a sandboxed agent). Nested
    coercion (``pipeline``/``phase3``/``memory_spec``) and unknown-field rejection
    are handled by pydantic when the ``TaskConfig`` is constructed.
    """
    # Validate required fields up front for a friendlier message than pydantic's.
    for key in ("data_path", "description"):
        if key not in raw:
            raise ValueError(f"Missing required config field: {key}")

    # Strip whitespace from top-level string values.
    cleaned = {k: v.strip() if isinstance(v, str) else v for k, v in raw.items()}

    return TaskConfig(**cleaned)


_TASK_CONFIG_ADAPTER = TypeAdapter(TaskConfig)


def config_to_jsonable(config: TaskConfig) -> dict[str, Any]:
    """Serialize a ``TaskConfig`` to a JSON-safe dict.

    Uses pydantic's JSON-mode dump so non-JSON field types are normalized
    (e.g. ``Path`` under ``memory_spec.remote.src`` → ``str``, ``Phase`` →
    its value). Prefer this over ``json.dumps(dataclasses.asdict(config))``,
    which raises ``TypeError`` on those values.
    """
    return _TASK_CONFIG_ADAPTER.dump_python(config, mode="json")


def split_frontmatter_from_config_body(text: str) -> tuple[str, str]:
    if not text.startswith("---\n"):
        raise ValueError("missing YAML frontmatter opening")
    if (end := text.find("\n---\n", 4)) == -1:
        raise ValueError("missing YAML frontmatter closing")
    return text[4:end], text[end + len("\n---\n") :]
