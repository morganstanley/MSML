"""Base classes for benchmark workspace generators."""

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from alpha_lab.benchmarks.agents import AgentConfig
from alpha_lab.config import TaskConfig, load_config


@dataclass(frozen=True, kw_only=True)
class WorkspaceGenerator(ABC):
    """Iterable that bootstraps benchmark workspaces.

    Subclasses must implement ``__iter__`` to yield zero-arg factories that
    bootstrap each workspace on demand, plus a ``bootstrap(...)`` method
    that produces a single workspace and returns its path. The ``bootstrap``
    signature is per-subclass since the per-item payload (registry row,
    seed, etc.) differs by generator.
    """

    workspace_root: str | Path
    overwrite: bool = False
    agent_config: AgentConfig | None = None
    config_overrides: dict | None = None

    def validate(self, workspace: str | Path) -> None:
        """Sanity-check a materialized workspace.

        Verifies the directory exists, ``config.json`` resolves to a real
        ``data_path``, and ``benchmark_manifest.json`` is a JSON object.
        """
        workspace = Path(workspace)
        if not workspace.is_dir():
            raise NotADirectoryError(f"workspace is not a directory: {workspace}")

        config = load_config(workspace / "config.json")
        data_path = Path(config.resolve_data_path(workspace))
        if not data_path.exists():
            raise FileNotFoundError(f"config data_path not found: {data_path}")

        manifest_path = workspace / "benchmark_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"benchmark manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict):
            raise ValueError(f"benchmark manifest must be a JSON object: {manifest_path}")

    @abstractmethod
    def __iter__(self) -> Iterator[Path]:
        """Materialize workspaces lazily, yielding each as it is produced."""

    def _write_workspace_config(self, workspace: Path, data: dict) -> dict:
        """Apply overrides, build the ``TaskConfig``, and write ``config.json``.

        Mutates ``data`` in place by merging ``agent_config`` then
        ``config_overrides``. Returns the serialized config dict for downstream
        use (e.g. by the benchmark manifest).
        """
        _apply_overrides(data, self.agent_config)
        _deep_merge(data, self.config_overrides)
        config = asdict(TaskConfig(**data))
        (workspace / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        return config


def _apply_overrides(data: dict, overrides: AgentConfig | None) -> None:
    """Merge non-``None`` :class:`AgentConfig` fields into ``data``."""
    if overrides is None:
        return
    for key, value in asdict(overrides).items():
        if value is not None:
            data[key] = value


def _deep_merge(dst: dict, src: dict | None) -> None:
    """Deep-merge ``src`` into ``dst`` in place; fail loudly on type mismatch."""
    if not src:
        return
    for key, value in src.items():
        if isinstance(value, dict):
            existing = dst.get(key)
            if existing is None:
                dst[key] = {}
            elif not isinstance(existing, dict):
                raise TypeError(
                    f"_deep_merge: key {key!r} is {type(existing).__name__} "
                    f"in dst but dict in src"
                )
            _deep_merge(dst[key], value)
        else:
            dst[key] = value
