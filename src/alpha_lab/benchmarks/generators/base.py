"""Base classes for benchmark workspace generators."""

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from alpha_lab.benchmarks.agents import AgentConfig
from alpha_lab.config import TaskConfig, config_to_jsonable, load_config
from alpha_lab.utils import atomic_write, deep_merge


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

        config = load_config(workspace / ".alpha_lab" / "config.json")
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
        deep_merge(data, self.config_overrides)
        # Store data_path absolute so the run's canonical config resolves it regardless
        # of where config.json is read from (it lives under .alpha_lab/, not the root).
        if data.get("data_path") and not Path(data["data_path"]).is_absolute():
            data["data_path"] = str((workspace / data["data_path"]).resolve())
        config = config_to_jsonable(TaskConfig(**data))
        config_path = workspace / ".alpha_lab" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(config_path, json.dumps(config, indent=2) + "\n")
        return config


def _apply_overrides(data: dict, overrides: AgentConfig | None) -> None:
    """Merge non-``None`` :class:`AgentConfig` fields into ``data``."""
    if overrides is None:
        return
    for key, value in asdict(overrides).items():
        if value is not None:
            data[key] = value


