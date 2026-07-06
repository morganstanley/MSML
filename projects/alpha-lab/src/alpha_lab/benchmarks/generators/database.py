"""Registry-backed benchmark workspace generator."""

import json
import shutil
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

from alpha_lab.benchmarks.generators.base import WorkspaceGenerator
from alpha_lab.benchmarks.manifest import benchmark_snapshot, write_benchmark_manifest
from alpha_lab.benchmarks.registry.models import Benchmark
from alpha_lab.benchmarks.registry.store import connect_registry, load_benchmarks


@dataclass(frozen=True, kw_only=True)
class RegistryGenerator(WorkspaceGenerator):
    """Materialize benchmark workspaces from rows of a SQLite registry."""

    registry: str | Path
    benchmark_ids: list[str] | None = None

    def bootstrap(self, benchmark: Benchmark) -> Path:
        """Bootstrap a single registry row into a benchmark workspace."""
        self._validate_benchmark(benchmark)
        workspace = Path(self.workspace_root) / benchmark.id
        if workspace.exists():
            if not self.overwrite:
                raise FileExistsError(
                    f"Workspace already exists: {workspace}. "
                    "Use --overwrite to replace it."
                )
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True)

        data_dir = workspace / "data"
        data_dir.mkdir(parents=True)
        if benchmark.data_path.is_dir():
            # Flatten the source directory's contents directly under data/
            # so prompts and engines can always assume {workspace}/data/<file>.
            for entry in benchmark.data_path.iterdir():
                (data_dir / entry.name).symlink_to(
                    entry, target_is_directory=entry.is_dir()
                )
            data_dst = data_dir
        else:
            data_dst = data_dir / benchmark.data_path.name
            data_dst.symlink_to(benchmark.data_path)

        # Copy any ``workspace_includes`` declared in the source workspace's
        # config.json. The source is presumed to be ``data_path.parent``
        # when ``data_path`` is a directory inside a workspace (marker:
        # ``benchmark_manifest.json``). Each include must exist or we raise.
        source_workspace = benchmark.data_path.parent
        if (source_workspace / "benchmark_manifest.json").is_file():
            source_cfg_path = source_workspace / "config.json"
            if source_cfg_path.is_file():
                source_cfg = json.loads(source_cfg_path.read_text())
                for entry_name in source_cfg.get("workspace_includes", []) or []:
                    src = source_workspace / entry_name
                    if not src.exists():
                        raise FileNotFoundError(
                            f"{benchmark.id}: workspace_includes entry "
                            f"{entry_name!r} missing from source workspace "
                            f"{source_workspace}"
                        )
                    dst = workspace / entry_name
                    if src.is_dir():
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)

        if benchmark.adapter_path is not None:
            adapter_dst = workspace / "adapter"
            if adapter_dst.exists():
                shutil.rmtree(adapter_dst)
            shutil.copytree(benchmark.adapter_path, adapter_dst)

        if benchmark.seed_path is not None:
            self._copy_seed(benchmark.seed_path, workspace)

        # Propagate workspace_includes from source config so downstream
        # tooling (e.g. another bootstrap from this workspace) sees it.
        source_includes: list[str] = []
        if (source_workspace / "benchmark_manifest.json").is_file():
            src_cfg = source_workspace / "config.json"
            if src_cfg.is_file():
                source_includes = list(
                    json.loads(src_cfg.read_text()).get("workspace_includes", []) or []
                )

        data: dict = {
            "data_path": str(data_dst),
            "description": benchmark.description,
            "target": benchmark.target,
            "provider": benchmark.provider,
            "model": benchmark.model,
            "reasoning_effort": benchmark.reasoning_effort,
            "domain": benchmark.domain or None,
            "shell_timeout": benchmark.shell_timeout,
            "tool_output_max_chars": benchmark.tool_output_max_chars,
            "pipeline": asdict(benchmark.pipeline),
            "workspace_includes": source_includes,
        }
        config = self._write_workspace_config(workspace, data)

        write_benchmark_manifest(
            workspace,
            source={
                "kind": "database",
                "registry": str(Path(self.registry).resolve()),
                "benchmark_id": benchmark.id,
            },
            benchmark=benchmark_snapshot(benchmark),
            materialized={
                "data_path": str(data_dst),
                "data_source": str(benchmark.data_path),
                "data_is_symlink": True,
                "adapter_source": (
                    str(benchmark.adapter_path) if benchmark.adapter_path else None
                ),
                "seed_source": (
                    str(benchmark.seed_path) if benchmark.seed_path else None
                ),
            },
            config=config,
        )
        self.validate(workspace)
        return workspace

    def __iter__(self) -> Iterator[Callable[[], Path]]:
        registry = Path(self.registry).resolve()
        if not registry.exists():
            raise FileNotFoundError(f"Registry not found: {registry}")
        conn = connect_registry(registry)
        try:
            benchmarks = load_benchmarks(conn, self.benchmark_ids)
        finally:
            conn.close()
        for benchmark in benchmarks:
            yield partial(self.bootstrap, benchmark)

    @staticmethod
    def _validate_benchmark(benchmark: Benchmark) -> None:
        """Verify benchmark filesystem references are absolute and present."""
        if not benchmark.data_path.is_absolute():
            raise ValueError(f"{benchmark.id}: data_path must be absolute")
        if not benchmark.data_path.exists():
            raise FileNotFoundError(
                f"{benchmark.id}: data_path not found: {benchmark.data_path}"
            )
        if benchmark.adapter_path is not None:
            if not benchmark.adapter_path.is_absolute():
                raise ValueError(f"{benchmark.id}: adapter_path must be absolute")
            if not benchmark.adapter_path.is_dir():
                raise FileNotFoundError(
                    f"{benchmark.id}: adapter_path must be a directory: "
                    f"{benchmark.adapter_path}"
                )
        if benchmark.seed_path is not None:
            if not benchmark.seed_path.is_absolute():
                raise ValueError(f"{benchmark.id}: seed_path must be absolute")
            if not benchmark.seed_path.exists():
                raise FileNotFoundError(
                    f"{benchmark.id}: seed_path not found: {benchmark.seed_path}"
                )

    @staticmethod
    def _copy_seed(seed_path: Path, workspace: Path) -> None:
        """Copy a seed file or unpack a seed directory into the workspace root."""
        if seed_path.is_dir():
            for item in seed_path.iterdir():
                dst = workspace / item.name
                if item.is_dir():
                    shutil.copytree(item, dst)
                else:
                    shutil.copy2(item, dst)
        else:
            shutil.copy2(seed_path, workspace / seed_path.name)


