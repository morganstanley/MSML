"""Registry-backed benchmark workspace generator."""

import shutil
from collections.abc import Iterator
from dataclasses import asdict, dataclass
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

    def materialize(self, benchmark: Benchmark) -> Path:
        """Materialize a single registry row into a benchmark workspace."""
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

        data_dst = workspace / "data" / benchmark.data_path.name
        data_dst.parent.mkdir(parents=True)
        data_dst.symlink_to(
            benchmark.data_path,
            target_is_directory=benchmark.data_path.is_dir(),
        )

        if benchmark.adapter_path is not None:
            shutil.copytree(benchmark.adapter_path, workspace / "adapter")

        if benchmark.seed_path is not None:
            self._copy_seed(benchmark.seed_path, workspace)

        data: dict = {
            "data_path": str(data_dst),
            "description": benchmark.description,
            "target": benchmark.target,
            "provider": benchmark.provider,
            "model": benchmark.model,
            "reasoning_effort": benchmark.reasoning_effort,
            "domain": benchmark.domain,
            "shell_timeout": benchmark.shell_timeout,
            "tool_output_max_chars": benchmark.tool_output_max_chars,
            "pipeline": asdict(benchmark.pipeline),
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
                "data_is_symlink": data_dst.is_symlink(),
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

    def __iter__(self) -> Iterator[Path]:
        registry = Path(self.registry).resolve()
        if not registry.exists():
            raise FileNotFoundError(f"Registry not found: {registry}")
        conn = connect_registry(registry)
        try:
            benchmarks = load_benchmarks(conn, self.benchmark_ids)
        finally:
            conn.close()
        for benchmark in benchmarks:
            yield self.materialize(benchmark)

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


