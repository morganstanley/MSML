"""Manifest helpers for benchmark runs and workspaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alpha_lab.benchmarks.paths import git_commit
from alpha_lab.benchmarks.registry.models import Benchmark


BENCHMARK_MANIFEST_NAME = "benchmark_manifest.json"
RUN_MANIFEST_NAME = "run_manifest.json"


def benchmark_snapshot(benchmark: Benchmark) -> dict[str, Any]:
    """Serialize the registry-only fields of a :class:`Benchmark` row.

    TaskConfig fields (description, target, domain, provider, model, etc.) are
    not duplicated here — they live in ``config.json`` next to the manifest.
    """
    return {
        "id": benchmark.id,
        "name": benchmark.name,
        "adapter_path": str(benchmark.adapter_path) if benchmark.adapter_path else None,
        "seed_path": str(benchmark.seed_path) if benchmark.seed_path else None,
        "notes": benchmark.notes,
    }


def write_benchmark_manifest(
    workspace: Path,
    *,
    source: dict[str, Any],
    benchmark: dict[str, Any],
    materialized: dict[str, Any],
    config: dict[str, Any],
    generator_info: dict[str, Any] | None = None,
) -> None:
    """Write the benchmark manifest for a materialized workspace.

    Args:
        workspace: Workspace directory.
        source: Provenance descriptor (e.g., registry row or generator profile).
        benchmark: Snapshot of the benchmark fields stored in the manifest.
        materialized: Filesystem layout descriptor (data/adapter/seed sources).
        config: Serialized :class:`alpha_lab.config.TaskConfig`.
        generator_info: Optional generator-specific block (used by synthetic generators).
    """
    manifest: dict[str, Any] = {
        "git_commit": git_commit(),
        "source": source,
        "benchmark": benchmark,
        "materialized": materialized,
        "run": {},
        "config": config,
    }
    if generator_info is not None:
        manifest["generator"] = generator_info
    (workspace / BENCHMARK_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def update_manifest_run(workspace: Path, run: dict[str, Any]) -> None:
    """Update the ``run`` block of an existing benchmark manifest in place."""
    path = workspace / BENCHMARK_MANIFEST_NAME
    data = json.loads(path.read_text())
    data["run"] = run
    path.write_text(json.dumps(data, indent=2) + "\n")
