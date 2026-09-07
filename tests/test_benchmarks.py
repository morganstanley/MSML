"""Tests for benchmark workspace generators and runners."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from alpha_lab.benchmarks.agents import AgentConfig
from alpha_lab.benchmarks.generators import (
    RegistryGenerator,
    WorkspaceGenerator,
)
from alpha_lab.benchmarks.paths import find_repo_root
from alpha_lab.benchmarks.registry.models import Benchmark
from alpha_lab.benchmarks.registry.seed import initialize_default_registry
from alpha_lab.benchmarks.registry.store import connect_registry, ensure_schema
from alpha_lab.benchmarks.run_benchmarks import main as benchmark_run_main
from alpha_lab.benchmarks.runners import LocalRunner


def test_initialize_default_registry_uses_absolute_paths(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite"

    initialize_default_registry(registry)

    conn = sqlite3.connect(registry)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT id, data_path FROM benchmarks").fetchall()
        table_names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    finally:
        conn.close()

    assert {row["id"] for row in rows} == {
        "demo_exchange",
        "llm_speedrun",
        "traffic",
    }
    assert all(Path(row["data_path"]).is_absolute() for row in rows)
    assert not {"benchmark_runs", "benchmark_experiment_summaries"} & table_names


def test_database_generator_materializes_demo_exchange(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite"
    workspace_root = tmp_path / "runs"

    initialize_default_registry(registry)
    generator = RegistryGenerator(
        registry=registry,
        workspace_root=workspace_root,
        benchmark_ids=["demo_exchange"],
    )

    assert isinstance(generator, WorkspaceGenerator)
    workspace = next(iter(generator))
    output_root = tmp_path / "results"
    exit_code = LocalRunner(
        output_root=output_root,
        script=find_repo_root() / "run.py",
        prepare_only=True,
    ).run(workspace)

    config_path = workspace / "config.json"
    manifest_path = workspace / "benchmark_manifest.json"
    data_path = workspace / "data" / "exchange_rates.csv"
    persisted = output_root / "demo_exchange"
    config = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    assert exit_code == 0
    assert data_path.is_symlink()
    assert (persisted / "data" / "exchange_rates.csv").is_symlink()
    assert not (workspace / "benchmark_run.log").exists()
    assert config["data_path"] == str(data_path)
    assert config["pipeline"]["phases"] == ["phase1", "phase2", "phase3"]
    assert manifest["source"] == {
        "kind": "database",
        "registry": str(registry.resolve()),
        "benchmark_id": "demo_exchange",
    }
    assert manifest["benchmark"]["id"] == "demo_exchange"
    assert manifest["materialized"]["data_is_symlink"] is True
    assert manifest["run"]["status"] == "prepared"


def test_database_generator_validates_missing_data_path(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite"
    _insert_test_benchmark(
        registry,
        Benchmark(
            id="missing",
            name="Missing data",
            data_path=tmp_path / "missing.csv",
            description="D",
            target="",
            domain="",
            provider="openai",
            model="gpt-5.2",
            reasoning_effort="low",
            shell_timeout=300,
            tool_output_max_chars=8000,
            pipeline={"phases": ["phase1"]},
            adapter_path=None,
            seed_path=None,
            notes="",
        ),
    )

    generator = RegistryGenerator(
        registry=registry,
        workspace_root=tmp_path / "runs",
        benchmark_ids=["missing"],
    )

    with pytest.raises(FileNotFoundError):
        next(iter(generator))


def test_workspace_generator_validates_workspace_root(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace-root"
    workspace_root.write_text("")
    generator = RegistryGenerator(
        registry=tmp_path / "missing.sqlite",
        workspace_root=workspace_root,
    )

    with pytest.raises(NotADirectoryError):
        generator.validate(workspace_root)


def test_workspace_generator_validates_config_data_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "config.json").write_text(
        json.dumps({"data_path": "missing.csv", "description": "D"}) + "\n"
    )
    (workspace / "benchmark_manifest.json").write_text("{}\n")
    generator = RegistryGenerator(
        registry=tmp_path / "missing.sqlite",
        workspace_root=tmp_path / "runs",
    )

    with pytest.raises(FileNotFoundError, match="config data_path"):
        generator.validate(workspace)


def test_database_generator_validates_task_config(tmp_path: Path) -> None:
    data_path = tmp_path / "data.csv"
    data_path.write_text("x,target\n1,0\n")
    registry = tmp_path / "registry.sqlite"
    _insert_test_benchmark(
        registry,
        Benchmark(
            id="bad_config",
            name="Bad config",
            data_path=data_path.resolve(),
            description="D",
            target="",
            domain="",
            provider="openai",
            model="gpt-5.2",
            reasoning_effort="low",
            shell_timeout=300,
            tool_output_max_chars=50,
            pipeline={"phases": ["phase1"]},
            adapter_path=None,
            seed_path=None,
            notes="",
        ),
    )
    generator = RegistryGenerator(
        registry=registry,
        workspace_root=tmp_path / "runs",
    )

    with pytest.raises(ValueError, match="tool_output_max_chars"):
        next(iter(generator))


def test_generated_data_can_feed_database_generator(tmp_path: Path) -> None:
    data_path = tmp_path / "synthetic.csv"
    data_path.write_text("x,target\n1,0\n2,1\n")
    registry = tmp_path / "registry.sqlite"
    _insert_test_benchmark(
        registry,
        Benchmark(
            id="tabular_smoke",
            name="Tabular smoke",
            data_path=data_path.resolve(),
            description="Synthetic tabular smoke dataset.",
            target="Predict target.",
            domain="tabular",
            provider="openai",
            model="gpt-5.2",
            reasoning_effort="low",
            shell_timeout=300,
            tool_output_max_chars=8000,
            pipeline={"phases": ["phase1"]},
            adapter_path=None,
            seed_path=None,
            notes="Generated test dataset.",
        ),
    )

    generator = RegistryGenerator(
        registry=registry,
        workspace_root=tmp_path / "runs",
        benchmark_ids=["tabular_smoke"],
    )
    workspace = next(iter(generator))
    exit_code = LocalRunner(
        output_root=tmp_path / "results",
        script=find_repo_root() / "run.py",
        prepare_only=True,
    ).run(workspace)

    assert exit_code == 0
    assert (workspace / "data" / "synthetic.csv").is_symlink()


def test_tabular_workspace_generator_yields_ready_workspaces(tmp_path: Path) -> None:
    pytest.importorskip("tabicl")
    from alpha_lab.benchmarks.generators import StructuralCausalGenerator
    suite_root = tmp_path / "suite"
    generator = StructuralCausalGenerator(
        workspace_root=suite_root,
        agent_config=AgentConfig(
            provider="openai",
            model="gpt-5.2",
            reasoning_effort="low",
            shell_timeout=300,
            tool_output_max_chars=8000,
        ),
        seed=100,
        count=2,
        max_features=4,
        max_classes=2,
        max_seq_len=64,
    )

    assert isinstance(generator, WorkspaceGenerator)
    workspaces = list(generator)
    workspace = workspaces[0]
    manifest = json.loads((workspace / "benchmark_manifest.json").read_text())
    config = json.loads((workspace / "config.json").read_text())

    assert workspaces == [suite_root / "tabular_100", suite_root / "tabular_101"]
    train_path = workspace / "data" / "train_data.npz"
    test_path = workspace / "data" / "test_data.npz"
    assert train_path.is_file()
    assert test_path.is_file()

    train = np.load(train_path)
    test = np.load(test_path)
    assert train["X"].ndim == 2
    assert train["y"].ndim == 1
    assert train["X"].shape[0] == train["y"].shape[0]
    assert test["X"].shape[1] == train["X"].shape[1]

    assert manifest["benchmark"]["id"] == "tabular_100"
    assert manifest["source"]["kind"] == "generator"
    assert manifest["source"]["seed"] == 100
    assert manifest["materialized"]["data_is_symlink"] is False
    assert config["provider"] == "openai"
    assert config["model"] == "gpt-5.2"


def test_local_runner_prepare_only_uses_full_workspace(tmp_path: Path) -> None:
    pytest.importorskip("tabicl")
    from alpha_lab.benchmarks.generators import StructuralCausalGenerator
    generator = StructuralCausalGenerator(
        workspace_root=tmp_path / "generated_suite",
        agent_config=AgentConfig(provider="openai", model="gpt-5.2"),
        seed=200,
        count=1,
        max_features=4,
        max_classes=2,
        max_seq_len=64,
    )

    exit_codes = LocalRunner(
        output_root=tmp_path / "results",
        script=find_repo_root() / "run.py",
        prepare_only=True,
    ).run_many(generator)

    workspace = tmp_path / "generated_suite" / "tabular_200"
    persisted = tmp_path / "results" / "tabular_200"
    manifest = json.loads((persisted / "benchmark_manifest.json").read_text())
    config = json.loads((persisted / "config.json").read_text())
    assert exit_codes == [0]
    assert manifest["run"]["status"] == "prepared"
    assert config["data_path"] == str((persisted / "data").resolve())


def test_run_main_resolves_imports_and_json_kwargs(tmp_path: Path) -> None:
    pytest.importorskip("tabicl")
    output_root = tmp_path / "results"
    workspace_parent = tmp_path / "tmp-workspaces"
    generator_kwargs = {"seed": 42, "count": 2, "max_features": 4, "max_classes": 2, "max_seq_len": 64}
    exit_code = benchmark_run_main(
        [
            "--generator",
            "alpha_lab.benchmarks.generators.structural_causal:StructuralCausalGenerator",
            "--generator-kwargs",
            json.dumps(generator_kwargs),
            "--agent-config",
            json.dumps({"provider": "openai", "model": "gpt-5.2"}),
            "--runner",
            "alpha_lab.benchmarks.runners:LocalRunner",
            "--runner-kwargs",
            json.dumps(
                {
                    "output_root": str(output_root),
                    "script": str(find_repo_root() / "run.py"),
                    "prepare_only": True,
                }
            ),
            "--workspace-root",
            str(workspace_parent),
            "--overwrite",
            "--num-workers",
            "2",
        ]
    )

    first_workspace = output_root / "tabular_42"
    run_manifest = json.loads((output_root / "run_manifest.json").read_text())
    assert exit_code == 0
    assert list(workspace_parent.iterdir()) == []
    assert run_manifest["generator"]["import"].endswith(":StructuralCausalGenerator")
    assert run_manifest["generator"]["kwargs"] == generator_kwargs
    assert run_manifest["runner"]["import"].endswith(":LocalRunner")
    assert run_manifest["agent"]["config"] == {
        "provider": "openai",
        "model": "gpt-5.2",
        "reasoning_effort": None,
        "shell_timeout": None,
        "tool_output_max_chars": None,
    }
    assert run_manifest["num_workers"] == 2
    assert (first_workspace / "config.json").exists()
    assert (first_workspace / "data" / "train_data.npz").is_file()
    assert (first_workspace / "data" / "test_data.npz").is_file()


def test_run_main_accepts_database_generator(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite"
    output_root = tmp_path / "db-results"
    workspace_parent = tmp_path / "db-tmp"
    initialize_default_registry(registry)

    exit_code = benchmark_run_main(
        [
            "--generator",
            "alpha_lab.benchmarks.generators.database:RegistryGenerator",
            "--generator-kwargs",
            json.dumps(
                {
                    "registry": str(registry),
                    "benchmark_ids": ["demo_exchange"],
                }
            ),
            "--agent-config",
            json.dumps({"model": "gpt-5.3-codex"}),
            "--runner",
            "alpha_lab.benchmarks.runners:LocalRunner",
            "--runner-kwargs",
            json.dumps(
                {
                    "output_root": str(output_root),
                    "script": str(find_repo_root() / "run.py"),
                    "prepare_only": True,
                }
            ),
            "--workspace-root",
            str(workspace_parent),
            "--overwrite",
        ]
    )

    workspace = output_root / "demo_exchange"
    config = json.loads((workspace / "config.json").read_text())
    manifest = json.loads((workspace / "benchmark_manifest.json").read_text())
    assert exit_code == 0
    assert list(workspace_parent.iterdir()) == []
    assert config["model"] == "gpt-5.3-codex"
    assert manifest["source"]["kind"] == "database"


def _insert_test_benchmark(registry: Path, benchmark: Benchmark) -> None:
    conn = connect_registry(registry)
    try:
        ensure_schema(conn)
        conn.execute(
            """
            INSERT INTO benchmarks (
                id, name, data_path, description, target, domain,
                provider, model, reasoning_effort, shell_timeout,
                tool_output_max_chars, pipeline_json, adapter_path, seed_path,
                enabled, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                benchmark.id,
                benchmark.name,
                str(benchmark.data_path),
                benchmark.description,
                benchmark.target,
                benchmark.domain,
                benchmark.provider,
                benchmark.model,
                benchmark.reasoning_effort,
                benchmark.shell_timeout,
                benchmark.tool_output_max_chars,
                json.dumps(asdict(benchmark.pipeline)),
                str(benchmark.adapter_path) if benchmark.adapter_path else None,
                str(benchmark.seed_path) if benchmark.seed_path else None,
                benchmark.notes,
            ),
        )
        conn.commit()
    finally:
        conn.close()
