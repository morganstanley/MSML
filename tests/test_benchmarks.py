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
from alpha_lab.benchmarks.registry.store import connect_registry, ensure_schema
from alpha_lab.benchmarks.scripts.run_benchmarks import main as benchmark_run_main
from alpha_lab.benchmarks.scripts.summarize_workspace import _load_db_info
from alpha_lab.benchmarks.runners import LocalRunner
from alpha_lab.experiment_db import ExperimentDB


def test_database_generator_materializes_synthetic_benchmark(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite"
    workspace_root = tmp_path / "runs"

    data_dir = _seed_synthetic_registry(tmp_path, registry)
    generator = RegistryGenerator(
        registry=registry,
        workspace_root=workspace_root,
        benchmark_ids=["gp_regression_smoke"],
    )

    assert isinstance(generator, WorkspaceGenerator)
    workspace = next(iter(generator))()
    exit_code = LocalRunner(
        script=find_repo_root() / "run.py",
        prepare_only=True,
    ).run(workspace)

    config_path = workspace / ".alpha_lab" / "config.json"
    manifest_path = workspace / "benchmark_manifest.json"
    flattened_data = workspace / "data"
    config = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    assert exit_code == 0
    # Source data dir contents are symlinked into workspace/data/ directly
    # (not nested under workspace/data/data/).
    for entry in data_dir.iterdir():
        assert (flattened_data / entry.name).is_symlink()
    assert not (workspace / "benchmark_run.log").exists()
    assert config["data_path"] == str(flattened_data)
    assert config["pipeline"]["phases"] == ["phase1", "phase2", "phase3"]
    assert manifest["source"] == {
        "kind": "database",
        "registry": str(registry.resolve()),
        "benchmark_id": "gp_regression_smoke",
    }
    assert manifest["benchmark"]["id"] == "gp_regression_smoke"
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
        next(iter(generator))()


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
    (workspace / ".alpha_lab").mkdir(parents=True)
    (workspace / ".alpha_lab" / "config.json").write_text(
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
        next(iter(generator))()


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
    workspace = next(iter(generator))()
    exit_code = LocalRunner(
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
    workspaces = [factory() for factory in generator]
    workspace = workspaces[0]
    manifest = json.loads((workspace / "benchmark_manifest.json").read_text())
    config = json.loads((workspace / ".alpha_lab" / "config.json").read_text())

    assert workspaces == [suite_root / "seed_100", suite_root / "seed_101"]
    train_path = workspace / "data" / "train_data.npz"
    test_path = workspace / "private" / "test_data.npz"
    assert train_path.is_file()
    assert test_path.is_file()

    train = np.load(train_path)
    test = np.load(test_path)
    assert train["X"].ndim == 2
    assert train["y"].ndim == 1
    assert train["X"].shape[0] == train["y"].shape[0]
    assert test["X"].shape[1] == train["X"].shape[1]

    assert manifest["benchmark"]["id"] == "seed_100"
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
        script=find_repo_root() / "run.py",
        prepare_only=True,
    ).run_many(generator)

    workspace = tmp_path / "generated_suite" / "seed_200"
    manifest = json.loads((workspace / "benchmark_manifest.json").read_text())
    config = json.loads((workspace / ".alpha_lab" / "config.json").read_text())
    assert exit_codes == [0]
    assert manifest["run"]["status"] == "prepared"
    assert config["data_path"] == str((workspace / "data").resolve())


def test_run_main_resolves_imports_and_json_kwargs(tmp_path: Path) -> None:
    pytest.importorskip("tabicl")
    workspace_root = tmp_path / "workspaces"
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
                    "script": str(find_repo_root() / "run.py"),
                    "prepare_only": True,
                }
            ),
            "--workspace-root",
            str(workspace_root),
            "--overwrite",
            "--num-workers",
            "2",
        ]
    )

    first_workspace = workspace_root / "seed_42"
    run_manifest = json.loads((workspace_root / "run_manifest.json").read_text())
    assert exit_code == 0
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
    assert (first_workspace / ".alpha_lab" / "config.json").exists()
    assert (first_workspace / "data" / "train_data.npz").is_file()
    assert (first_workspace / "private" / "test_data.npz").is_file()


def test_run_main_accepts_database_generator(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite"
    workspace_root = tmp_path / "workspaces"
    _seed_synthetic_registry(tmp_path, registry)

    exit_code = benchmark_run_main(
        [
            "--generator",
            "alpha_lab.benchmarks.generators.database:RegistryGenerator",
            "--generator-kwargs",
            json.dumps(
                {
                    "registry": str(registry),
                    "benchmark_ids": ["gp_regression_smoke"],
                }
            ),
            "--agent-config",
            json.dumps({"model": "gpt-5.3-codex"}),
            "--runner",
            "alpha_lab.benchmarks.runners:LocalRunner",
            "--runner-kwargs",
            json.dumps(
                {
                    "script": str(find_repo_root() / "run.py"),
                    "prepare_only": True,
                }
            ),
            "--workspace-root",
            str(workspace_root),
            "--overwrite",
        ]
    )

    workspace = workspace_root / "gp_regression_smoke"
    config = json.loads((workspace / ".alpha_lab" / "config.json").read_text())
    manifest = json.loads((workspace / "benchmark_manifest.json").read_text())
    assert exit_code == 0
    assert config["model"] == "gpt-5.3-codex"
    assert manifest["source"]["kind"] == "database"


def test_load_suite_config_bundled() -> None:
    from alpha_lab.benchmarks.scripts.create_suite import _load_suite_config

    cfg = _load_suite_config("gp_blackbox/smoke_test")
    assert "generator" in cfg
    assert cfg["generator"].endswith("GPBlackboxGenerator")
    assert isinstance(cfg["generator_kwargs"], dict)
    assert isinstance(cfg["config_overrides"], dict)


def test_load_suite_config_external(tmp_path: Path) -> None:
    from alpha_lab.benchmarks.scripts.create_suite import _load_suite_config

    yaml_path = tmp_path / "external.yaml"
    yaml_path.write_text(
        "my_group:\n"
        "  generator: alpha_lab.benchmarks.generators.gp_blackbox:GPBlackboxGenerator\n"
        "  tiny:\n"
        "    generator_kwargs:\n"
        "      seed: 1\n"
        "      count: 1\n"
        "    config_overrides: {}\n"
    )
    cfg = _load_suite_config(f"{yaml_path}:my_group/tiny")
    assert cfg["generator"].endswith("GPBlackboxGenerator")
    assert cfg["generator_kwargs"] == {"seed": 1, "count": 1}
    assert cfg["config_overrides"] == {}


def test_load_suite_config_external_missing_file(tmp_path: Path) -> None:
    from alpha_lab.benchmarks.scripts.create_suite import _load_suite_config

    with pytest.raises(FileNotFoundError):
        _load_suite_config(f"{tmp_path / 'nope.yaml'}:any/key")


def test_load_suite_config_unknown_key() -> None:
    from alpha_lab.benchmarks.scripts.create_suite import _load_suite_config

    with pytest.raises(ValueError, match="Unknown suite"):
        _load_suite_config("gp_blackbox/does_not_exist")


def test_gp_blackbox_generator_bootstrap_end_to_end(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys
    from alpha_lab.benchmarks.generators import GPBlackboxGenerator

    suite_root = tmp_path / "suite"
    generator = GPBlackboxGenerator(
        workspace_root=suite_root,
        seed=7,
        count=1,
        n_dims=1,
        n_rff=32,
        eval_limit=2,
        noise_variance=0.01,
        rel_lengthscale=0.3,
        n_restarts=4,
        n_sobol=64,
    )
    [workspace] = [factory() for factory in generator]

    # Workspace layout
    assert (workspace / "harness" / "blackbox.py").is_file()
    assert (workspace / "private" / "objective.py").is_file()
    assert (workspace / "private" / "problem_state.json").is_file()
    assert (workspace / "public" / "problem_state.json").is_file()

    # Private state schema
    private = json.loads((workspace / "private" / "problem_state.json").read_text())
    assert set(private) == {
        "eval_limit", "history", "latent_parameters",
        "noise_generator", "noise_parameters",
    }
    assert private["history"] == []
    assert private["eval_limit"] == 2
    assert private["noise_generator"].endswith(":crn_noise")
    assert set(private["noise_parameters"]) == {"seed", "variance"}
    assert "noise_variance" not in private["latent_parameters"]

    # Public state schema (no f, no latent_parameters, no noise spec)
    public = json.loads((workspace / "public" / "problem_state.json").read_text())
    assert set(public) == {"n_dims", "eval_limit", "history"}
    assert public["history"] == []

    # End-to-end evaluation via subprocess (fresh interpreter binds WORKSPACE
    # from env at import time, mirroring the worker setup).
    script = (
        "import os, sys, json, numpy as np\n"
        f"sys.path.insert(0, {str(workspace)!r})\n"
        "from private import objective\n"
        "y1 = objective.evaluate(np.array([0.5]))\n"
        "y2 = objective.evaluate(np.array([0.5]))\n"
        "print(json.dumps({'y1': y1, 'y2': y2}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "ALPHALAB_WORKSPACE": str(workspace)},
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])

    # Re-query at same x produces different noisy y; private history has both
    # entries with identical f, distinct y.
    assert payload["y1"] != payload["y2"]
    private_after = json.loads((workspace / "private" / "problem_state.json").read_text())
    assert len(private_after["history"]) == 2
    assert private_after["history"][0]["f"] == private_after["history"][1]["f"]
    assert private_after["history"][0]["y"] != private_after["history"][1]["y"]

    # Public history has no f, mirrored x/y from private.
    public_after = json.loads((workspace / "public" / "problem_state.json").read_text())
    assert len(public_after["history"]) == 2
    assert all("f" not in h for h in public_after["history"])


class TestRank:
    """Tests for report._rank()."""

    def test_maximize(self) -> None:
        from alpha_lab.benchmarks.scripts.create_report import _rank

        assert _rank([0.5, 0.9, 0.1], "maximize") == [2, 1, 3]

    def test_minimize(self) -> None:
        from alpha_lab.benchmarks.scripts.create_report import _rank

        assert _rank([0.5, 0.9, 0.1], "minimize") == [2, 3, 1]

    def test_all_none(self) -> None:
        from alpha_lab.benchmarks.scripts.create_report import _rank

        ranks = _rank([None, None, None], "maximize")
        assert all(r == 4 for r in ranks)

    def test_mixed_none(self) -> None:
        from alpha_lab.benchmarks.scripts.create_report import _rank

        ranks = _rank([0.5, None, 0.9], "maximize")
        assert ranks == [2, 3, 1]

    def test_single_value(self) -> None:
        from alpha_lab.benchmarks.scripts.create_report import _rank

        assert _rank([0.5], "maximize") == [1]

    def test_single_none(self) -> None:
        from alpha_lab.benchmarks.scripts.create_report import _rank

        assert _rank([None], "maximize") == [2]


def _seed_synthetic_registry(
    tmp_path: Path,
    registry: Path,
    *,
    benchmark_id: str = "gp_regression_smoke",
) -> Path:
    """Materialize a tiny gp_regression workspace and register it.

    Returns the path to the synthetic data directory that the registry
    row references; tests can use it to compute expected symlink names.
    """
    from alpha_lab.benchmarks.generators import GPRegressionGenerator

    seed_root = tmp_path / "synthetic_seed"
    generator = GPRegressionGenerator(
        workspace_root=seed_root,
        seed=42,
        count=1,
        n_dims=1,
        n_train=8,
        n_test=8,
    )
    seed_workspace = next(iter(generator))()
    data_dir = (seed_workspace / "data").resolve()

    _insert_test_benchmark(
        registry,
        Benchmark(
            id=benchmark_id,
            name="GP regression smoke",
            data_path=data_dir,
            description="Synthetic GP-regression smoke benchmark.",
            target="Predict y from X.",
            domain="",
            provider="openai",
            model="gpt-5.2",
            reasoning_effort="low",
            shell_timeout=300,
            tool_output_max_chars=8000,
            pipeline={"phases": ["phase1", "phase2", "phase3"]},
            adapter_path=None,
            seed_path=None,
            notes="Synthetic test fixture (gp_regression).",
        ),
    )
    return data_dir


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


class TestLoadDbInfoMetricExtraction:
    """_load_db_info's own use of the shared find_metric_values matcher (see
    TestFindMetricValues in test_experiment_db.py for the matcher's own unit
    tests -- it now lives in alpha_lab.experiment_db, shared with
    leaderboard(), dispatcher.py, strategist.py, and output_generator.py)."""

    def test_picks_up_prefixed_and_nested_metrics_previously_silently_skipped(
        self, tmp_path: Path
    ) -> None:
        db = ExperimentDB(str(tmp_path / ".alpha_lab" / "experiments.db"))
        flat = db.create("flat", "d", "h", "{}")
        db.set_results(flat, json.dumps({"mse": 0.5}))
        prefixed = db.create("prefixed", "d", "h", "{}")
        db.set_results(prefixed, json.dumps({"val/mse": 0.05}))
        nested = db.create("nested", "d", "h", "{}")
        db.set_results(nested, json.dumps({"best_gp": {"loocv_mse": 0.01}}))

        info = _load_db_info(tmp_path, extract_key="mse", metric_direction="minimize", top_k=5)
        assert info["best_metric_value"] == 0.01
        assert info["best_experiment"] == "nested"
        assert {e["name"] for e in info["top_k_raw"]} == {"flat", "prefixed", "nested"}

    def test_direction_aware_selection_within_one_experiment(self, tmp_path: Path) -> None:
        """Minimize picks the lowest match, maximize the highest, among
        multiple candidates within the same match tier (here, two
        case-variant top-level exact keys)."""
        db = ExperimentDB(str(tmp_path / ".alpha_lab" / "experiments.db"))
        exp = db.create("multi", "d", "h", "{}")
        db.set_results(exp, json.dumps({"mse": 0.5, "MSE": 0.2}))

        info_min = _load_db_info(tmp_path, extract_key="mse", metric_direction="minimize", top_k=5)
        assert info_min["best_metric_value"] == 0.2

        info_max = _load_db_info(tmp_path, extract_key="mse", metric_direction="maximize", top_k=5)
        assert info_max["best_metric_value"] == 0.5

    def test_top_level_exact_wins_over_nested_exact(self, tmp_path: Path) -> None:
        """A one-level-nested exact match (e.g. an alternative validation
        split reported under a sub-object) must never outrank the
        experiment's own top-level exact score, regardless of direction --
        same priority rule as exact-over-variant, one level deeper.
        Real regression found on live data during PROJ-765 review."""
        db = ExperimentDB(str(tmp_path / ".alpha_lab" / "experiments.db"))
        exp = db.create("multi", "d", "h", "{}")
        db.set_results(exp, json.dumps({"mse": 0.5, "best_gp": {"mse": 0.2}}))

        info_min = _load_db_info(tmp_path, extract_key="mse", metric_direction="minimize", top_k=5)
        assert info_min["best_metric_value"] == 0.5

        info_max = _load_db_info(tmp_path, extract_key="mse", metric_direction="maximize", top_k=5)
        assert info_max["best_metric_value"] == 0.5

    def test_exact_match_used_even_when_variant_would_flatter_more(self, tmp_path: Path) -> None:
        """A same-tokened neighbor field (e.g. a baseline-to-beat) must never
        outrank the experiment's own exactly-named score, even when it would
        be more flattering for the metric's direction. Regression case from
        real on-prem data reviewed on #279."""
        db = ExperimentDB(str(tmp_path / ".alpha_lab" / "experiments.db"))
        exp = db.create("own_vs_baseline", "d", "h", "{}")
        db.set_results(exp, json.dumps({"rmse": 0.5, "baseline_rmse_to_beat": 0.3}))

        info = _load_db_info(tmp_path, extract_key="rmse", metric_direction="minimize", top_k=5)
        assert info["best_metric_value"] == 0.5
