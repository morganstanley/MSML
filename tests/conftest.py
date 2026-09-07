"""Shared fixtures for the alpha-lab test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Block tiktoken before any alpha_lab import so tests use the deterministic
# character-based fallback in alpha_lab.context. Avoids the Azure-blob BPE
# fetch (which hangs on locked-down networks) and removes the need for a
# checked-in BPE cache fixture.
sys.modules["tiktoken"] = None  # type: ignore[assignment]

# Neutralize litellm's GitHub pricing fetch. pytest-env in pyproject.toml
# also sets this; setdefault keeps it idempotent.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

# Run agents in-process by default: the suite drives AgentLoops with fake providers
# that can't cross a bwrap subprocess boundary. Sandbox-specific tests opt back in.
os.environ.setdefault("ALPHALAB_AGENT_NOSANDBOX", "1")

# Never emit token-usage metrics to Cortex during tests. Propagates to any
# subprocesses the integration tests spawn. token_metrics' own tests override
# this explicitly where they exercise the enabled path.
os.environ.setdefault("ALPHALAB_TOKEN_METRICS_DISABLED", "1")

import pytest  # noqa: E402

from alpha_lab import deps  # noqa: E402
from alpha_lab.adapter import DomainAdapter, ExperimentStructure, MetricConfig  # noqa: E402
from alpha_lab.config import Phase3Config, PipelineConfig, TaskConfig  # noqa: E402
from alpha_lab.constants import Phase  # noqa: E402
from alpha_lab.experiment_db import ExperimentDB  # noqa: E402


class _FakeExecutor:
    """No-op executor injected into the default RunDeps so tests need no real pool."""

    def total_slots(self) -> int:
        return 0

    def cleanup_all(self) -> None:
        pass


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "no_run_deps: do not publish the default RunDeps for this test"
    )


@pytest.fixture(autouse=True)
def published_run_deps(request: pytest.FixtureRequest, tmp_path: Path):
    """Publish a default RunDeps for each test so code that reads ``deps.get()`` works
    without per-test wiring.

    Tests that exercise the deps lifecycle itself opt out with
    ``@pytest.mark.no_run_deps``; tests needing specific executors or a particular
    memory workspace nest their own ``with RunDeps(...)``, which overrides this default.
    """
    if request.node.get_closest_marker("no_run_deps"):
        yield None
        return
    cfg = TaskConfig(
        data_path="d",
        description="x",
        pipeline=PipelineConfig(
            phases=[Phase.PHASE3], phase3=Phase3Config(gpu_ids=[], cpu_enabled=False)
        ),
    )
    with deps.RunDeps(
        cfg, run_id="test", workspace=tmp_path / ".run", _gpu_executor=_FakeExecutor()
    ) as rd:
        yield rd


@pytest.fixture()
def adapter() -> DomainAdapter:
    """A minimal DomainAdapter with default metric and experiment structure."""
    return DomainAdapter(
        metric=MetricConfig(),
        experiment=ExperimentStructure(
            required_files=["strategy.py", "run_experiment.py"],
        ),
    )


@pytest.fixture()
def tmp_workspace(tmp_path: Path) -> str:
    """Create a temporary workspace directory with basic structure."""
    ws = str(tmp_path / "workspace")
    os.makedirs(ws, exist_ok=True)
    return ws


@pytest.fixture(autouse=True)
def deterministic_memory_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid network embedding calls in tests with a deterministic, unit vector."""
    import numpy as np

    from alpha_lab.embeddings import EmbeddingStore

    def fake_embed(self: EmbeddingStore, text: str) -> np.ndarray:
        buckets = [0.0] * self.model.dim
        for word in (text or "").lower().replace("/", " ").replace("-", " ").split():
            buckets[sum(ord(ch) for ch in word) % len(buckets)] += 1.0
        vector = np.asarray(buckets, dtype=np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    monkeypatch.setattr(EmbeddingStore, "embed", fake_embed)


@pytest.fixture()
def db(tmp_path: Path) -> ExperimentDB:
    """Create an ExperimentDB with a fresh temporary database."""
    # The db lives under .alpha_lab/ (as in a real workspace), so the leaderboard's
    # sentinel lookup resolves the workspace as db_path.parent.parent.
    db_path = str(tmp_path / ".alpha_lab" / "test_experiments.db")
    return ExperimentDB(db_path)


@pytest.fixture()
def populated_db(db: ExperimentDB) -> ExperimentDB:
    """An ExperimentDB pre-populated with experiments in various states.

    Rows with `results_json` set also have the canonical-run sentinel created.
    """
    from alpha_lab.experiment_db import CANONICAL_RUN_COMPLETE_SENTINEL
    workspace = Path(db.db_path).parent.parent

    def _sentinel(name: str) -> None:
        d = workspace / "experiments" / name / "results"
        d.mkdir(parents=True, exist_ok=True)
        (d / CANONICAL_RUN_COMPLETE_SENTINEL).touch()

    # to_implement
    db.create("exp_xgboost_baseline", "XGBoost baseline", "Trees work", '{"model": "xgboost"}')
    # implemented
    db.create("exp_lstm_v1", "LSTM first try", "RNNs generalize", '{"model": "lstm"}')
    db.update_status(2, "implemented")
    # checked
    db.create("exp_tft_v1", "TFT model", "Attention helps", '{"model": "tft"}')
    db.update_status(3, "implemented")
    db.update_status(3, "checked")
    # running (with slurm job)
    db.create("exp_nbeats_v1", "N-BEATS", "Basis expansion", '{"model": "nbeats"}')
    db.update_status(4, "implemented")
    db.update_status(4, "checked")
    db.update_status(4, "queued")
    db.set_slurm_job(4, "12345")
    db.update_status(4, "running", started_at=1000.0)
    # finished (with results)
    db.create("exp_tcn_v1", "TCN model", "Dilated convolutions", '{"model": "tcn"}')
    db.update_status(5, "implemented")
    db.update_status(5, "checked")
    db.update_status(5, "queued")
    db.update_status(5, "running", started_at=1000.0)
    db.update_status(5, "finished", finished_at=2000.0)
    db.set_results(5, '{"sharpe": 1.5, "max_drawdown": -0.12, "mae": 0.03}')
    _sentinel("exp_tcn_v1")
    # analyzed
    db.create("exp_deepar_v1", "DeepAR", "Probabilistic", '{"model": "deepar"}')
    db.update_status(6, "implemented")
    db.update_status(6, "checked")
    db.update_status(6, "queued")
    db.update_status(6, "running", started_at=1000.0)
    db.update_status(6, "finished", finished_at=2000.0)
    db.set_results(6, '{"sharpe": 2.1, "max_drawdown": -0.08, "mae": 0.02}')
    _sentinel("exp_deepar_v1")
    db.update_status(6, "analyzed")
    # done
    db.create("exp_patchtst_v1", "PatchTST", "Patch attention", '{"model": "patchtst"}')
    db.update_status(7, "implemented")
    db.update_status(7, "checked")
    db.update_status(7, "queued")
    db.update_status(7, "running", started_at=1000.0)
    db.update_status(7, "finished", finished_at=2000.0)
    db.set_results(7, '{"sharpe": 0.8, "max_drawdown": -0.20, "mae": 0.05}')
    _sentinel("exp_patchtst_v1")
    db.update_status(7, "analyzed")
    db.update_status(7, "done")
    return db
