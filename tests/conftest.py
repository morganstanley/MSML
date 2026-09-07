"""Shared fixtures for the alpha-lab test suite."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from alpha_lab.experiment_db import ExperimentDB


@pytest.fixture()
def tmp_workspace(tmp_path: Path) -> str:
    """Create a temporary workspace directory with basic structure."""
    ws = str(tmp_path / "workspace")
    os.makedirs(ws, exist_ok=True)
    return ws


@pytest.fixture()
def db(tmp_path: Path) -> ExperimentDB:
    """Create an ExperimentDB with a fresh temporary database."""
    db_path = str(tmp_path / "test_experiments.db")
    return ExperimentDB(db_path)


@pytest.fixture()
def populated_db(db: ExperimentDB) -> ExperimentDB:
    """An ExperimentDB pre-populated with experiments in various states."""
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
    # analyzed
    db.create("exp_deepar_v1", "DeepAR", "Probabilistic", '{"model": "deepar"}')
    db.update_status(6, "implemented")
    db.update_status(6, "checked")
    db.update_status(6, "queued")
    db.update_status(6, "running", started_at=1000.0)
    db.update_status(6, "finished", finished_at=2000.0)
    db.set_results(6, '{"sharpe": 2.1, "max_drawdown": -0.08, "mae": 0.02}')
    db.update_status(6, "analyzed")
    # done
    db.create("exp_patchtst_v1", "PatchTST", "Patch attention", '{"model": "patchtst"}')
    db.update_status(7, "implemented")
    db.update_status(7, "checked")
    db.update_status(7, "queued")
    db.update_status(7, "running", started_at=1000.0)
    db.update_status(7, "finished", finished_at=2000.0)
    db.set_results(7, '{"sharpe": 0.8, "max_drawdown": -0.20, "mae": 0.05}')
    db.update_status(7, "analyzed")
    db.update_status(7, "done")
    return db
