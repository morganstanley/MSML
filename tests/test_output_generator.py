"""Tests for OutputGenerator's live status report."""

from __future__ import annotations

from pathlib import Path

from alpha_lab.adapter import DomainAdapter, ExperimentStructure, MetricConfig
from alpha_lab.experiment_db import ExperimentDB
from alpha_lab.output_generator import OutputGenerator


def _generator(tmp_path: Path, adapter: DomainAdapter) -> OutputGenerator:
    return OutputGenerator(workspace=tmp_path, adapter=adapter)


def _db(tmp_path: Path) -> ExperimentDB:
    return ExperimentDB(str(tmp_path / ".alpha_lab" / "experiments.db"))


class TestGenerateStatusReportTopModels:
    """PROJ-765: generate_status_report's "top models by primary metric"
    section had the same flat-key bug -- a variant-spelled metric was
    silently excluded from the live status report/dashboard entirely."""

    def test_variant_key_spelling_included(self, tmp_path: Path) -> None:
        adapter = DomainAdapter(
            metric=MetricConfig(primary_metric="mse", direction="minimize"),
            experiment=ExperimentStructure(required_files=[]),
        )
        db = _db(tmp_path)
        db.set_results(db.create("gp_run", "D", "H", "{}"), '{"val/mse": 0.05}')

        report = _generator(tmp_path, adapter).generate_status_report()

        names = [e["name"] for e in report["experiments"]["top_models"]]
        assert "gp_run" in names
        entry = next(e for e in report["experiments"]["top_models"] if e["name"] == "gp_run")
        assert entry["mse"] == 0.05

    def test_exact_key_still_wins_over_flattering_variant(self, tmp_path: Path) -> None:
        adapter = DomainAdapter(
            metric=MetricConfig(primary_metric="rmse", direction="minimize"),
            experiment=ExperimentStructure(required_files=[]),
        )
        db = _db(tmp_path)
        db.set_results(
            db.create("own_vs_baseline", "D", "H", "{}"),
            '{"rmse": 0.5, "baseline_rmse_to_beat": 0.1}',
        )

        report = _generator(tmp_path, adapter).generate_status_report()

        entry = next(
            e for e in report["experiments"]["top_models"] if e["name"] == "own_vs_baseline"
        )
        assert entry["rmse"] == 0.5

    def test_missing_metric_excluded_from_top_models(self, tmp_path: Path) -> None:
        adapter = DomainAdapter(
            metric=MetricConfig(primary_metric="mse", direction="minimize"),
            experiment=ExperimentStructure(required_files=[]),
        )
        db = _db(tmp_path)
        db.set_results(db.create("no_metric", "D", "H", "{}"), '{"unrelated_field": 1.0}')

        report = _generator(tmp_path, adapter).generate_status_report()

        names = [e["name"] for e in report["experiments"]["top_models"]]
        assert "no_metric" not in names

    def test_no_db_reports_unavailable(self, tmp_path: Path) -> None:
        adapter = DomainAdapter(
            metric=MetricConfig(primary_metric="mse", direction="minimize"),
            experiment=ExperimentStructure(required_files=[]),
        )
        report = _generator(tmp_path, adapter).generate_status_report()
        assert report["experiments"]["available"] is False
