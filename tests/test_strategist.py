"""Tests for the Strategist's own context-building logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from alpha_lab.adapter import DomainAdapter
from alpha_lab.experiment_db import CANONICAL_RUN_COMPLETE_SENTINEL, ExperimentDB
from alpha_lab.strategist import Strategist


def _touch_sentinel(db: ExperimentDB, exp_name: str) -> None:
    """Create the canonical-run sentinel a row needs to appear on the leaderboard."""
    workspace = Path(db.db_path).parent.parent
    results = workspace / "experiments" / exp_name / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / CANONICAL_RUN_COMPLETE_SENTINEL).touch()


def _strategist(db: ExperimentDB, adapter: DomainAdapter) -> Strategist:
    return Strategist(
        provider=MagicMock(), db=db,
        event_callback=lambda event: None, adapter=adapter,
    )


class TestBuildContextLeaderboardDisplay:
    """PROJ-765: the Strategist's own context previously re-parsed the
    leaderboard's results with a flat key lookup, showing "?" instead of the
    real value whenever an experiment reported a variant key spelling --
    degrading the Strategist's own reasoning input every turn."""

    def test_variant_key_spelling_shown_correctly(
        self, db: ExperimentDB, adapter: DomainAdapter,
    ) -> None:
        # Default adapter fixture: primary_metric="sharpe", direction="maximize".
        strategist = _strategist(db, adapter)
        exp_id = db.create("gp_run", "D", "H", "{}")
        db.set_results(exp_id, '{"val/sharpe": 2.5}')
        _touch_sentinel(db, "gp_run")

        context = strategist._build_context()

        leaderboard_section = context.split("## Leaderboard")[1]
        assert "gp_run" in leaderboard_section
        assert "Sharpe: 2.5" in leaderboard_section
        assert "Sharpe: ?" not in leaderboard_section

    def test_exact_key_still_wins_over_flattering_variant(
        self, db: ExperimentDB, adapter: DomainAdapter,
    ) -> None:
        strategist = _strategist(db, adapter)
        exp_id = db.create("gp_run", "D", "H", "{}")
        db.set_results(exp_id, '{"sharpe": 1.0, "champion_sharpe": 99.0}')
        _touch_sentinel(db, "gp_run")

        context = strategist._build_context()

        leaderboard_section = context.split("## Leaderboard")[1]
        assert "Sharpe: 1.0" in leaderboard_section

    def test_missing_metric_falls_back_to_question_mark(
        self, db: ExperimentDB, adapter: DomainAdapter,
    ) -> None:
        strategist = _strategist(db, adapter)
        exp_id = db.create("no_metric", "D", "H", "{}")
        db.set_results(exp_id, '{"unrelated_field": 1.0}')
        _touch_sentinel(db, "no_metric")

        context = strategist._build_context()

        leaderboard_section = context.split("## Leaderboard")[1]
        assert "Sharpe: ?" in leaderboard_section
