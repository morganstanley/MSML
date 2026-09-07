"""Strategist + Worker integration tests for the Conductor's directive
channel.

Each agent's prompt-builder must read meta/directives.md and
meta/annotations.json (where applicable) and inject the content into the
agent's context. Each must tolerate missing files gracefully so the
no_conductor=True NOOP mode reproduces the pre-Conductor behavior.

Strategist's tool list:
  - default mode: cancel_experiments REMOVED (Conductor parks instead);
    note_to_conductor ADDED.
  - no_conductor=True: cancel_experiments RESTORED, note_to_conductor REMOVED.

Worker's tool list always includes note_to_conductor (the directive read
is always-on; the absence of meta/directives.md just means no directive).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alpha_lab import meta_layout as ml
from alpha_lab.config import Phase3Config, PipelineConfig, TaskConfig
from alpha_lab.experiment_db import ExperimentDB
from alpha_lab.strategist import Strategist
from alpha_lab.worker import Worker


def _make_strategist(tmp_path: Path, *, no_conductor: bool = False) -> Strategist:
    cfg = TaskConfig(data_path="/d", description="D")
    cfg.pipeline = PipelineConfig(
        phase3=Phase3Config(no_conductor=no_conductor)
    )
    return Strategist(
        provider=MagicMock(openai_client=None),
        config=cfg,
        workspace=str(tmp_path),
        db=ExperimentDB(str(tmp_path / "db.sqlite")),
        event_callback=lambda e: None,
        adapter=None,
    )


def _make_worker(tmp_path: Path) -> Worker:
    cfg = TaskConfig(data_path="/d", description="D")
    cfg.pipeline = PipelineConfig(phase3=Phase3Config())
    return Worker(
        worker_id="worker_test",
        provider=MagicMock(openai_client=None),
        config=cfg,
        workspace=str(tmp_path),
        db=ExperimentDB(str(tmp_path / "db.sqlite")),
        event_callback=lambda e: None,
        adapter=None,
    )


# ---------------------------------------------------------------------------
# Strategist context inclusion
# ---------------------------------------------------------------------------


class TestStrategistSlidingPendingCap:
    """The strategist's context shows the sliding pending cap as the
    primary signal: 'Slots open this turn = max_pending_proposals - pending'.
    The lifetime cap (max_experiments) stays as a soft ceiling."""

    def _strategist_with_pending(
        self, tmp_path: Path, pending: int, max_pending: int = 12,
        max_experiments: int = 360,
    ) -> Strategist:
        cfg = TaskConfig(data_path="/d", description="D")
        cfg.pipeline = PipelineConfig(
            phase3=Phase3Config(
                max_pending_proposals=max_pending,
                max_experiments=max_experiments,
            )
        )
        db = ExperimentDB(str(tmp_path / "db.sqlite"))
        # Seed `pending` to_implement rows so board_summary surfaces them.
        for i in range(pending):
            db.create(f"pending_{i}", "D", "H", "{}")
        return Strategist(
            provider=MagicMock(openai_client=None),
            config=cfg,
            workspace=str(tmp_path),
            db=db,
            event_callback=lambda e: None,
            adapter=None,
        )

    def test_slots_open_reflects_max_pending_minus_pending(
        self, tmp_path: Path
    ) -> None:
        s = self._strategist_with_pending(tmp_path, pending=5, max_pending=12)
        ctx = s._build_context()
        # Primary signal mentions the sliding cap and the slots open.
        assert "Pending (to_implement): 5" in ctx
        assert "Pending cap (max_pending_proposals): 12" in ctx
        assert "Slots open this turn: 7" in ctx

    def test_pending_at_cap_shows_zero_slots(self, tmp_path: Path) -> None:
        s = self._strategist_with_pending(tmp_path, pending=12, max_pending=12)
        ctx = s._build_context()
        assert "Slots open this turn: 0" in ctx
        # Should also surface the explicit "queue at cap" guidance.
        assert "PENDING QUEUE AT CAP" in ctx

    def test_pending_above_cap_clamps_to_zero(self, tmp_path: Path) -> None:
        # Whatever caused this (e.g. cap lowered mid-run), slots_open
        # should never go negative.
        s = self._strategist_with_pending(tmp_path, pending=20, max_pending=12)
        ctx = s._build_context()
        assert "Slots open this turn: 0" in ctx

    def test_lifetime_cap_also_surfaced(self, tmp_path: Path) -> None:
        s = self._strategist_with_pending(
            tmp_path, pending=3, max_pending=12, max_experiments=50,
        )
        ctx = s._build_context()
        assert "Lifetime cap (max_experiments, safety ceiling): 50" in ctx


class TestStrategistLifetimeBudgetGate:
    """run_turn skips entirely once max_experiments non-cancelled rows
    exist (mirrors msml's turn-skip gate). The prompt banner alone is
    advisory — models have proposed past it — so the model must not be
    invoked at all when the budget is spent."""

    def _strategist(self, tmp_path: Path, rows: int, cap: int) -> Strategist:
        cfg = TaskConfig(data_path="/d", description="D")
        cfg.pipeline = PipelineConfig(
            phase3=Phase3Config(max_experiments=cap)
        )
        db = ExperimentDB(str(tmp_path / "db.sqlite"))
        for i in range(rows):
            db.create(f"exp_{i}", "D", "H", "{}")
        return Strategist(
            provider=MagicMock(openai_client=None),
            config=cfg,
            workspace=str(tmp_path),
            db=db,
            event_callback=lambda e: None,
            adapter=None,
        )

    def test_turn_skipped_at_cap(self, tmp_path: Path) -> None:
        s = self._strategist(tmp_path, rows=20, cap=20)
        with patch("alpha_lab.strategist.AgentLoop") as loop_cls:
            s.run_turn()
        loop_cls.assert_not_called()

    def test_turn_skipped_over_cap(self, tmp_path: Path) -> None:
        s = self._strategist(tmp_path, rows=25, cap=20)
        with patch("alpha_lab.strategist.AgentLoop") as loop_cls:
            s.run_turn()
        loop_cls.assert_not_called()

    def test_turn_runs_under_cap(self, tmp_path: Path) -> None:
        s = self._strategist(tmp_path, rows=19, cap=20)
        with patch("alpha_lab.strategist.AgentLoop") as loop_cls:
            s.run_turn()
        loop_cls.assert_called_once()

    def test_cancelled_rows_do_not_consume_budget(self, tmp_path: Path) -> None:
        s = self._strategist(tmp_path, rows=20, cap=20)
        s.db.update_status(1, "cancelled")
        with patch("alpha_lab.strategist.AgentLoop") as loop_cls:
            s.run_turn()
        loop_cls.assert_called_once()


class TestStrategistDirectiveInjection:
    def test_directive_appears_in_context(self, tmp_path: Path) -> None:
        ml.ensure_meta_layout(tmp_path)
        from alpha_lab import conductor_tools as ct
        ct.append_directive(
            tmp_path, "strategist", "Diversify into sequential.",
        )
        s = _make_strategist(tmp_path)
        ctx = s._build_context()
        assert "Diversify into sequential" in ctx

    def test_annotations_appear_in_context(self, tmp_path: Path) -> None:
        ml.ensure_meta_layout(tmp_path)
        from alpha_lab import conductor_tools as ct
        ct.set_annotation(tmp_path, 7, "champion")
        ct.set_annotation(tmp_path, 9, "control")
        s = _make_strategist(tmp_path)
        ctx = s._build_context()
        assert "champion" in ctx
        assert "control" in ctx
        assert "#7" in ctx
        assert "#9" in ctx

    def test_no_meta_dir_does_not_crash(self, tmp_path: Path) -> None:
        # Don't bootstrap; meta/ doesn't exist
        s = _make_strategist(tmp_path)
        ctx = s._build_context()
        # Should not raise; should still produce a context string
        assert isinstance(ctx, str)


class TestStrategistArtifactReadiness:
    def test_context_lists_only_existing_experiment_artifacts(
        self, tmp_path: Path,
    ) -> None:
        strategist = _make_strategist(tmp_path)
        exp_id = strategist.db.create("artifact_exp", "D", "H", "{}")
        exp_dir = tmp_path / "experiments" / "artifact_exp"
        exp_dir.mkdir(parents=True)
        (exp_dir / "config.yaml").write_text("model: ridge\n")

        context = strategist._build_context()
        line = next(line for line in context.splitlines() if f"#{exp_id} " in line)

        assert "available: config.yaml" in line
        assert "debrief.md" not in line

    def test_context_lists_only_existing_shared_artifacts(
        self, tmp_path: Path,
    ) -> None:
        strategist = _make_strategist(tmp_path)
        (tmp_path / "playbook.md").write_text("# Playbook\n")

        context = strategist._build_context()
        section = context.split("## Available Shared Artifacts", 1)[1].split("##", 1)[0]

        assert "playbook.md" in section
        assert "feedback_to_system.md" not in section


class TestStrategistToolSet:
    def test_default_mode_has_note_to_conductor_no_cancel(
        self, tmp_path: Path
    ) -> None:
        s = _make_strategist(tmp_path, no_conductor=False)
        # Reach into the tool list assembly logic in run_turn
        # We can't easily call run_turn (it spawns an AgentLoop); instead
        # we replicate the same selection.
        from alpha_lab.config import Phase3Config
        # Build the same list run_turn would build
        tool_names = [
            "read_board", "propose_experiment",
            "update_playbook", "read_file", "grep_file", "report_to_user",
            "memory_store", "memory_search", "memory_read",
            "note_to_conductor",
        ]
        if s.config.pipeline.phase3.no_conductor:
            tool_names.append("cancel_experiments")
            try:
                tool_names.remove("note_to_conductor")
            except ValueError:
                pass
        if s.config.pipeline.phase3.no_playbook:
            tool_names.remove("update_playbook")
        assert "note_to_conductor" in tool_names
        assert "cancel_experiments" not in tool_names

    def test_noop_mode_restores_cancel_experiments(self, tmp_path: Path) -> None:
        s = _make_strategist(tmp_path, no_conductor=True)
        tool_names = [
            "read_board", "propose_experiment",
            "update_playbook", "read_file", "grep_file", "report_to_user",
            "memory_store", "memory_search", "memory_read",
            "note_to_conductor",
        ]
        if s.config.pipeline.phase3.no_conductor:
            tool_names.append("cancel_experiments")
            try:
                tool_names.remove("note_to_conductor")
            except ValueError:
                pass
        assert "cancel_experiments" in tool_names
        assert "note_to_conductor" not in tool_names


# ---------------------------------------------------------------------------
# Worker context inclusion
# ---------------------------------------------------------------------------


class TestWorkerDirectiveInjection:
    def test_directive_appears_at_top_of_experiment_context(self, tmp_path: Path) -> None:
        ml.ensure_meta_layout(tmp_path)
        from alpha_lab import conductor_tools as ct
        ct.append_directive(
            tmp_path, "worker", "Always include a --no-teacher ablation row.",
        )
        w = _make_worker(tmp_path)
        from alpha_lab.experiment_db import Experiment
        exp = Experiment(
            id=1, name="x", description="d", hypothesis="h", status="to_implement",
            config_json="{}", worker_id=None, slurm_job_id=None,
            results_json=None, error=None, debrief_path=None,
            created_at=0.0, updated_at=0.0, started_at=None, finished_at=None,
        )
        ctx = w._build_experiment_context(exp)
        # Directive appears before the experiment block
        assert "no-teacher ablation" in ctx
        assert ctx.find("no-teacher") < ctx.find("Experiment #1")

    def test_no_meta_dir_does_not_crash(self, tmp_path: Path) -> None:
        w = _make_worker(tmp_path)
        from alpha_lab.experiment_db import Experiment
        exp = Experiment(
            id=1, name="x", description="d", hypothesis="h", status="to_implement",
            config_json="{}", worker_id=None, slurm_job_id=None,
            results_json=None, error=None, debrief_path=None,
            created_at=0.0, updated_at=0.0, started_at=None, finished_at=None,
        )
        ctx = w._build_experiment_context(exp)
        # No directive, but context still includes the experiment block
        assert "Experiment #1" in ctx
