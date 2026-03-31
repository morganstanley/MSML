"""Tests for Phase 2 pipeline: state detection, review parsing, test detection."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from alpha_lab.pipeline import (
    Pipeline,
    detect_phase1_complete,
    detect_phase2_progress,
)


# ---------------------------------------------------------------------------
# Phase 1 detection
# ---------------------------------------------------------------------------


class TestDetectPhase1Complete:
    def test_complete_workspace(self, tmp_workspace: str) -> None:
        (Path(tmp_workspace) / "learnings.md").write_text("# Key findings\n- Found stuff")
        report_dir = Path(tmp_workspace) / "data_report"
        report_dir.mkdir()
        (report_dir / "schema.md").write_text("# Schema")
        assert detect_phase1_complete(tmp_workspace) is True

    def test_empty_learnings(self, tmp_workspace: str) -> None:
        (Path(tmp_workspace) / "learnings.md").write_text("")
        report_dir = Path(tmp_workspace) / "data_report"
        report_dir.mkdir()
        (report_dir / "schema.md").write_text("# Schema")
        assert detect_phase1_complete(tmp_workspace) is False

    def test_no_learnings(self, tmp_workspace: str) -> None:
        report_dir = Path(tmp_workspace) / "data_report"
        report_dir.mkdir()
        (report_dir / "schema.md").write_text("# Schema")
        assert detect_phase1_complete(tmp_workspace) is False

    def test_no_report_dir(self, tmp_workspace: str) -> None:
        (Path(tmp_workspace) / "learnings.md").write_text("# Findings")
        assert detect_phase1_complete(tmp_workspace) is False

    def test_empty_report_dir(self, tmp_workspace: str) -> None:
        (Path(tmp_workspace) / "learnings.md").write_text("# Findings")
        (Path(tmp_workspace) / "data_report").mkdir()
        assert detect_phase1_complete(tmp_workspace) is False

    def test_report_dir_no_md_files(self, tmp_workspace: str) -> None:
        (Path(tmp_workspace) / "learnings.md").write_text("# Findings")
        report_dir = Path(tmp_workspace) / "data_report"
        report_dir.mkdir()
        (report_dir / "data.csv").write_text("a,b")  # not .md
        assert detect_phase1_complete(tmp_workspace) is False


# ---------------------------------------------------------------------------
# Phase 2 progress detection
# ---------------------------------------------------------------------------


class TestDetectPhase2Progress:
    def test_nothing_built(self, tmp_workspace: str) -> None:
        assert detect_phase2_progress(tmp_workspace) == "builder"

    def test_partial_build(self, tmp_workspace: str) -> None:
        bt = Path(tmp_workspace) / "backtest"
        bt.mkdir()
        (bt / "strategy.py").write_text("class Strategy: pass")
        # Missing engine.py and metrics.py
        assert detect_phase2_progress(tmp_workspace) == "builder"

    def test_full_build_no_review(self, tmp_workspace: str) -> None:
        bt = Path(tmp_workspace) / "backtest"
        bt.mkdir()
        for f in ["strategy.py", "engine.py", "metrics.py"]:
            (bt / f).write_text(f"# {f}")
        assert detect_phase2_progress(tmp_workspace) == "critic"

    def test_review_needs_fixes(self, tmp_workspace: str) -> None:
        bt = Path(tmp_workspace) / "backtest"
        bt.mkdir()
        for f in ["strategy.py", "engine.py", "metrics.py"]:
            (bt / f).write_text(f"# {f}")
        (bt / "review.md").write_text("## Review\nVerdict: NEEDS FIXES\n- Fix line 42")
        assert detect_phase2_progress(tmp_workspace) == "builder"

    def test_review_passes_no_tests(self, tmp_workspace: str) -> None:
        bt = Path(tmp_workspace) / "backtest"
        bt.mkdir()
        for f in ["strategy.py", "engine.py", "metrics.py"]:
            (bt / f).write_text(f"# {f}")
        (bt / "review.md").write_text("## Review\nVERDICT: PASS\nAll good!")
        assert detect_phase2_progress(tmp_workspace) == "tester"


# ---------------------------------------------------------------------------
# Review verdict parsing
# ---------------------------------------------------------------------------


class TestReviewPasses:
    """Test the Pipeline._review_passes() method via a minimal Pipeline instance."""

    @pytest.fixture()
    def pipeline(self, tmp_workspace: str) -> Pipeline:
        """Create a minimal pipeline for testing _review_passes."""
        from unittest.mock import MagicMock

        from alpha_lab.config import TaskConfig

        provider = MagicMock()
        config = TaskConfig(data_path="/tmp/data", description="Test")
        return Pipeline(
            provider=provider,
            config=config,
            workspace=tmp_workspace,
            event_callback=lambda e: None,
        )

    def _write_review(self, workspace: str, content: str) -> None:
        bt = Path(workspace) / "backtest"
        bt.mkdir(exist_ok=True)
        (bt / "review.md").write_text(content)

    def test_verdict_pass(self, pipeline: Pipeline) -> None:
        self._write_review(pipeline.workspace, "## Review\nVERDICT: PASS\nAll clear.")
        assert pipeline._review_passes() is True

    def test_verdict_pass_heading(self, pipeline: Pipeline) -> None:
        self._write_review(pipeline.workspace, "# PASS\nEverything looks good.")
        assert pipeline._review_passes() is True

    def test_verdict_bold_pass(self, pipeline: Pipeline) -> None:
        self._write_review(pipeline.workspace, "**PASS**\nApproved.")
        assert pipeline._review_passes() is True

    def test_verdict_needs_fixes(self, pipeline: Pipeline) -> None:
        self._write_review(pipeline.workspace, "VERDICT: NEEDS FIXES\n- Fix X")
        assert pipeline._review_passes() is False

    def test_no_false_positive_on_passed(self, pipeline: Pipeline) -> None:
        """The word 'passed' should not trigger a false positive PASS verdict."""
        self._write_review(
            pipeline.workspace,
            "Some tests passed but there are issues.\nNEEDS FIXES"
        )
        assert pipeline._review_passes() is False

    def test_no_review_file(self, pipeline: Pipeline) -> None:
        # No review.md at all — no file should not pass
        assert pipeline._review_passes() is False

    def test_bare_pass_on_own_line(self, pipeline: Pipeline) -> None:
        self._write_review(pipeline.workspace, "Review done.\n\nPASS\n")
        assert pipeline._review_passes() is True


# ---------------------------------------------------------------------------
# Test pass detection
# ---------------------------------------------------------------------------


class TestTestsPass:
    """Test the Pipeline._tests_pass() method."""

    @pytest.fixture()
    def pipeline(self, tmp_workspace: str) -> Pipeline:
        from unittest.mock import MagicMock

        from alpha_lab.config import TaskConfig

        provider = MagicMock()
        config = TaskConfig(data_path="/tmp/data", description="Test")
        return Pipeline(
            provider=provider,
            config=config,
            workspace=tmp_workspace,
            event_callback=lambda e: None,
        )

    def test_passing_output(self, pipeline: Pipeline) -> None:
        output = "test_foo.py PASSED\ntest_bar.py PASSED\n\n[exit code: 0]"
        assert pipeline._tests_pass(output) is True

    def test_failing_output(self, pipeline: Pipeline) -> None:
        output = "test_foo.py FAILED\n\n[exit code: 1]"
        assert pipeline._tests_pass(output) is False

    def test_no_false_positive_exit_code_in_content(self, pipeline: Pipeline) -> None:
        """[exit code: 0] in test stdout should NOT cause false positive."""
        output = (
            "Testing: output contains [exit code: 0] as text\n"
            "test_foo.py FAILED\n\n[exit code: 1]"
        )
        assert pipeline._tests_pass(output) is False

    def test_trailing_whitespace(self, pipeline: Pipeline) -> None:
        output = "all good\n[exit code: 0]  \n"
        # rstrip() is applied, so trailing whitespace/newlines are handled
        assert pipeline._tests_pass(output) is True
