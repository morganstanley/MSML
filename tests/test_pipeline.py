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

    def test_heading_pass(self, pipeline: Pipeline) -> None:
        self._write_review(pipeline.workspace, "# PASS\n\nReview body.")
        assert pipeline._review_passes() is True

    def test_heading_needs_fixes(self, pipeline: Pipeline) -> None:
        self._write_review(
            pipeline.workspace,
            "# NEEDS FIXES\n\n- Issue 1\n- Issue 2",
        )
        assert pipeline._review_passes() is False

    def test_bare_needs_fixes_trailing(self, pipeline: Pipeline) -> None:
        self._write_review(
            pipeline.workspace,
            "Review notes here.\n\nNEEDS FIXES\n",
        )
        assert pipeline._review_passes() is False

    def test_instruction_heading_doesnt_override_verdict(self, pipeline: Pipeline) -> None:
        """Example `## PASS` in instructions must not override a NEEDS FIXES verdict in the tail."""
        # Long body (>500 chars) so the early `## PASS` is NOT in the tail window.
        body = (
            "# Review template\n\n"
            "The verdict section should use one of these formats:\n\n"
            "## PASS\n\nor\n\n"
            "## NEEDS FIXES\n\n"
            "Pick the appropriate heading based on the issues found.\n\n"
            + ("Detailed analysis follows.\n" * 40)
            + "\n---\n\nVerdict: NEEDS FIXES\n\n- Bug in foo\n- Edge case in bar\n"
        )
        self._write_review(pipeline.workspace, body)
        assert pipeline._review_passes() is False

    def test_instruction_bare_line_doesnt_override_verdict(self, pipeline: Pipeline) -> None:
        """Example bare `PASS` / `NEEDS FIXES` lines in instructions must not override verdict."""
        body = (
            "# How to conclude\n\nEnd your review with a bare verdict line:\n\n"
            "PASS\n\nor\n\nNEEDS FIXES\n\n"
            + ("Analysis line.\n" * 50)
            + "\n---\n\nFinal verdict: PASS\n"
        )
        self._write_review(pipeline.workspace, body)
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


class TestSaveCanonicalBaselinesEntryModule:
    """Covers the framework-aware entry_module selection in save_canonical_baselines."""

    def _make_pipeline(self, tmp_workspace, adapter):
        from unittest.mock import MagicMock

        from alpha_lab.config import TaskConfig

        provider = MagicMock()
        config = TaskConfig(data_path="/tmp/data", description="Test")
        pipe = Pipeline(
            provider=provider,
            config=config,
            workspace=tmp_workspace,
            event_callback=lambda e: None,
        )
        pipe.adapter = adapter
        return pipe

    def _run_and_capture(self, pipe, monkeypatch):
        captured = {}

        def fake_run(cmd, *args, **kwargs):
            captured["cmd"] = cmd
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            result.stdout = ""
            return result

        import subprocess as _sp
        from unittest.mock import MagicMock
        monkeypatch.setattr(_sp, "run", fake_run)
        pipe.save_canonical_baselines()
        return captured.get("cmd", [])

    def test_picks_run_star_py_from_framework_files(self, tmp_workspace, monkeypatch):
        from unittest.mock import MagicMock
        adapter = MagicMock()
        adapter.experiment.framework_dir = "benchmark"
        adapter.experiment.framework_files = ["bench.py", "run_bench.py"]
        fw = Path(tmp_workspace) / "benchmark"
        fw.mkdir(parents=True, exist_ok=True)
        (fw / "run_bench.py").write_text("# runner")
        pipe = self._make_pipeline(tmp_workspace, adapter)
        cmd = self._run_and_capture(pipe, monkeypatch)
        assert "benchmark.run_bench" in cmd

    def test_falls_back_to_run_backtest_when_present(self, tmp_workspace, monkeypatch):
        from unittest.mock import MagicMock
        adapter = MagicMock()
        adapter.experiment.framework_dir = "backtest"
        adapter.experiment.framework_files = ["strategy.py"]  # no run_*.py
        fw = Path(tmp_workspace) / "backtest"
        fw.mkdir(parents=True, exist_ok=True)
        (fw / "run_backtest.py").write_text("# runner")
        pipe = self._make_pipeline(tmp_workspace, adapter)
        cmd = self._run_and_capture(pipe, monkeypatch)
        assert "backtest.run_backtest" in cmd

    def test_default_when_no_adapter(self, tmp_workspace, monkeypatch):
        pipe = self._make_pipeline(tmp_workspace, adapter=None)
        cmd = self._run_and_capture(pipe, monkeypatch)
        assert "backtest.run_backtest" in cmd


class TestExtractVerdictDecorations:
    """An archived GLM critic wrote 'Verdict: 🟢 PASS' at the TOP of a long
    review; the old parser required only colon/space/asterisk after the
    label and tail-restricted every fallback, so 7 clean builds were
    re-'fixed' across an 8-round loop (phase-2 study, 2026-08-13)."""

    def test_emoji_decorated_pass_at_top_of_long_review(self):
        from alpha_lab.pipeline import _extract_verdict
        content = "Verdict: 🟢 PASS\n\n" + ("detail line\n" * 200)
        assert _extract_verdict(content) == "PASS"

    def test_emoji_decorated_needs_fixes(self):
        from alpha_lab.pipeline import _extract_verdict
        content = "## Verdict\n\n🔴 NEEDS FIXES\n\n" + ("x\n" * 100)
        assert _extract_verdict(content) == "NEEDS FIXES"

    def test_plain_forms_still_work(self):
        from alpha_lab.pipeline import _extract_verdict
        assert _extract_verdict("Final verdict: PASS") == "PASS"
        assert _extract_verdict("Verdict: NEEDS FIXES") == "NEEDS FIXES"
        long_tail_bold = ("y\n" * 100) + "**NEEDS FIXES**\n"
        assert _extract_verdict(long_tail_bold) == "NEEDS FIXES"

    def test_no_verdict_stays_unclear(self):
        from alpha_lab.pipeline import _extract_verdict
        assert _extract_verdict("nothing conclusive here") == "UNCLEAR"
