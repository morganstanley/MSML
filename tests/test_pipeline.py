"""Tests for Phase 2 pipeline: state detection, review parsing, test detection."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from types import ModuleType

import pytest

from alpha_lab.pipeline import (
    Pipeline,
    StepResult,
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
        assert not (Path(pipeline.workspace) / ".memory").exists()

    def test_decorated_bold_pass_above_long_tail(self, pipeline: Pipeline) -> None:
        """A real critic ended its review '**✅ PASS** — the harness is
        correct...' followed by a >500-char evidence table; the emoji broke
        the bold pattern, the tail window saw only the table, and a passing
        run was aborted (2026-08-08)."""
        table = "| check | result |\n" + "| x | ok |\n" * 40
        self._write_review(
            pipeline.workspace,
            "## Final review\n\n**✅ PASS** — the harness is correct, safe, "
            "and reproducible; no code changes are required.\n\n" + table,
        )
        assert pipeline._review_passes() is True

    def test_decorated_verdict_outranks_tail_bold_mention(
            self, pipeline: Pipeline) -> None:
        """An explicit opening-line verdict must beat a bold mention that
        merely lands inside the 500-char tail window."""
        filler = "analysis line\n" * 40
        self._write_review(
            pipeline.workspace,
            "**✅ NEEDS FIXES** — leakage in the split.\n\n" + filler +
            "\nSee the **PASS** criteria section above for what passing "
            "would have required.",
        )
        assert pipeline._review_passes() is False

    def test_unclear_file_falls_back_to_critic_summary(
            self, pipeline: Pipeline) -> None:
        self._write_review(pipeline.workspace,
                           "| check | result |\n| x | ok |\n" * 30)
        assert pipeline._review_passes() is False
        assert pipeline._review_passes(
            "Review complete. Verdict: PASS — no changes needed.") is True

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

    def test_review_path_cannot_escape_workspace(self, pipeline: Pipeline) -> None:
        from unittest.mock import MagicMock

        outside = Path(pipeline.workspace).parent / "outside_review.md"
        outside.write_text("VERDICT: PASS\noutside workspace")
        adapter = MagicMock()
        adapter.experiment.framework_dir = ".."
        adapter.phase2_review_file = outside.name
        pipeline.adapter = adapter

        content = pipeline._read_review()

        assert "outside workspace" not in content
        assert content == f"(no {outside.name} found)"

    def test_review_read_error_returns_no_file(self, pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch) -> None:
        self._write_review(pipeline.workspace, "VERDICT: PASS\n")
        review_path = Path(pipeline.workspace) / "backtest" / "review.md"
        original_read_text = Path.read_text

        def flaky_read_text(self: Path, *args, **kwargs) -> str:
            if self.resolve() == review_path.resolve():
                raise OSError("removed")
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", flaky_read_text)

        assert pipeline._read_review() == "(no review.md found)"

    def test_review_read_replaces_invalid_utf8(self, pipeline: Pipeline) -> None:
        bt = Path(pipeline.workspace) / "backtest"
        bt.mkdir(exist_ok=True)
        (bt / "review.md").write_bytes(b"VERDICT: PASS\nNon-UTF-8: \xff")

        content = pipeline._read_review()

        assert "VERDICT: PASS" in content
        assert "Non-UTF-8: \ufffd" in content

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


class TestPhase2Outcome:
    @pytest.fixture()
    def pipeline(self, tmp_workspace: str) -> Pipeline:
        from unittest.mock import MagicMock

        from alpha_lab.config import TaskConfig

        return Pipeline(
            provider=MagicMock(),
            config=TaskConfig(data_path="/tmp/data", description="Test"),
            workspace=tmp_workspace,
            event_callback=lambda event: None,
        )

    def test_already_complete_returns_true(
        self, pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "alpha_lab.pipeline.detect_phase2_progress",
            lambda *_args, **_kwargs: "done",
        )

        assert pipeline.run_phase2() is True

    def test_failed_critic_gate_returns_false(
        self, pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "alpha_lab.pipeline.detect_phase2_progress",
            lambda *_args, **_kwargs: "builder",
        )
        monkeypatch.setattr(
            pipeline,
            "_run_step",
            lambda step, **_kwargs: StepResult(step=step.name, completed=True),
        )
        monkeypatch.setattr(pipeline, "_review_passes",
                            lambda *a, **k: False)

        assert pipeline.run_phase2() is False


@pytest.mark.no_run_deps
class TestRunOutcomePropagation:
    @staticmethod
    def _write_config(tmp_path: Path, phases: list[str]) -> Path:
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(
            {
                "data_path": str(tmp_path / "data.csv"),
                "description": "Runner outcome test",
                "provider": "bedrock",
                "pipeline": {"phases": phases},
            }
        ))
        return config_path

    @staticmethod
    def _patch_run_dependencies(
        monkeypatch: pytest.MonkeyPatch,
        adapter: object,
    ) -> None:
        from alpha_lab import providers

        class SupervisorStub:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def validate_adapter(self) -> None:
                pass

            def review_phase2(self) -> None:
                pass

        monkeypatch.setattr(
            providers,
            "get_provider",
            lambda *_args, **_kwargs: object(),
        )
        monkeypatch.setattr(
            "alpha_lab.phase0.run_phase0",
            lambda *_args, **_kwargs: adapter,
        )
        monkeypatch.setattr("alpha_lab.supervisor.Supervisor", SupervisorStub)
        monkeypatch.setattr("alpha_lab.run.detect_phase1_complete", lambda *_args: True)

    @staticmethod
    def _invoke_run_main(
        run_module: ModuleType,
        config_path: Path,
        workspace: Path,
    ) -> None:
        callback = run_module.run_main.callback
        assert callback is not None
        arguments = {
            "config_path": str(config_path),
            "workspace": str(workspace),
            "verbose": False,
            "run_id": "outcome-test",
            "run_id_prefix": None,
            "enable_intake": False,
            "mlflow_enabled": False,
        }
        if "overwrite_config" in inspect.signature(callback).parameters:
            arguments["overwrite_config"] = False
        callback(**arguments)

    def test_phase2_failure_raises_from_run_main(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        adapter: object,
    ) -> None:
        from unittest.mock import MagicMock

        from alpha_lab import run as run_module

        config_path = self._write_config(tmp_path, ["phase2"])
        workspace = tmp_path / "phase2-workspace"
        self._patch_run_dependencies(monkeypatch, adapter)
        pipeline = MagicMock()
        pipeline.run_phase2.return_value = False
        monkeypatch.setattr(run_module, "Pipeline", MagicMock(return_value=pipeline))

        with pytest.raises(RuntimeError, match="Phase 2 did not complete successfully"):
            self._invoke_run_main(run_module, config_path, workspace)

    def test_phase3_dispatcher_crash_raises_from_run_main(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        adapter: object,
    ) -> None:
        from unittest.mock import MagicMock

        from alpha_lab import dispatcher as dispatcher_module
        from alpha_lab import run as run_module

        config_path = self._write_config(tmp_path, ["phase3"])
        workspace = tmp_path / "phase3-workspace"
        self._patch_run_dependencies(monkeypatch, adapter)
        dispatcher = MagicMock()
        dispatcher.crashed = True
        monkeypatch.setattr(
            dispatcher_module,
            "Dispatcher",
            MagicMock(return_value=dispatcher),
        )

        with pytest.raises(RuntimeError, match="Phase 3 dispatcher crashed unexpectedly"):
            self._invoke_run_main(run_module, config_path, workspace)

        dispatcher.run.assert_called_once_with()
        dispatcher.stop.assert_called_once_with()


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
