"""Tests for benchmark workspace validators.

The synthetic workspaces these tests build mirror the real on-disk schema:

- ``adapter/manifest.json`` matches what ``save_adapter()`` writes:
  ``domain_name`` + ``metric.primary_metric`` + ``experiment.framework_dir`` +
  ``phase2_review_file``.
- Phase 2 framework lives at ``{workspace}/{framework_dir}/``  — default
  ``backtest/`` for the built-in time-series adapter.
- Phase 3 ``experiments.db`` rows use the kanban statuses from
  ``KANBAN_COLUMNS`` in ``experiment_db.py`` (``done`` / ``analyzed`` are the
  terminal success states; ``running`` / ``queued`` etc. are in-flight).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from alpha_lab.benchmarks.validators import (
    EXPERIMENT_COMPLETED_STATUSES,
    EXPERIMENT_IN_FLIGHT_STATUSES,
    TOOL_ERROR_MARKER,
    Check,
    ValidationResult,
    validate_workspace,
)


# ---------------------------------------------------------------------------
# Workspace builders — assemble synthetic post-run workspaces on disk
# ---------------------------------------------------------------------------


def _write_config(ws: Path, phases: list[str]) -> None:
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "config.json").write_text(json.dumps({
        "data_path": "data.csv",
        "description": "test",
        "pipeline": {"phases": phases},
    }))


def _write_adapter(
    ws: Path,
    *,
    valid: bool = True,
    framework_dir: str = "backtest",
    review_file: str = "review.md",
) -> None:
    """Write an adapter/manifest.json matching ``adapter_loader.save_adapter()``."""
    adapter = ws / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    if valid:
        manifest = {
            "domain_name": "time_series",
            "domain_description": "test",
            "phase2_framework_description": "test framework",
            "phase2_review_file": review_file,
            "metric": {
                "primary_metric": "rmse",
                "direction": "minimize",
                "extract_key": "rmse",
                "display_name": "RMSE",
                "secondary_metrics": [],
            },
            "experiment": {
                "required_files": ["model.py"],
                "entry_point": "run_experiment.py",
                "results_dir": "results",
                "results_file": "metrics.json",
                "framework_dir": framework_dir,
                "framework_files": [],
            },
        }
    else:
        manifest = {"domain_name": "time_series"}  # missing required keys
    (adapter / "manifest.json").write_text(json.dumps(manifest))


def _write_phase1(ws: Path, *, learnings: str = "important", reports: int = 1) -> None:
    (ws / "learnings.md").write_text(learnings)
    if reports:
        report_dir = ws / "data_report"
        report_dir.mkdir(parents=True, exist_ok=True)
        for i in range(reports):
            (report_dir / f"report_{i}.md").write_text(f"report {i}")


def _write_phase2(
    ws: Path,
    *,
    framework_dir: str = "backtest",
    review_file: str = "review.md",
    review_verdict: str = "PASS",
    py_files: int = 2,
    test_files: int = 1,
) -> None:
    """Build a Phase 2 framework dir matching the adapter's configuration."""
    framework = ws / framework_dir
    framework.mkdir(parents=True, exist_ok=True)
    for i in range(py_files):
        (framework / f"module_{i}.py").write_text("# placeholder\n")
    tests_dir = framework / "tests"
    tests_dir.mkdir(exist_ok=True)
    for i in range(test_files):
        (tests_dir / f"test_module_{i}.py").write_text("def test_x(): pass\n")
    (framework / review_file).write_text(f"Verdict: {review_verdict}\n")


def _write_phase3(
    ws: Path,
    *,
    done: int = 1,
    analyzed: int = 0,
    running: int = 0,
    queued: int = 0,
    finished: int = 0,
) -> None:
    """Write an experiments.db with rows in the requested kanban statuses."""
    db_path = ws / "experiments.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE experiments (id INTEGER PRIMARY KEY, name TEXT, status TEXT)")
    counter = 0
    for status, n in (
        ("done", done),
        ("analyzed", analyzed),
        ("running", running),
        ("queued", queued),
        ("finished", finished),
    ):
        for _ in range(n):
            counter += 1
            cur.execute(
                "INSERT INTO experiments(name, status) VALUES (?, ?)",
                (f"exp_{counter}", status),
            )
    conn.commit()
    conn.close()


def _write_logs(ws: Path, *, contents: str = '{"event": "ok"}\n') -> None:
    logs = ws / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "pipeline.jsonl").write_text(contents)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_validate_clean_workspace_passes(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_config(ws, phases=["phase1", "phase2", "phase3"])
    _write_adapter(ws)
    _write_phase1(ws)
    _write_phase2(ws)  # framework_dir defaults to "backtest"
    _write_phase3(ws, done=1)
    _write_logs(ws)

    result = validate_workspace(ws)
    assert isinstance(result, ValidationResult)
    failed = [c.name for c in result.checks if not c.ok]
    assert result.ok, f"unexpected failures: {failed}"


def test_validate_missing_config_fails(tmp_path: Path) -> None:
    result = validate_workspace(tmp_path / "no_config")
    assert not result.ok
    assert result.checks[0].name == "config.json present"
    assert not result.checks[0].ok


def test_validate_invalid_config_fails(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "config.json").write_text("not json {")
    result = validate_workspace(ws)
    assert not result.ok
    names = {c.name for c in result.checks}
    assert "config.json parses" in names


def test_validate_missing_adapter_manifest(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_config(ws, phases=["phase1"])
    # no adapter directory at all
    _write_phase1(ws)
    _write_logs(ws)
    result = validate_workspace(ws)
    assert not result.ok
    failures = {c.name for c in result.checks if not c.ok}
    assert "adapter/manifest.json exists" in failures


def test_validate_adapter_missing_required_keys(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_config(ws, phases=["phase1"])
    _write_adapter(ws, valid=False)
    _write_phase1(ws)
    _write_logs(ws)
    result = validate_workspace(ws)
    assert not result.ok
    failures = {c.name for c in result.checks if not c.ok}
    assert "adapter manifest has required keys" in failures


def test_validate_adapter_missing_primary_metric(tmp_path: Path) -> None:
    """Adapter has the right top-level keys but metric.primary_metric is empty."""
    ws = tmp_path / "ws"
    _write_config(ws, phases=["phase1"])
    adapter = ws / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "manifest.json").write_text(json.dumps({
        "domain_name": "time_series",
        "metric": {"primary_metric": ""},  # blank
        "experiment": {"framework_dir": "backtest"},
    }))
    _write_phase1(ws)
    _write_logs(ws)
    result = validate_workspace(ws)
    assert not result.ok
    failures = {c.name for c in result.checks if not c.ok}
    assert "adapter metric has primary_metric" in failures


def test_validate_phase1_empty_learnings_fails(tmp_path: Path) -> None:
    """A whitespace-only learnings.md should fail, not just truly-empty."""
    ws = tmp_path / "ws"
    _write_config(ws, phases=["phase1"])
    _write_adapter(ws)
    _write_phase1(ws, learnings="   \n\t\n")  # only whitespace
    _write_logs(ws)
    result = validate_workspace(ws)
    assert not result.ok
    failures = {c.name for c in result.checks if not c.ok}
    assert "learnings.md non-empty" in failures


def test_validate_phase1_missing_data_report_fails(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_config(ws, phases=["phase1"])
    _write_adapter(ws)
    _write_phase1(ws, reports=0)
    _write_logs(ws)
    result = validate_workspace(ws)
    assert not result.ok
    failures = {c.name for c in result.checks if not c.ok}
    assert "data_report/*.md produced" in failures


def test_validate_phase2_default_uses_backtest_dir(tmp_path: Path) -> None:
    """When manifest says framework_dir='backtest', that's the dir checked."""
    ws = tmp_path / "ws"
    _write_config(ws, phases=["phase2"])
    _write_adapter(ws, framework_dir="backtest")
    _write_phase2(ws, framework_dir="backtest")
    _write_logs(ws)
    result = validate_workspace(ws)
    failed = [c.name for c in result.checks if not c.ok]
    assert result.ok, f"unexpected failures: {failed}"


def test_validate_phase2_uses_adapter_configured_dir(tmp_path: Path) -> None:
    """Custom adapter with framework_dir='harness' should look there."""
    ws = tmp_path / "ws"
    _write_config(ws, phases=["phase2"])
    _write_adapter(ws, framework_dir="harness")
    _write_phase2(ws, framework_dir="harness")
    _write_logs(ws)
    result = validate_workspace(ws)
    assert result.ok, [c for c in result.checks if not c.ok]
    # Confirm the right dir was probed
    names = " ".join(c.name for c in result.checks)
    assert "harness/" in names


def test_validate_phase2_rejects_absolute_framework_dir(tmp_path: Path) -> None:
    """Manifest with an absolute framework_dir must be refused — would let
    the validator read outside the workspace.
    """
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    (ws / "config.json").write_text(json.dumps({
        "data_path": "data.csv",
        "description": "test",
        "pipeline": {"phases": ["phase2"]},
    }))
    adapter = ws / "adapter"
    adapter.mkdir()
    (adapter / "manifest.json").write_text(json.dumps({
        "domain_name": "x",
        "metric": {"primary_metric": "rmse"},
        "experiment": {"framework_dir": "/etc"},
    }))
    _write_logs(ws)
    result = validate_workspace(ws)
    assert not result.ok
    failures = {c.name for c in result.checks if not c.ok}
    assert "adapter framework_dir is workspace-relative" in failures


def test_validate_phase2_rejects_traversal_review_file(tmp_path: Path) -> None:
    """Manifest with phase2_review_file containing '..' must be refused."""
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    (ws / "config.json").write_text(json.dumps({
        "data_path": "data.csv",
        "description": "test",
        "pipeline": {"phases": ["phase2"]},
    }))
    adapter = ws / "adapter"
    adapter.mkdir()
    (adapter / "manifest.json").write_text(json.dumps({
        "domain_name": "x",
        "metric": {"primary_metric": "rmse"},
        "experiment": {"framework_dir": "backtest"},
        "phase2_review_file": "../../../etc/passwd",
    }))
    _write_logs(ws)
    result = validate_workspace(ws)
    assert not result.ok
    failures = {c.name for c in result.checks if not c.ok}
    assert "adapter phase2_review_file is workspace-relative" in failures


def test_validate_phase2_uses_adapter_review_file_name(tmp_path: Path) -> None:
    """Custom adapter with phase2_review_file='audit.md' is honoured."""
    ws = tmp_path / "ws"
    _write_config(ws, phases=["phase2"])
    _write_adapter(ws, framework_dir="backtest", review_file="audit.md")
    _write_phase2(ws, framework_dir="backtest", review_file="audit.md")
    _write_logs(ws)
    result = validate_workspace(ws)
    assert result.ok, [c for c in result.checks if not c.ok]


def test_validate_phase2_no_pass_verdict_fails(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_config(ws, phases=["phase2"])
    _write_adapter(ws)
    _write_phase2(ws, review_verdict="FAIL")
    _write_logs(ws)
    result = validate_workspace(ws)
    assert not result.ok
    failures = {c.name for c in result.checks if not c.ok}
    assert any("verdict is PASS" in f for f in failures)


@pytest.mark.parametrize("verdict_text", [
    "Verdict: PASS",
    "**Verdict**: PASS",
    "verdict:**PASS**",
])
def test_validate_phase2_pass_verdict_variants(tmp_path: Path, verdict_text: str) -> None:
    """The verdict regex accepts the three template variants."""
    ws = tmp_path / "ws"
    _write_config(ws, phases=["phase2"])
    _write_adapter(ws)
    framework = ws / "backtest"
    framework.mkdir(parents=True)
    (framework / "model.py").write_text("# ok\n")
    (framework / "eval.py").write_text("# ok\n")
    (framework / "tests").mkdir()
    (framework / "tests" / "test_x.py").write_text("def test(): pass\n")
    (framework / "review.md").write_text(verdict_text + "\n")
    _write_logs(ws)
    result = validate_workspace(ws)
    assert result.ok, [c for c in result.checks if not c.ok]


def test_validate_phase3_completed_status_done(tmp_path: Path) -> None:
    """Real experiments terminate at 'done' or 'analyzed' (not 'completed')."""
    ws = tmp_path / "ws"
    _write_config(ws, phases=["phase3"])
    _write_adapter(ws)
    _write_phase3(ws, done=2)
    _write_logs(ws)
    result = validate_workspace(ws)
    assert result.ok, [c for c in result.checks if not c.ok]


def test_validate_phase3_completed_status_analyzed(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_config(ws, phases=["phase3"])
    _write_adapter(ws)
    _write_phase3(ws, analyzed=1)
    _write_logs(ws)
    result = validate_workspace(ws)
    assert result.ok, [c for c in result.checks if not c.ok]


def test_validate_phase3_no_terminal_rows_fails(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_config(ws, phases=["phase3"])
    _write_adapter(ws)
    _write_phase3(ws, done=0, analyzed=0, finished=1)  # finished is pre-analysis
    _write_logs(ws)
    result = validate_workspace(ws)
    assert not result.ok
    failures = {c.name for c in result.checks if not c.ok}
    assert "at least one experiment reached done/analyzed" in failures


def test_validate_phase3_in_flight_status_fails(tmp_path: Path) -> None:
    """Even with a successful experiment, an in-flight row is suspicious."""
    ws = tmp_path / "ws"
    _write_config(ws, phases=["phase3"])
    _write_adapter(ws)
    _write_phase3(ws, done=1, running=2)
    _write_logs(ws)
    result = validate_workspace(ws)
    assert not result.ok
    failures = {c.name for c in result.checks if not c.ok}
    assert "no experiments stuck mid-pipeline" in failures


def test_validate_phase3_queued_also_in_flight(tmp_path: Path) -> None:
    """`queued` counts as in-flight too — dispatcher should drain everything."""
    ws = tmp_path / "ws"
    _write_config(ws, phases=["phase3"])
    _write_adapter(ws)
    _write_phase3(ws, done=1, queued=1)
    _write_logs(ws)
    result = validate_workspace(ws)
    assert not result.ok


def test_validate_phase3_finished_also_in_flight(tmp_path: Path) -> None:
    """`finished` is post-execution but pre-analysis — counts as in-flight.

    Regression test: an earlier version of the validator excluded
    ``finished`` from in-flight statuses, so a workspace with a terminal
    ``done`` row plus a stuck ``finished`` row (the analyzer crashed)
    would incorrectly pass.
    """
    ws = tmp_path / "ws"
    _write_config(ws, phases=["phase3"])
    _write_adapter(ws)
    _write_phase3(ws, done=1, finished=1)
    _write_logs(ws)
    result = validate_workspace(ws)
    assert not result.ok
    failures = {c.name for c in result.checks if not c.ok}
    assert "no experiments stuck mid-pipeline" in failures


def test_validate_tool_error_in_logs_fails(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_config(ws, phases=["phase1"])
    _write_adapter(ws)
    _write_phase1(ws)
    _write_logs(
        ws,
        contents=f'{{"output": "{TOOL_ERROR_MARKER} something broke"}}\n',
    )
    result = validate_workspace(ws)
    assert not result.ok
    failures = {c.name for c in result.checks if not c.ok}
    assert f"no {TOOL_ERROR_MARKER} strings in logs" in failures


def test_validate_phases_missing_defaults_to_phase1(tmp_path: Path) -> None:
    """When ``pipeline.phases`` is missing, the validator must mirror
    TaskConfig's default of ``["phase1"]`` so Phase 1 artifacts are still
    required and a missing learnings.md doesn't silently pass.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    # Omit pipeline entirely
    (ws / "config.json").write_text(json.dumps({
        "data_path": "data.csv",
        "description": "test",
    }))
    _write_adapter(ws)
    # No learnings.md / data_report — Phase 1 should still fail
    _write_logs(ws)
    result = validate_workspace(ws)
    assert not result.ok
    failures = {c.name for c in result.checks if not c.ok}
    assert "learnings.md non-empty" in failures


def test_validate_skips_phases_not_configured(tmp_path: Path) -> None:
    """Phase 3 artifacts shouldn't be required when phase3 isn't in the config."""
    ws = tmp_path / "ws"
    _write_config(ws, phases=["phase1"])  # no phase3
    _write_adapter(ws)
    _write_phase1(ws)
    _write_logs(ws)
    # Deliberately omit experiments.db; result should still be ok.
    result = validate_workspace(ws)
    assert result.ok, [c for c in result.checks if not c.ok]
    names = {c.name for c in result.checks}
    assert "experiments.db exists" not in names


def test_validation_result_to_dict_round_trip(tmp_path: Path) -> None:
    """to_dict() output is JSON-serializable for stamping into the manifest."""
    ws = tmp_path / "ws"
    _write_config(ws, phases=["phase1"])
    _write_adapter(ws)
    _write_phase1(ws)
    _write_logs(ws)
    result = validate_workspace(ws)
    payload = result.to_dict()
    decoded = json.loads(json.dumps(payload))
    assert decoded["ok"] is True
    assert decoded["checks"]
    assert all({"name", "ok", "detail"} <= set(c) for c in decoded["checks"])


def test_check_dataclass_defaults() -> None:
    c = Check(name="x", ok=True)
    assert c.detail == ""


def test_exposed_status_constants_match_experiment_db() -> None:
    """Guard against drift between the validator and ExperimentDB's schema."""
    from alpha_lab.experiment_db import KANBAN_COLUMNS

    # Every status the validator knows about must be a real kanban column.
    for status in EXPERIMENT_COMPLETED_STATUSES + EXPERIMENT_IN_FLIGHT_STATUSES:
        assert status in KANBAN_COLUMNS, (
            f"{status!r} not in KANBAN_COLUMNS — validator and ExperimentDB drifted"
        )
