"""Tests for the tool system: schemas, execution, path traversal protection."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from alpha_lab.experiment_db import ExperimentDB
from alpha_lab.tools import (
    TOOL_REGISTRY,
    _resolve_in_workspace,
    _truncate_output,
    execute_shell,
    execute_tool,
    get_tool_schemas,
    grep_files,
    parse_tool_args,
    read_file,
    read_image_base64,
)


# ---------------------------------------------------------------------------
# Path resolution / traversal protection
# ---------------------------------------------------------------------------


class TestResolveInWorkspace:
    """Test the _resolve_in_workspace helper."""

    def test_relative_path_inside(self, tmp_workspace: str) -> None:
        (Path(tmp_workspace) / "file.txt").write_text("hi")
        result = _resolve_in_workspace("file.txt", tmp_workspace)
        assert result is not None
        assert result.name == "file.txt"

    def test_absolute_path_inside(self, tmp_workspace: str) -> None:
        p = Path(tmp_workspace) / "file.txt"
        p.write_text("hi")
        result = _resolve_in_workspace(str(p), tmp_workspace)
        assert result is not None

    def test_path_traversal_rejected(self, tmp_workspace: str) -> None:
        result = _resolve_in_workspace("../../etc/passwd", tmp_workspace)
        assert result is None

    def test_absolute_path_outside_rejected(self, tmp_workspace: str) -> None:
        result = _resolve_in_workspace("/etc/passwd", tmp_workspace)
        assert result is None

    def test_symlink_escape_rejected(self, tmp_workspace: str) -> None:
        link_path = Path(tmp_workspace) / "escape_link"
        try:
            link_path.symlink_to("/etc")
        except OSError:
            pytest.skip("Cannot create symlinks")
        result = _resolve_in_workspace("escape_link/passwd", tmp_workspace)
        assert result is None

    def test_nested_path_inside(self, tmp_workspace: str) -> None:
        subdir = Path(tmp_workspace) / "sub" / "dir"
        subdir.mkdir(parents=True)
        (subdir / "data.csv").write_text("a,b\n1,2")
        result = _resolve_in_workspace("sub/dir/data.csv", tmp_workspace)
        assert result is not None


# ---------------------------------------------------------------------------
# Tool argument parsing
# ---------------------------------------------------------------------------


class TestParseToolArgs:
    def test_valid_json(self) -> None:
        assert parse_tool_args('{"key": "value"}') == {"key": "value"}

    def test_empty_string(self) -> None:
        assert parse_tool_args("") == {}

    def test_invalid_json(self) -> None:
        assert parse_tool_args("not json") == {}

    def test_nested_json(self) -> None:
        args = parse_tool_args('{"config": {"lr": 0.001, "epochs": 10}}')
        assert args["config"]["lr"] == 0.001


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------


class TestTruncateOutput:
    def test_short_text_unchanged(self) -> None:
        text = "short output"
        assert _truncate_output(text) == text

    def test_long_text_truncated(self) -> None:
        text = "x" * 50_000
        result = _truncate_output(text)
        assert len(result) < len(text)
        assert "truncated" in result

    def test_preserves_start_and_end(self) -> None:
        text = "START" + "x" * 50_000 + "END"
        result = _truncate_output(text)
        assert result.startswith("START")
        assert result.endswith("END")


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_read_existing_file(self, tmp_workspace: str) -> None:
        (Path(tmp_workspace) / "hello.txt").write_text("line1\nline2\nline3")
        result = read_file("hello.txt", tmp_workspace)
        assert "line1" in result
        assert "line2" in result
        assert "lines 1-3 of 3" in result

    def test_read_nonexistent_file(self, tmp_workspace: str) -> None:
        result = read_file("nope.txt", tmp_workspace)
        assert "[ERROR]" in result
        assert "not found" in result.lower()

    def test_read_with_offset_and_limit(self, tmp_workspace: str) -> None:
        lines = "\n".join(f"line {i}" for i in range(100))
        (Path(tmp_workspace) / "big.txt").write_text(lines)
        result = read_file("big.txt", tmp_workspace, offset=10, limit=5)
        assert "line 10" in result
        assert "lines 11-15" in result

    def test_read_path_traversal_blocked(self, tmp_workspace: str) -> None:
        result = read_file("../../etc/passwd", tmp_workspace)
        assert "[ERROR]" in result
        assert "outside workspace" in result.lower()

    def test_read_directory_rejected(self, tmp_workspace: str) -> None:
        (Path(tmp_workspace) / "subdir").mkdir()
        result = read_file("subdir", tmp_workspace)
        assert "[ERROR]" in result


# ---------------------------------------------------------------------------
# grep_files
# ---------------------------------------------------------------------------


class TestGrepFiles:
    def test_grep_finds_pattern(self, tmp_workspace: str) -> None:
        (Path(tmp_workspace) / "code.py").write_text("def foo():\n    return 42\n")
        result = grep_files("return 42", tmp_workspace)
        assert "return 42" in result

    def test_grep_no_match(self, tmp_workspace: str) -> None:
        (Path(tmp_workspace) / "code.py").write_text("hello world")
        result = grep_files("zzz_not_found_zzz", tmp_workspace)
        assert "No matches" in result

    def test_grep_with_include(self, tmp_workspace: str) -> None:
        (Path(tmp_workspace) / "code.py").write_text("needle")
        (Path(tmp_workspace) / "data.csv").write_text("needle")
        result = grep_files("needle", tmp_workspace, include="*.py")
        assert "code.py" in result

    def test_grep_path_traversal_blocked(self, tmp_workspace: str) -> None:
        result = grep_files("root", tmp_workspace, path="../../etc")
        assert "[ERROR]" in result


# ---------------------------------------------------------------------------
# execute_shell
# ---------------------------------------------------------------------------


class TestExecuteShell:
    def test_simple_command(self, tmp_workspace: str) -> None:
        result = execute_shell("echo hello", tmp_workspace)
        assert "hello" in result
        assert "[exit code: 0]" in result

    def test_command_stderr(self, tmp_workspace: str) -> None:
        result = execute_shell("echo err >&2", tmp_workspace)
        assert "[stderr]" in result
        assert "err" in result

    def test_command_failure(self, tmp_workspace: str) -> None:
        result = execute_shell("false", tmp_workspace)
        assert "[exit code: 1]" in result

    def test_timeout(self, tmp_workspace: str) -> None:
        result = execute_shell("sleep 10", tmp_workspace, timeout=1)
        assert "[ERROR]" in result
        assert "timed out" in result.lower()

    def test_timeout_clamped(self, tmp_workspace: str) -> None:
        # timeout=0 should be clamped to 1
        result = execute_shell("echo fast", tmp_workspace, timeout=0)
        assert "fast" in result

    def test_cwd_is_workspace(self, tmp_workspace: str) -> None:
        result = execute_shell("pwd", tmp_workspace)
        # The resolved path might differ from tmp_workspace if there are symlinks,
        # but the output should contain the workspace directory name
        assert Path(tmp_workspace).name in result

    def test_timeout_kills_child_process_group(self, tmp_workspace: str) -> None:
        """Spawn a shell that spawns a child `sleep`; on timeout both must be reaped.

        Regression guard for the `start_new_session=True` + `os.killpg()` path:
        without it the child `sleep` would outlive the parent shell.
        """
        import os
        import signal
        import time

        pid_file = Path(tmp_workspace) / "child.pid"
        # Parent shell spawns a detached child `sleep` and records its PID.
        # Both the shell and the sleep share the process group started by
        # start_new_session=True; killpg(SIGKILL) must take both down.
        cmd = (
            f"sleep 30 & echo $! > {pid_file}; "
            f"wait"
        )
        result = execute_shell(cmd, tmp_workspace, timeout=1)
        assert "[ERROR]" in result
        assert "timed out" in result.lower()
        assert pid_file.exists(), "child never started; test setup is wrong"
        child_pid = int(pid_file.read_text().strip())

        def _alive_and_not_zombie(pid: int) -> bool:
            # os.kill(pid, 0) also succeeds for zombies — which would make
            # this test flaky if the child is reaped a moment late. Read
            # /proc/<pid>/status and exclude the Z state so zombies count
            # as dead for this test.
            try:
                with open(f"/proc/{pid}/status") as f:
                    for line in f:
                        if line.startswith("State:"):
                            return "Z" not in line.split(None, 1)[1]
                return True
            except (FileNotFoundError, ProcessLookupError):
                return False

        for _ in range(20):
            if not _alive_and_not_zombie(child_pid):
                return  # child is gone (or zombie) — group kill worked
            time.sleep(0.05)
        # If we fell out of the loop, the child is still alive — clean up
        # so we don't leak a 30s sleep, then fail the test.
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise AssertionError(
            f"child pid {child_pid} still alive after timeout — process group not killed"
        )


class TestExecuteToolShellTimeout:
    """Regression tests for shell_timeout normalization in execute_tool."""

    def test_shell_timeout_none_falls_back_to_default(self, tmp_workspace: str) -> None:
        """shell_timeout=None (e.g., from JSON with a missing field) must not crash."""
        result = execute_tool(
            "shell_exec",
            {"command": "echo ok"},
            workspace=tmp_workspace,
            shell_timeout=None,  # type: ignore[arg-type]
        )
        assert "ok" in result["output"]
        assert "[ERROR]" not in result["output"]

    def test_shell_timeout_numeric_string_is_coerced(self, tmp_workspace: str) -> None:
        """shell_timeout as a numeric string (e.g., YAML-loaded) is coerced to int."""
        result = execute_tool(
            "shell_exec",
            {"command": "echo ok"},
            workspace=tmp_workspace,
            shell_timeout="300",  # type: ignore[arg-type]
        )
        assert "ok" in result["output"]
        assert "[ERROR]" not in result["output"]

    def test_shell_timeout_non_numeric_string_falls_back_to_default(
        self, tmp_workspace: str
    ) -> None:
        """A non-numeric string falls back to the default (not a crash)."""
        result = execute_tool(
            "shell_exec",
            {"command": "echo ok"},
            workspace=tmp_workspace,
            shell_timeout="not-a-number",  # type: ignore[arg-type]
        )
        assert "ok" in result["output"]
        assert "[ERROR]" not in result["output"]

    def test_shell_timeout_zero_falls_back_to_default(self, tmp_workspace: str) -> None:
        """shell_timeout=0 or negative must fall back to the default, not clamp to 1s."""
        result = execute_tool(
            "shell_exec",
            {"command": "echo ok"},
            workspace=tmp_workspace,
            shell_timeout=0,
        )
        assert "ok" in result["output"]
        assert "[ERROR]" not in result["output"]


# ---------------------------------------------------------------------------
# view_image / read_image_base64
# ---------------------------------------------------------------------------


class TestReadImageBase64:
    def test_read_png(self, tmp_workspace: str) -> None:
        # Write a minimal valid PNG (1x1 pixel)
        import base64
        # Minimal PNG header
        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        (Path(tmp_workspace) / "test.png").write_bytes(png_data)
        b64, media = read_image_base64("test.png", tmp_workspace)
        assert media == "image/png"
        assert len(b64) > 0

    def test_read_unsupported_format(self, tmp_workspace: str) -> None:
        (Path(tmp_workspace) / "test.bmp").write_bytes(b"BM")
        with pytest.raises(ValueError, match="Unsupported"):
            read_image_base64("test.bmp", tmp_workspace)

    def test_read_nonexistent_image(self, tmp_workspace: str) -> None:
        with pytest.raises(FileNotFoundError):
            read_image_base64("missing.png", tmp_workspace)

    def test_path_traversal_blocked(self, tmp_workspace: str) -> None:
        with pytest.raises(ValueError, match="outside workspace"):
            read_image_base64("../../etc/passwd", tmp_workspace)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_all_tools_have_required_fields(self) -> None:
        for name, schema in TOOL_REGISTRY.items():
            assert schema["type"] == "function", f"{name} missing type"
            assert schema["name"] == name, f"{name} name mismatch"
            assert "parameters" in schema, f"{name} missing parameters"
            assert "description" in schema, f"{name} missing description"

    def test_get_tool_schemas_subset(self) -> None:
        schemas = get_tool_schemas(["shell_exec", "read_file"])
        assert len(schemas) == 2
        names = {s["name"] for s in schemas}
        assert names == {"shell_exec", "read_file"}

    def test_get_tool_schemas_with_web_search(self) -> None:
        schemas = get_tool_schemas(["shell_exec"], include_web_search=True)
        types = {s.get("type") for s in schemas}
        assert "web_search_preview" in types

    def test_get_tool_schemas_unknown_ignored(self) -> None:
        schemas = get_tool_schemas(["shell_exec", "nonexistent_tool"])
        assert len(schemas) == 1


# ---------------------------------------------------------------------------
# execute_tool dispatch
# ---------------------------------------------------------------------------


class TestExecuteTool:
    def test_shell_exec(self, tmp_workspace: str) -> None:
        result = execute_tool("shell_exec", {"command": "echo hi"}, tmp_workspace)
        assert "hi" in result["output"]

    def test_report_to_user(self, tmp_workspace: str) -> None:
        result = execute_tool("report_to_user", {"summary": "All done"}, tmp_workspace)
        assert result["done"] is True
        assert result["summary"] == "All done"

    def test_ask_user_with_fn(self, tmp_workspace: str) -> None:
        fn = lambda q: "user answer"
        result = execute_tool("ask_user", {"question": "Q?"}, tmp_workspace, ask_user_fn=fn)
        assert result["output"] == "user answer"

    def test_ask_user_without_fn(self, tmp_workspace: str) -> None:
        result = execute_tool("ask_user", {"question": "Q?"}, tmp_workspace)
        assert "[ERROR]" in result["output"]

    def test_unknown_tool(self, tmp_workspace: str) -> None:
        result = execute_tool("nonexistent_tool", {}, tmp_workspace)
        assert "[ERROR]" in result["output"]
        assert "Unknown tool" in result["output"]

    def test_read_file_tool(self, tmp_workspace: str) -> None:
        (Path(tmp_workspace) / "data.txt").write_text("content here")
        result = execute_tool("read_file", {"path": "data.txt"}, tmp_workspace)
        assert "content here" in result["output"]

    def test_grep_file_tool(self, tmp_workspace: str) -> None:
        (Path(tmp_workspace) / "code.py").write_text("import pandas")
        result = execute_tool("grep_file", {"pattern": "pandas"}, tmp_workspace)
        assert "pandas" in result["output"]


# ---------------------------------------------------------------------------
# Phase 3 tools
# ---------------------------------------------------------------------------


class TestPhase3Tools:
    @pytest.fixture()
    def db(self, tmp_path: Path) -> ExperimentDB:
        return ExperimentDB(str(tmp_path / "test.db"))

    def test_propose_experiment(self, tmp_workspace: str, db: ExperimentDB) -> None:
        result = execute_tool(
            "propose_experiment",
            {
                "name": "test_xgboost",
                "description": "XGBoost baseline",
                "hypothesis": "Trees work",
                "config": '{"model": "xgboost"}',
            },
            tmp_workspace,
            db=db,
        )
        assert "created" in result["output"].lower()
        assert "test_xgboost" in result["output"]

    def test_propose_refused_at_lifetime_budget(
            self, tmp_workspace: str, db: ExperimentDB) -> None:
        """The turn-entry gate can't stop one in-flight turn from proposing
        past max_experiments (one glm run proposed 72 rows past the prompt
        banner); the tool itself must refuse at the cap."""
        from types import SimpleNamespace
        db._task_config = SimpleNamespace(  # type: ignore[attr-defined]
            pipeline=SimpleNamespace(
                phase3=SimpleNamespace(max_experiments=2)))
        for i in range(2):
            r = execute_tool(
                "propose_experiment",
                {"name": f"e{i}", "description": "d", "hypothesis": "h",
                 "config": "{}"},
                tmp_workspace, db=db)
            assert "created" in r["output"].lower()
        r = execute_tool(
            "propose_experiment",
            {"name": "over_cap", "description": "d", "hypothesis": "h",
             "config": "{}"},
            tmp_workspace, db=db)
        assert "[REFUSED]" in r["output"]
        assert db.get(3) is None
        # propose_variant is a new row too — same refusal
        r = execute_tool(
            "propose_variant",
            {"base_experiment_id": 1, "name": "e0_v2", "intent": "x"},
            tmp_workspace, db=db)
        assert "[REFUSED]" in r["output"]

    def test_propose_experiment_sanitizes_name(self, tmp_workspace: str, db: ExperimentDB) -> None:
        result = execute_tool(
            "propose_experiment",
            {
                "name": "../../evil/path.sh",
                "description": "D",
                "hypothesis": "H",
                "config": "{}",
            },
            tmp_workspace,
            db=db,
        )
        assert "created" in result["output"].lower()
        # Slashes and dots should be replaced
        exp = db.get(1)
        assert "/" not in exp.name
        assert ".." not in exp.name

    def test_propose_experiment_no_db(self, tmp_workspace: str) -> None:
        result = execute_tool(
            "propose_experiment",
            {"name": "x", "description": "D", "hypothesis": "H", "config": "{}"},
            tmp_workspace,
            db=None,
        )
        assert "[ERROR]" in result["output"]

    # ------------------------------------------------------------------
    # propose_variant — directory-copy spawn of a child experiment.
    # ------------------------------------------------------------------

    def _seed_base_experiment(self, tmp_workspace: str, db: ExperimentDB) -> int:
        """Helper: create a base experiment row + on-disk directory with
        the canonical artifacts the variant tool expects to copy."""
        base_id = db.create("base_xgb", "XGB baseline", "Trees work", "{}")
        base_dir = Path(tmp_workspace) / "experiments" / "base_xgb"
        base_dir.mkdir(parents=True)
        (base_dir / "strategy.py").write_text("# base strategy\n")
        (base_dir / "run_experiment.py").write_text("# entry point\n")
        (base_dir / "config.yaml").write_text("hidden: 64\nlr: 0.001\n")
        # Outputs that should NOT be copied to the variant dir
        (base_dir / "results").mkdir()
        (base_dir / "results" / "metrics.json").write_text('{"sharpe": 1.2}')
        (base_dir / "logs").mkdir()
        (base_dir / "logs" / "stdout.log").write_text("base run output\n")
        (base_dir / "__pycache__").mkdir()
        (base_dir / "__pycache__" / "strategy.cpython-312.pyc").write_text("bytecode")
        return base_id

    def test_propose_variant_copies_code_excludes_outputs(
        self, tmp_workspace: str, db: ExperimentDB
    ) -> None:
        base_id = self._seed_base_experiment(tmp_workspace, db)
        result = execute_tool(
            "propose_variant",
            {
                "base_experiment_id": base_id,
                "name": "base_xgb_hidden_128",
                "hypothesis": "Larger hidden should help #" + str(base_id),
                "what_changes": "change hidden from 64 to 128 in config.yaml",
            },
            tmp_workspace,
            db=db,
        )
        assert "[ERROR]" not in result["output"]
        # New row created with parent_id pointing at the base
        var = db.get(2)
        assert var is not None
        assert var.parent_id == base_id
        assert var.name == "base_xgb_hidden_128"
        assert var.status == "to_implement"
        # Directory copied: code present, outputs excluded
        var_dir = Path(tmp_workspace) / "experiments" / "base_xgb_hidden_128"
        assert (var_dir / "strategy.py").exists()
        assert (var_dir / "run_experiment.py").exists()
        assert (var_dir / "config.yaml").exists()
        assert not (var_dir / "results").exists()
        assert not (var_dir / "logs").exists()
        assert not (var_dir / "__pycache__").exists()
        # .variant_intent.md mentions the base and the change
        intent = (var_dir / ".variant_intent.md").read_text()
        assert "hidden from 64 to 128" in intent
        assert f"#{base_id}" in intent
        assert "base_xgb" in intent

    def test_propose_variant_excludes_parent_output_artifacts(
        self, tmp_workspace: str, db: ExperimentDB
    ) -> None:
        """A fresh variant must NOT inherit the parent's OUTPUT artifacts
        (debrief.md, run_status.json, analysis/, local_job*.out) — otherwise
        it looks already-run/already-analyzed and the parent's findings get
        misattributed. The INPUTS (code, config) must still copy."""
        base_id = self._seed_base_experiment(tmp_workspace, db)
        base_dir = Path(tmp_workspace) / "experiments" / "base_xgb"
        # Seed the parent's OUTPUT artifacts (the A1 bug copied these).
        (base_dir / "debrief.md").write_text("# Debrief — Experiment base_xgb\n")
        (base_dir / "run_status.json").write_text('{"exp_name": "base_xgb", "status": "COMPLETED"}')
        (base_dir / "local_job.abcd1234.out").write_text("base stdout\n")
        (base_dir / "local_job.out").write_text("symlinkish\n")
        (base_dir / "analysis").mkdir()
        (base_dir / "analysis" / "residuals.py").write_text("# base analysis\n")

        result = execute_tool(
            "propose_variant",
            {
                "base_experiment_id": base_id,
                "name": "base_xgb_v2",
                "hypothesis": "tweak #" + str(base_id),
                "what_changes": "change lr",
            },
            tmp_workspace,
            db=db,
        )
        assert "[ERROR]" not in result["output"]
        vd = Path(tmp_workspace) / "experiments" / "base_xgb_v2"
        # INPUTS copied
        assert (vd / "strategy.py").exists()
        assert (vd / "run_experiment.py").exists()
        assert (vd / "config.yaml").exists()
        # OUTPUTS excluded — the A1 fix
        assert not (vd / "debrief.md").exists(), "must not inherit parent debrief"
        assert not (vd / "run_status.json").exists(), "must not inherit parent run_status"
        assert not (vd / "analysis").exists(), "must not inherit parent analysis/"
        assert not (vd / "local_job.abcd1234.out").exists(), "must not inherit parent job output"
        assert not (vd / "local_job.out").exists()

    def test_propose_variant_rejects_missing_base(
        self, tmp_workspace: str, db: ExperimentDB
    ) -> None:
        result = execute_tool(
            "propose_variant",
            {
                "base_experiment_id": 999,
                "name": "v",
                "hypothesis": "h",
                "what_changes": "x",
            },
            tmp_workspace,
            db=db,
        )
        assert "[ERROR]" in result["output"]
        assert "not found" in result["output"].lower()

    def test_propose_variant_rejects_missing_base_directory(
        self, tmp_workspace: str, db: ExperimentDB
    ) -> None:
        # Row exists in DB but no on-disk dir — variant tool refuses.
        base_id = db.create("phantom", "D", "H", "{}")
        result = execute_tool(
            "propose_variant",
            {
                "base_experiment_id": base_id,
                "name": "phantom_v",
                "hypothesis": "h",
                "what_changes": "x",
            },
            tmp_workspace,
            db=db,
        )
        assert "[ERROR]" in result["output"]
        assert "experiments/phantom" in result["output"]

    def test_propose_variant_rejects_name_collision(
        self, tmp_workspace: str, db: ExperimentDB
    ) -> None:
        base_id = self._seed_base_experiment(tmp_workspace, db)
        # Pre-create a directory at the variant's target path
        (Path(tmp_workspace) / "experiments" / "base_xgb_v2").mkdir()
        result = execute_tool(
            "propose_variant",
            {
                "base_experiment_id": base_id,
                "name": "base_xgb_v2",
                "hypothesis": "h",
                "what_changes": "x",
            },
            tmp_workspace,
            db=db,
        )
        assert "[ERROR]" in result["output"]
        assert "already exists" in result["output"]

    def test_propose_variant_rejects_same_name_as_base(
        self, tmp_workspace: str, db: ExperimentDB
    ) -> None:
        base_id = self._seed_base_experiment(tmp_workspace, db)
        result = execute_tool(
            "propose_variant",
            {
                "base_experiment_id": base_id,
                "name": "base_xgb",
                "hypothesis": "h",
                "what_changes": "x",
            },
            tmp_workspace,
            db=db,
        )
        assert "[ERROR]" in result["output"]

    def test_propose_variant_honors_fan_out_cap(
        self, tmp_workspace: str, db: ExperimentDB
    ) -> None:
        # Attach a TaskConfig-like object with max_variants_per_base=2 so
        # the cap fires after 2 variants exist.
        class _P3:
            max_variants_per_base = 2

        class _Pipeline:
            phase3 = _P3()

        class _Cfg:
            pipeline = _Pipeline()

        db._task_config = _Cfg()  # type: ignore[attr-defined]

        base_id = self._seed_base_experiment(tmp_workspace, db)
        for i in range(2):
            result = execute_tool(
                "propose_variant",
                {
                    "base_experiment_id": base_id,
                    "name": f"base_xgb_v_{i}",
                    "hypothesis": "h",
                    "what_changes": f"change {i}",
                },
                tmp_workspace,
                db=db,
            )
            assert "[ERROR]" not in result["output"]
        # Third should be refused.
        result = execute_tool(
            "propose_variant",
            {
                "base_experiment_id": base_id,
                "name": "base_xgb_v_3",
                "hypothesis": "h",
                "what_changes": "change 3",
            },
            tmp_workspace,
            db=db,
        )
        assert "[ERROR]" in result["output"]
        assert "cap" in result["output"].lower()

    def test_update_playbook(self, tmp_workspace: str) -> None:
        result = execute_tool(
            "update_playbook",
            {"content": "# Playbook\n\n## What works\n- LSTMs"},
            tmp_workspace,
        )
        assert "updated" in result["output"].lower()
        content = (Path(tmp_workspace) / "playbook.md").read_text()
        assert "LSTMs" in content

    def test_update_playbook_preserves_analyzer_appends_below_sentinel(
        self, tmp_workspace: str
    ) -> None:
        """The strategist's overwrite must NOT lose analyzer appends that
        landed since its last consolidation. The tool reads the current
        file just before writing and preserves anything below the
        ANALYZER-APPENDS-BELOW sentinel."""
        playbook = Path(tmp_workspace) / "playbook.md"
        # First strategist write creates the sentinel.
        execute_tool(
            "update_playbook",
            {"content": "# Playbook v1\n- rule 1"},
            tmp_workspace,
        )
        content = playbook.read_text()
        assert "ANALYZER-APPENDS-BELOW" in content
        # Analyzer appends a note via O_APPEND-style shell write (simulated).
        with playbook.open("a") as f:
            f.write("\nNOTE FROM ANALYZER: avoid X without clipping\n")
        # Strategist overwrites with v2; the analyzer note must survive.
        execute_tool(
            "update_playbook",
            {"content": "# Playbook v2\n- rule 1\n- rule 2"},
            tmp_workspace,
        )
        new_content = playbook.read_text()
        assert "Playbook v2" in new_content
        assert "rule 2" in new_content
        assert "NOTE FROM ANALYZER" in new_content
        # The new sentinel is still present so future appends keep working.
        assert "ANALYZER-APPENDS-BELOW" in new_content

    def test_update_playbook_strips_sentinel_from_incoming_content(
        self, tmp_workspace: str
    ) -> None:
        """Regression: the strategist commonly reads playbook.md and writes
        the whole thing back as ``content`` — including the sentinel. The
        previous implementation re-added the sentinel each time, so it grew
        by one per write (observed: 15 sentinels after 14 strategist writes
        in workspace_etfflow_grok). The tool must strip the embedded
        sentinel from incoming content before appending its own.
        """
        playbook = Path(tmp_workspace) / "playbook.md"
        SENT_MARK = "ANALYZER-APPENDS-BELOW"
        # Three writes in a row, each with embedded sentinel + trailing junk
        # (simulating "strategist read the file and pasted it back").
        for i in range(3):
            current = playbook.read_text() if playbook.exists() else ""
            # Strategist's content is the existing file + a new rule.
            new_body = current + f"\n- new rule v{i}"
            execute_tool(
                "update_playbook",
                {"content": new_body},
                tmp_workspace,
            )
            n = playbook.read_text().count(SENT_MARK)
            assert n == 1, f"after write {i}: expected 1 sentinel, got {n}"

    def test_update_playbook_collapses_legacy_sentinel_variant(
        self, tmp_workspace: str
    ) -> None:
        """Regression (A4): a playbook carrying a LEGACY sentinel variant
        whose text differs from the canonical string
        ('<!-- ANALYZER-APPENDS-BELOW (do not delete this line) -->') used to
        survive forever because the dedup keyed on the exact canonical string.
        The regex-based dedup must collapse ALL variants to a single canonical
        sentinel, while preserving analyzer appends below.
        """
        playbook = Path(tmp_workspace) / "playbook.md"
        legacy = "<!-- ANALYZER-APPENDS-BELOW (do not delete this line) -->"
        canonical = (
            "<!-- ANALYZER-APPENDS-BELOW (handled by update_playbook tool; "
            "appends below this line are preserved across strategist writes "
            "and folded into the main body on the next consolidation) -->"
        )
        # Seed a playbook with BOTH sentinel variants + a real analyzer append.
        playbook.write_text(
            "# Playbook v0\n- old rule\n\n"
            f"{legacy}\n\n{canonical}\n\nNOTE FROM ANALYZER: keep clipping\n"
        )
        # Strategist edits the BODY (above the sentinels) and pastes the whole
        # file back — the realistic flow. New rule goes in the body; the
        # sentinels + analyzer note are pasted verbatim after it.
        strategist_content = (
            "# Playbook v0\n- old rule\n- new rule\n\n"
            f"{legacy}\n\n{canonical}\n\nNOTE FROM ANALYZER: keep clipping\n"
        )
        execute_tool(
            "update_playbook",
            {"content": strategist_content},
            tmp_workspace,
        )
        out = playbook.read_text()
        # Exactly ONE sentinel survives (the legacy variant collapsed away)...
        assert out.count("ANALYZER-APPENDS-BELOW") == 1, out
        assert "(do not delete this line)" not in out, "legacy variant must be gone"
        # ...the analyzer append below is preserved...
        assert "NOTE FROM ANALYZER: keep clipping" in out
        # ...and the new body is there.
        assert "new rule" in out

    def test_update_playbook_clears_below_sentinel_when_strategist_consolidates(
        self, tmp_workspace: str
    ) -> None:
        """If the strategist's body already absorbs prior analyzer appends
        and the file no longer has any below-sentinel content, the new
        write should leave only the sentinel + empty area below."""
        playbook = Path(tmp_workspace) / "playbook.md"
        # First write: creates sentinel, no appends below.
        execute_tool(
            "update_playbook",
            {"content": "# Playbook v1\n- rule 1"},
            tmp_workspace,
        )
        # Second write without any appends below — file should have only the
        # new body + sentinel.
        execute_tool(
            "update_playbook",
            {"content": "# Playbook v2\n- consolidated"},
            tmp_workspace,
        )
        content = playbook.read_text()
        # Partition at the closing `-->` of the sentinel comment so we can
        # check that nothing meaningful follows it.
        before, sep, after = content.partition("-->")
        assert sep == "-->"
        assert "Playbook v2" in before
        assert "consolidated" in before
        assert after.strip() == ""

    def test_update_playbook_backs_up_prior_version(
        self, tmp_workspace: str
    ) -> None:
        """An LLM error that overwrites playbook.md with garbage must
        not destroy accumulated strategist context — the prior version
        is backed up to ``meta/backups/playbook_<ts>.md`` before write."""
        from alpha_lab.meta_layout import backups_dir
        # Seed an initial playbook
        Path(tmp_workspace, "playbook.md").write_text(
            "# Original playbook\n- valuable accumulated insight"
        )
        # Overwrite via tool
        execute_tool(
            "update_playbook",
            {"content": "# New playbook"},
            tmp_workspace,
        )
        # New content is in place
        assert "New playbook" in (Path(tmp_workspace) / "playbook.md").read_text()
        # Backup exists with the original content
        backup_files = list(backups_dir(tmp_workspace).glob("playbook_*.md"))
        assert len(backup_files) == 1, f"expected 1 backup, got {backup_files}"
        assert "valuable accumulated insight" in backup_files[0].read_text()

    def test_update_playbook_first_time_no_backup(
        self, tmp_workspace: str
    ) -> None:
        """First write (no existing playbook) skips the backup step
        without erroring."""
        from alpha_lab.meta_layout import backups_dir
        # No prior playbook.md exists
        assert not (Path(tmp_workspace) / "playbook.md").exists()
        execute_tool(
            "update_playbook",
            {"content": "# Initial playbook"},
            tmp_workspace,
        )
        assert (Path(tmp_workspace) / "playbook.md").exists()
        # No backup created (nothing to back up)
        backup_files = list(backups_dir(tmp_workspace).glob("playbook_*.md"))
        assert backup_files == []

    def test_read_board(self, tmp_workspace: str, db: ExperimentDB) -> None:
        db.create("exp_a", "D", "H", "{}")
        result = execute_tool("read_board", {}, tmp_workspace, db=db)
        assert "Board Summary" in result["output"]
        assert "exp_a" in result["output"]

    def test_read_board_no_db(self, tmp_workspace: str) -> None:
        result = execute_tool("read_board", {}, tmp_workspace, db=None)
        assert "[ERROR]" in result["output"]

    def test_update_experiment_status(self, tmp_workspace: str, db: ExperimentDB) -> None:
        exp_id = db.create("upd_exp", "D", "H", "{}")
        result = execute_tool(
            "update_experiment",
            {"experiment_id": exp_id, "status": "implemented"},
            tmp_workspace,
            db=db,
        )
        assert "updated" in result["output"].lower()
        assert db.get(exp_id).status == "implemented"

    def test_update_experiment_with_results(self, tmp_workspace: str, db: ExperimentDB) -> None:
        exp_id = db.create("res_exp", "D", "H", "{}")
        result = execute_tool(
            "update_experiment",
            {"experiment_id": exp_id, "results": '{"sharpe": 1.2}'},
            tmp_workspace,
            db=db,
        )
        assert "results set" in result["output"]
        assert db.get(exp_id).results_json == '{"sharpe": 1.2}'

    def test_update_experiment_not_found(self, tmp_workspace: str, db: ExperimentDB) -> None:
        result = execute_tool(
            "update_experiment",
            {"experiment_id": 9999, "status": "done"},
            tmp_workspace,
            db=db,
        )
        assert "[ERROR]" in result["output"]
        assert "not found" in result["output"].lower()

    def test_update_experiment_no_db(self, tmp_workspace: str) -> None:
        result = execute_tool(
            "update_experiment",
            {"experiment_id": 1},
            tmp_workspace,
            db=None,
        )
        assert "[ERROR]" in result["output"]

    def test_update_experiment_explicit_clear_error(
        self, tmp_workspace: str, db: ExperimentDB,
    ) -> None:
        """Passing ``error=""`` clears a stale error string. Without this
        a failed-then-recovered experiment would show as a red row in the
        GUI even though the run succeeded."""
        exp_id = db.create("recovered", "D", "H", "{}")
        db.set_error(exp_id, "first attempt failed")
        assert db.get(exp_id).error == "first attempt failed"
        result = execute_tool(
            "update_experiment",
            {"experiment_id": exp_id, "error": ""},
            tmp_workspace,
            db=db,
        )
        assert "error cleared" in result["output"]
        assert (db.get(exp_id).error or "") == ""

    def test_update_experiment_auto_clears_error_on_success_transition(
        self, tmp_workspace: str, db: ExperimentDB,
    ) -> None:
        """A transition to ``analyzed``/``done`` carrying a results payload
        is the worker asserting success — the stale error from an earlier
        failure should be cleared automatically so the GUI reflects truth."""
        exp_id = db.create("recovered2", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        db.update_status(exp_id, "checked")
        db.update_status(exp_id, "queued")
        db.update_status(exp_id, "running")
        db.update_status(exp_id, "finished")
        db.set_error(exp_id, "earlier SLURM FAILED")
        result = execute_tool(
            "update_experiment",
            {
                "experiment_id": exp_id,
                "status": "analyzed",
                "results": '{"sharpe": 1.5}',
            },
            tmp_workspace,
            db=db,
        )
        assert "auto-cleared" in result["output"]
        row = db.get(exp_id)
        assert (row.error or "") == ""
        assert row.status == "analyzed"

    def test_update_experiment_preserves_intentional_error(
        self, tmp_workspace: str, db: ExperimentDB,
    ) -> None:
        """If the worker explicitly passes a new error string in the same
        call, the auto-clear must not fire — the worker's explicit
        intent wins."""
        exp_id = db.create("intentional", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        db.update_status(exp_id, "checked")
        db.update_status(exp_id, "queued")
        db.update_status(exp_id, "running")
        db.update_status(exp_id, "finished")
        db.set_error(exp_id, "old failure")
        result = execute_tool(
            "update_experiment",
            {
                "experiment_id": exp_id,
                "status": "analyzed",
                "results": '{"sharpe": 0.1}',
                "error": "new analysis failure",
            },
            tmp_workspace,
            db=db,
        )
        assert "error set" in result["output"]
        assert db.get(exp_id).error == "new analysis failure"


# ---------------------------------------------------------------------------
# Conductor tools — dispatch tests. These exercise the full execute_tool
# entry points, complementing the unit tests in test_conductor_tools.py
# (which test the underlying helpers directly).
# ---------------------------------------------------------------------------


class TestConductorToolsDispatch:
    @pytest.fixture()
    def db(self, tmp_path: Path) -> ExperimentDB:
        d = ExperimentDB(str(tmp_path / "c.db"))
        d.create("first", "Desc", "H", '{"k": "v"}')
        d.create("second", "Desc 2", "H", "{}")
        return d

    @pytest.fixture()
    def workspace(self, tmp_path: Path) -> str:
        return str(tmp_path)

    def test_park_then_unpark_via_dispatch(self, workspace: str, db: ExperimentDB) -> None:
        r = execute_tool(
            "park_experiment",
            {"experiment_id": 1, "reason": "saturated", "evidence": "leaderboard plateau"},
            workspace,
            db=db,
        )
        assert "parked #1" in r["output"]
        assert db.get(1).parked_at is not None
        # Excluded from active queue
        assert [e.id for e in db.list_by_status("to_implement")] == [2]
        # Audit log written
        from alpha_lab import meta_layout as ml
        log = ml.meta_log_jsonl_path(workspace).read_text()
        assert "park" in log
        # Unpark restores
        r = execute_tool(
            "unpark_experiment",
            {"experiment_id": 1, "reason": "reconsidered", "evidence": "new note from strategist"},
            workspace,
            db=db,
        )
        assert "unparked #1" in r["output"]
        assert db.get(1).parked_at is None
        assert {e.id for e in db.list_by_status("to_implement")} == {1, 2}

    def test_park_unknown_id_errors_cleanly(self, workspace: str, db: ExperimentDB) -> None:
        r = execute_tool(
            "park_experiment",
            {"experiment_id": 9999, "reason": "r", "evidence": "e"},
            workspace,
            db=db,
        )
        assert "[ERROR]" in r["output"]

    def test_set_priority_via_dispatch(self, workspace: str, db: ExperimentDB) -> None:
        r = execute_tool(
            "set_priority",
            {"experiment_id": 2, "priority": 10, "reason": "promote", "evidence": "user request"},
            workspace,
            db=db,
        )
        assert "set priority of #2 to 10" in r["output"]
        # Surfaces above #1 in queue
        assert [e.id for e in db.list_by_status("to_implement")] == [2, 1]

    def test_annotate_via_dispatch(self, workspace: str, db: ExperimentDB) -> None:
        r = execute_tool(
            "annotate_experiment",
            {"experiment_id": 1, "label": "champion", "reason": "best so far", "evidence": "P@5=0.18"},
            workspace,
            db=db,
        )
        assert "champion" in r["output"]
        from alpha_lab import conductor_tools as ct
        assert ct.read_annotations(workspace).get("1") == "champion"

    def test_annotate_clear(self, workspace: str, db: ExperimentDB) -> None:
        from alpha_lab import conductor_tools as ct
        ct.set_annotation(workspace, 1, "champion")
        r = execute_tool(
            "annotate_experiment",
            {"experiment_id": 1, "label": "", "reason": "demoted", "evidence": "new champion arrived"},
            workspace,
            db=db,
        )
        assert "cleared" in r["output"]
        assert ct.read_annotations(workspace) == {}

    def test_issue_directive_writes_to_directives_md(
        self, workspace: str, db: ExperimentDB
    ) -> None:
        r = execute_tool(
            "issue_directive",
            {
                "target_role": "strategist",
                "message": "diversify into sequential models",
                "reason": "leaderboard saturated on lambdarank",
                "evidence": "milestone 43 plateau",
            },
            workspace,
            db=db,
        )
        # New output includes the auto-generated directive id and scope.
        assert "recorded for strategist" in r["output"]
        assert "scope=standing" in r["output"]
        from alpha_lab import meta_layout as ml
        text = ml.directives_path(workspace).read_text()
        assert "sequential models" in text
        assert "scope=standing" in text

    def test_issue_directive_with_one_shot_scope(
        self, workspace: str, db: ExperimentDB
    ) -> None:
        r = execute_tool(
            "issue_directive",
            {
                "target_role": "strategist",
                "message": "propose 3 exploration experiments",
                "reason": "covered base, want breadth",
                "evidence": "ack",
                "scope": "one-shot",
            },
            workspace,
            db=db,
        )
        assert "scope=one-shot" in r["output"]
        from alpha_lab import meta_layout as ml
        assert "scope=one-shot" in ml.directives_path(workspace).read_text()

    def test_ack_directive_writes_jsonl(
        self, workspace: str, db: ExperimentDB
    ) -> None:
        from alpha_lab import conductor_tools as ct
        did = ct.append_directive(
            workspace, "strategist", "do thing", scope="one-shot",
        )
        r = execute_tool(
            "ack_directive",
            {"directive_id": did, "action_taken": "did the thing"},
            workspace,
            db=db,
            caller_role="strategist",
            caller_id="strategist",
        )
        assert "ack recorded" in r["output"]
        acks = ct.read_directive_acks(workspace)
        assert len(acks) == 1
        assert acks[0]["directive_id"] == did
        assert acks[0]["actor_role"] == "strategist"
        # filtering should now remove it for strategist role
        active = ct.directives_for_role(workspace, "strategist")
        assert not any(d["id"] == did for d in active)

    def test_write_note_to_user(self, workspace: str) -> None:
        r = execute_tool(
            "write_note_to_user",
            {"message": "Plateau observed at P@5=0.183"},
            workspace,
        )
        assert "appended" in r["output"]
        from alpha_lab import meta_layout as ml
        assert "Plateau" in ml.notes_to_user_path(workspace).read_text()

    def test_set_throttle(self, workspace: str) -> None:
        r = execute_tool(
            "set_throttle",
            {"gpu": "slow", "reason": "GPU oversubscribed", "evidence": "load=12 on 4 cores"},
            workspace,
        )
        assert "gpu=slow" in r["output"]
        from alpha_lab import meta_layout as ml
        assert ml.read_throttle(workspace)["gpu"] == "slow"

    def test_read_meta_log_dispatch(self, workspace: str, db: ExperimentDB) -> None:
        # Seed a few entries via park
        for eid in (1, 2):
            execute_tool(
                "park_experiment",
                {"experiment_id": eid, "reason": "r", "evidence": "e"},
                workspace,
                db=db,
            )
        r = execute_tool("read_meta_log", {"last_n": 5}, workspace, db=db)
        # Output is JSON of entries
        import json
        entries = json.loads(r["output"])
        assert len(entries) == 2
        assert entries[0]["decision_type"] == "park"

    def test_read_user_instructions_dispatch(self, workspace: str) -> None:
        from alpha_lab import meta_layout as ml
        ml.ensure_meta_layout(workspace)
        ml.from_user_path(workspace).write_text("please prioritize cold-client experiments")
        r = execute_tool(
            "read_user_instructions",
            {"mark_seen": True},
            workspace,
        )
        import json
        payload = json.loads(r["output"])
        assert payload["is_new"] is True
        assert "cold-client" in payload["content"]
        # On the second read after mark_seen, is_new should be False
        r2 = execute_tool("read_user_instructions", {}, workspace)
        payload2 = json.loads(r2["output"])
        assert payload2["is_new"] is False

    def test_read_system_load_dispatch(self, workspace: str) -> None:
        r = execute_tool("read_system_load", {}, workspace)
        assert isinstance(r["output"], str)
        assert "throttle" in r["output"]

    def test_peek_experiment_log_dispatch(self, workspace: str, tmp_path: Path) -> None:
        exp_dir = Path(workspace) / "experiments" / "my_exp"
        exp_dir.mkdir(parents=True)
        (exp_dir / "local_job.out").write_text("\n".join(f"epoch {i}" for i in range(20)))
        r = execute_tool(
            "peek_experiment_log",
            {"experiment_name": "my_exp", "last_n_lines": 5},
            workspace,
        )
        assert "epoch 19" in r["output"]

    def test_kill_experiment_parks_and_writes_marker(
        self, workspace: str, db: ExperimentDB
    ) -> None:
        r = execute_tool(
            "kill_experiment",
            {"experiment_id": 1, "reason": "doomed", "evidence": "loss curve flat"},
            workspace,
            db=db,
        )
        assert "kill requested" in r["output"]
        assert db.get(1).parked_at is not None
        from alpha_lab import meta_layout as ml
        marker_path = ml.meta_dir(workspace) / "kill_requests.jsonl"
        assert marker_path.exists()
        assert "1" in marker_path.read_text()

    def test_backup_path(self, workspace: str) -> None:
        (Path(workspace) / "stale.parquet").write_text("cache contents")
        r = execute_tool(
            "backup_path",
            {"path": "stale.parquet", "reason": "before delete"},
            workspace,
        )
        assert "backed up stale.parquet" in r["output"]
        assert (Path(workspace) / "stale.parquet").exists()  # Not deleted

    def test_delete_path_with_backup(self, workspace: str) -> None:
        (Path(workspace) / "stale.parquet").write_text("cache contents")
        r = execute_tool(
            "delete_path",
            {"path": "stale.parquet", "reason": "stale", "evidence": "no live exp uses it"},
            workspace,
        )
        assert "deleted" in r["output"]
        assert not (Path(workspace) / "stale.parquet").exists()

    def test_delete_path_protected_refused(self, workspace: str) -> None:
        from alpha_lab import meta_layout as ml
        ml.ensure_meta_layout(workspace)
        r = execute_tool(
            "delete_path",
            {"path": "meta", "reason": "r", "evidence": "e"},
            workspace,
        )
        assert "[ERROR]" in r["output"]
        assert "protected" in r["output"]

    def test_request_phase_rewind(self, workspace: str) -> None:
        (Path(workspace) / "adapter").mkdir()
        (Path(workspace) / "adapter" / "manifest.json").write_text("{}")
        r = execute_tool(
            "request_phase_rewind",
            {
                "target_phase": "phase0",
                "reason": "domain mismatch",
                "evidence": "demonstrated by script meta/scratch/x.py output",
            },
            workspace,
        )
        assert "phase rewind requested" in r["output"]

    def test_request_phase_rewind_invalid(self, workspace: str) -> None:
        r = execute_tool(
            "request_phase_rewind",
            {"target_phase": "phase42", "reason": "r", "evidence": "e"},
            workspace,
        )
        assert "[ERROR]" in r["output"]

    def test_read_experiment_dispatch(self, workspace: str, db: ExperimentDB) -> None:
        r = execute_tool("read_experiment", {"experiment_id": 1}, workspace, db=db)
        import json
        payload = json.loads(r["output"])
        assert payload["id"] == 1
        assert payload["name"] == "first"
        assert "paths" in payload
        assert "flags" in payload

    def test_note_to_conductor_with_caller_role(self, workspace: str) -> None:
        r = execute_tool(
            "note_to_conductor",
            {"message": "Don't preempt #181 until it has finished training."},
            workspace,
            caller_role="strategist",
        )
        assert "from strategist" in r["output"]
        from alpha_lab import meta_layout as ml
        inbox = ml.notes_inbox_path(workspace).read_text()
        assert "strategist" in inbox
        assert "preempt #181" in inbox

    # ------------------------------------------------------------------
    # request_run_end — Conductor's graceful end-of-run.
    # ------------------------------------------------------------------

    def _make_cfg_like(
        self,
        min_runtime_hours: float = 0.0,
        min_analyzed_before_end: int = 0,
        allow_conductor_end_run: bool = True,
    ):
        class _Cfg:
            pass
        c = _Cfg()
        c.min_runtime_hours = min_runtime_hours
        c.min_analyzed_before_end = min_analyzed_before_end
        c.allow_conductor_end_run = allow_conductor_end_run
        return c

    def test_request_run_end_refused_without_task_config(
        self, workspace: str, db: ExperimentDB
    ) -> None:
        # No _task_config attached → refuses for safety.
        r = execute_tool(
            "request_run_end",
            {"reason": "r", "evidence": "e"},
            workspace,
            db=db,
        )
        assert "[ERROR]" in r["output"]
        assert "TaskConfig" in r["output"]

    def test_request_run_end_refused_without_start_ts(
        self, workspace: str, db: ExperimentDB
    ) -> None:
        db._task_config = self._make_cfg_like()  # type: ignore[attr-defined]
        # Don't write run_state.json — the helper refuses.
        r = execute_tool(
            "request_run_end",
            {"reason": "r", "evidence": "e"},
            workspace,
            db=db,
        )
        assert "[ERROR]" in r["output"]
        assert "Dispatcher start" in r["output"]

    def test_request_run_end_refused_below_min_runtime(
        self, workspace: str, db: ExperimentDB
    ) -> None:
        import time as _t
        from alpha_lab import meta_layout as _ml
        db._task_config = self._make_cfg_like(  # type: ignore[attr-defined]
            min_runtime_hours=1000.0,  # extremely high; will refuse
            min_analyzed_before_end=0,
        )
        # Pretend the dispatcher started right now.
        _ml.write_dispatcher_start_ts(workspace, _t.time())
        r = execute_tool(
            "request_run_end",
            {"reason": "r", "evidence": "e"},
            workspace,
            db=db,
        )
        assert "[ERROR]" in r["output"]
        assert "min_runtime_hours" in r["output"]

    def test_request_run_end_refused_below_min_analyzed(
        self, workspace: str, db: ExperimentDB
    ) -> None:
        import time as _t
        from alpha_lab import meta_layout as _ml
        db._task_config = self._make_cfg_like(  # type: ignore[attr-defined]
            min_runtime_hours=0.0,
            min_analyzed_before_end=100,
        )
        _ml.write_dispatcher_start_ts(workspace, _t.time() - 7200)  # 2h ago
        # Board has fewer than 100 analyzed (the `workspace` fixture seeds
        # only a couple of rows).
        r = execute_tool(
            "request_run_end",
            {"reason": "r", "evidence": "e"},
            workspace,
            db=db,
        )
        assert "[ERROR]" in r["output"]
        assert "min_analyzed_before_end" in r["output"]

    def test_request_run_end_refused_when_not_allowed(
        self, workspace: str, db: ExperimentDB
    ) -> None:
        import time as _t
        from alpha_lab import meta_layout as _ml
        db._task_config = self._make_cfg_like(  # type: ignore[attr-defined]
            allow_conductor_end_run=False,
        )
        _ml.write_dispatcher_start_ts(workspace, _t.time() - 7200)
        r = execute_tool(
            "request_run_end",
            {"reason": "r", "evidence": "e"},
            workspace,
            db=db,
        )
        assert "[ERROR]" in r["output"]
        assert "allow_conductor_end_run" in r["output"]

    def test_request_run_end_writes_marker_on_success(
        self, workspace: str, db: ExperimentDB
    ) -> None:
        import time as _t
        import json as _json
        from alpha_lab import meta_layout as _ml
        from alpha_lab.conductor_tools import RUN_END_MARKER
        db._task_config = self._make_cfg_like(  # type: ignore[attr-defined]
            min_runtime_hours=0.0,
            min_analyzed_before_end=0,
            allow_conductor_end_run=True,
        )
        _ml.write_dispatcher_start_ts(workspace, _t.time() - 7200)
        r = execute_tool(
            "request_run_end",
            {"reason": "plateau", "evidence": "see meta/scratch/x.py output"},
            workspace,
            db=db,
        )
        assert "[ERROR]" not in r["output"]
        marker = _ml.meta_dir(workspace) / RUN_END_MARKER
        assert marker.exists()
        payload = _json.loads(marker.read_text())
        assert payload["reason"] == "plateau"
        assert "elapsed_hours_at_request" in payload
        assert payload["elapsed_hours_at_request"] >= 1.9
