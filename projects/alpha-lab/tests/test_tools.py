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

    def test_update_playbook(self, tmp_workspace: str) -> None:
        result = execute_tool(
            "update_playbook",
            {"content": "# Playbook\n\n## What works\n- LSTMs"},
            tmp_workspace,
        )
        assert "updated" in result["output"].lower()
        content = (Path(tmp_workspace) / "playbook.md").read_text()
        assert "LSTMs" in content

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
