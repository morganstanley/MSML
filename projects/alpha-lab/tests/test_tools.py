"""Tests for the tool system: schemas, execution, path traversal protection."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from alpha_lab import deps
from alpha_lab.config import Phase3Config, PipelineConfig, TaskConfig
from alpha_lab.experiment_db import ExperimentDB
from alpha_lab.tools import (
    TOOLS_DIR,
    WEB_SEARCH_TOOL,
    ToolDefinition,
    execute_tool,
    get_tool_schemas,
    load_all_tools,
    load_tool,
    load_tools,
    parse_tool_args,
)
from alpha_lab.tools.execution import (
    MAX_OUTPUT_CHARS,
    _resolve_in_workspace,
    _truncate_output,
    execute_shell,
    grep_files,
    read_file,
    read_image_base64,
)


class _FakeExec:
    def __init__(self, slots: int) -> None:
        self._slots = slots

    def total_slots(self) -> int:
        return self._slots


def _deps() -> deps.RunDeps:
    """Minimal active-run scope so propose_experiment can validate resources."""
    cfg = TaskConfig(
        data_path="d",
        description="x",
        pipeline=PipelineConfig(phases=["phase3"], phase3=Phase3Config(gpu_ids=[])),
    )
    return deps.RunDeps(cfg, gpu_executor=_FakeExec(1), cpu_executor=_FakeExec(1))


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
        for tool in load_all_tools():
            schema = tool.schema
            assert schema["type"] == "function", f"{tool.name} missing type"
            assert schema["name"] == tool.name, f"{tool.name} name mismatch"
            assert "parameters" in schema, f"{tool.name} missing parameters"
            assert "description" in schema, f"{tool.name} missing description"

    def test_get_tool_schemas_subset(self) -> None:
        schemas = get_tool_schemas([load_tool("shell_exec"), load_tool("read_file")])
        assert len(schemas) == 2
        names = {s["name"] for s in schemas}
        assert names == {"shell_exec", "read_file"}

    def test_get_tool_schemas_with_web_search(self) -> None:
        schemas = get_tool_schemas([load_tool("shell_exec")], include_web_search=True)
        types = {s.get("type") for s in schemas}
        assert "web_search_preview" in types


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
        with _deps():
            result = execute_tool(
                "propose_experiment",
                {
                    "name": "test_xgboost",
                    "description": "XGBoost baseline",
                    "hypothesis": "Trees work",
                    "config": '{"resource": "gpu", "model": "xgboost"}',
                },
                tmp_workspace,
                db=db,
            )
        assert "created" in result["output"].lower()
        assert "test_xgboost" in result["output"]

    def test_propose_experiment_sanitizes_name(self, tmp_workspace: str, db: ExperimentDB) -> None:
        with _deps():
            result = execute_tool(
                "propose_experiment",
                {
                    "name": "../../evil/path.sh",
                    "description": "D",
                    "hypothesis": "H",
                    "config": '{"resource": "gpu"}',
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

    def test_read_board(self, tmp_workspace: str, db: ExperimentDB, adapter) -> None:
        db.create("exp_a", "D", "H", "{}")
        result = execute_tool("read_board", {}, tmp_workspace, db=db, adapter=adapter)
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

    def test_update_experiment_smoke_flagged(
        self, tmp_workspace: str, db: ExperimentDB
    ) -> None:
        """Smoke-test results should be auto-tagged and produce a warning."""
        exp_id = db.create("smoke_exp", "D", "H", "{}")
        result = execute_tool(
            "update_experiment",
            {
                "experiment_id": exp_id,
                "results": '{"sharpe": 5.0, "smoke_test": true}',
            },
            tmp_workspace,
            db=db,
        )
        assert "WARNING" in result["output"]
        assert "smoke" in result["output"].lower()
        # The stored results should have _smoke_flagged injected
        exp = db.get(exp_id)
        stored = json.loads(exp.results_json)
        assert stored["_smoke_flagged"] is True
        assert stored["smoke_test"] is True

    def test_update_experiment_non_smoke_no_warning(
        self, tmp_workspace: str, db: ExperimentDB
    ) -> None:
        """Normal results should not produce a smoke warning."""
        exp_id = db.create("good_exp", "D", "H", "{}")
        result = execute_tool(
            "update_experiment",
            {
                "experiment_id": exp_id,
                "results": '{"sharpe": 1.2}',
            },
            tmp_workspace,
            db=db,
        )
        assert "WARNING" not in result["output"]
        assert db.get(exp_id).results_json == '{"sharpe": 1.2}'

    def test_update_experiment_no_db(self, tmp_workspace: str) -> None:
        result = execute_tool(
            "update_experiment",
            {"experiment_id": 1},
            tmp_workspace,
            db=None,
        )
        assert "[ERROR]" in result["output"]


# ---------------------------------------------------------------------------
# load_tool / load_tools — registry/*.md loading
# ---------------------------------------------------------------------------


class TestLoadTool:
    def test_load_tool_returns_tool_definition(self) -> None:
        tool = load_tool("shell_exec")
        assert isinstance(tool, ToolDefinition)
        assert tool.name == "shell_exec"

    def test_load_tool_missing_name_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_tool("nonexistent_tool")

    def test_load_tools_skips_names_without_registry_file(self) -> None:
        tools = load_tools(["shell_exec", "web_search"])
        assert len(tools) == 1
        assert tools[0].name == "shell_exec"

    def test_load_all_tools_returns_tool_definitions(self) -> None:
        tools = load_all_tools()
        assert tools
        assert all(isinstance(tool, ToolDefinition) for tool in tools)


# ---------------------------------------------------------------------------
# ToolDefinition — build_from_config validation, schema shape, hashability
# ---------------------------------------------------------------------------


class TestToolDefinition:
    def test_schema_shape(self) -> None:
        tool = ToolDefinition.build_from_config(
            {
                "name": "my_tool",
                "description": "Does a thing.",
                "metadata": {
                    "parameters": {
                        "type": "object",
                        "properties": {"foo": {"type": "string"}},
                        "required": ["foo"],
                    }
                },
            }
        )
        assert tool.schema == {
            "type": "function",
            "name": "my_tool",
            "description": "Does a thing.",
            "parameters": {
                "type": "object",
                "properties": {"foo": {"type": "string"}},
                "required": ["foo"],
            },
        }

    def test_missing_name_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="name"):
            ToolDefinition.build_from_config(
                {"description": "d", "metadata": {"parameters": {}}}
            )

    def test_missing_description_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="description"):
            ToolDefinition.build_from_config(
                {"name": "my_tool", "metadata": {"parameters": {}}}
            )

    def test_missing_metadata_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="metadata"):
            ToolDefinition.build_from_config(
                {"name": "my_tool", "description": "d"}
            )

    def test_missing_parameters_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="parameters"):
            ToolDefinition.build_from_config(
                {"name": "my_tool", "description": "d", "metadata": {}}
            )

    def test_is_hashable_despite_parameters_dict(self) -> None:
        tool = load_tool("shell_exec")
        assert hash(tool) == hash(load_tool("shell_exec"))
        assert {tool, tool} == {tool}


# ---------------------------------------------------------------------------
# get_tool_schemas — extra cases (existing class covers subset/web/unknown)
# ---------------------------------------------------------------------------


class TestGetToolSchemasExtra:
    def test_preserves_requested_order(self) -> None:
        schemas = get_tool_schemas(
            [load_tool("report_to_user"), load_tool("shell_exec"), load_tool("ask_user")]
        )
        assert [s["name"] for s in schemas] == [
            "report_to_user",
            "shell_exec",
            "ask_user",
        ]

    def test_empty_list_returns_empty(self) -> None:
        assert get_tool_schemas([]) == []

    def test_empty_list_with_web_search_returns_only_web_search(self) -> None:
        schemas = get_tool_schemas([], include_web_search=True)
        assert schemas == [WEB_SEARCH_TOOL]

    def test_web_search_appended_after_named_tools(self) -> None:
        schemas = get_tool_schemas([load_tool("shell_exec")], include_web_search=True)
        assert schemas[0]["name"] == "shell_exec"
        assert schemas[-1] == WEB_SEARCH_TOOL


# ---------------------------------------------------------------------------
# _truncate_output — boundary case (existing class covers under/over/preserves)
# ---------------------------------------------------------------------------


class TestTruncateOutputBoundary:
    def test_text_exactly_at_threshold_unchanged(self) -> None:
        text = "x" * MAX_OUTPUT_CHARS
        assert _truncate_output(text) == text

    def test_text_one_over_threshold_truncated(self) -> None:
        text = "x" * (MAX_OUTPUT_CHARS + 1)
        result = _truncate_output(text)
        assert "truncated" in result
        assert "1 chars" in result  # exactly 1 char dropped


# ---------------------------------------------------------------------------
# parse_tool_args — non-dict JSON case
# ---------------------------------------------------------------------------


class TestParseToolArgsExtra:
    def test_non_dict_json_returned_as_is(self) -> None:
        # Current contract: json.loads succeeds, return value passed through
        # even when it is not a dict. Verifying the implemented behavior.
        assert parse_tool_args("[1, 2, 3]") == [1, 2, 3]
        assert parse_tool_args("42") == 42


# ---------------------------------------------------------------------------
# Adapter tools via execute_tool
# ---------------------------------------------------------------------------


def _seed_adapter(workspace: str) -> Path:
    """Create a minimal adapter dir under workspace with manifest + one prompt."""
    adapter_dir = Path(workspace) / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "manifest.json").write_text(
        json.dumps({"domain_name": "test", "version": 1})
    )
    (adapter_dir / "phase1.md").write_text("# Phase 1 prompt body\nExplore the data.")
    return adapter_dir


class TestAdapterToolsViaExecuteTool:
    def test_read_adapter_returns_manifest_and_prompt(self, tmp_workspace: str) -> None:
        _seed_adapter(tmp_workspace)
        result = execute_tool("read_adapter", {}, tmp_workspace)
        assert "manifest.json" in result["output"]
        assert "phase1.md" in result["output"]
        assert "Explore the data" in result["output"]

    def test_read_adapter_missing_directory_returns_error_string(
        self, tmp_workspace: str
    ) -> None:
        # No adapter dir created — must return a sensible error, not raise.
        result = execute_tool("read_adapter", {}, tmp_workspace)
        assert "[ERROR]" in result["output"]
        assert "adapter" in result["output"].lower()

    def test_read_reference_adapter_time_series_smoke(self, tmp_workspace: str) -> None:
        # Real built-in reference under src/alpha_lab/adapters/time_series/.
        result = execute_tool(
            "read_reference_adapter", {"name": "time_series"}, tmp_workspace
        )
        assert "[ERROR]" not in result["output"]
        # Header line includes the reference adapter name; manifest section
        # is always emitted.
        assert "time_series" in result["output"]
        assert "manifest" in result["output"]

    def test_write_adapter_file_creates_file(self, tmp_workspace: str) -> None:
        result = execute_tool(
            "write_adapter_file",
            {"filename": "domain_knowledge.md", "content": "Hello domain."},
            tmp_workspace,
        )
        assert "[ERROR]" not in result["output"]
        written = Path(tmp_workspace) / "adapter" / "domain_knowledge.md"
        assert written.read_text() == "Hello domain."

    def test_write_adapter_file_rejects_non_allowlisted_filename(
        self, tmp_workspace: str
    ) -> None:
        # Path-traversal attempts and unknown filenames alike are rejected
        # because write_adapter_file checks against ADAPTER_FILES allowlist.
        result = execute_tool(
            "write_adapter_file",
            {"filename": "../escape.md", "content": "evil"},
            tmp_workspace,
        )
        assert "[ERROR]" in result["output"]
        assert "Invalid adapter filename" in result["output"]
        assert not (Path(tmp_workspace) / "adapter" / "..").exists() or not (
            Path(tmp_workspace).parent / "escape.md"
        ).exists()

    def test_patch_adapter_file_overwrites_content(self, tmp_workspace: str) -> None:
        # Initialize a real git repo so the internal `git add && git commit`
        # checkpoint has somewhere to land. The function swallows git errors,
        # but a real repo exercises the happy path of the side effect.
        adapter_dir = _seed_adapter(tmp_workspace)
        # Replace phase1.md (which is in the ADAPTER_FILES allowlist) instead
        # of manifest.json (which is also allowlisted) — either works.
        subprocess_run = __import__("subprocess").run
        subprocess_run(
            ["git", "init", "-q"], cwd=tmp_workspace, check=True, capture_output=True
        )
        subprocess_run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A"],
            cwd=tmp_workspace,
            check=True,
            capture_output=True,
        )
        subprocess_run(
            [
                "git",
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "-q",
                "-m",
                "initial",
            ],
            cwd=tmp_workspace,
            check=True,
            capture_output=True,
        )

        result = execute_tool(
            "patch_adapter_file",
            {
                "filename": "phase1.md",
                "content": "Patched body.",
                "reason": "test patch",
            },
            tmp_workspace,
        )
        assert "[ERROR]" not in result["output"]
        assert "Patched adapter/phase1.md" in result["output"]
        assert (adapter_dir / "phase1.md").read_text() == "Patched body."


# ---------------------------------------------------------------------------
# Memory tools (memory_store / memory_search / memory_read)
# ---------------------------------------------------------------------------


class TestMemoryToolsViaExecuteTool:
    def test_store_then_read_round_trip(self, tmp_workspace: str) -> None:
        store_result = execute_tool(
            "memory_store",
            {
                "content": "The quick brown fox jumps over the lazy dog.",
                "tags": ["animal", "pangram"],
                "summary": "classic pangram",
            },
            tmp_workspace,
        )
        # Output looks like "Memory #1 stored."
        assert "stored" in store_result["output"].lower()
        assert "#1" in store_result["output"]

        read_result = execute_tool(
            "memory_read", {"memory_id": 1}, tmp_workspace
        )
        assert read_result["output"] == "The quick brown fox jumps over the lazy dog."

    def test_search_finds_stored_entry_by_keyword(self, tmp_workspace: str) -> None:
        execute_tool(
            "memory_store",
            {
                "content": "irrelevant body",
                "tags": ["modeling"],
                "summary": "lstm hyperparameter sweep results",
            },
            tmp_workspace,
        )
        execute_tool(
            "memory_store",
            {
                "content": "irrelevant body",
                "tags": ["data"],
                "summary": "outlier detection on exchange rates",
            },
            tmp_workspace,
        )

        # Search matches summary + tags — "lstm" only appears in the first entry.
        result = execute_tool(
            "memory_search", {"query": "lstm"}, tmp_workspace
        )
        assert "lstm hyperparameter sweep results" in result["output"]
        assert "outlier detection on exchange rates" not in result["output"]

    def test_read_missing_id_returns_not_found_string(self, tmp_workspace: str) -> None:
        result = execute_tool("memory_read", {"memory_id": 999}, tmp_workspace)
        assert "[ERROR]" in result["output"]
        assert "not found" in result["output"].lower()

    def test_memory_state_isolated_per_workspace(self, tmp_path: Path) -> None:
        ws_a = str(tmp_path / "ws_a")
        ws_b = str(tmp_path / "ws_b")
        os.makedirs(ws_a)
        os.makedirs(ws_b)

        execute_tool(
            "memory_store",
            {"content": "only in A", "tags": [], "summary": "a-summary"},
            ws_a,
        )
        # Workspace B sees an empty store, regardless of A's state.
        result = execute_tool("memory_search", {"query": "a-summary"}, ws_b)
        assert "No matching memories found" in result["output"]


# ---------------------------------------------------------------------------
# cancel_experiments
# ---------------------------------------------------------------------------


class TestCancelExperiments:
    @pytest.fixture()
    def db(self, tmp_path: Path) -> ExperimentDB:
        return ExperimentDB(str(tmp_path / "test.db"))

    def test_cancel_marks_to_implement_experiment_cancelled(
        self, tmp_workspace: str, db: ExperimentDB
    ) -> None:
        exp_id = db.create("to_cancel", "D", "H", "{}")
        # Default status of a freshly-created experiment is "to_implement".
        assert db.get(exp_id).status == "to_implement"

        result = execute_tool(
            "cancel_experiments",
            {"experiment_ids": [exp_id], "reason": "duplicate hypothesis"},
            tmp_workspace,
            db=db,
        )
        assert "Cancelled 1 experiments" in result["output"]
        assert db.get(exp_id).status == "cancelled"

    def test_cancel_nonexistent_id_returned_in_skipped_list(
        self, tmp_workspace: str, db: ExperimentDB
    ) -> None:
        result = execute_tool(
            "cancel_experiments",
            {"experiment_ids": [9999]},
            tmp_workspace,
            db=db,
        )
        # Source uses "Skipped N: ..." for unfound / non-cancellable ids.
        assert "Skipped" in result["output"]
        assert "9999" in result["output"]
        assert "not found" in result["output"]

    def test_cancel_no_db_returns_error(self, tmp_workspace: str) -> None:
        result = execute_tool(
            "cancel_experiments",
            {"experiment_ids": [1]},
            tmp_workspace,
            db=None,
        )
        assert "[ERROR]" in result["output"]


# ---------------------------------------------------------------------------
# reality_check
# ---------------------------------------------------------------------------


class TestRealityCheck:
    def test_missing_experiment_name_returns_error(self, tmp_workspace: str) -> None:
        result = execute_tool("reality_check", {}, tmp_workspace)
        assert "[ERROR]" in result["output"]
        assert "experiment_name" in result["output"]

    def test_missing_experiment_directory_returns_error(
        self, tmp_workspace: str
    ) -> None:
        # No experiments/ dir exists — source returns a path-not-found error
        # rather than calling into validation.
        result = execute_tool(
            "reality_check",
            {"experiment_name": "does_not_exist"},
            tmp_workspace,
        )
        assert "[ERROR]" in result["output"]
        assert "Experiment directory not found" in result["output"]
        assert "does_not_exist" in result["output"]


# ---------------------------------------------------------------------------
# Flat registry: each <tool>.md filename matches its frontmatter name
# ---------------------------------------------------------------------------


class TestFlatRegistry:
    def test_filename_matches_frontmatter_name(self) -> None:
        for path in TOOLS_DIR.iterdir():
            if not path.name.endswith(".md"):
                continue
            tool = load_tool(path.name[:-3])
            assert tool.name == path.name[:-3]


# ---------------------------------------------------------------------------
# Dispatcher truncation contract: shell_exec output > MAX_OUTPUT_CHARS
# ---------------------------------------------------------------------------


class TestExecuteToolTruncatesShellOutput:
    def test_long_shell_output_is_truncated(self, tmp_workspace: str) -> None:
        # 40k chars > MAX_OUTPUT_CHARS (30k) — must trigger _truncate_output.
        # Use sys.executable to avoid PATH ambiguity.
        import sys as _sys
        cmd = f"{_sys.executable} -c \"print('x' * 40000)\""
        result = execute_tool(
            "shell_exec", {"command": cmd}, tmp_workspace
        )
        out = result["output"]
        # _truncate_output marker shape: "[... truncated N chars ...]"
        assert "truncated" in out
        assert "chars" in out
        # The dropped-chars count produced by _truncate_output is computed
        # from the *pre-truncation* length, but the returned string also
        # adds the truncation marker. Bound the length by MAX_OUTPUT_CHARS
        # plus a generous marker allowance (~200 chars covers the marker).
        assert len(out) <= MAX_OUTPUT_CHARS + 200
