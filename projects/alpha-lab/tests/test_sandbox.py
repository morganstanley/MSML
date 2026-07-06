"""Tests for agent sandboxing: availability gating, mount derivation, runtime payload,
in-process fallback, and (bwrap-gated) real filesystem isolation."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
import subprocess
import sys
import time
import types
from unittest.mock import MagicMock

import pytest

from alpha_lab import deps, tracing
from alpha_lab.sandboxing import sandbox
from alpha_lab.benchmarks.paths import find_repo_root
from alpha_lab.config import TaskConfig, task_config_from_mapping
from alpha_lab.events import MetricsEvent, StatusEvent
from alpha_lab.metrics import MetricsCollector


# --- is_available -----------------------------------------------------------

def test_is_available_true_when_bwrap_present_and_not_disabled(monkeypatch) -> None:
    monkeypatch.setattr(sandbox.shutil, "which", lambda _name: "/usr/bin/bwrap")
    monkeypatch.delenv(sandbox.NOSANDBOX_ENV, raising=False)
    assert sandbox._is_available() is True


def test_is_available_false_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(sandbox.shutil, "which", lambda _name: "/usr/bin/bwrap")
    monkeypatch.setenv(sandbox.NOSANDBOX_ENV, "1")
    assert sandbox._is_available() is False


def test_is_available_false_when_bwrap_missing(monkeypatch) -> None:
    monkeypatch.setattr(sandbox.shutil, "which", lambda _name: None)
    monkeypatch.delenv(sandbox.NOSANDBOX_ENV, raising=False)
    assert sandbox._is_available() is False


# --- child payload serialization --------------------------------------------

def test_child_payload_serializes_config_and_carries_workspace(tmp_path, monkeypatch) -> None:
    """run_agent's bwrap path puts a JSON-round-trippable payload on the child's stdin:
    config as a mapping the child rebuilds, and the workspace as a plain string."""
    monkeypatch.setattr(sandbox, "_is_available", lambda: True)
    captured: dict = {}

    def fake_popen(argv, **kwargs):
        proc = MagicMock()
        proc.poll.return_value = None
        return proc

    monkeypatch.setattr(sandbox, "_popen", fake_popen)
    monkeypatch.setattr(sandbox, "_relay", lambda *a, **k: None)
    monkeypatch.setattr(
        sandbox.AgentRunHandle, "_attach_proc", lambda self, proc, pl: captured.update(pl)
    )

    config = TaskConfig(data_path=str(tmp_path / "x.csv"), description="desc")
    workspace = str(tmp_path / "ws")
    with deps.RunDeps(config, workspace=workspace, gpu_executor=MagicMock()):
        sandbox.run_agent("phase3/strategist", MagicMock(), initial_message="go")

    assert json.loads(json.dumps(captured)) == captured  # JSON round-trippable
    child_runtime = captured["runtime"]
    assert task_config_from_mapping(child_runtime["config"]) == config
    assert child_runtime["workspace"] == workspace


# --- bwrap argv builder -----------------------------------------------------

def test_build_bwrap_prefix_flags() -> None:
    prefix = sandbox._build_bwrap_prefix(
        ro_paths=["/data/x.csv"],
        rw_paths=["/ws"],
        chdir="/ws",
        project_ro_root="/repo",
        needs_gpu=False,
    )
    text = " ".join(prefix)
    assert prefix[0] == "bwrap"
    assert "--unshare-pid" in prefix
    assert "--die-with-parent" in prefix
    assert "--ro-bind /repo /repo" in text
    assert "--ro-bind-try /data/x.csv /data/x.csv" in text
    assert "--bind-try /ws /ws" in text
    assert "--chdir /ws" in text
    assert prefix[-1] == "--"
    assert "--dev-bind" not in text  # no GPU when needs_gpu is False


def test_build_bwrap_prefix_includes_gpu_args(monkeypatch) -> None:
    monkeypatch.setattr(sandbox, "_nvidia_args", lambda: ["--dev-bind", "/dev/nvidia0", "/dev/nvidia0"])
    prefix = sandbox._build_bwrap_prefix(
        ro_paths=[], rw_paths=["/ws"], chdir="/ws", project_ro_root="/repo", needs_gpu=True
    )
    assert "--dev-bind /dev/nvidia0 /dev/nvidia0" in " ".join(prefix)


# --- in-process fallback ----------------------------------------------------

def test_run_agent_falls_back_in_process(monkeypatch) -> None:
    # conftest sets ALPHALAB_AGENT_NOSANDBOX=1, so is_available() is False here.
    assert not sandbox._is_available()
    fake_agent = MagicMock()
    captured = {}

    def fake_build_agent(agent_definition, **kwargs):
        captured["agent_definition"] = agent_definition
        captured["kwargs"] = kwargs
        return fake_agent

    monkeypatch.setattr(sandbox, "build_agent", fake_build_agent)
    config = TaskConfig(data_path="data/x.csv", description="desc")

    with deps.RunDeps(config, workspace="/ws", gpu_executor=MagicMock()):
        handle = sandbox.run_agent(
            "phase3/strategist", MagicMock(), initial_message="go"
        )

    fake_agent.run.assert_called_once_with("go")
    assert handle._agent is fake_agent
    assert captured["agent_definition"].name == "strategist"
    # config/workspace/api_key are no longer passed to build_agent; it sources them from the
    # active RunDeps. initial_message is a run-only value, never a build kwarg.
    assert "config" not in captured["kwargs"]
    assert "workspace" not in captured["kwargs"]
    assert "api_key" not in captured["kwargs"]
    assert "initial_message" not in captured["kwargs"]


# --- AgentRunHandle control buffering ----------------------------------

def test_stop_buffered_until_agent_attached() -> None:
    handle = sandbox.AgentRunHandle()
    handle.stop()  # no target yet — must be buffered, not dropped
    agent = MagicMock()
    handle._attach_agent(agent)
    agent.stop.assert_called_once_with()


def test_buffered_control_written_after_payload_on_proc() -> None:
    handle = sandbox.AgentRunHandle()
    handle.provide_answer("42")  # buffered before proc exists
    writes: list[str] = []
    proc = MagicMock()
    proc.poll.return_value = None
    proc.stdin.write.side_effect = lambda s: writes.append(s)
    handle._attach_proc(proc, {"agent_id": "x", "runtime": {}})
    # payload line first, then the buffered answer
    assert json.loads(writes[0]) == {"agent_id": "x", "runtime": {}}
    assert json.loads(writes[1]) == {"type": "answer", "text": "42"}


# --- metrics merge across the boundary ---------------------------------

class _FakeProc:
    def __init__(self, stdout_lines: list[str]) -> None:
        self.stdout = iter(stdout_lines)
        self.stderr = iter([])

    def wait(self) -> int:
        return 0


def test_relay_merges_child_metrics() -> None:
    counters = MetricsCollector()
    counters.record_api_call(10, 5, cache_read_tokens=2)
    lines = [
        json.dumps(MetricsEvent(counters=asdict(counters.counters())).to_dict()) + "\n",
        json.dumps(StatusEvent(status="done").to_dict()) + "\n",
    ]
    parent = MetricsCollector()
    handle = sandbox.AgentRunHandle()
    sandbox._relay(_FakeProc(lines), lambda ev: None, handle, parent)
    assert parent.counters().total_input_tokens == 10
    assert parent.counters().api_call_count == 1
    assert handle._done_observed is True


def test_relay_merges_incremental_metric_deltas_additively() -> None:
    """#2: the child emits per-call deltas; the parent must sum them, so a hard crash
    after the second call still leaves the first call's counters merged."""
    first = MetricsCollector()
    first.record_api_call(10, 5)
    second = MetricsCollector()
    second.record_api_call(3, 2)
    lines = [
        json.dumps(MetricsEvent(counters=asdict(first.counters())).to_dict()) + "\n",
        json.dumps(MetricsEvent(counters=asdict(second.counters())).to_dict()) + "\n",
    ]
    parent = MetricsCollector()
    sandbox._relay(_FakeProc(lines), lambda ev: None, sandbox.AgentRunHandle(), parent)
    assert parent.counters().total_input_tokens == 13
    assert parent.counters().api_call_count == 2


# --- run_agent owns the handle ------------------------------------------

def test_run_agent_publishes_and_clears_owner_handle(monkeypatch) -> None:
    owner = types.SimpleNamespace(_run_handle=None)
    seen = {}

    def fake_build_agent(agent_definition, **kwargs):
        agent = MagicMock()
        agent.run.side_effect = lambda _msg: seen.update(during=owner._run_handle)
        return agent

    monkeypatch.setattr(sandbox, "build_agent", fake_build_agent)
    config = TaskConfig(data_path="d.csv", description="x")
    with deps.RunDeps(config, workspace="/ws", gpu_executor=MagicMock()):
        handle = sandbox.run_agent(
            "phase3/strategist", MagicMock(), initial_message="go", owner=owner,
        )

    assert seen["during"] is handle  # published during the run
    assert owner._run_handle is None  # cleared afterwards


def test_shell_less_agent_mounts_declared_paths_but_not_workspace_root(tmp_path, monkeypatch) -> None:
    """A shell-less agent (strategist) mounts exactly its declared footprint paths — its
    board/memory paths under the workspace — but never the workspace root, since it has no
    ``.`` footprint. (The board/memory are still reached at runtime via the parent proxy.)"""
    monkeypatch.setattr(sandbox, "_is_available", lambda: True)
    captured = {}

    def fake_popen(argv, **kwargs):
        captured.update(kwargs)
        proc = MagicMock()
        proc.poll.return_value = None
        return proc

    monkeypatch.setattr(sandbox, "_popen", fake_popen)
    monkeypatch.setattr(sandbox, "_relay", lambda *a, **k: None)

    workspace = tmp_path / "ws"
    config = TaskConfig(data_path=str(tmp_path / "d.csv"), description="d")
    with deps.RunDeps(config, workspace=str(workspace), gpu_executor=MagicMock()):
        sandbox.run_agent(
            "phase3/strategist", MagicMock(),
            initial_message="go",
            db_path=str(workspace / "experiments.db"),
        )

    rw = captured["rw_paths"]
    assert str(workspace / "experiments.db") in rw
    assert str(workspace / ".memory") in rw
    assert str(workspace) not in rw  # shell-less: the root is never mounted


def test_run_agent_chdirs_to_workspace_and_passes_it_to_child(tmp_path, monkeypatch) -> None:
    """run_agent uses the caller's workspace as-is: chdir is the workspace, and the child
    payload carries it as a plain string (no serialization)."""
    monkeypatch.setattr(sandbox, "_is_available", lambda: True)
    workspace = tmp_path / "ws"

    captured = {}
    payload = {}

    def fake_popen(argv, **kwargs):
        captured.update(kwargs)
        proc = MagicMock()
        proc.poll.return_value = None
        return proc

    monkeypatch.setattr(sandbox, "_popen", fake_popen)
    monkeypatch.setattr(sandbox, "_relay", lambda *a, **k: None)
    monkeypatch.setattr(
        sandbox.AgentRunHandle, "_attach_proc", lambda self, proc, pl: payload.update(pl)
    )

    config = TaskConfig(data_path=str(tmp_path / "d.csv"), description="d")
    with deps.RunDeps(config, workspace=str(workspace), gpu_executor=MagicMock()):
        sandbox.run_agent("phase3/strategist", MagicMock(), initial_message="go")

    assert captured["chdir"] == str(workspace)
    assert payload["runtime"]["workspace"] == str(workspace)


# --- RW bind-source creation ------------------------------------------------

def test_create_rw_sources_makes_dirs_and_touches_files(tmp_path) -> None:
    memory_dir = tmp_path / ".memory"
    playbook = tmp_path / "playbook.md"
    sandbox._create_rw_sources({memory_dir, playbook})
    assert memory_dir.is_dir()  # no extension -> directory
    assert playbook.is_file()  # has an extension -> touched as a file


def test_create_rw_sources_leaves_existing_path(tmp_path) -> None:
    # An existing path (e.g. a hardlinked file) is left as-is, not re-created.
    playbook = tmp_path / "playbook.md"
    original = tmp_path / "orig.md"
    original.write_text("x")
    os.link(original, playbook)
    sandbox._create_rw_sources({playbook})
    assert playbook.is_file()
    assert playbook.read_text() == "x"


# --- bwrap-gated filesystem isolation ---------------------------------------

@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not available")
def test_sandbox_isolates_filesystem(tmp_path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "marker.txt").write_text("hi")
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "s.txt").write_text("x")

    probe = (
        "import os, sys; "
        "print('marker=%s secret=%s' % "
        "(os.path.exists(sys.argv[1]), os.path.exists(sys.argv[2])))"
    )
    proc = sandbox._popen(
        [sys.executable, "-c", probe, str(workspace / "marker.txt"), str(secret / "s.txt")],
        ro_paths=[],
        rw_paths=[str(workspace)],
        chdir=str(workspace),
        project_ro_root=str(find_repo_root()),
        needs_gpu=False,
    )
    proc.stdin.close()
    out = proc.stdout.read()
    proc.wait()
    assert "marker=True" in out  # workspace is mounted
    assert "secret=False" in out  # unmounted host path is invisible


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not available")
def test_sandbox_sqlite_wal_and_learnings_writes(tmp_path) -> None:
    """Regression #1/#2: with the workspace dir RW-mounted, SQLite WAL (sidecar
    creation) and learnings.md writes succeed inside the sandbox."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    probe = (
        "import os, sqlite3, sys; ws = sys.argv[1]; "
        "c = sqlite3.connect(os.path.join(ws, 'experiments.db')); "
        "c.execute('PRAGMA journal_mode=WAL'); "
        "c.execute('CREATE TABLE t(x)'); c.execute('INSERT INTO t VALUES (1)'); "
        "c.commit(); c.close(); "
        "open(os.path.join(ws, 'learnings.md'), 'w').write('ok'); "
        "print('wrote')"
    )
    proc = sandbox._popen(
        [sys.executable, "-c", probe, str(workspace)],
        ro_paths=[],
        rw_paths=[str(workspace)],  # what run_agent derives (DB-file parent → workspace root)
        chdir=str(workspace),
        project_ro_root=str(find_repo_root()),
        needs_gpu=False,
    )
    proc.stdin.close()
    out = proc.stdout.read()
    err = proc.stderr.read()
    rc = proc.wait()
    assert rc == 0, err
    assert "wrote" in out
    assert (workspace / "experiments.db").exists()
    assert (workspace / "learnings.md").read_text() == "ok"


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not available")
def test_sandbox_follows_symlink_inside_mounted_dir(tmp_path) -> None:
    """A symlink inside a mounted dir is followable iff its target is also mounted."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    external = tmp_path / "ext"
    external.mkdir()
    (external / "r.txt").write_text("reachable")
    (workspace / "link").symlink_to(external, target_is_directory=True)
    probe = "import sys; print(open(sys.argv[1]).read())"

    def run(rw_paths: list[str]) -> str:
        proc = sandbox._popen(
            [sys.executable, "-c", probe, str(workspace / "link" / "r.txt")],
            ro_paths=[],
            rw_paths=rw_paths,
            chdir=str(workspace),
            project_ro_root=str(find_repo_root()),
            needs_gpu=False,
        )
        proc.stdin.close()
        out = proc.stdout.read()
        proc.wait()
        return out

    assert "reachable" not in run([str(workspace)])  # target unmounted -> dangling
    extra = {external}
    assert "reachable" in run([str(workspace), *(str(path) for path in extra)])


# --- OTel context propagation across the boundary ----------------------

def test_inject_extract_round_trips_active_trace() -> None:
    """The child reconstructs the parent's trace from the injected carrier, so its
    spans nest under the parent run rather than starting an orphan trace."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    tracer = TracerProvider().get_tracer("test")
    with tracer.start_as_current_span("parent") as span:
        carrier = tracing.inject_context()
    assert "traceparent" in carrier

    extracted = trace.get_current_span(tracing.context_from_carrier(carrier))
    assert extracted.get_span_context().trace_id == span.get_span_context().trace_id


# --- forceful stop escalation ------------------------------------------

def test_stop_escalates_to_terminate_then_kill(monkeypatch) -> None:
    """A child that ignores the stdin `stop` (e.g. wedged in startup) must be
    escalated: SIGTERM after the grace, SIGKILL if it still won't die."""
    monkeypatch.setattr(sandbox, "_STOP_GRACE_SECONDS", 0.05)
    monkeypatch.setattr(sandbox, "_STOP_KILL_GRACE_SECONDS", 0.05)
    proc = MagicMock()
    proc.poll.return_value = None
    proc.wait.side_effect = subprocess.TimeoutExpired(cmd="child", timeout=0.05)

    handle = sandbox.AgentRunHandle()
    handle._attach_proc(proc, {"agent_id": "x", "runtime": {}})
    handle.stop()

    deadline = time.time() + 2.0
    while time.time() < deadline and not proc.kill.called:
        time.sleep(0.02)
    proc.terminate.assert_called()
    proc.kill.assert_called()


# --- parent services the child's backend requests (single connection owner) ----------

def test_service_backend_request_routes_to_target_and_encodes() -> None:
    db = MagicMock()
    db.board_summary.return_value = {"done": 2}
    handle = sandbox.AgentRunHandle()
    sent: list = []
    handle._send_backend_response = sent.append  # capture the reply
    sandbox._service_backend_request(
        {"type": "backend_request", "req_id": "r1", "target": "experiments",
         "method": "board_summary", "args": [], "kwargs": {}},
        {"experiments": db}, handle,
    )
    db.board_summary.assert_called_once_with()
    assert sent[0]["ok"] is True and sent[0]["result"] == {"done": 2}


def test_service_backend_request_routes_memory_target() -> None:
    memory = MagicMock()
    memory.store.return_value = 7
    handle = sandbox.AgentRunHandle()
    sent: list = []
    handle._send_backend_response = sent.append
    sandbox._service_backend_request(
        {"req_id": "r", "target": "memory", "method": "store", "args": [], "kwargs": {"summary": "s"}},
        {"experiments": MagicMock(), "memory": memory}, handle,
    )
    memory.store.assert_called_once_with(summary="s")
    assert sent[0]["ok"] is True and sent[0]["result"] == 7


def test_service_backend_request_rejects_private_method_and_unknown_target() -> None:
    handle = sandbox.AgentRunHandle()
    sent: list = []
    handle._send_backend_response = sent.append
    sandbox._service_backend_request({"req_id": "r", "target": "experiments", "method": "_connect",
                                      "args": [], "kwargs": {}}, {"experiments": MagicMock()}, handle)
    assert sent[-1]["ok"] is False
    sandbox._service_backend_request({"req_id": "r", "target": "nope", "method": "create",
                                      "args": [], "kwargs": {}}, {}, handle)
    assert sent[-1]["ok"] is False


def test_service_backend_request_reports_exception() -> None:
    db = MagicMock()
    db.create.side_effect = ValueError("dup name")
    handle = sandbox.AgentRunHandle()
    sent: list = []
    handle._send_backend_response = sent.append
    sandbox._service_backend_request(
        {"req_id": "r", "target": "experiments", "method": "create", "args": ["n"], "kwargs": {}},
        {"experiments": db}, handle,
    )
    assert sent[0]["ok"] is False and "dup name" in sent[0]["error"]


def test_relay_services_backend_request_inline() -> None:
    db = MagicMock()
    db.board_summary.return_value = {"x": 1}
    lines = [
        json.dumps({"type": "backend_request", "req_id": "r", "target": "experiments",
                    "method": "board_summary", "args": [], "kwargs": {}}) + "\n",
        json.dumps(StatusEvent(status="done").to_dict()) + "\n",
    ]
    handle = sandbox.AgentRunHandle()
    sent: list = []
    handle._send_backend_response = sent.append
    events: list = []
    sandbox._relay(_FakeProc(lines), events.append, handle, None, backends={"experiments": db})
    assert sent and sent[0]["ok"] and sent[0]["result"] == {"x": 1}
    assert handle._done_observed is True  # the backend_request line didn't disrupt event flow
