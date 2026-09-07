"""bwrap sandboxing for agent runs.

When bwrap is available, :func:`run_agent` executes an agent in a subprocess
(:mod:`alpha_lab.sandboxing.runner`) confined to its workspace (RW), with the agent's
``blocked_paths`` overlaid empty + read-only so it can neither read nor clobber their real
contents — always including ``.alpha_lab`` (the parent-owned board/memory, reached only via
the gateway) and by default the held-out ``private`` set. Dirs are blocked with a read-only
tmpfs, files with a ``/dev/null`` bind. Plus the repo (RO), venv (RO), auth-token cache (RO),
and — when the agent runs GPU code — the NVIDIA devices. Events are relayed from the child's
stdout; ``ask_user`` and ``stop`` are relayed to its stdin. When bwrap is absent (or
``ALPHALAB_AGENT_NOSANDBOX`` is set), the agent runs in-process exactly as before.

Namespace policy: only the PID namespace is unshared. The network and user namespaces
are inherited so the agent keeps its Kerberos/SCV identity and can reach the LLM gateway.
"""

from __future__ import annotations

import functools
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from alpha_lab import deps
from alpha_lab import mlflow_logger
from alpha_lab.agents import load_agent
from alpha_lab.agents.factory import build_agent
from alpha_lab.benchmarks.paths import find_repo_root
from alpha_lab.config import config_to_jsonable
from alpha_lab.sandboxing.db_proxy import decode_result, encode_result
from alpha_lab.memory import MemoryStore
from alpha_lab import token_metrics
from alpha_lab.events import (
    AgentEvent,
    ErrorEvent,
    MetricsEvent,
    StatusEvent,
    TokenUsageEvent,
    event_from_dict,
)
from alpha_lab.process_control import escalate_termination
from alpha_lab.tracing import inject_context

logger = logging.getLogger("alpha_lab.sandboxing.sandbox")

RUNNER_MODULE = "alpha_lab.sandboxing.runner"
NOSANDBOX_ENV = "ALPHALAB_AGENT_NOSANDBOX"
TOKEN_CACHE_DIR = "/var/tmp/svcacct"  # default on-prem bearer-token cache dir; see _token_cache_dirs TODO: Remove this in favor of a per-agent token cache under the workspace, populated by the parent and mounted RO


def _token_cache_dirs() -> list[str]:
    """Directories holding the on-prem bearer-token cache (best-effort RO mounts).

    Resolved through the auth module rather than hardcoded, so a ``TOKEN_FILEPATH``
    override reaches sandboxed children too — the child inherits the env var, so
    binding only the default dir would leave it pointing at an unmounted path.
    Imported lazily to keep the heavier ``providers`` package off the import path
    of callers that only sandbox subprocesses.
    """
    try:
        from ..providers.utils.auth import token_cache_candidates
    except ImportError:
        return [TOKEN_CACHE_DIR]
    # dict.fromkeys dedupes while preserving the candidate order.
    return list(dict.fromkeys(str(p.parent) for p in token_cache_candidates()))
# Kerberos credentials the agent's on-prem tooling authenticates with: the ticket cache(s)
# in /tmp (KRB5CCNAME default) and the renewable-ticket spool. /tmp is a tmpfs in the sandbox,
# so these binds layer on top of it.
KERBEROS_CC_GLOB = "/tmp/krb5cc*"
KERBEROS_TICKET_SPOOL = "/var/spool/tickets"
# cinit credential dir (holds the mTLS client cert+key, etc.) 
CINIT_CC_ENV = "CINITCCNAME"
_NVIDIA_PROC = "/proc/driver/nvidia"


def _int_env(name: str, default: int) -> int:
    """Read a positive integer override from the environment, else ``default``."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default
    return value if value > 0 else default


_MAX_RELAYED_STDERR = _int_env("ALPHALAB_SANDBOX_MAX_RELAYED_STDERR", 4000)
# stop(): grace for the child to exit cooperatively, then to honor SIGTERM, before SIGKILL.
# The cooperative grace must exceed a normal shutdown (final metric flush + run_deps.close()
# + MLflow run finalization over the network), so a slow-but-correct child isn't killed.
_STOP_GRACE_SECONDS = _int_env("ALPHALAB_SANDBOX_STOP_GRACE_SECONDS", 30)
_STOP_KILL_GRACE_SECONDS = _int_env("ALPHALAB_SANDBOX_STOP_KILL_GRACE_SECONDS", 5)


def _is_available() -> bool:
    """True iff bwrap is on PATH and sandboxing is not explicitly disabled."""
    return shutil.which("bwrap") is not None and not os.environ.get(NOSANDBOX_ENV)


class AgentRunHandle:
    """External control surface for an in-flight :func:`run_agent` call.

    ``run_agent`` allocates one, attaches either the in-process ``AgentLoop`` or the
    subprocess ``Popen`` once it exists, and ``stop`` / ``provide_answer`` dispatch to
    whichever is active. Control messages issued before a target is attached are buffered
    and flushed on attach, so a stop sent during spawn/setup is never dropped.
    """

    def __init__(self) -> None:
        self._agent: Any | None = None
        self._proc: subprocess.Popen | None = None
        self._done_observed = False
        self._lock = threading.Lock()
        self._pending: list[dict] = []

    def stop(self) -> None:
        self._record({"type": "stop"})

    def provide_answer(self, text: str) -> None:
        self._record({"type": "answer", "text": text})

    @property
    def done(self) -> bool:
        if self._agent is not None:
            return bool(self._agent._done)
        return self._done_observed

    def _record(self, message: dict) -> None:
        with self._lock:
            if self._agent is not None:
                self._dispatch_to_agent(message)
            elif self._proc is not None:
                self._send_to_proc(message)
            else:
                self._pending.append(message)  # no target yet — apply on attach

    def _dispatch_to_agent(self, message: dict) -> None:
        if message["type"] == "stop":
            self._agent.stop()
        elif message["type"] == "answer":
            self._agent.provide_answer(message["text"])

    def _send_to_proc(self, message: dict) -> None:
        """Write a control message to the child; for ``stop``, also escalate to a forced
        terminate/kill if the child doesn't exit (it may be wedged and not reading stdin)."""
        self._write_message(self._proc, message)
        if message.get("type") == "stop":
            self._escalate_stop()

    def _send_backend_response(self, message: dict) -> None:
        """Reply to a child's backend_request. Locked so it serializes with answer/stop
        writes (the relay thread sends these while the orchestrator thread may send control)."""
        with self._lock:
            self._write_message(self._proc, message)

    @staticmethod
    def _write_message(proc: subprocess.Popen | None, message: dict) -> None:
        if proc is None or proc.stdin is None or proc.poll() is not None:
            return
        try:
            proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            pass

    def _escalate_stop(self) -> None:
        """Give the child a grace period to stop cooperatively, then terminate, then kill.

        Runs off-thread: ``_record`` holds ``self._lock`` and the grace can be tens of
        seconds. Single-PID signals suffice — bwrap's ``--unshare-pid`` tears down the
        whole namespace when its supervisor process dies.
        """
        proc = self._proc
        if proc is None:
            return
        threading.Thread(
            target=escalate_termination,
            args=(proc,),
            kwargs={
                "cooperative_grace": _STOP_GRACE_SECONDS,
                "term_grace": _STOP_KILL_GRACE_SECONDS,
                "kill_grace": _STOP_KILL_GRACE_SECONDS,
            },
            daemon=True,
        ).start()

    def _attach_agent(self, agent: Any) -> None:
        with self._lock:
            self._agent = agent
            for message in self._pending:
                self._dispatch_to_agent(message)
            self._pending.clear()

    def _attach_proc(self, proc: subprocess.Popen, payload: dict) -> None:
        # Write the spec payload before taking the lock: it can be large (initial_message /
        # extra_context) and a blocking write must not stall a concurrent stop() waiting on
        # the lock. Control buffered meanwhile lands in _pending and is flushed after.
        self._write_message(proc, payload)
        with self._lock:
            self._proc = proc
            for message in self._pending:
                self._send_to_proc(message)
            self._pending.clear()


def _venv_mount_paths() -> list[str]:
    """RO paths that make the venv/base Python interpreter exec'able in the sandbox."""
    paths = [sys.prefix]
    if sys.base_prefix != sys.prefix:
        paths.append(sys.base_prefix)
    return paths


@functools.lru_cache(maxsize=1)
def _nvidia_args() -> tuple[str, ...]:
    """bwrap args binding the NVIDIA devices and driver proc dir.

    Device creation belongs to the host. Re-running ``nvidia-modprobe`` here can replace
    device nodes that active sandboxes have already bind-mounted.
    """
    if not Path(_NVIDIA_PROC).is_dir():
        return ()
    gpu_devices = sorted(glob.glob("/dev/nvidia[0-9]*"))
    required_devices = [
        "/dev/nvidiactl",
        "/dev/nvidia-uvm",
    ]
    missing = [device for device in required_devices if not Path(device).exists()]
    if not gpu_devices:
        missing.append("/dev/nvidiaN")
    if missing:
        raise RuntimeError(
            "GPU sandbox requires host-created NVIDIA device nodes; missing: "
            + ", ".join(missing)
        )
    devices = [
        *required_devices,
        "/dev/nvidia-uvm-tools",
        "/dev/nvidia-caps",
        *gpu_devices,
    ]
    args: list[str] = []
    for device in devices:
        if Path(device).exists():
            args += ["--dev-bind", device, device]
    args += ["--ro-bind", _NVIDIA_PROC, _NVIDIA_PROC]  # after --proc; NVML reads it
    return tuple(args)


def _build_bwrap_prefix(
    *,
    ro_paths: list[str],
    rw_paths: list[str],
    blocked_paths: list[str],
    chdir: str,
    project_ro_root: str,
    needs_gpu: bool,
) -> list[str]:
    prefix = [
        "bwrap",
        "--unshare-pid",
        "--die-with-parent",
        "--new-session",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/etc", "/etc",
        "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/sbin", "/sbin",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--ro-bind", "/sys", "/sys",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
    ]
    if needs_gpu:
        prefix += _nvidia_args()
    prefix += ["--ro-bind", project_ro_root, project_ro_root]
    for venv_path in _venv_mount_paths():
        prefix += ["--ro-bind", venv_path, venv_path]
    for ro_path in ro_paths:
        prefix += ["--ro-bind-try", ro_path, ro_path]
    for rw_path in rw_paths:
        prefix += ["--bind-try", rw_path, rw_path]
    # Block each path with an empty, read-only overlay (placed after the RW workspace bind so
    # the overlay wins over the real contents). Dirs get a tmpfs remounted read-only; files
    # get a /dev/null bind — either way the agent can't read the real contents or write the
    # path. Classify by the real on-disk type when the path exists (accurate
    # for dotted dirs / extensionless files), falling back to the suffix heuristic only for a
    # not-yet-created path.
    for blocked_path in blocked_paths:
        p = Path(blocked_path)
        is_file = p.is_file() if p.exists() else bool(p.suffix)
        if is_file:
            prefix += ["--ro-bind", "/dev/null", blocked_path]
        else:
            prefix += ["--tmpfs", blocked_path, "--remount-ro", blocked_path]
    prefix += ["--chdir", chdir, "--"]
    return prefix


def _popen(
    argv: list[str],
    *,
    ro_paths: list[str],
    rw_paths: list[str],
    blocked_paths: list[str],
    chdir: str,
    project_ro_root: str,
    needs_gpu: bool,
    **kwargs: Any,
) -> subprocess.Popen:
    """Spawn *argv* under bwrap with the given mounts; stdin/stdout/stderr are pipes."""
    command = _build_bwrap_prefix(
        ro_paths=ro_paths,
        rw_paths=rw_paths,
        blocked_paths=blocked_paths,
        chdir=chdir,
        project_ro_root=project_ro_root,
        needs_gpu=needs_gpu,
    ) + argv
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        **kwargs,
    )


# Runtime keys that are NOT build_agent kwargs: the run-only initial_message, and the
# run-level values build_agent reads from the active RunDeps (deps.get()) rather than receiving.
_NON_BUILD_KWARGS = frozenset({"initial_message", "config", "workspace", "api_key"})


def build_kwargs_from_runtime(runtime: dict[str, Any]) -> dict[str, Any]:
    """The runtime values that are ``build_agent`` kwargs. Excludes ``initial_message`` (run-only)
    and ``config``/``workspace``/``api_key`` (build_agent reads those from the active RunDeps).
    Shared by the in-process path and the bwrap child."""
    return {key: value for key, value in runtime.items() if key not in _NON_BUILD_KWARGS}


def _create_rw_sources(rw_paths: Iterable[Path]) -> None:
    """Create each read-write bind source so bwrap can bind it. Existing paths are left
    as-is; a missing path with a file extension (e.g. ``experiments.db``, ``playbook.md``)
    is touched as a file, anything else is created as a directory."""
    for path in rw_paths:
        if path.exists():
            continue
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        else:
            path.mkdir(parents=True, exist_ok=True)


def _service_backend_request(data: dict, backends: dict, handle: AgentRunHandle) -> None:
    """Apply a sandboxed child's store call against the parent's single connection and reply.

    Routes by ``target`` ("experiments"/"memory") to the backend the parent owns. The
    result/error is sent back only after the write commits, so the child's blocked tool sees
    the real outcome (see :mod:`alpha_lab.sandboxing.db_proxy`)."""
    req_id = data.get("req_id")
    target = data.get("target", "")
    method_name = data.get("method", "")
    backend = backends.get(target)
    if backend is None or method_name.startswith("_"):
        handle._send_backend_response({
            "type": "backend_response", "req_id": req_id, "ok": False,
            "error": f"backend {target!r} method {method_name!r} unavailable in parent",
        })
        return
    try:
        args = decode_result(data.get("args", []))
        kwargs = decode_result(data.get("kwargs", {}))
        result = getattr(backend, method_name)(*args, **kwargs)
        response = {"type": "backend_response", "req_id": req_id, "ok": True,
                    "result": encode_result(result)}
    except Exception as exc:
        response = {"type": "backend_response", "req_id": req_id, "ok": False,
                    "error": f"{type(exc).__name__}: {exc}"}
    handle._send_backend_response(response)


def _relay(
    proc: subprocess.Popen,
    event_callback: Callable[[AgentEvent], None],
    handle: AgentRunHandle,
    metrics: Any | None,
    backends: dict | None = None,
) -> None:
    """Forward the child's stdout JSONL events; merge its metrics; surface stderr on error.

    Also services the child's ``backend_request`` lines against ``backends`` (the parent owns
    the only connection to each store).
    """
    stderr_chunks: list[str] = []

    def drain_stderr() -> None:
        for line in proc.stderr:  # type: ignore[union-attr]
            stderr_chunks.append(line)
            logger.info("%s", line.rstrip())

    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    stderr_thread.start()

    for line in proc.stdout:  # type: ignore[union-attr]
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("ignoring non-JSON child stdout: %s", line)
            continue
        if data.get("type") == "backend_request":
            _service_backend_request(data, backends or {}, handle)
            continue
        event = event_from_dict(data)
        if isinstance(event, TokenUsageEvent):
            # Cortex-only control signal: apply to the parent's single per-run counter
            # and do NOT forward downstream (keeps it out of the logs and dashboard).
            token_metrics.apply(
                event.model, event.system,
                event.input_tokens, event.output_tokens,
                event.cache_read, event.cache_write,
            )
            continue
        if isinstance(event, StatusEvent) and event.status == "done":
            handle._done_observed = True
        elif isinstance(event, MetricsEvent) and metrics is not None:
            metrics.merge_counters(event.counters)  # fold the child's run into the parent
        event_callback(event)

    return_code = proc.wait()
    stderr_thread.join(timeout=5)
    if return_code != 0:
        detail = "".join(stderr_chunks)[-_MAX_RELAYED_STDERR:]
        event_callback(ErrorEvent(message=f"Sandboxed agent exited {return_code}: {detail}"))
        event_callback(StatusEvent(status="error", detail=f"sandboxed agent exit {return_code}"))


def run_agent(
    agent_id: str,
    event_callback: Callable[[AgentEvent], None],
    *,
    initial_message: str,
    extra_context: str | None = None,
    log_name: str | None = None,
    tools_include: tuple[str, ...] | None = None,
    db_path: str | None = None,
    mlflow_run_target: str | None = None,
    provider: Any | None = None,
    db: Any | None = None,
    adapter: Any | None = None,
    metrics: Any | None = None,
    owner: object | None = None,
    handle_attr: str = "_run_handle",
) -> AgentRunHandle:
    """Run an agent, sandboxed under bwrap when available, else in-process.

    Must run within an active :class:`~alpha_lab.deps.RunDeps`: ``config``/``workspace``/
    ``api_key`` are sourced from ``deps.get(strict=True)`` (and serialized into ``runtime`` so the child
    can rebuild its own RunDeps). Owns the run handle: allocates it, publishes it on
    ``owner.<handle_attr>`` and clears it on return, then returns it. Passing ``owner`` is
    REQUIRED for ``stop()``/``done`` on the orchestrator to reach the run — the handle must be
    published *before* this call blocks on the run, so the returned value alone cannot serve a
    concurrent ``stop()``. Per-owner reentrancy is bounded by callers' busy-guards (one agent
    per orchestrator at a time).
    """
    run = deps.get(strict=True)
    runtime = {
        "workspace": str(run.workspace),
        "initial_message": initial_message,
        "config": run.config,
        "db_path": db_path,
        "extra_context": extra_context,
        "log_name": log_name,
        "tools_include": tools_include,
        "mlflow_run_target": mlflow_run_target,
        "api_key": run.api_key,
    }
    agent_definition = load_agent(agent_id)
    handle = AgentRunHandle()
    if owner is not None:
        setattr(owner, handle_attr, handle)
    try:
        if not _is_available():
            agent = build_agent(
                agent_definition,
                event_callback=event_callback,
                provider=provider,
                db=db,
                adapter=adapter,
                metrics=metrics,
                **build_kwargs_from_runtime(runtime),
            )
            handle._attach_agent(agent)
            agent.run(runtime["initial_message"])
            return handle

        # The agent is scoped to its workspace (RW); its blocked_paths (always including
        # .alpha_lab, which houses the board and memory) are overlaid empty + read-only. Those
        # stores are reached only through the parent gateway (below), never the on-disk files.
        workspace = runtime["workspace"]
        workspace_path = Path(workspace)
        _create_rw_sources({workspace_path})
        rw_paths = [str(workspace_path)]
        config = runtime["config"]
        ro_paths = []
        if config.data_path:
            ro_paths.append(config.data_path)
        ro_paths += _token_cache_dirs()
        ro_paths += [*sorted(glob.glob(KERBEROS_CC_GLOB)), KERBEROS_TICKET_SPOOL]
        if cinit_dir := os.environ.get(CINIT_CC_ENV):
            ro_paths.append(cinit_dir)
        blocked_paths = [
            str(workspace_path / name) for name in agent_definition.blocked_paths
        ]

        proc = _popen(
            [sys.executable, "-m", RUNNER_MODULE],
            ro_paths=ro_paths,
            rw_paths=rw_paths,
            blocked_paths=blocked_paths,
            chdir=str(workspace),
            project_ro_root=str(find_repo_root()),
            needs_gpu=agent_definition.needs_gpu,
        )

        # The workspace crosses the boundary as a plain string; only config needs encoding.
        config_payload = config_to_jsonable(runtime["config"])
        child_runtime = {
            **runtime,
            "config": config_payload,
        }
        payload = {
            "agent_id": agent_id,
            "runtime": child_runtime,
            "otel_carrier": inject_context(),
            # Propagate the active pipeline-Run context so the child's agent_trace builds an
            # AGENT-rooted trace linked to the same Run (matching the in-process structure).
            "mlflow_context": mlflow_logger.export_run_context(),
        }

        handle._attach_proc(proc, payload)

        backends: dict[str, Any] = {"memory": deps.memory_store}
        if db is not None:
            backends["experiments"] = db

        _relay(proc, event_callback, handle, metrics, backends=backends)
        return handle
    finally:
        if owner is not None:
            setattr(owner, handle_attr, None)
