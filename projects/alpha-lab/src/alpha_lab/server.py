"""FastAPI + WebSocket server for alpha-lab web dashboard.

Pure viewer mode: point at a --workspace to browse files and monitor a
running or completed analysis.  The pipeline is started separately via
``python run.py`` — this server never starts or stops it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from alpha_lab.events import AgentEvent, FileChangedEvent
from alpha_lab.metrics import MetricsCollector

logger = logging.getLogger("alpha_lab.server")


# ---------------------------------------------------------------------------
# Dashboard State
# ---------------------------------------------------------------------------


class DashboardState:
    """Holds async event queue and references used by read-only endpoints."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[AgentEvent] | None = None
        self.workspace: str = ""
        self.db: Any = None  # ExperimentDB, loaded lazily from workspace

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue = asyncio.Queue()

    def _try_load_db(self) -> None:
        """Load experiment DB from workspace if it exists."""
        if self.db is not None or not self.workspace:
            return
        db_path = os.path.join(self.workspace, "experiments.db")
        if os.path.exists(db_path):
            from alpha_lab.experiment_db import ExperimentDB
            self.db = ExperimentDB(db_path)


# ---------------------------------------------------------------------------
# Log Tailer — streams live JSONL events from the pipeline into the WS queue
# ---------------------------------------------------------------------------


class LogTailer:
    """Tails all .jsonl files in {workspace}/logs/ and pushes new events."""

    def __init__(
        self,
        workspace: str,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue,
    ) -> None:
        self.logs_dir = Path(workspace) / "logs"
        self._loop = loop
        self._queue = queue
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Track file positions: {path: offset}
        self._offsets: dict[str, int] = {}

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._tail, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _tail(self) -> None:
        """Poll for new lines in all .jsonl files every 0.5s."""
        while not self._stop_event.is_set():
            if self.logs_dir.exists():
                for log_path in self.logs_dir.glob("*.jsonl"):
                    self._read_new_lines(str(log_path))
            self._stop_event.wait(0.5)

    def _read_new_lines(self, path: str) -> None:
        offset = self._offsets.get(path, 0)
        try:
            with open(path) as f:
                f.seek(offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Create a minimal AgentEvent-like object for the broadcast loop
                        evt = _DictEvent(data)
                        self._loop.call_soon_threadsafe(self._queue.put_nowait, evt)
                    except json.JSONDecodeError:
                        pass
                self._offsets[path] = f.tell()
        except Exception:
            pass


class _DictEvent:
    """Thin wrapper so pre-parsed dicts work with the broadcast loop."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def to_dict(self) -> dict:
        return self._data


# ---------------------------------------------------------------------------
# File Watcher
# ---------------------------------------------------------------------------


class FileWatcher:
    """Watches workspace for file changes using watchfiles."""

    def __init__(
        self,
        workspace: str,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[AgentEvent],
    ) -> None:
        self.workspace = workspace
        self._loop = loop
        self._queue = queue
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _watch(self) -> None:
        try:
            from watchfiles import watch, Change

            for changes in watch(
                self.workspace,
                stop_event=self._stop_event,
                debounce=500,
                recursive=True,
            ):
                for change_type, path_str in changes:
                    rel_path = os.path.relpath(path_str, self.workspace)

                    if any(
                        part.startswith(".") or part == "__pycache__"
                        for part in Path(rel_path).parts
                    ):
                        continue

                    change_name = {
                        Change.added: "added",
                        Change.modified: "modified",
                        Change.deleted: "deleted",
                    }.get(change_type, "modified")

                    event = FileChangedEvent(
                        change=change_name,
                        path=rel_path,
                    )
                    self._loop.call_soon_threadsafe(
                        self._queue.put_nowait, event
                    )
        except Exception as e:
            logger.error(f"File watcher error: {e}")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(title="Alpha Lab")

_server_start_time: float = 0.0
_metrics_collector = MetricsCollector()

manager = DashboardState()
file_watcher: FileWatcher | None = None
log_tailer: LogTailer | None = None
connected_websockets: set[WebSocket] = set()
event_history: list[dict] = []  # All events, replayed to new WS connections
MAX_EVENT_HISTORY = 10_000  # Cap to prevent OOM on long runs
_broadcast_task: asyncio.Task | None = None  # prevent GC of broadcast loop
_status_report_task: asyncio.Task | None = None  # prevent GC of timer

# Set by serve_main.
_workspace: str = ""
_adapter_cache: dict = {}  # {"adapter": DomainAdapter | None}


def _load_jsonl_history() -> None:
    """Load events from all .jsonl log files into event_history."""
    logs_dir = Path(_workspace) / "logs"
    if not logs_dir.exists():
        return
    total_loaded = 0
    for log_path in sorted(logs_dir.glob("*.jsonl")):
        loaded = 0
        errors = 0
        try:
            with open(log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event_history.append(json.loads(line))
                        loaded += 1
                    except json.JSONDecodeError:
                        errors += 1
            if loaded:
                logger.info(f"Loaded {loaded} events from {log_path.name}" +
                             (f" ({errors} malformed lines skipped)" if errors else ""))
            total_loaded += loaded
        except Exception as e:
            logger.warning(f"Failed to load {log_path.name}: {e}")
    if total_loaded:
        # Sort by timestamp so events from multiple files are in order
        event_history.sort(key=lambda e: e.get("timestamp", 0))


@app.on_event("startup")
async def startup() -> None:
    global file_watcher, log_tailer, _broadcast_task, _status_report_task, _server_start_time

    _server_start_time = time.time()
    loop = asyncio.get_event_loop()
    manager.set_loop(loop)

    _broadcast_task = asyncio.create_task(_broadcast_loop())
    _status_report_task = asyncio.create_task(_status_report_timer())

    # Try to load adapter for metric-aware endpoints
    if _workspace:
        try:
            from alpha_lab.adapter_loader import resolve_adapter
            _adapter_cache["adapter"] = resolve_adapter(_workspace)
        except Exception as e:
            logger.warning(f"Failed to load adapter at startup: {e}")
            _adapter_cache["adapter"] = None

    # Load previous run's events if available
    if _workspace:
        _load_jsonl_history()

    # Load experiment DB if it exists
    manager._try_load_db()

    # Start log tailer — streams live JSONL events from the pipeline
    if _workspace:
        log_tailer = LogTailer(_workspace, loop, manager._queue)
        # Set offsets to end of existing files so we only get new events
        logs_dir = Path(_workspace) / "logs"
        if logs_dir.exists():
            for lp in logs_dir.glob("*.jsonl"):
                log_tailer._offsets[str(lp)] = lp.stat().st_size
        log_tailer.start()

    # Always start file watcher if workspace exists
    if _workspace and Path(_workspace).exists():
        file_watcher = FileWatcher(_workspace, loop, manager._queue)
        file_watcher.start()


async def _broadcast_loop() -> None:
    """Consume events from the queue and broadcast to all WebSockets."""
    while True:
        if manager._queue is None:
            await asyncio.sleep(0.1)
            continue

        try:
            event = await asyncio.wait_for(manager._queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue

        data = event.to_dict()
        event_history.append(data)
        if len(event_history) > MAX_EVENT_HISTORY:
            # Trim oldest 20% to avoid trimming every iteration
            trim_count = MAX_EVENT_HISTORY // 5
            del event_history[:trim_count]

        dead: set[WebSocket] = set()
        for ws in list(connected_websockets):  # snapshot to avoid set-changed-during-iteration
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        connected_websockets.difference_update(dead)


async def _generate_and_broadcast_status() -> None:
    """Generate a status report and broadcast it to all connected clients."""
    if not _workspace:
        return
    try:
        from alpha_lab.output_generator import OutputGenerator
        gen = OutputGenerator(_workspace, adapter=_adapter_cache.get("adapter"))
        report = gen.generate_status_report()
        data = {"type": "status_report", "timestamp": time.time(), **report}
        event_history.append(data)

        dead: set[WebSocket] = set()
        for ws in list(connected_websockets):
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        connected_websockets.difference_update(dead)
    except Exception as e:
        logger.error(f"Status report broadcast failed: {e}")


_STATUS_REPORT_INTERVAL = 600  # seconds (10 minutes)


async def _status_report_timer() -> None:
    """Auto-generate status reports every 10 minutes while agent is running."""
    while True:
        await asyncio.sleep(_STATUS_REPORT_INTERVAL)
        if connected_websockets:
            await _generate_and_broadcast_status()


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


# Event types the frontend actually renders (skip bloated internal events)
_REPLAY_TYPES = {
    "status", "tool_call", "tool_result", "phase", "experiment",
    "board_summary", "question", "error", "file_changed", "status_report",
}

# Max size (chars) for a single replayed event payload
_REPLAY_MAX_CHARS = 20_000


def _slim_event(data: dict) -> dict | None:
    """Return a trimmed copy of *data* suitable for replay, or None to skip."""
    etype = data.get("type", "")

    # Skip event types the frontend ignores (api_request, api_response, etc.)
    if etype not in _REPLAY_TYPES:
        return None

    # Truncate oversized string fields (e.g. tool outputs)
    slim = dict(data)
    for key in ("output", "arguments", "content"):
        val = slim.get(key)
        if isinstance(val, str) and len(val) > _REPLAY_MAX_CHARS:
            slim[key] = val[:_REPLAY_MAX_CHARS] + "\n··· truncated for replay ···"
    return slim


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()

    # Add client first so it receives live events during replay
    connected_websockets.add(ws)

    # Replay a trimmed snapshot of history (skip bloated internal events)
    history_snapshot = list(event_history)
    for data in history_snapshot:
        slim = _slim_event(data)
        if slim is None:
            continue
        try:
            await ws.send_json(slim)
        except Exception:
            connected_websockets.discard(ws)
            return

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")

            if msg_type == "generate_status":
                asyncio.ensure_future(_generate_and_broadcast_status())

    except WebSocketDisconnect:
        pass
    finally:
        connected_websockets.discard(ws)


# ---------------------------------------------------------------------------
# REST API endpoints
# ---------------------------------------------------------------------------


@app.get("/api/status")
async def get_status() -> JSONResponse:
    return JSONResponse({
        "workspace": _workspace,
    })


@app.get("/api/files")
async def get_files() -> JSONResponse:
    """Return workspace directory tree as nested JSON."""
    if not _workspace or not Path(_workspace).exists():
        return JSONResponse({"children": []})

    def build_tree(dir_path: Path, rel_prefix: str = "") -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        try:
            entries = sorted(dir_path.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        except PermissionError:
            return items

        for entry in entries:
            if entry.name.startswith(".") or entry.name == "__pycache__":
                continue

            rel = f"{rel_prefix}/{entry.name}" if rel_prefix else entry.name

            if entry.is_dir():
                items.append({
                    "name": entry.name,
                    "path": rel,
                    "type": "directory",
                    "children": build_tree(entry, rel),
                })
            else:
                items.append({
                    "name": entry.name,
                    "path": rel,
                    "type": "file",
                })

        return items

    tree = build_tree(Path(_workspace))
    return JSONResponse({"children": tree})


@app.get("/api/file/{path:path}")
async def get_file(path: str) -> JSONResponse:
    """Return file content. Text as JSON, images as base64."""
    if not _workspace:
        return JSONResponse({"error": "No workspace"}, status_code=400)

    full_path = (Path(_workspace) / path).resolve()
    workspace_resolved = Path(_workspace).resolve()
    try:
        full_path.relative_to(workspace_resolved)
    except ValueError:
        return JSONResponse({"error": "Path traversal denied"}, status_code=403)

    if not full_path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)

    if not full_path.is_file():
        return JSONResponse({"error": "Not a file"}, status_code=400)

    suffix = full_path.suffix.lower()
    image_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
    }

    if suffix in image_types:
        import base64
        data = base64.b64encode(full_path.read_bytes()).decode("ascii")
        return JSONResponse({
            "type": "image",
            "media_type": image_types[suffix],
            "data": data,
            "name": full_path.name,
        })

    try:
        content = full_path.read_text(errors="replace")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse({
        "type": "text",
        "content": content,
        "name": full_path.name,
    })


# ---------------------------------------------------------------------------
# Health & Metrics endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def get_health() -> JSONResponse:
    """Return structured health status of all components."""
    uptime = time.time() - _server_start_time if _server_start_time else 0.0

    # Database status
    manager._try_load_db()
    db_status: dict[str, Any] = {"status": "not_configured"}
    if manager.db is not None:
        try:
            board = manager.db.board_summary()
            db_status = {"status": "ok", "board": board}
        except Exception as e:
            db_status = {"status": "error", "error": str(e)}

    overall = "ok"
    if db_status.get("status") == "error":
        overall = "degraded"

    return JSONResponse({
        "status": overall,
        "uptime_seconds": round(uptime, 2),
        "components": {
            "database": db_status,
            "metrics": _metrics_collector.snapshot(),
        },
    })


@app.get("/api/metrics")
async def get_metrics() -> JSONResponse:
    """Return MetricsCollector snapshot."""
    return JSONResponse(_metrics_collector.snapshot())


# ---------------------------------------------------------------------------
# Experiment API endpoints (Phase 3)
# ---------------------------------------------------------------------------


@app.get("/api/experiments")
async def get_experiments() -> JSONResponse:
    """Return all experiments from the board."""
    manager._try_load_db()
    if manager.db is None:
        return JSONResponse({"experiments": []})
    experiments = manager.db.list_all()
    return JSONResponse({
        "experiments": [
            {
                "id": exp.id,
                "name": exp.name,
                "description": exp.description,
                "hypothesis": exp.hypothesis,
                "status": exp.status,
                "config_json": exp.config_json,
                "worker_id": exp.worker_id,
                "slurm_job_id": exp.slurm_job_id,
                "results_json": exp.results_json,
                "error": exp.error,
                "debrief_path": exp.debrief_path,
                "created_at": exp.created_at,
                "updated_at": exp.updated_at,
                "started_at": exp.started_at,
                "finished_at": exp.finished_at,
            }
            for exp in experiments
        ],
        "summary": manager.db.board_summary(),
    })


@app.get("/api/experiments/{exp_id}")
async def get_experiment(exp_id: int) -> JSONResponse:
    """Return a single experiment's details."""
    manager._try_load_db()
    if manager.db is None:
        return JSONResponse({"error": "No experiment database"}, status_code=404)
    exp = manager.db.get(exp_id)
    if exp is None:
        return JSONResponse({"error": "Experiment not found"}, status_code=404)
    return JSONResponse({
        "id": exp.id,
        "name": exp.name,
        "description": exp.description,
        "hypothesis": exp.hypothesis,
        "status": exp.status,
        "config_json": exp.config_json,
        "worker_id": exp.worker_id,
        "slurm_job_id": exp.slurm_job_id,
        "results_json": exp.results_json,
        "error": exp.error,
        "debrief_path": exp.debrief_path,
        "created_at": exp.created_at,
        "updated_at": exp.updated_at,
        "started_at": exp.started_at,
        "finished_at": exp.finished_at,
    })


@app.post("/api/experiments/{exp_id}/cancel")
async def cancel_experiment(exp_id: int) -> JSONResponse:
    """Cancel an experiment by setting its status to 'cancelled'."""
    manager._try_load_db()
    if manager.db is None:
        return JSONResponse({"error": "No experiment database"}, status_code=404)
    exp = manager.db.get(exp_id)
    if exp is None:
        return JSONResponse({"error": "Experiment not found"}, status_code=404)
    terminal = {"done", "cancelled", "analyzed"}
    if exp.status in terminal:
        return JSONResponse(
            {"error": f"Cannot cancel experiment in '{exp.status}' state"},
            status_code=409,
        )
    try:
        manager.db.update_status(exp_id, "cancelled")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({"status": "cancelled", "id": exp_id})


@app.get("/api/status-report")
async def get_status_report() -> JSONResponse:
    """Generate and return a structured status report."""
    if not _workspace:
        return JSONResponse({"error": "No workspace"}, status_code=400)
    try:
        from alpha_lab.output_generator import OutputGenerator
        gen = OutputGenerator(_workspace, adapter=_adapter_cache.get("adapter"))
        report = gen.generate_status_report()
        return JSONResponse(report)
    except Exception as e:
        logger.error(f"Status report generation failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/leaderboard")
async def get_leaderboard() -> JSONResponse:
    """Return top experiments by primary metric."""
    manager._try_load_db()
    if manager.db is None:
        return JSONResponse({"leaderboard": []})
    import json as _json

    # Use adapter metric if available
    _metric = "sharpe"
    _direction = "maximize"
    _display_name = "Sharpe"
    _adapter = _adapter_cache.get("adapter")
    if _adapter is not None:
        _metric = _adapter.metric.primary_metric
        _direction = _adapter.metric.direction
        _display_name = _adapter.metric.display_name

    leaders = manager.db.leaderboard(_metric, 20)

    def _safe_metrics(results_json: str | None) -> dict:
        if not results_json:
            return {}
        try:
            return _json.loads(results_json)
        except (ValueError, TypeError):
            return {}

    return JSONResponse({
        "metric": _metric,
        "direction": _direction,
        "display_name": _display_name,
        "leaderboard": [
            {
                "id": exp.id,
                "name": exp.name,
                "status": exp.status,
                "metrics": _safe_metrics(exp.results_json),
            }
            for exp in leaders
        ],
    })


# ---------------------------------------------------------------------------
# Chat endpoint for status queries
# ---------------------------------------------------------------------------


@app.post("/api/chat")
async def chat_query(body: dict) -> JSONResponse:
    """Answer questions about the system status using an LLM.

    Accepts {"question": "what's happening?"} and returns {"answer": "..."}.
    """
    if not _workspace:
        return JSONResponse({"error": "Workspace not configured"}, status_code=400)

    question = body.get("question", "").strip()
    if not question:
        return JSONResponse({"error": "No question provided"}, status_code=400)

    try:
        # Build context from current state
        context_parts = []

        # Board summary
        manager._try_load_db()
        if manager.db is not None:
            summary = manager.db.board_summary()
            context_parts.append("## Experiment Board Status")
            for status, count in sorted(summary.items()):
                context_parts.append(f"  {status}: {count}")

            # Leaderboard
            _chat_metric = "sharpe"
            _chat_metric_display = "Sharpe"
            if _adapter_cache.get("adapter") is not None:
                _chat_metric = _adapter_cache["adapter"].metric.primary_metric
                _chat_metric_display = _adapter_cache["adapter"].metric.display_name
            leaders = manager.db.leaderboard(_chat_metric, 5)
            if leaders:
                context_parts.append(f"\n## Top 5 Experiments (by {_chat_metric_display})")
                for i, exp in enumerate(leaders, 1):
                    metrics = {}
                    if exp.results_json:
                        try:
                            metrics = json.loads(exp.results_json)
                        except (ValueError, TypeError):
                            pass
                    val = metrics.get(_chat_metric, "?")
                    context_parts.append(f"  {i}. {exp.name} — {_chat_metric_display}: {val}")

            # Recent errors
            all_exps = manager.db.list_all()
            errors = [e for e in all_exps if e.error][-5:]
            if errors:
                context_parts.append("\n## Recent Errors")
                for exp in errors:
                    context_parts.append(f"  #{exp.id} {exp.name}: {exp.error[:100]}...")

        # System status
        context_parts.append(f"\n## System Status")
        context_parts.append(f"  Workspace: {_workspace}")

        # Read recent log entries
        log_path = Path(_workspace) / "logs" / "dispatcher.jsonl"
        if log_path.exists():
            try:
                lines = log_path.read_text().strip().split("\n")[-20:]
                context_parts.append("\n## Recent Dispatcher Log (last 20 entries)")
                for line in lines:
                    try:
                        entry = json.loads(line)
                        action = entry.get("action", "unknown")
                        context_parts.append(f"  {action}: {json.dumps({k: v for k, v in entry.items() if k not in ['t', 'action']})[:100]}")
                    except (ValueError, TypeError):
                        pass
            except Exception:
                pass

        context = "\n".join(context_parts)

        # Call LLM to answer
        from alpha_lab.client import get_provider
        chat_provider = get_provider("openai")
        try:
            answer = chat_provider.complete(
                model="gpt-4.1-mini",
                system=(
                    "You are a helpful assistant for Alpha Lab, an autonomous ML research system. "
                    "Answer the user's question about the system status based on the context provided. "
                    "Be concise and specific. If you don't have enough information, say so."
                ),
                messages=[
                    {"role": "user", "content": f"## Current System State\n{context}\n\n## User Question\n{question}"},
                ],
            )
        finally:
            if hasattr(chat_provider, 'openai_client'):
                try:
                    chat_provider.openai_client.close()
                except Exception:
                    pass

        return JSONResponse({"answer": answer})

    except Exception as e:
        logger.error(f"Chat query failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Static files (must be last)
# ---------------------------------------------------------------------------

_frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def serve_main() -> None:
    """CLI entry point for the web dashboard server.

    This server is a passive viewer — it does not start or stop the pipeline.
    Run the pipeline separately via ``python run.py``, then point this server
    at the same workspace to monitor progress.
    """
    parser = argparse.ArgumentParser(
        prog="alpha-lab-serve",
        description="Alpha Lab Web Dashboard — passive viewer for a running or completed pipeline",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="Workspace directory path to monitor",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    global _workspace
    _workspace = os.path.abspath(args.workspace)
    manager.workspace = _workspace

    print(f"Workspace: {_workspace}")

    Path(_workspace).mkdir(parents=True, exist_ok=True)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    serve_main()
