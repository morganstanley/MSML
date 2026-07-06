"""Child entrypoint for a sandboxed agent run: ``python -m alpha_lab.sandboxing.runner``.

Reads one JSON payload line on stdin (``{"agent_id", "runtime"}``), rebuilds the agent
via the shared :func:`alpha_lab.agents.factory.build_agent`, runs it, relays events as
JSONL on stdout, and accepts ``answer``/``stop``/``backend_response`` control messages on stdin.

 The agent never opens ``experiments.db`` / ``memory.db`` itself (unsafe across processes, esp. on NFS):
 its store tool-calls go through :class:`~alpha_lab.sandboxing.db_proxy.BackendProxy` instances that the
 parent services against the single connection it owns.

stdout carries event JSONL and ``backend_request`` lines; logging and tracebacks go to stderr.
"""

from __future__ import annotations

import json
import logging
import queue
import sys
import threading
import uuid
from dataclasses import asdict

from alpha_lab import deps, mlflow_logger
from alpha_lab.agents import load_agent
from alpha_lab.sandboxing.db_proxy import BackendProxy, ProxyChannel
from alpha_lab.agents.factory import build_agent
from alpha_lab.sandboxing.sandbox import build_kwargs_from_runtime
from alpha_lab.config import task_config_from_mapping
from alpha_lab.events import AgentEvent, MetricsEvent, QuestionEvent
from alpha_lab.metrics import MetricsCollector
from alpha_lab.tracing import (
    attach_context,
    context_from_carrier,
    detach_context,
    init_tracing,
)

logger = logging.getLogger("alpha_lab.sandboxing.runner")
_ANSWER_POLL_SECONDS = 1.0


def _write_line(payload: dict) -> None:
    """Write one JSONL line to stdout (events and db_requests; the parent's relay reads these)."""
    sys.stdout.write(json.dumps(payload, default=str) + "\n")
    sys.stdout.flush()


def _emit(event: AgentEvent) -> None:
    _write_line(event.to_dict())


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    payload = json.loads(sys.stdin.readline())
    agent_id = payload["agent_id"]
    runtime = payload["runtime"]

    agent_definition = load_agent(agent_id)
    # config crossed the boundary serialized; rebuild the live object once. The workspace
    # is a plain string and needs no reconstruction.
    config = task_config_from_mapping(runtime["config"])
    runtime["config"] = config

    # Restore the observability the parent had, mirroring run.py's mutually-exclusive
    # backend selection (--mlflow ⇒ MLflow only, no OTel TracerProvider), then nest the
    # child's spans under the parent run's trace via the propagated context.
    if mlflow_logger.is_active():
        mlflow_logger.configure_sdk()
    else:
        init_tracing()
    otel_token = attach_context(context_from_carrier(payload.get("otel_carrier", {})))

    # Publish the child's own RunDeps (separate process; the parent's global can't cross), so
    # deps.get()-using tools and build_agent below read config/workspace/api_key from it.
    run_deps = deps.RunDeps(
        config, workspace=runtime["workspace"], api_key=runtime["api_key"]
    )
    run_deps.open()

    # The parent's MetricsCollector can't cross the process boundary; record into a local
    # one and emit incremental deltas (merged additively by the parent) so a hard crash
    # loses at most the last recorded counter rather than the whole run's.
    def flush_metrics() -> None:
        nonlocal last_counters
        current, delta = metrics.delta_since(last_counters)
        if not delta.is_zero():
            # asdict only here: MetricsEvent crosses the process boundary as JSON.
            _emit(MetricsEvent(counters=asdict(delta)))
            last_counters = current

    # Flush from the recording site (on_record) rather than guessing which events change
    # counters — robust if a new counter is recorded outside the API/error path, and it
    # ships each delta as it happens (no end-of-run flush needed).
    metrics = MetricsCollector(on_record=flush_metrics)
    last_counters = metrics.counters()  # zero baseline; advanced on each flush

    # The child reaches the parent's stores through proxies over stdio, never the files
    # (two processes on one SQLite DB is unsafe, esp. on NFS). One channel multiplexes them.
    build_kwargs = build_kwargs_from_runtime(runtime)
    channel = ProxyChannel(_write_line)
    db_proxy = BackendProxy(channel, "experiments") if build_kwargs.get("db_path") else None
    memory_proxy = BackendProxy(channel, "memory")

    try:
        agent = build_agent(
            agent_definition,
            event_callback=_emit,
            metrics=metrics,
            db=db_proxy,
            memory_store=memory_proxy,
            **build_kwargs,
        )

        answers: queue.Queue[str] = queue.Queue(maxsize=1)

        def ask_user_via_stdio(question: str) -> str:
            agent.emit(QuestionEvent(question_id=str(uuid.uuid4()), question=question))
            while not agent._stop_requested:
                try:
                    return answers.get(timeout=_ANSWER_POLL_SECONDS)
                except queue.Empty:
                    continue
            return "(agent stopped)"

        agent._ask_user_fn = ask_user_via_stdio

        def read_control() -> None:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = message.get("type")
                if kind == "answer":
                    try:
                        answers.put_nowait(message.get("text", ""))
                    except queue.Full:
                        logger.warning("dropping unsolicited answer")
                elif kind == "stop":
                    agent.stop()
                elif kind == "backend_response":
                    channel.deliver(message)
            # stdin closed = parent gone: unblock any store call still waiting on a reply.
            channel.fail_all("parent channel closed before backend_response")

        threading.Thread(target=read_control, daemon=True).start()
        agent.run(runtime["initial_message"])
    finally:
        detach_context(otel_token)
        run_deps.close()


if __name__ == "__main__":
    main()
