"""Tests for the sandboxed-agent store proxy (child side)."""

from __future__ import annotations

import threading
import time

import pytest

from alpha_lab.sandboxing.db_proxy import (
    BackendProxy,
    ProxyChannel,
    decode_result,
    encode_result,
)
from alpha_lab.experiment_db import Experiment
from alpha_lab.memory import MemoryEntry


def _experiment(**over) -> Experiment:
    base = dict(
        id=1, name="e", description="d", hypothesis="h", status="done",
        config_json="{}", worker_id=None, slurm_job_id=None, results_json=None,
        error=None, debrief_path=None, created_at=0.0, updated_at=0.0,
        started_at=None, finished_at=None,
    )
    base.update(over)
    return Experiment(**base)


def _memory_entry(**over) -> MemoryEntry:
    base = dict(id=1, tags=["a"], summary="s", created_at="t", file="f")
    base.update(over)
    return MemoryEntry(**base)


def test_codec_roundtrips_each_known_dataclass_lists_and_scalars() -> None:
    assert decode_result(encode_result(_experiment(id=7))) == _experiment(id=7)
    assert decode_result(encode_result(_memory_entry(id=3))) == _memory_entry(id=3)
    mixed = [_memory_entry(id=1), _memory_entry(id=2)]
    assert [e.id for e in decode_result(encode_result(mixed))] == [1, 2]
    # scalars / plain dicts (e.g. board_summary) pass through untouched
    assert decode_result(encode_result(5)) == 5
    assert decode_result(encode_result({"done": 3})) == {"done": 3}
    assert decode_result(encode_result(None)) is None


def test_proxy_forwards_with_target_and_decodes_reply() -> None:
    captured: dict = {}

    def send(request: dict) -> None:
        captured.update(request)
        channel.deliver({"req_id": request["req_id"], "ok": True,
                         "result": encode_result(_memory_entry(id=9))})

    channel = ProxyChannel(send)
    memory = BackendProxy(channel, "memory")
    entry = memory.read(9)
    assert captured["type"] == "backend_request"
    assert captured["target"] == "memory" and captured["method"] == "read"
    assert captured["args"] == [9]
    assert isinstance(entry, MemoryEntry) and entry.id == 9


def test_two_backends_share_one_channel() -> None:
    def send(request: dict) -> None:
        result = 42 if request["target"] == "experiments" else "note"
        channel.deliver({"req_id": request["req_id"], "ok": True, "result": result})

    channel = ProxyChannel(send)
    assert BackendProxy(channel, "experiments").create("n", "d", "h", "{}") == 42
    assert BackendProxy(channel, "memory").read(1) == "note"


def test_call_raises_on_error_reply() -> None:
    def send(request: dict) -> None:
        channel.deliver({"req_id": request["req_id"], "ok": False, "error": "duplicate name"})

    channel = ProxyChannel(send)
    with pytest.raises(RuntimeError, match="duplicate name"):
        BackendProxy(channel, "experiments").create("n", "d", "h", "{}")


def test_fail_all_unblocks_an_inflight_call() -> None:
    channel = ProxyChannel(lambda request: None)  # never replies
    box: dict = {}

    def call() -> None:
        try:
            BackendProxy(channel, "memory").search("q")
        except Exception as exc:  # noqa: BLE001
            box["error"] = str(exc)

    worker = threading.Thread(target=call)
    worker.start()
    for _ in range(200):
        if channel._pending:
            break
        time.sleep(0.01)
    channel.fail_all("parent channel closed")
    worker.join(timeout=2)
    assert "parent channel closed" in box["error"]


def test_call_after_close_raises_immediately() -> None:
    channel = ProxyChannel(lambda request: None)
    channel.fail_all("closed")
    with pytest.raises(RuntimeError, match="closed"):
        BackendProxy(channel, "memory").store(content="x", tags=[], summary="s")
