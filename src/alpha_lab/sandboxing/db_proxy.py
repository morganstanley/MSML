"""Child-side stand-ins for the parent's SQLite-backed stores (experiments.db, memory.db).

A sandboxed agent must never open a SQLite file itself: two processes on one SQLite DB is
unsafe (corruption on NFS, where WAL/`fcntl` locking don't work). Instead the child forwards
every store method call to the parent — the single connection owner — over the stdio channel
and blocks for the committed result, exactly like the ``ask_user`` round-trip.

One :class:`ProxyChannel` multiplexes all backends; a :class:`BackendProxy` is the per-store
stand-in. Requests are tagged with a ``target`` ("experiments"/"memory") and matched to
replies by ``req_id``; the parent services them in :mod:`alpha_lab.sandboxing.sandbox`. Dataclass
results (``Experiment``, ``Memory``) cross the boundary via
:func:`encode_result`/:func:`decode_result`.
"""

from __future__ import annotations

import dataclasses
import queue
import threading
import uuid
from typing import Any, Callable

from pydantic import BaseModel

from alpha_lab.experiment_db import Experiment
from alpha_lab.memory import Memory

# Dataclasses a store method may return, keyed by type name so results round-trip as the
# real objects the tools expect (not bare dicts).
_RESULT_TYPES: dict[str, type] = {"Experiment": Experiment, "Memory": Memory}
_TYPE_KEY = "__type__"
_FIELDS_KEY = "fields"


def encode_result(value: Any) -> Any:
    """Make a store method's return value JSON-safe (known dataclasses → tagged dict)."""
    if type(value).__name__ in _RESULT_TYPES:
        if dataclasses.is_dataclass(value):
            fields = dataclasses.asdict(value)
        elif isinstance(value, BaseModel):
            fields = value.model_dump(mode="json")
        else:
            fields = None
        if fields is not None:
            return {_TYPE_KEY: type(value).__name__, _FIELDS_KEY: fields}
    if isinstance(value, (list, tuple)):
        return [encode_result(item) for item in value]
    if isinstance(value, dict):
        return {key: encode_result(item) for key, item in value.items()}
    return value


def decode_result(value: Any) -> Any:
    """Inverse of :func:`encode_result` — rebuild dataclass instances for the tools."""
    if isinstance(value, dict) and _TYPE_KEY in value:
        return _RESULT_TYPES[value[_TYPE_KEY]](**value[_FIELDS_KEY])
    if isinstance(value, list):
        return [decode_result(item) for item in value]
    if isinstance(value, dict):
        return {key: decode_result(item) for key, item in value.items()}
    return value


class ProxyChannel:
    """Multiplexes synchronous store calls from the (single-threaded) agent to the parent.

    ``deliver`` (parent reply) and ``fail_all`` (channel closed) are driven by the child's
    stdin reader. One channel serves every backend; per-backend stand-ins are
    :class:`BackendProxy` instances.
    """

    def __init__(self, send: Callable[[dict], None]) -> None:
        self._send = send
        self._lock = threading.Lock()
        self._pending: dict[str, queue.Queue] = {}
        self._closed_error: str | None = None

    def call(self, target: str, method: str, args: tuple, kwargs: dict) -> Any:
        waiter: queue.Queue = queue.Queue(maxsize=1)
        req_id = uuid.uuid4().hex
        with self._lock:
            if self._closed_error is not None:
                raise RuntimeError(self._closed_error)
            self._pending[req_id] = waiter
        self._send({"type": "backend_request", "req_id": req_id, "target": target,
                    "method": method, "args": encode_result(list(args)),
                    "kwargs": encode_result(kwargs)})
        response = waiter.get()  # blocks until the parent commits + replies, or fail_all
        if not response.get("ok"):
            raise RuntimeError(response.get("error", f"{target}.{method} proxy call failed"))
        return decode_result(response.get("result"))

    def deliver(self, response: dict) -> None:
        """Hand a parent ``backend_response`` to the waiting call."""
        with self._lock:
            waiter = self._pending.pop(response.get("req_id"), None)
        if waiter is not None:
            waiter.put(response)

    def fail_all(self, error: str) -> None:
        """Unblock every in-flight call with an error (parent gone / channel closed)."""
        with self._lock:
            self._closed_error = error
            waiters = list(self._pending.values())
            self._pending.clear()
        for waiter in waiters:
            waiter.put({"ok": False, "error": error})


class BackendProxy:
    """Stand-in for one backend ('experiments'/'memory'): forwards each method call through
    the channel to the parent (the single connection owner) and blocks for the result."""

    def __init__(self, channel: ProxyChannel, target: str) -> None:
        self._channel = channel
        self._target = target

    def close(self) -> None:
        """No-op: the parent owns every real backend's lifecycle.

        Without this explicit method, ``__getattr__`` would forward
        ``close()`` to the parent, and a child tearing itself down would
        close the parent's shared embedding client under every other agent
        still using it.
        """

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*args: Any, **kwargs: Any) -> Any:
            return self._channel.call(self._target, name, args, kwargs)

        return call
