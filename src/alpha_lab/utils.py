"""Generic utilities shared across alpha_lab modules."""

from __future__ import annotations

import functools
import importlib
import importlib.util
import json
import os
import random
import re
import stat
import subprocess
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from typing import Any, TypeAlias
from urllib.parse import urlparse

import numpy as np
import numpy.typing as npt

from alpha_lab import deps
from alpha_lab.experiment_db import BUSY_STATUSES

_FALLBACK_CHARS_PER_TOKEN = 4
"""Rough chars/token, used only when tiktoken's encoding can't be loaded."""


def get_timestamp() -> str:
    """Current UTC time as an ISO-8601 ``YYYY-MM-DDTHH:MM:SS`` string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


@functools.lru_cache(maxsize=1)
def _token_encoder() -> Any | None:
    """The ``cl100k_base`` encoder, or ``None`` if tiktoken/its vocab can't load."""
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(msg: str) -> int:
    """Count tokens in ``msg`` — exact via tiktoken, char-estimate as fallback."""
    encoder = _token_encoder()
    if encoder is not None:
        return len(encoder.encode(msg))
    return -(-len(msg) // _FALLBACK_CHARS_PER_TOKEN)


def truncate_text(msg: str, max_tokens: int) -> str:
    """Truncate ``msg`` to at most ``max_tokens`` tokens.

    Exact when tiktoken is available (cut on the token boundary); otherwise a
    proportional character estimate, leaving any residual overflow to the caller.
    """
    encoder = _token_encoder()
    if encoder is not None:
        tokens = encoder.encode(msg)
        return msg if len(tokens) <= max_tokens else encoder.decode(tokens[:max_tokens])
    return msg[: max_tokens * _FALLBACK_CHARS_PER_TOKEN]


class SSHMeta(type):
    _pattern = re.compile(r"^[A-Za-z0-9_.-]+@[^:]+:.+")

    def __instancecheck__(self, instance):
        return isinstance(instance, str) and self._pattern.match(instance) is not None


class SSH(str, metaclass=SSHMeta): ...


class URLMeta(type):
    def __instancecheck__(self, instance):
        if not isinstance(instance, str):
            return False
        try:
            result = urlparse(instance)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False


class URL(str, metaclass=URLMeta):
    """String subclass that registers all URL-like strings as instances."""


NetworkPath: TypeAlias = SSH | URL


def topk(
    values: npt.ArrayLike,
    k: int,
    *,
    sort: bool = True,
    descending: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``k`` most extreme values and their indices.

    Selection uses ``np.argpartition`` (O(n)); ``sort=True`` then orders the
    ``k`` results (``descending`` by default). Indices point into ``values``.

    Args:
        values: 1-D array-like to select from.
        k: number of values to return (clamped to ``len(values)``).
        sort: whether to order the ``k`` results by value.
        descending: largest-first when True, smallest-first when False.

    Returns:
        ``(topk_values, topk_indices)``.
    """
    values = np.asarray(values)
    k = min(k, values.shape[0])
    if k <= 0:
        return np.empty(0, dtype=values.dtype), np.empty(0, dtype=np.intp)

    if descending:
        selected = np.argpartition(values, -k)[-k:]
    else:
        selected = np.argpartition(values, k - 1)[:k]

    if sort:
        order = np.argsort(values[selected])
        if descending:
            order = order[::-1]
        selected = selected[order]

    return values[selected], selected


def detect_gpu_ids() -> list[int]:
    """Auto-detect available GPU indices via nvidia-smi. Returns [] on failure.

    Stateless, so run startup can resolve ``gpu_ids: "auto"`` without constructing an
    executor.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode != 0:
            return []
        return [
            int(x.strip()) for x in result.stdout.strip().split("\n")
            if x.strip().isdigit()
        ]
    except (OSError, subprocess.TimeoutExpired):
        # nvidia-smi missing (FileNotFoundError ⊂ OSError) or timed out; anything else
        # is unexpected and should surface rather than be silently swallowed.
        return []


def experiment_resource(exp: Any) -> str:
    """Resource type ("gpu"/"cpu") for an experiment; untagged/invalid -> "gpu"."""
    try:
        cfg = json.loads(getattr(exp, "config_json", None) or "{}")
        raw = cfg.get("resource") if isinstance(cfg, dict) else None
    except (json.JSONDecodeError, TypeError):
        raw = None
    rtype = raw.lower() if isinstance(raw, str) else "gpu"
    return rtype if rtype in ("gpu", "cpu") else "gpu"


def slot_states(db: Any) -> dict[str, dict[str, int]]:
    """Per-type slot capacity ``{type: {total, busy, free}}`` from executors + board.

    Reads the run deps; a type is omitted when it has no executor or zero capacity
    (e.g. ``gpu_ids=[]`` is CPU-only, so GPU reports 0 slots) — an omitted type is not
    a proposable resource. "busy" counts experiments occupying a slot (``BUSY_STATUSES``),
    bucketed by resource type.
    """
    d = deps.get()
    busy = {"gpu": 0, "cpu": 0}
    for exp in db.list_by_status(*BUSY_STATUSES):
        busy[experiment_resource(exp)] += 1
    out: dict[str, dict[str, int]] = {}
    for rtype, ex in (("gpu", d.gpu_executor), ("cpu", d.cpu_executor)):
        if ex is None:
            continue
        total = ex.total_slots()
        if total == 0:
            continue
        out[rtype] = {"total": total, "busy": busy[rtype], "free": max(0, total - busy[rtype])}
    return out


def worker_states(db: Any) -> dict[str, int]:
    """``{busy, free}`` workers from assigned ``worker_id`` rows; count from run config."""
    worker_count = deps.config.pipeline.phase3.worker_count
    assigned = {
        exp.worker_id
        for exp in db.list_all()
        if exp.worker_id is not None
    }
    return {"busy": len(assigned), "free": max(0, worker_count - len(assigned))}


def resolve_import(
    import_path: str,
    types: type | tuple[type, ...] | None = None,
) -> Any:
    """Resolve ``"module:object"`` (or ``"module.object"``) to a Python object.

    Three accepted forms:

    - ``"pkg.module:Object"`` -- preferred, explicit
    - ``"pkg.module.Object"`` -- legacy fallback; last dot splits
    - ``"/abs/or/rel/path.py:Object"`` -- load the file via
      :mod:`importlib.util.spec_from_file_location`; useful for generators
      and helpers that live outside the installed package

    Path-loaded modules are not registered in :data:`sys.modules` under a
    stable name, so they cannot be pickled across processes. This is fine
    for thread-based pools but breaks under :mod:`multiprocessing.Pool`.

    Args:
        import_path: Dotted/colon-qualified module + attribute, or
            ``"path.py:attr"`` for ad-hoc file loads.
        types: Optional type or tuple of types the resolved object must
            satisfy. Class objects must be subclasses of one of the types;
            non-class objects must be instances.

    Returns:
        The resolved attribute.

    Raises:
        ValueError: ``import_path`` is not parseable.
        FileNotFoundError: Path-mode and the file does not exist.
        ImportError: Path-mode loader could not be constructed.
        AttributeError: The module has no such attribute.
        TypeError: The resolved object does not satisfy ``types``.
    """
    module_name, sep, object_name = import_path.partition(":")
    if not sep:
        module_name, _, object_name = import_path.rpartition(".")
    if not module_name or not object_name:
        raise ValueError(
            f"Import path must be 'module:object' or 'module.object': {import_path!r}"
        )

    if module_name.endswith(".py") or os.sep in module_name:
        path = Path(module_name).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Module file not found: {path}")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create spec for {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)

    if not hasattr(module, object_name):
        raise AttributeError(
            f"Module {module_name!r} does not have attribute {object_name!r}."
        )

    obj = getattr(module, object_name)
    if types is None:
        return obj
    if isinstance(obj, type) and issubclass(obj, types):
        return obj
    if not isinstance(obj, type) and isinstance(obj, types):
        return obj
    raise TypeError(f"{module_name}.{object_name} does not satisfy {types}.")


def deep_merge(dst: dict, src: dict | None) -> None:
    """Deep-merge ``src`` into ``dst`` in place; fail loudly on type mismatch.

    Nested dicts are merged recursively; every other value replaces the one in
    ``dst``. Raises ``TypeError`` if a key is a dict in ``src`` but a non-dict,
    non-missing value in ``dst``.
    """
    if not src:
        return
    for key, value in src.items():
        if isinstance(value, dict):
            existing = dst.get(key)
            if existing is None:
                dst[key] = {}
            elif not isinstance(existing, dict):
                raise TypeError(
                    f"deep_merge: key {key!r} is {type(existing).__name__} "
                    f"in dst but dict in src"
                )
            deep_merge(dst[key], value)
        else:
            dst[key] = value



def atomic_write(dest: Path, data: str, *, fsync: bool = True) -> None:
    """Write ``data`` to ``dest`` atomically via temp-file + ``os.replace``.

    The destination is left as either its previous contents or the full new contents —
    never a partially-written file — even if the process is interrupted mid-write. The
    temp file is created alongside ``dest`` (same directory, so the rename stays on one
    filesystem) and removed if anything fails before the replace.

    Permissions match :meth:`pathlib.Path.write_text`: a newly created file gets the
    umask-respecting default (the temp file is opened with mode ``0o666``, which the
    kernel masks), and overwriting an existing file preserves that file's mode rather
    than leaking the temp file's restrictive default.

    Args:
        dest: File to write. Its parent directory must already exist.
        data: The complete text to write; callers handle their own serialization.
        fsync: When True (default), flush the file's data to physical storage before the
            rename so the contents survive power loss, and best-effort flush the
            containing directory so the rename entry is durable too (on filesystems that
            support directory fsync). Set False on hot paths where crash-atomicity is
            enough and the fsync round-trips are not worth their cost.
    """
    dest = Path(dest)
    # Open the temp with an explicit mode so the kernel applies the umask (matching
    # write_text) instead of tempfile.mkstemp's fixed 0o600. O_EXCL + a random name
    # guards against collisions.
    tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    # Hand the fd to a file object up front. If os.fdopen fails it hasn't taken
    # ownership of fd, so close it explicitly to avoid leaking the descriptor.
    try:
        handle = os.fdopen(fd, "w", encoding="utf-8")
    except BaseException:
        os.close(fd)
        Path(tmp).unlink(missing_ok=True)
        raise
    try:
        with handle as f:
            f.write(data)
            f.flush()
            if fsync:
                os.fsync(f.fileno())
        # Best-effort: preserve the destination's existing permissions on overwrite.
        # A permission/ACL edge case (or a missing dest) shouldn't fail the write —
        # fall back to the umask-default mode the temp was created with.
        try:
            os.chmod(tmp, stat.S_IMODE(dest.stat().st_mode))
        except OSError:
            pass
        os.replace(tmp, dest)
        if fsync:
            # Best-effort: persist the rename itself so the new entry survives a crash.
            # This runs *after* the successful replace, so a failure here (e.g. a
            # filesystem that can't fsync a directory) must not surface as a write
            # failure — the file is already in place.
            try:
                dir_fd = os.open(dest.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class RetryTimer:
    def __init__(
        self,
        tries: int = 3,
        delay: float = 1.0,
        power: float = 2.0,
        max_delay: float = 60.0,
        randomness: float = 0.1,
    ) -> None:
        """Exponential backoff timer for retry loops.

        Args:
            tries: Maximum number of attempts before giving up.
            delay: Initial delay in seconds.
            power: Exponential growth factor (default 2.0).
            max_delay: Maximum delay in seconds (default 60.0).
            randomness: Fractional jitter to apply to the delay (default 0.1).
        """
        self.delay = delay
        self.power = power
        self.tries = tries
        self.max_delay = max_delay
        self.randomness = randomness

    def __iter__(self) -> Iterator[int]:
        delay = self.delay
        for attempt in range(self.tries):
            yield attempt
            if attempt == self.tries - 1:
                break

            frac = 1 + self.randomness * (2 * random.random() - 1)
            sleep(min(frac * delay, self.max_delay))

            delay *= self.power
