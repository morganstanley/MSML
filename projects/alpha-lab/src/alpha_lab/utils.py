"""Generic utilities shared across alpha_lab modules."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from alpha_lab import deps
from alpha_lab.experiment_db import BUSY_STATUSES


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
    worker_count = deps.get().config.pipeline.phase3.worker_count
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
