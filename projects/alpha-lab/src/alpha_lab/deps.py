"""Run-scoped dependencies, exposed via a single module global.

One config and at most one executor of each type live for the whole run. ``RunDeps`` is
the run's deps plus its own lifecycle: use it as a context manager
(``with RunDeps(config): ...``) to publish the deps for the block and tear the executors
down on exit. Internal callers read them via ``deps.get()``; capacity readers over these
deps live in ``utils`` (``slot_states``/``worker_states``).

The published deps live in a plain module global, so they're visible to every thread in
the process for free — the strategist/worker threads just call ``deps.get()``. This assumes
a single run per process (the model today); per-thread/per-run isolation would need a
context var, which we can reintroduce behind this same API if that ever becomes a need.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alpha_lab.config import TaskConfig

logger = logging.getLogger("alpha_lab.deps")


@dataclass(frozen=True)
class RunDeps:
    """The run's config, workspace, and executors, published as the active deps for its
    duration.

    Build with ``RunDeps(config, workspace=…, api_key=…)`` — executors are built from config;
    inject them to skip the build (tests). ``cpu_executor`` stays ``None`` when CPU is disabled.
    ``workspace``/``api_key`` are the run-level workspace root and provider API key, read by
    ``run_agent``/``build_agent`` via ``deps.get()``. Use as a context manager to publish the
    deps and tear executors down on exit; ``open``/``close`` are the manual escape hatch for any
    non-block caller.
    """

    config: TaskConfig
    gpu_executor: Any = None
    cpu_executor: Any | None = None
    workspace: str = ""
    api_key: str | None = None
    # Stack of values ``_current`` held when each ``open()`` shadowed it, so ``close()``
    # restores rather than nulls. A stack (vs a single slot) keeps open/close balanced even
    # if the same instance is re-entered; a list mutated in place needs no unfreezing.
    _prev: list[RunDeps | None] = field(
        default_factory=list, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # Lazy imports: a top-level deps -> executor modules -> utils -> deps would cycle.
        from alpha_lab.local_cpu import LocalCPUManager
        from alpha_lab.local_gpu import LocalGPUManager
        from alpha_lab.slurm import SlurmManager

        p3 = self.config.pipeline.phase3
        if self.gpu_executor is None:
            if p3.executor == "local":
                gpu = LocalGPUManager(
                    gpu_ids=p3.gpu_ids,
                    max_per_gpu=p3.max_per_gpu,
                    time_limit_seconds=p3.time_limit_seconds,
                    python_executable=p3.python_executable,
                )
            else:
                gpu = SlurmManager(
                    partitions=p3.slurm_partitions,
                    gpu_per_job=p3.gpu_per_job,
                    max_gpus=p3.max_concurrent_gpus,
                    time_limit=p3.slurm_time_limit,
                    python_executable=p3.python_executable,
                )
            object.__setattr__(self, "gpu_executor", gpu)
        if self.cpu_executor is not None and not p3.cpu_enabled:
            raise ValueError(
                "RunDeps: a cpu_executor was provided but "
                "pipeline.phase3.cpu_enabled is False — contradictory config."
            )
        if self.cpu_executor is None and p3.cpu_enabled:
            object.__setattr__(self, "cpu_executor", LocalCPUManager(
                max_parallel=p3.cpu_max_parallel,
                time_limit_seconds=p3.cpu_time_limit_seconds,
                python_executable=p3.python_executable,
            ))

    def open(self) -> RunDeps:
        """Publish self as the active run's deps; return self."""
        global _current
        self._prev.append(_current)
        _current = self
        return self

    def close(self) -> None:
        """Restore the previously active deps; tear executors down once fully unwound."""
        global _current
        if _current is not self:
            raise RuntimeError(
                "RunDeps.close(): this instance is not the active deps "
                "(out-of-order open/close)"
            )
        _current = self._prev.pop()
        if self._prev:
            return
        for ex in (self.gpu_executor, self.cpu_executor):
            cleanup = getattr(ex, "cleanup_all", None)
            if cleanup is None:
                continue
            try:
                cleanup()
            except Exception as e:
                logger.warning("Executor cleanup failed: %s", e)

    def __enter__(self) -> RunDeps:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()


_current: RunDeps | None = None


def get(strict: bool = True) -> RunDeps | None:
    """The active run's deps.

    Args:
        strict: When True (default), raise if no deps are published — internal callers
            must run inside a ``with RunDeps(...)`` scope. When False, return ``None``
            instead of raising.
    """
    if _current is None and strict:
        raise LookupError("No active RunDeps — call within `with RunDeps(config): ...`")
    return _current


def __getattr__(name: str) -> object:
    """Convenience: ``deps.config`` is shorthand for ``deps.get().config``.

    Delegates attribute access on this module to the active ``RunDeps``, so callers
    can read ``deps.config``/``deps.gpu_executor`` without the ``.get()``. Inherits
    ``get()``'s fail-loud behavior (raises if no run is active). Real module names
    (``get``, ``RunDeps``, ``_current``) resolve normally; dunders are excluded so
    import machinery isn't intercepted.
    """
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return getattr(get(), name)
