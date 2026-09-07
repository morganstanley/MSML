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
from dataclasses import dataclass, field, replace
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING, Any

from alpha_lab.constants import DEFAULT, DefaultType

if TYPE_CHECKING:
    from alpha_lab.config import TaskConfig
    from alpha_lab.memory import MemoryStore

logger = logging.getLogger("alpha_lab.deps")


@dataclass(frozen=True)
class RunDeps:
    """The run's config, workspace, and executors, published as the active deps for its
    duration.

    Build with ``RunDeps(config, workspace=…, api_key=…)`` — the executors and memory store are
    built lazily from config on first property access (so a scope can be opened cheaply, e.g.
    around intake, without spinning them up), and cached. Inject them to skip the build (tests,
    sandbox proxies) via the ``_gpu_executor``/``_cpu_executor``/``_memory_store`` fields.
    ``cpu_executor`` stays ``None`` when CPU is disabled. ``workspace``/``api_key`` are the
    run-level workspace root and provider API key, read by ``run_agent``/``build_agent`` via
    ``deps.get()``. Use as a context manager to publish the deps and tear built executors down on
    exit; ``open``/``close`` are the manual escape hatch for any non-block caller.
    """

    config: TaskConfig
    workspace: str | Path
    run_id: str = ""
    api_key: str | None = None
    owns_temp: bool = False
    # Lazily built on first access via the public gpu_executor/cpu_executor/memory_store
    # properties; DEFAULT means "not built yet" (distinct from a legitimately-None
    # cpu_executor when CPU is disabled). Real fields — not plain attrs — so
    # dataclasses.replace/asdict carry an injected or already-built value. Inject to skip
    # the build (tests, sandbox proxies) via these underscore names.
    _gpu_executor: Any = DEFAULT
    _cpu_executor: Any = DEFAULT
    _memory_store: MemoryStore | DefaultType = DEFAULT
    # Stack of values ``_current`` held when each ``open()`` shadowed it, so ``close()``
    # restores rather than nulls. A stack (vs a single slot) keeps open/close balanced even
    # if the same instance is re-entered; a list mutated in place needs no unfreezing.
    _prev: list[RunDeps | None] = field(
        default_factory=list, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace", Path(self.workspace))
        # An injected cpu_executor contradicting config is a hard error, surfaced eagerly
        # at construction rather than lazily on first access.
        p3 = self.config.pipeline.phase3
        if self._cpu_executor is not DEFAULT and self._cpu_executor is not None and not p3.cpu_enabled:
            raise ValueError(
                "RunDeps: a cpu_executor was provided but "
                "pipeline.phase3.cpu_enabled is False — contradictory config."
            )

    @property
    def gpu_executor(self) -> Any:
        """The GPU/SLURM executor, built from config on first access and cached."""
        if self._gpu_executor is DEFAULT:
            # Lazy imports: a top-level deps -> executor modules -> utils -> deps would cycle.
            p3 = self.config.pipeline.phase3
            if p3.executor == "local":
                from alpha_lab.local_gpu import LocalGPUManager
                built: Any = LocalGPUManager(
                    gpu_ids=p3.gpu_ids,
                    max_per_gpu=p3.max_per_gpu,
                    time_limit_seconds=p3.time_limit_seconds,
                    python_executable=p3.python_executable,
                )
            else:
                from alpha_lab.slurm import SlurmManager
                built = SlurmManager(
                    partitions=p3.slurm_partitions,
                    gpu_per_job=p3.gpu_per_job,
                    max_gpus=p3.max_concurrent_gpus,
                    time_limit=p3.slurm_time_limit,
                    python_executable=p3.python_executable,
                )
            object.__setattr__(self, "_gpu_executor", built)
        return self._gpu_executor

    @property
    def cpu_executor(self) -> Any | None:
        """The CPU executor (``None`` when CPU is disabled), built on first access and cached."""
        if self._cpu_executor is DEFAULT:
            p3 = self.config.pipeline.phase3
            if p3.cpu_enabled:
                from alpha_lab.local_cpu import LocalCPUManager
                built: Any = LocalCPUManager(
                    max_parallel=p3.cpu_max_parallel,
                    time_limit_seconds=p3.cpu_time_limit_seconds,
                    python_executable=p3.python_executable,
                )
            else:
                built = None
            object.__setattr__(self, "_cpu_executor", built)
        return self._cpu_executor

    @property
    def memory_store(self) -> MemoryStore:
        """The run's memory store, built from config on first access and cached."""
        if self._memory_store is DEFAULT:
            from alpha_lab.databases import ModelDB
            from alpha_lab.git import GitRepository, GitPointer
            from alpha_lab.memory import Memory, MemoryStore
            remote = self.config.memory_spec.remote
            if remote is None:
                remote = GitPointer(ref=self.run_id or None)
            elif remote.ref is None:
                remote = replace(remote, ref=self.run_id)

            repo = GitRepository(
                root=Path(self.workspace) / ".alpha_lab" / "memory",
                spec=replace(self.config.memory_spec, remote=remote),
            )
            if not repo.can_commit:
                logger.warning(
                    "Memory git commits disabled: git user.name/user.email are not configured"
                )
            db = ModelDB(
                root=repo.root,
                model_type=Memory,
                embedding="text-embedding-3-large",
                repo=repo,
                build=True,  # the run's writer store: eager build off the write path
            )
            object.__setattr__(self, "_memory_store", MemoryStore(db))
        return self._memory_store

    def update_config(self, updates: dict[str, Any]) -> TaskConfig:
        """Deep-merge ``updates`` into the run config, validate, and apply in place.

        Side-effect-free with respect to the filesystem — persisting the result is the
        caller's job. Nested dicts merge (``pipeline``/``pipeline.phase3``); scalars and
        lists replace. Validation — including loud rejection of unknown fields at every
        level (``extra="forbid"``) — is handled when pydantic reconstructs the config.
        The validated fields are copied onto the live ``config`` object so existing
        ``deps.config`` holders observe the change.
        """
        import dataclasses

        from alpha_lab.config import task_config_from_mapping
        from alpha_lab.utils import deep_merge

        merged = dataclasses.asdict(self.config)
        deep_merge(merged, updates)
        validated = task_config_from_mapping(merged)
        for field_ in dataclasses.fields(self.config):
            setattr(self.config, field_.name, getattr(validated, field_.name))
        return self.config

    @property
    def temp_dir(self) -> Path:
        return Path(self.workspace, ".alpha_lab", "tmp")

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

        # Always close a present store, reading the private field so close()
        # never builds a store just to clean it up (same rule as the
        # executors below). Every store implements close() for its own
        # lifecycle: the sandboxed child's BackendProxy close() is a no-op,
        # so a child can never tear down the parent's shared embedding
        # client through the forwarding proxy.
        if self._memory_store is not DEFAULT:
            try:
                self._memory_store.close()
            except Exception as e:
                logger.warning("Memory embedding client cleanup failed: %s", e)

        # Tear down only executors that were actually built — reading the private fields
        # (not the properties) so close() never builds an executor just to clean it up.
        for ex in (self._gpu_executor, self._cpu_executor):
            if ex is DEFAULT:
                continue
            cleanup = getattr(ex, "cleanup_all", None)
            if cleanup is None:
                continue
            try:
                cleanup()
            except Exception as e:
                logger.warning("Executor cleanup failed: %s", e)

        if self.owns_temp and self.temp_dir.exists():
            rmtree(self.temp_dir, ignore_errors=True)

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
