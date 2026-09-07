"""SQLite experiment database for Phase 3 kanban tracking."""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger("alpha_lab.experiment_db")


KANBAN_COLUMNS = (
    "to_implement",
    "implemented",
    "checked",
    "queued",
    "running",
    "cancelling",
    "finished",
    "analyzed",
    "done",
    "cancelled",  # Experiments pruned by strategist
)

# Forward edges of the kanban DAG. update_status() silently no-ops any
# backwards move so a repeated update_experiment(status="checked") tool
# call from the worker LLM cannot bounce an in-flight experiment back to
# checked and trigger a duplicate dispatcher submission. The fixer's
# finished -> checked path is kept open so retries still work, and
# any -> cancelled is allowed so the strategist can prune.
KANBAN_FORWARD_EDGES = {
    "to_implement": {"implemented", "checked", "cancelled"},
    "implemented":  {"checked", "to_implement", "cancelled"},
    "checked":      {"queued", "to_implement", "cancelled"},
    "queued":       {"running", "cancelling", "finished", "cancelled"},
    "running":      {"cancelling", "finished", "cancelled"},
    "cancelling":   {"checked"},
    "finished":     {"analyzed", "checked", "cancelled"},  # fixer can re-check
    "analyzed":     {"done", "cancelled"},
    "done":         set(),
    "cancelled":    set(),
}

# Statuses at which the row is fully done with worker activity. Reaching any
# of these via update_status() implicitly releases any worker_id still
# attached to the row — otherwise terminated experiments can show "still
# assigned to worker X" indefinitely, which has surfaced as an integrity
# violation in production DBs.
_TERMINAL_STATUSES = frozenset({"finished", "analyzed", "done", "cancelled"})


# Self-labelling keys an experiment runtime may attach to a non-canonical
# results document. Generic across domains: any adapter that doesn't set
# these keys is unaffected (the guard is a no-op). Anything that DOES set
# them is asserting "this is not a full canonical run" and must not be
# stored as canonical results.
_NON_CANONICAL_STATUS_VALUES = {"smoke_complete", "smoke", "dry_run", "partial"}


def is_execution_failure(error: str | None, results_json: str | None) -> bool:
    """Return whether an experiment failed before producing results."""
    return bool(error) and not results_json


def _classify_non_canonical_results(results_json: str | None) -> str | None:
    """Return a short refusal reason if ``results_json`` self-identifies as
    smoke / dry-run / partial, else ``None``.

    Conservative: only triggers on explicit self-labelling keys
    (``smoke``, ``run_scope``, ``partial``, ``status``,
    ``canonical_full_run``). Numeric metrics alone never trip this guard.
    Adapters that don't set any of these keys are unaffected.
    """
    if not results_json:
        return None
    try:
        payload = json.loads(results_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("smoke"):
        return "smoke=true"
    if payload.get("partial"):
        return "partial=true"
    # An explicit ``canonical_full_run: false`` is the experiment-runtime
    # asserting "this is not a canonical artifact" — usually written by a
    # gate that aborted before training. Accept missing/None unchanged
    # (backward compatible: only refuse the explicit-False case).
    canonical_flag = payload.get("canonical_full_run")
    if canonical_flag is False:
        return "canonical_full_run=false"
    run_scope = payload.get("run_scope")
    if run_scope is not None and run_scope != "full":
        return f"run_scope={run_scope!r}"
    status = payload.get("status")
    if isinstance(status, str) and status in _NON_CANONICAL_STATUS_VALUES:
        return f"status={status!r}"
    return None


# Bookkeeping keys excluded from the disk-consistency comparison: timing
# fields legitimately differ between a worker's payload and the artifact.
_CONSISTENCY_EXEMPT_KEYS = {
    "smoke", "partial", "run_scope", "status", "canonical_full_run",
    "error", "traceback", "native_fallback_used", "started_at",
    "finished_at", "wall_seconds",
}


def _results_conflict_with_disk(
    workspace: Path, exp_name: str | None, payload: dict,
) -> str | None:
    """Refusal reason when a results payload's numbers disagree with the
    experiment's own canonical ``results/metrics.json`` on disk.

    The payload usually arrives through the worker LLM, which can retype
    or round numbers; the board must never disagree with the artifact
    (matches the post-hoc audit's ``db_file_metric_mismatches`` check:
    shared non-bookkeeping numeric keys, isclose rel 1e-9 / abs 1e-12).
    Missing or unparseable file → no opinion (None)."""
    fpath = (workspace / "experiments" / str(exp_name or "")
             / "results" / "metrics.json")
    try:
        file_metrics = json.loads(fpath.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(file_metrics, dict):
        return None
    for key, val in payload.items():
        if key in _CONSISTENCY_EXEMPT_KEYS:
            continue
        fv = file_metrics.get(key)
        if (
            isinstance(val, (int, float)) and not isinstance(val, bool)
            and isinstance(fv, (int, float)) and not isinstance(fv, bool)
            and not math.isclose(float(val), float(fv),
                                 rel_tol=1e-9, abs_tol=1e-12)
        ):
            return (
                f"metric_conflicts_with_disk (key {key!r}: payload has "
                f"{val!r}, results/metrics.json has {fv!r} — report the "
                "file's numbers verbatim)"
            )
    return None


@dataclass
class Experiment:
    id: int
    name: str
    description: str
    hypothesis: str
    status: str
    config_json: str
    worker_id: str | None
    slurm_job_id: str | None
    results_json: str | None
    error: str | None
    debrief_path: str | None
    created_at: float
    updated_at: float
    started_at: float | None
    finished_at: float | None
    fix_attempts: int = 0  # Number of times fixer has tried to fix this experiment
    # MLflow back-reference cache. Populated by _ensure_mlflow_run when MLflow
    # is active; NULL otherwise. Lets workers and update_experiment look up
    # the sub-run UUID in O(1).
    mlflow_run_uuid: str | None = None
    mlflow_artifact_uri: str | None = None
    # Conductor-administered fields. Default 0 / NULL means no Conductor influence;
    # the dispatcher's queue ordering reduces to plain created_at ASC and no rows
    # are excluded — preserves current behavior when no_conductor=True.
    priority: int = 0          # Higher runs first; ties broken by created_at ASC
    parked_at: float | None = None  # Soft-cancel timestamp; dispatcher skips parked rows
    # Variant parentage: when the strategist used `propose_variant` to spawn this
    # experiment from an existing one, parent_id is the source experiment's id.
    # NULL for ordinary `propose_experiment` rows. Lets the analyzer/conductor
    # walk variant families and the dispatcher cap variant fan-out per base.
    parent_id: int | None = None


_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    hypothesis TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'to_implement',
    config_json TEXT NOT NULL DEFAULT '{}',
    worker_id TEXT,
    slurm_job_id TEXT,
    results_json TEXT,
    error TEXT,
    debrief_path TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    fix_attempts INTEGER NOT NULL DEFAULT 0,
    priority INTEGER NOT NULL DEFAULT 0,
    parked_at REAL,
    parent_id INTEGER,
    mlflow_run_uuid TEXT,
    mlflow_artifact_uri TEXT,
    launch_attempts INTEGER NOT NULL DEFAULT 0
);
"""


# Columns that may be missing on existing DBs and are added by the migration in
# _init_db. Each entry is the full ALTER TABLE column definition. Order matters
# for upgrading from pre-fix_attempts → pre-conductor → current schema.
_MIGRATION_COLUMNS: tuple[str, ...] = (
    "fix_attempts INTEGER NOT NULL DEFAULT 0",
    "priority INTEGER NOT NULL DEFAULT 0",
    "parked_at REAL",
    "parent_id INTEGER",
    "mlflow_run_uuid TEXT",
    "mlflow_artifact_uri TEXT",
    "launch_attempts INTEGER NOT NULL DEFAULT 0",
)


def _row_to_experiment(row: sqlite3.Row) -> Experiment:
    # Handle columns that may be missing on existing DBs (the migration in
    # _init_db adds them on startup, but we read defensively in case a caller
    # constructed a DB outside of our class).
    def _safe(key: str, default: object) -> object:
        try:
            val = row[key]
        except (KeyError, IndexError):
            return default
        return default if val is None and isinstance(default, (int, float)) and key in {"fix_attempts", "priority"} else val

    return Experiment(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        hypothesis=row["hypothesis"],
        status=row["status"],
        config_json=row["config_json"],
        worker_id=row["worker_id"],
        slurm_job_id=row["slurm_job_id"],
        results_json=row["results_json"],
        error=row["error"],
        debrief_path=row["debrief_path"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        fix_attempts=_safe("fix_attempts", 0),  # type: ignore[arg-type]
        priority=_safe("priority", 0),  # type: ignore[arg-type]
        parked_at=_safe("parked_at", None),  # type: ignore[arg-type]
        parent_id=_safe("parent_id", None),  # type: ignore[arg-type]
        mlflow_run_uuid=_safe("mlflow_run_uuid", None),  # type: ignore[arg-type]
        mlflow_artifact_uri=_safe("mlflow_artifact_uri", None),  # type: ignore[arg-type]
    )


class ExperimentDB:
    """Thread-safe SQLite database for experiment tracking.

    Uses WAL mode for concurrent reads and a threading lock for writes.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("PRAGMA journal_mode=DELETE")
                conn.execute(_CREATE_TABLE)
                # Migrations for DBs created against older schemas. Each ALTER
                # is wrapped because SQLite raises OperationalError on duplicate
                # add — the existing fix_attempts migration uses the same idiom
                # inline; we centralize it here.
                for col_def in _MIGRATION_COLUMNS:
                    try:
                        conn.execute(f"ALTER TABLE experiments ADD COLUMN {col_def}")
                    except sqlite3.OperationalError:
                        pass  # Column already exists
                conn.commit()
            finally:
                conn.close()

    def create(
        self,
        name: str,
        description: str,
        hypothesis: str,
        config_json: str,
        parent_id: int | None = None,
    ) -> int:
        """Insert a new experiment row.

        ``parent_id`` is set only for variant rows spawned via the strategist's
        ``propose_variant`` tool; for ordinary ``propose_experiment`` calls it
        stays NULL and the row behaves as a top-level experiment.
        """
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "INSERT INTO experiments "
                    "(name, description, hypothesis, config_json, "
                    " created_at, updated_at, parent_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (name, description, hypothesis, config_json, now, now, parent_id),
                )
                conn.commit()
                return cur.lastrowid  # type: ignore[return-value]
            finally:
                conn.close()

    def count_variants_of(self, base_exp_id: int) -> int:
        """Count rows whose parent_id equals ``base_exp_id``.

        Used by ``propose_variant`` to enforce the ``max_variants_per_base``
        cap and by the conductor for variant-family audits.
        Returns 0 for any base id with no variants (or on missing
        ``parent_id`` column on legacy DBs — the migration in _init_db has
        already added the column on first open).
        """
        conn = self._connect()
        try:
            try:
                row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM experiments WHERE parent_id = ?",
                    (base_exp_id,),
                ).fetchone()
                return int(row["cnt"]) if row else 0
            except sqlite3.OperationalError:
                # ``parent_id`` missing — extremely defensive; _init_db adds
                # it. Returning 0 keeps callers from crashing on a fresh DB
                # that has not yet been opened by ``__init__``.
                return 0
        finally:
            conn.close()

    def get(self, exp_id: int) -> Experiment | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (exp_id,)
            ).fetchone()
            return _row_to_experiment(row) if row else None
        finally:
            conn.close()

    _ALLOWED_UPDATE_COLS = frozenset({
        "started_at", "finished_at", "debrief_path",
    })

    def update_status(self, exp_id: int, status: str, **kwargs: object) -> str:
        """Apply a status transition, honoring the forward-only kanban guard.

        Return value (lets the caller — typically the `update_experiment`
        tool dispatch — surface a clear message to the LLM):

          * ``"applied"``  — the row's status moved from the prior value
            to the requested one.
          * ``"idempotent"`` — the row was already at the requested status;
            no-op. The LLM may be re-emitting the same transition; not
            harmful but worth surfacing so it can stop.
          * ``"blocked:<current>"`` — the requested status is not a
            forward successor of the current one. No-op. The LLM should
            not retry this transition; the row stays at <current>.
        """
        if status not in KANBAN_COLUMNS:
            raise ValueError(f"Invalid status: {status}")
        now = time.time()
        sets = ["status = ?", "updated_at = ?"]
        vals: list[object] = [status, now]
        for k, v in kwargs.items():
            if k not in self._ALLOWED_UPDATE_COLS:
                raise ValueError(f"update_status: disallowed column '{k}'")
            sets.append(f"{k} = ?")
            vals.append(v)
        vals.append(exp_id)
        with self._lock:
            conn = self._connect()
            try:
                # Forward-only transition guard: silently no-op if the
                # requested status is not a successor of the current one.
                # Idempotent same-status calls are also no-ops. This prevents
                # the worker LLM's redundant update_experiment("checked")
                # calls from racing the dispatcher into duplicate submissions.
                # The return value lets the tool dispatcher tell the LLM
                # exactly what happened so it can stop retrying.
                row = conn.execute(
                    "SELECT status FROM experiments WHERE id = ?", (exp_id,)
                ).fetchone()
                if row is not None:
                    cur_status = row["status"]
                    if cur_status == status:
                        return "idempotent"
                    if status not in KANBAN_FORWARD_EDGES.get(cur_status, set()):
                        logger.info(
                            "blocked transition #%d %s -> %s (no-op)",
                            exp_id, cur_status, status,
                        )
                        return f"blocked:{cur_status}"
                # When transitioning to a terminal state, the row is done
                # with worker activity. Release the worker_id atomically so
                # downstream observers (dashboards, integrity audits) don't
                # see "status=finished AND worker_id=worker_4". This used
                # to be inconsistent in 3 cond rows + 1 o47 row.
                if status in _TERMINAL_STATUSES and "worker_id" not in kwargs:
                    sets.append("worker_id = ?")
                    vals.insert(-1, None)  # before exp_id at end
                conn.execute(
                    f"UPDATE experiments SET {', '.join(sets)} WHERE id = ?",
                    vals,
                )
                conn.commit()
                return "applied"
            finally:
                conn.close()

    def assign_worker(self, exp_id: int, worker_id: str) -> None:
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE experiments SET worker_id = ?, updated_at = ? WHERE id = ?",
                    (worker_id, now, exp_id),
                )
                conn.commit()
            finally:
                conn.close()

    def release_worker(self, exp_id: int) -> None:
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE experiments SET worker_id = NULL, updated_at = ? WHERE id = ?",
                    (now, exp_id),
                )
                conn.commit()
            finally:
                conn.close()

    def set_slurm_job(self, exp_id: int, job_id: str) -> None:
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE experiments SET slurm_job_id = ?, updated_at = ? WHERE id = ?",
                    (job_id, now, exp_id),
                )
                conn.commit()
            finally:
                conn.close()

    def set_mlflow_run(
        self, exp_id: int, run_uuid: str, artifact_uri: str,
    ) -> None:
        """Persist the MLflow sub-run UUID + artifact URI for this experiment."""
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE experiments SET mlflow_run_uuid = ?, "
                    "mlflow_artifact_uri = ?, updated_at = ? WHERE id = ?",
                    (run_uuid, artifact_uri, now, exp_id),
                )
                conn.commit()
            finally:
                conn.close()

    def set_results(
        self, exp_id: int, results_json: str, *, refuse_if_error: bool = False,
    ) -> str:
        """Persist a canonical-results payload for ``exp_id``.

        Returns ``"applied"`` on success or ``"refused:<reason>"`` when the
        payload looks like a smoke / dry-run / partial artifact rather than
        a canonical full-run result. The check is conservative: it triggers
        ONLY on explicit self-labelling keys that adapters may set on their
        own results documents (``smoke``, ``run_scope``, ``partial``,
        ``status``). Domains that don't use these keys are unaffected — the
        guard is a no-op for them.

        Rationale: smoke metrics shipped from the experiment-runtime contain
        real numeric fields (``sharpe``, ``mae``, …) that look identical to
        canonical numbers, so downstream consumers (board, reporter,
        strategist, leaderboard) can't distinguish them after the fact. The
        safe place to refuse is the choke-point that writes them to the DB.

        Recognised refusal signals on the top-level payload:

        * ``smoke``: any truthy value
        * ``run_scope``: any value other than ``"full"`` (e.g. ``"smoke"``,
          ``"dry"``, ``"partial"``)
        * ``partial``: any truthy value
        * ``status``: equal to ``"smoke_complete"``, ``"smoke"``,
          ``"dry_run"``, or ``"partial"``

        On refusal the row is left untouched and the caller (typically the
        ``update_experiment`` tool) is expected to surface the message back
        to the worker LLM so it can re-run for a canonical artifact.

        ``refuse_if_error`` is used by zombie recovery so an older results
        file cannot atomically override a deliberate quarantine.
        """
        # Validate JSON parseable. In production, one row in workspace_etfflow_cond
        # ended up with a truncated string (broken at char 1400). Reject obviously
        # invalid payloads at the choke-point rather than store unparseable bytes
        # that downstream consumers will all fail on.
        try:
            parsed = json.loads(results_json)
        except (TypeError, ValueError) as e:
            return f"refused:invalid_json ({type(e).__name__}: {str(e)[:80]})"
        # Reject obviously-empty payloads. `{}` looks completed to a casual
        # observer but contributes nothing to the leaderboard or convergence
        # tracking — 150 such rows accumulated in workspace_etfflow_g55.
        # Require at least one non-trivial key beyond bookkeeping fields.
        if isinstance(parsed, dict):
            _BOOKKEEPING = {"smoke", "partial", "run_scope", "status",
                            "canonical_full_run", "error", "traceback",
                            "native_fallback_used", "started_at",
                            "finished_at", "wall_seconds"}
            non_bookkeeping = [k for k in parsed if k not in _BOOKKEEPING]
            if not non_bookkeeping:
                return "refused:empty_results (no non-bookkeeping keys)"
        refusal = _classify_non_canonical_results(results_json)
        if refusal:
            return f"refused:{refusal}"
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                # Auto-promote zombie rows: if the row has been sitting at
                # ``implemented`` or ``checked`` and a canonical results
                # payload finally lands, advance it straight to ``finished``
                # so the analyzer can pick it up. Without this, rows whose
                # canonical run was executed out-of-band (worker-launched,
                # restart-recovered, or completed before the executor
                # signal got back to the dispatcher) get stranded mid-kanban
                # and waste worker turns on idempotent re-confirmations.
                #
                # Bypasses the forward-only update_status guard
                # deliberately — that guard exists to stop LLM workers from
                # skipping stages, but here we are asserting a verified
                # canonical artifact via the smoke-guard above. Generic
                # across domains: it's a single canonical-results signal,
                # not coupled to any particular metric or task.
                row = conn.execute(
                    "SELECT name, status, finished_at, error FROM experiments "
                    "WHERE id = ?",
                    (exp_id,),
                ).fetchone()
                if (
                    refuse_if_error
                    and row is not None
                    and row["status"] in ("implemented", "checked")
                    and row["error"]
                ):
                    return "refused:row_has_error"
                # Headline-consistency guard: this payload usually arrives
                # through the worker LLM, which can drift from the canonical
                # on-disk artifact (an audit found board rows whose metric
                # disagreed with their own results/metrics.json). When the
                # canonical file exists, every shared non-bookkeeping
                # numeric key must match it — otherwise refuse and make the
                # worker report the file's numbers verbatim.
                if isinstance(parsed, dict) and row is not None:
                    conflict = _results_conflict_with_disk(
                        Path(self.db_path).parent, row["name"], parsed)
                    if conflict:
                        return f"refused:{conflict}"
                promote = (
                    row is not None
                    and row["status"] in ("implemented", "checked")
                )
                if promote:
                    finished_at = row["finished_at"] or now
                    conn.execute(
                        "UPDATE experiments SET results_json = ?, status = ?, "
                        "finished_at = ?, updated_at = ? WHERE id = ?",
                        (results_json, "finished", finished_at, now, exp_id),
                    )
                    logger.info(
                        "Auto-promoted #%d %s -> finished on canonical "
                        "results write",
                        exp_id, row["status"],
                    )
                else:
                    conn.execute(
                        "UPDATE experiments SET results_json = ?, updated_at = ? WHERE id = ?",
                        (results_json, now, exp_id),
                    )
                conn.commit()
            finally:
                conn.close()
        return "applied:promoted" if promote else "applied"

    def set_error(self, exp_id: int, error_msg: str) -> None:
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE experiments SET error = ?, updated_at = ? WHERE id = ?",
                    (error_msg, now, exp_id),
                )
                conn.commit()
            finally:
                conn.close()

    def set_error_and_finish(self, exp_id: int, error_msg: str) -> bool:
        """Finish an active job without overwriting a concurrent cancellation.

        The current-state predicate is part of the same SQLite statement as
        the write. A late executor result therefore cannot replace a
        ``cancelling`` state that won the race.
        """
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "UPDATE experiments SET error = ?, status = 'finished', "
                    "finished_at = ?, worker_id = NULL, updated_at = ? "
                    "WHERE id = ? "
                    "AND status IN ('checked', 'queued', 'running') "
                    "AND parked_at IS NULL",
                    (error_msg, now, now, exp_id),
                )
                conn.commit()
                return cursor.rowcount == 1
            finally:
                conn.close()

    def increment_launch_attempts(self, exp_id: int) -> int:
        """Increment launch_attempts and return the new value.

        One increment per executor submission, so re-executions of the same
        row are visible in the DB instead of only as extra ``*job*.out``
        files in the experiment directory (a real run relaunched one row 15
        times with no database trace).
        """
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE experiments SET launch_attempts = "
                    "launch_attempts + 1, updated_at = ? WHERE id = ?",
                    (now, exp_id),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT launch_attempts FROM experiments WHERE id = ?",
                    (exp_id,),
                ).fetchone()
                return row["launch_attempts"] if row else 0
            finally:
                conn.close()

    def increment_fix_attempts(self, exp_id: int) -> int:
        """Increment fix_attempts counter and return new value."""
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                # Try to add column if it doesn't exist (migration for old DBs)
                try:
                    conn.execute("ALTER TABLE experiments ADD COLUMN fix_attempts INTEGER NOT NULL DEFAULT 0")
                    conn.commit()
                except sqlite3.OperationalError:
                    pass  # Column already exists

                conn.execute(
                    "UPDATE experiments SET fix_attempts = fix_attempts + 1, updated_at = ? WHERE id = ?",
                    (now, exp_id),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT fix_attempts FROM experiments WHERE id = ?", (exp_id,)
                ).fetchone()
                return row["fix_attempts"] if row else 0
            finally:
                conn.close()

    def list_by_status(
        self, *statuses: str, include_parked: bool = False
    ) -> list[Experiment]:
        """Rows with status in the given set, ordered for the dispatcher.

        Order is ``priority DESC, created_at ASC`` — when no Conductor has
        touched any row, ``priority`` is 0 everywhere and the result reduces
        to ``created_at ASC`` (current behavior). Parked rows (``parked_at``
        non-null) are excluded by default so the dispatcher skips them; the
        Conductor's tooling passes ``include_parked=True`` for queries that
        need to see soft-cancelled rows.
        """
        if not statuses:
            return []
        placeholders = ", ".join("?" for _ in statuses)
        parked_clause = "" if include_parked else " AND parked_at IS NULL"
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT * FROM experiments WHERE status IN ({placeholders})"
                f"{parked_clause} "
                "ORDER BY priority DESC, created_at ASC",
                statuses,
            ).fetchall()
            return [_row_to_experiment(r) for r in rows]
        finally:
            conn.close()

    def park(self, exp_id: int) -> str | None:
        """Soft-cancel an experiment. Dispatcher skips parked rows in
        list_by_status. Reversible via unpark. The reason for parking is
        recorded by the Conductor in meta_log.jsonl, not in the DB — the
        DB only tracks state, not narrative.

        An active job first enters ``cancelling``. The dispatcher confirms
        executor exit before returning it to ``checked`` while leaving it
        parked. This prevents a late FAILED/CANCELLED poll from overwriting
        the parking decision and keeps the row safely resumable.
        """
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE experiments SET parked_at = ?, "
                    "status = CASE WHEN status IN ('queued', 'running') "
                    "AND slurm_job_id IS NOT NULL THEN 'cancelling' "
                    "ELSE status END, updated_at = ? "
                    "WHERE id = ?",
                    (now, now, exp_id),
                )
                row = conn.execute(
                    "SELECT status FROM experiments WHERE id = ?", (exp_id,)
                ).fetchone()
                conn.commit()
                return row["status"] if row else None
            finally:
                conn.close()

    def finalize_park(self, exp_id: int) -> bool:
        """Return a stopped job to the parked, resumable ``checked`` state."""
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "UPDATE experiments SET status = 'checked', "
                    "slurm_job_id = NULL, worker_id = NULL, error = NULL, "
                    "started_at = NULL, finished_at = NULL, updated_at = ? "
                    "WHERE id = ? AND status = 'cancelling' "
                    "AND parked_at IS NOT NULL",
                    (now, exp_id),
                )
                conn.commit()
                return cursor.rowcount == 1
            finally:
                conn.close()

    def unpark(self, exp_id: int) -> bool:
        """Restore a parked experiment after any active job has stopped."""
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "UPDATE experiments SET parked_at = NULL, updated_at = ? "
                    "WHERE id = ? AND status != 'cancelling'",
                    (now, exp_id),
                )
                conn.commit()
                return cursor.rowcount == 1
            finally:
                conn.close()

    def set_priority(self, exp_id: int, priority: int) -> None:
        """Override an experiment's queue priority. Higher runs first."""
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE experiments SET priority = ?, updated_at = ? "
                    "WHERE id = ?",
                    (priority, now, exp_id),
                )
                conn.commit()
            finally:
                conn.close()

    def clear_block(self, exp_id: int) -> bool:
        """Clear a stale ``blocked:`` error so the dispatcher can assign the row
        again, without changing its status, priority, or economics.

        Only acts on a row the dispatcher currently treats as externally blocked
        (``_is_externally_blocked``): status in ``to_implement`` / ``implemented``
        with an ``error`` that, after stripping leading whitespace, starts with
        ``blocked:``. Returns True iff such a row was cleared; False if the row
        was not in a blocked state (nothing to do). The reason is recorded by
        the Conductor in ``meta_log.jsonl``, not in the DB.

        This is the recovery for a *transient* precondition (e.g. a shared
        fixture that has since gone green) whose ``blocked:`` marker was never
        cleared: the worker that set it only runs when assigned, but the
        dispatcher never assigns a blocked row, so the flag is otherwise
        self-perpetuating. Clearing ``error`` alone requeues the row in place —
        its ``implemented`` / ``to_implement`` status is already assignable.
        """
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "UPDATE experiments SET error = NULL, updated_at = ? "
                    "WHERE id = ? AND status IN ('to_implement', 'implemented') "
                    "AND ltrim(error) LIKE 'blocked:%'",
                    (now, exp_id),
                )
                conn.commit()
                return cursor.rowcount == 1
            finally:
                conn.close()

    def list_all(self) -> list[Experiment]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY created_at ASC"
            ).fetchall()
            return [_row_to_experiment(r) for r in rows]
        finally:
            conn.close()

    def list_experiments(
        self,
        limit: int = 50,
        offset: int = 0,
        status_filter: str | None = None,
        name_search: str | None = None,
    ) -> tuple[list[Experiment], int]:
        """Paginated experiment query. Returns (experiments, total_count)."""
        conn = self._connect()
        try:
            where_clauses: list[str] = []
            params: list[str | int] = []

            if status_filter:
                where_clauses.append("status = ?")
                params.append(status_filter)
            if name_search:
                where_clauses.append("name LIKE ?")
                params.append(f"%{name_search}%")

            where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

            total = conn.execute(
                f"SELECT COUNT(*) FROM experiments{where_sql}", params
            ).fetchone()[0]

            # Secondary sort by id breaks ties on updated_at, which can
            # collide because time.time() resolution is coarser than the
            # rate at which rows update during a busy dispatcher. Without
            # a stable tiebreaker, pagination can skip or double-count
            # rows across pages.
            rows = conn.execute(
                f"SELECT * FROM experiments{where_sql} ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()

            return [_row_to_experiment(r) for r in rows], total
        finally:
            conn.close()

    def board_summary(self) -> dict[str, int]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM experiments GROUP BY status"
            ).fetchall()
            return {row["status"]: row["cnt"] for row in rows}
        finally:
            conn.close()

    def active_to_implement_count(self) -> int:
        # Excludes Conductor-parked rows (parked_at IS NOT NULL). Used by the
        # strategist's sliding-cap math so soft-cancelled rows don't eat slots.
        conn = self._connect()
        try:
            return conn.execute(
                "SELECT COUNT(*) FROM experiments "
                "WHERE status='to_implement' AND parked_at IS NULL"
            ).fetchone()[0]
        finally:
            conn.close()

    def leaderboard(
        self,
        metric_key: str = "sharpe",
        top_n: int = 10,
        direction: str = "maximize",
    ) -> list[Experiment]:
        """Top experiments by ``metric_key``, honoring the metric direction.

        ``direction`` must be ``"maximize"`` or ``"minimize"``. Rows with a
        missing, non-numeric, or NaN metric sort last in either direction.
        """
        minimize = direction == "minimize"
        worst = float("inf") if minimize else float("-inf")
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM experiments WHERE results_json IS NOT NULL "
                "ORDER BY updated_at DESC"
            ).fetchall()
            experiments = [_row_to_experiment(r) for r in rows]

            def sort_key(exp: Experiment) -> float:
                try:
                    results = json.loads(exp.results_json or "{}")
                    if not isinstance(results, dict):
                        return worst
                    val = float(results.get(metric_key, worst))
                except (json.JSONDecodeError, ValueError, TypeError):
                    return worst
                # Normalize NaN to the worst sentinel: json.loads accepts
                # "NaN"/"Infinity" via parse_constant, and a NaN key would
                # make Python's sort order non-deterministic (NaN compares
                # false against itself).
                if math.isnan(val):
                    return worst
                return val

            experiments.sort(key=sort_key, reverse=not minimize)
            return experiments[:top_n]
        finally:
            conn.close()

    def count_active_gpus(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM experiments "
                "WHERE status IN ('queued', 'running')"
            ).fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()

    def stale_workers(self, timeout_s: int = 1800) -> list[Experiment]:
        cutoff = time.time() - timeout_s
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM experiments "
                "WHERE worker_id IS NOT NULL "
                "AND status IN ('to_implement', 'implemented', 'finished') "
                "AND updated_at < ?",
                (cutoff,),
            ).fetchall()
            return [_row_to_experiment(r) for r in rows]
        finally:
            conn.close()
