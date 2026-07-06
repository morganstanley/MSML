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
from typing import Literal

logger = logging.getLogger("alpha_lab.experiment_db")


def is_smoke_result(results_json: str | None) -> bool:
    """Return True if the results JSON contains ``smoke_test: true``.

    This is the single source of truth for deciding whether metrics came
    from a quick smoke run rather than a full GPU execution.  The flag
    is expected to be set by the experiment's ``run_experiment.py`` when
    invoked with ``--smoke``.
    """
    if not results_json:
        return False
    try:
        data = json.loads(results_json)
        if isinstance(data, dict):
            return data.get("smoke_test") is True
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return False


# Sentinel file written into <experiment>/results/ by the launcher when
# the experiment subprocess exits with code 0 and metrics.json is present.
CANONICAL_RUN_COMPLETE_SENTINEL = ".canonical_run_complete"


KANBAN_COLUMNS = (
    "to_implement",
    "implemented",
    "checked",
    "queued",
    "running",
    "finished",
    "analyzed",
    "done",
    "cancelled",  # Experiments pruned by strategist
)

# Statuses that reserve a compute slot for capacity accounting — from acceptance
# through job completion; pre-submission states (to_implement/implemented/checked)
# reserve capacity, they don't yet occupy a running slot.
BUSY_STATUSES = ("to_implement", "implemented", "checked", "queued", "running")


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
    # MLflow back-reference cache. Populated by _attach_mlflow_run_to_experiment
    # when MLflow is active; NULL otherwise (no MLflow). Lets workers and
    # update_experiment look up the sub-run UUID in O(1).
    mlflow_run_uuid: str | None = None
    mlflow_artifact_uri: str | None = None


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
    mlflow_run_uuid TEXT,
    mlflow_artifact_uri TEXT
);
"""


def _row_to_experiment(row: sqlite3.Row) -> Experiment:
    # Tolerate older DBs missing newer columns.
    def _opt(col: str, default: object = None) -> object:
        try:
            return row[col]
        except (KeyError, IndexError):
            return default

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
        fix_attempts=_opt("fix_attempts", 0),  # type: ignore[arg-type]
        mlflow_run_uuid=_opt("mlflow_run_uuid"),  # type: ignore[arg-type]
        mlflow_artifact_uri=_opt("mlflow_artifact_uri"),  # type: ignore[arg-type]
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
                # Migrate older DBs that predate columns added later. ALTER
                # TABLE ADD COLUMN is idempotent-safe via try/except —
                # SQLite raises OperationalError when the column exists.
                for col_def in (
                    "mlflow_run_uuid TEXT",
                    "mlflow_artifact_uri TEXT",
                ):
                    try:
                        conn.execute(f"ALTER TABLE experiments ADD COLUMN {col_def}")
                    except sqlite3.OperationalError:
                        pass  # already present
                conn.commit()
            finally:
                conn.close()

    def create(
        self,
        name: str,
        description: str,
        hypothesis: str,
        config_json: str,
    ) -> int:
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "INSERT INTO experiments "
                    "(name, description, hypothesis, config_json, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (name, description, hypothesis, config_json, now, now),
                )
                conn.commit()
                return cur.lastrowid  # type: ignore[return-value]
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

    def update_status(self, exp_id: int, status: str, **kwargs: object) -> None:
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
                conn.execute(
                    f"UPDATE experiments SET {', '.join(sets)} WHERE id = ?",
                    vals,
                )
                conn.commit()
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

    def set_results(self, exp_id: int, results_json: str) -> None:
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE experiments SET results_json = ?, updated_at = ? WHERE id = ?",
                    (results_json, now, exp_id),
                )
                conn.commit()
            finally:
                conn.close()

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

    def set_error_and_finish(self, exp_id: int, error_msg: str) -> None:
        """Set error and update status to 'finished' atomically in one transaction."""
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE experiments SET error = ?, status = 'finished', "
                    "finished_at = ?, updated_at = ? WHERE id = ?",
                    (error_msg, now, now, exp_id),
                )
                conn.commit()
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

    def list_by_status(self, *statuses: str) -> list[Experiment]:
        if not statuses:
            return []
        placeholders = ", ".join("?" for _ in statuses)
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT * FROM experiments WHERE status IN ({placeholders}) "
                "ORDER BY created_at ASC",
                statuses,
            ).fetchall()
            return [_row_to_experiment(r) for r in rows]
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

    def leaderboard(
        self,
        metric_key: str,
        top_n: int,
        direction: Literal["maximize", "minimize"],
    ) -> list[Experiment]:
        """Return top experiments sorted by metric_key.

        Args:
            metric_key: JSON key to extract from results_json.
            top_n: Number of experiments to return.
            direction: "maximize" or "minimize".
        """
        if direction not in ("maximize", "minimize"):
            raise ValueError(f"direction must be 'maximize' or 'minimize', got {direction!r}")
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM experiments "
                "WHERE results_json IS NOT NULL "
                "ORDER BY updated_at DESC"
            ).fetchall()
            experiments = [_row_to_experiment(r) for r in rows]

            # Filter by the launcher-written canonical-run sentinel.
            # Reject names containing path separators or `..` parts so a
            # malformed name can't make the existence check resolve outside
            # <workspace>/experiments/<name>/.
            workspace = Path(self.db_path).parent
            experiments_dir = workspace / "experiments"
            before_sentinel = len(experiments)

            filtered: list[Experiment] = []
            for e in experiments:
                if "/" in e.name or "\\" in e.name:
                    continue
                try:
                    if ".." in Path(e.name).parts:
                        continue
                    if (
                        experiments_dir / e.name / "results" / CANONICAL_RUN_COMPLETE_SENTINEL
                    ).is_file():
                        filtered.append(e)
                except (OSError, ValueError):
                    continue
            experiments = filtered
            dropped = before_sentinel - len(experiments)
            if dropped:
                logger.debug(
                    "Leaderboard: excluded %d row(s) with no canonical-run sentinel",
                    dropped,
                )

            # Smoke-test filter.
            before = len(experiments)
            experiments = [
                e for e in experiments
                if not is_smoke_result(e.results_json)
            ]
            filtered = before - len(experiments)
            if filtered:
                logger.debug(
                    "Leaderboard: excluded %d smoke-test result(s)", filtered
                )

            worst = float("-inf") if direction == "maximize" else float("inf")

            def sort_key(exp: Experiment) -> float:
                try:
                    results = json.loads(exp.results_json or "{}")
                    if not isinstance(results, dict):
                        return worst
                    val = float(results.get(metric_key, worst))
                except (json.JSONDecodeError, ValueError, TypeError):
                    return worst
                if math.isnan(val):
                    return worst
                return val

            experiments.sort(key=sort_key, reverse=(direction != "minimize"))
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
