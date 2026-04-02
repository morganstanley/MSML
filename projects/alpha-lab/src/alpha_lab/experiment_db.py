"""SQLite experiment database for Phase 3 kanban tracking."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass


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
    fix_attempts INTEGER NOT NULL DEFAULT 0
);
"""


def _row_to_experiment(row: sqlite3.Row) -> Experiment:
    # Handle missing fix_attempts column for existing DBs
    try:
        fix_attempts = row["fix_attempts"]
    except (KeyError, IndexError):
        fix_attempts = 0

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
        fix_attempts=fix_attempts,
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
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(_CREATE_TABLE)
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

    def board_summary(self) -> dict[str, int]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM experiments GROUP BY status"
            ).fetchall()
            return {row["status"]: row["cnt"] for row in rows}
        finally:
            conn.close()

    def leaderboard(self, metric_key: str = "sharpe", top_n: int = 10) -> list[Experiment]:
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
                    return float(results.get(metric_key, float("-inf")))
                except (json.JSONDecodeError, ValueError, TypeError):
                    return float("-inf")

            experiments.sort(key=sort_key, reverse=True)
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
