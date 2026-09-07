"""SQLite registry access."""

from __future__ import annotations

import json
import sqlite3
from importlib.resources import files
from pathlib import Path
from typing import Any

from alpha_lab.benchmarks.registry.models import Benchmark
from alpha_lab.config import PipelineConfig


def _schema_text() -> str:
    """Read ``schema.sql`` from the packaged registry resources."""
    return files("alpha_lab.benchmarks.registry").joinpath("schema.sql").read_text()


def connect_registry(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_schema_text())
    conn.commit()


def insert_benchmark_row(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO benchmarks (
            id, name, data_path, description, target, domain,
            provider, model, reasoning_effort, shell_timeout,
            tool_output_max_chars, pipeline_json, adapter_path, seed_path,
            enabled, notes, created_at, updated_at, creator, owner
        )
        VALUES (
            :id, :name, :data_path, :description, :target,
            :domain, :provider, :model, :reasoning_effort, :shell_timeout,
            :tool_output_max_chars, :pipeline_json, :adapter_path,
            :seed_path, :enabled, :notes, :created_at, :updated_at,
            :creator, :owner
        )
        """,
        row,
    )


def load_benchmarks(
    conn: sqlite3.Connection,
    benchmark_ids: list[str] | None,
) -> list[Benchmark]:
    if benchmark_ids:
        placeholders = ", ".join("?" for _ in benchmark_ids)
        rows = conn.execute(
            f"SELECT * FROM benchmarks WHERE id IN ({placeholders}) ORDER BY id",
            benchmark_ids,
        ).fetchall()
        found = {row["id"] for row in rows}
        missing = sorted(set(benchmark_ids) - found)
        if missing:
            raise ValueError(f"Unknown benchmark id(s): {', '.join(missing)}")
    else:
        rows = conn.execute(
            "SELECT * FROM benchmarks WHERE enabled = 1 ORDER BY id"
        ).fetchall()

    benchmarks: list[Benchmark] = []
    for row in rows:
        try:
            pipeline_raw = json.loads(row["pipeline_json"])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Benchmark {row['id']} has invalid pipeline_json: {exc}"
            ) from exc
        if not isinstance(pipeline_raw, dict):
            raise ValueError(f"Benchmark {row['id']} pipeline_json must be an object")
        pipeline = PipelineConfig(**pipeline_raw)
        benchmarks.append(
            Benchmark(
                id=row["id"],
                name=row["name"],
                data_path=Path(row["data_path"]),
                description=row["description"],
                target=row["target"],
                domain=row["domain"],
                provider=row["provider"],
                model=row["model"],
                reasoning_effort=row["reasoning_effort"],
                shell_timeout=row["shell_timeout"],
                tool_output_max_chars=row["tool_output_max_chars"],
                pipeline=pipeline,
                adapter_path=Path(row["adapter_path"]) if row["adapter_path"] else None,
                seed_path=Path(row["seed_path"]) if row["seed_path"] else None,
                notes=row["notes"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                creator=row["creator"],
                owner=row["owner"],
            )
        )
    return benchmarks
