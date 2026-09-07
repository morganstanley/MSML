"""Helpers for creating local registries from checkout-local examples/configs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from alpha_lab.benchmarks.paths import find_repo_root
from alpha_lab.benchmarks.registry.store import connect_registry, ensure_schema


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _seed_row(benchmark_id: str, name: str, config_path: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    data_path = Path(config["data_path"])
    if not data_path.is_absolute():
        data_path = (find_repo_root() / data_path).resolve()
    pipeline = config.get("pipeline", {"phases": ["phase1"]})
    return {
        "id": benchmark_id,
        "name": name,
        "data_path": str(data_path),
        "description": config["description"],
        "target": config.get("target", ""),
        "domain": config.get("domain", ""),
        "provider": config.get("provider", "openai"),
        "model": config.get("model", "gpt-5.2"),
        "reasoning_effort": config.get("reasoning_effort", "low"),
        "shell_timeout": config.get("shell_timeout", 300),
        "tool_output_max_chars": config.get("tool_output_max_chars", 8000),
        "pipeline_json": json.dumps(pipeline),
        "adapter_path": None,
        "seed_path": None,
        "enabled": 1,
        "notes": f"Seeded from {config_path.relative_to(find_repo_root())}",
        "created_at": None,
        "updated_at": None,
        "creator": None,
        "owner": None,
    }


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


def initialize_default_registry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect_registry(path)
    try:
        ensure_schema(conn)
        rows = [
            _seed_row(
                "demo_exchange",
                "Demo exchange rates",
                find_repo_root() / "data" / "demo_exchange_config.json",
            ),
            _seed_row(
                "llm_speedrun",
                "LLM speedrun",
                find_repo_root() / "data" / "llm_speedrun_config.json",
            ),
            _seed_row(
                "traffic",
                "Traffic forecasting",
                find_repo_root() / "data" / "paper_traffic_gpt.json",
            ),
        ]
        for row in rows:
            insert_benchmark_row(conn, row)
        conn.commit()
    finally:
        conn.close()
