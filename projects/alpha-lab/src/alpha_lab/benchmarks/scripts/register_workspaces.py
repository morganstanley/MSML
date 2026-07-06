"""Register existing Alpha Lab workspaces into a benchmark suite.

Bootstraps each source workspace (full copy by default) into the suite's
workspaces/ directory and inserts a registry row into suite.db. Pass
``--symlink-data`` to symlink data files instead of copying.
"""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alpha_lab.benchmarks.manifest import (
    benchmark_snapshot,
    write_benchmark_manifest,
)
from alpha_lab.benchmarks.registry.models import Benchmark
from alpha_lab.benchmarks.registry.store import (
    connect_registry,
    ensure_schema,
    insert_benchmark_row,
)
from alpha_lab.benchmarks.scripts.copy_workspace import (
    copy_workspace as _copy,
)

LOGGER = logging.getLogger(__name__)

SUITE_DB_NAME = "suite.db"
WORKSPACES_SUBDIR = "workspaces"

_REQUIRED_CONFIG_FIELDS = ("data_path", "description", "provider", "model")


def _validate_external_config(value: str) -> dict[str, Any]:
    """Load and validate an external config from ``--config``.

    Accepts either a JSON-encoded dict or a path to a JSON file. Tries the
    inline-JSON interpretation first; falls back to treating the value as a
    file path. Raises if neither interpretation yields a valid config dict.

    Args:
        value: Inline JSON dict or path to a JSON file.

    Returns:
        Parsed config dict.

    Raises:
        ValueError: If neither interpretation is valid, or the parsed
            object is not a dict, or required fields are missing.
    """
    data: Any
    try:
        data = json.loads(value)
        if not isinstance(data, dict):
            raise ValueError("inline JSON must decode to an object")
    except (json.JSONDecodeError, ValueError):
        path = Path(value)
        if not path.is_file():
            raise ValueError(
                f"--config {value!r} is neither valid JSON nor an existing file."
            )
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"--config file {path} is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise ValueError(f"--config file must contain a JSON object: {path}")

    for field in _REQUIRED_CONFIG_FIELDS:
        if field not in data:
            raise ValueError(
                f"--config missing required field {field!r}"
            )
    return data


def _build_registry_row(
    workspace_id: str,
    config: dict[str, Any],
    data_path: str,
    adapter_path: Path | None,
    *,
    created_at: str,
    creator: str,
    owner: str,
) -> dict[str, Any]:
    """Build a registry row dict from a workspace's config.json.

    Args:
        workspace_id: Registry ID (source workspace directory name).
        config: Parsed config.json.
        data_path: Resolved absolute data_path.
        adapter_path: Absolute path to the copied adapter dir, or None.
        created_at: ISO timestamp.
        creator: Username of the person running the script.
        owner: Suite owner.

    Returns:
        Row dict compatible with insert_benchmark_row.
    """
    pipeline = config.get("pipeline", {"phases": ["phase1"]})
    return {
        "id": workspace_id,
        "name": config.get("name", workspace_id),
        "data_path": data_path,
        "description": config["description"],
        "target": config.get("target", ""),
        "domain": config.get("domain", ""),
        "provider": config["provider"],
        "model": config["model"],
        "reasoning_effort": config.get("reasoning_effort", "low"),
        "shell_timeout": config.get("shell_timeout", 300),
        "tool_output_max_chars": config.get("tool_output_max_chars", 8000),
        "pipeline_json": json.dumps(pipeline),
        "adapter_path": str(adapter_path) if adapter_path else None,
        "seed_path": None,
        "enabled": 1,
        "notes": "",
        "created_at": created_at,
        "updated_at": created_at,
        "creator": creator,
        "owner": owner,
    }


def register(
    src_workspace: Path,
    suite_dir: Path,
    conn: sqlite3.Connection,
    *,
    symlink_data: bool,
    overwrite: bool,
    created_at: str,
    creator: str,
    owner: str,
    external_config: dict[str, Any] | None = None,
) -> None:
    """Register one workspace into the suite.

    Args:
        src_workspace: Source workspace directory.
        suite_dir: Suite root (contains suite.db and workspaces/).
        conn: Open registry SQLite connection.
        symlink_data: If True, symlink the source's ``data/`` instead of
            copying. Default copies.
        overwrite: Whether to replace an existing destination workspace.
        created_at: ISO timestamp for the registry row.
        creator: Registering user.
        owner: Suite owner.
        external_config: If provided, use this as config.json instead of
            reading from the source workspace.

    Raises:
        FileExistsError: If destination workspace exists and overwrite is False.
        FileNotFoundError: If config.json or data_path is missing.
        ValueError: If config.json is malformed.
    """
    workspace_id = src_workspace.name
    dst_workspace = suite_dir / WORKSPACES_SUBDIR / workspace_id

    if dst_workspace.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination workspace already exists: {dst_workspace}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(dst_workspace)

    dst_workspace = _copy(
        source=src_workspace,
        output_dir=suite_dir / WORKSPACES_SUBDIR,
        name=workspace_id,
        symlink_data=symlink_data,
        config=external_config,
    )

    config = json.loads((dst_workspace / "config.json").read_text())
    data_path = config["data_path"]
    adapter_dir = dst_workspace / "adapter"
    adapter_path = adapter_dir.resolve() if adapter_dir.is_dir() else None

    row = _build_registry_row(
        workspace_id,
        config,
        data_path,
        adapter_path,
        created_at=created_at,
        creator=creator,
        owner=owner,
    )

    benchmark = Benchmark(
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
        pipeline=json.loads(row["pipeline_json"]),
        adapter_path=adapter_path,
        seed_path=None,
        notes=row["notes"],
    )

    # Read the pre-copy data_path from the source (or override config) so the
    # manifest's data_source points at the original data location, not at the
    # destination path that copy_workspace wrote into the workspace's config.
    if external_config is not None:
        original_data_path = external_config["data_path"]
    else:
        src_config = json.loads((src_workspace / "config.json").read_text())
        original_data_path = src_config["data_path"]

    write_benchmark_manifest(
        dst_workspace,
        source={"kind": "registered", "source_workspace": str(src_workspace.resolve())},
        benchmark=benchmark_snapshot(benchmark),
        materialized={
            "data_path": data_path,
            "data_source": str(
                (src_workspace / original_data_path).resolve()
                if not Path(original_data_path).is_absolute()
                else original_data_path
            ),
            "data_copied": not symlink_data,
            "adapter_source": str(src_workspace / "adapter") if adapter_path else None,
        },
        config=config,
    )

    insert_benchmark_row(conn, row)
    LOGGER.info("[registered] %s -> %s", src_workspace.name, dst_workspace)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Register existing Alpha Lab workspaces into a benchmark suite."
    )
    parser.add_argument(
        "--workspaces",
        nargs="+",
        required=True,
        type=Path,
        metavar="WORKSPACE",
        help="Source workspace directories to register.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Suite root directory (suite.db and workspaces/ are written here).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="CONFIG",
        help=(
            "External config to apply to each workspace. Accepts either a "
            "JSON-encoded dict or a path to a JSON file. Required config "
            "fields must be present; otherwise fails loudly."
        ),
    )
    parser.add_argument(
        "--symlink-data",
        action="store_true",
        dest="symlink_data",
        help="Symlink data files into the suite. Default: copy.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing destination workspaces and registry rows.",
    )
    parser.add_argument(
        "--owner",
        default=None,
        help="Suite owner (defaults to current user).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    # Filter to directories, warn on non-directories.
    sources: list[Path] = []
    for ws_path in args.workspaces:
        src = ws_path.resolve()
        if not src.is_dir():
            LOGGER.warning("Skipping non-directory: %s", src)
            continue
        sources.append(src)

    if not sources:
        LOGGER.error("No valid workspace directories found.")
        return 2

    # Validate external config if provided. Accepts inline JSON or file path.
    external_config: dict[str, Any] | None = None
    if args.config is not None:
        try:
            external_config = _validate_external_config(args.config)
        except ValueError as exc:
            LOGGER.error("%s", exc)
            return 2

    suite_dir = args.output_dir.resolve()
    suite_db = suite_dir / SUITE_DB_NAME
    (suite_dir / WORKSPACES_SUBDIR).mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    creator = getpass.getuser()
    owner = args.owner or creator

    conn = connect_registry(suite_db)
    try:
        ensure_schema(conn)
        for src in sources:
            try:
                register(
                    src,
                    suite_dir,
                    conn,
                    symlink_data=args.symlink_data,
                    overwrite=args.overwrite,
                    created_at=now,
                    creator=creator,
                    owner=owner,
                    external_config=external_config,
                )
            except (FileExistsError, FileNotFoundError, ValueError) as exc:
                LOGGER.error("%s", exc)
                return 2
        conn.commit()
    finally:
        conn.close()

    LOGGER.info("Suite written to %s", suite_dir)
    return 0


def cli(argv: list[str] | None = None) -> int:
    """Simplified CLI: ``--src WS... --dest PATH [--config JSON_OR_PATH] [...]``."""
    parser = argparse.ArgumentParser(
        description="Register existing Alpha Lab workspaces into a benchmark suite."
    )
    parser.add_argument(
        "--src", nargs="+", required=True, type=Path, metavar="WORKSPACE",
        help="One or more source workspace directories.",
    )
    parser.add_argument(
        "--dest", type=Path, required=True,
        help="Destination suite directory.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="External config: inline JSON dict or path to a JSON file.",
    )
    parser.add_argument("--symlink-data", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--owner", default=None)
    args = parser.parse_args(argv)

    inner: list[str] = ["--workspaces", *[str(p) for p in args.src]]
    inner += ["--output-dir", str(args.dest)]
    if args.config is not None:
        inner += ["--config", args.config]
    if args.symlink_data:
        inner.append("--symlink-data")
    if args.overwrite:
        inner.append("--overwrite")
    if args.owner is not None:
        inner += ["--owner", args.owner]
    return main(inner)


if __name__ == "__main__":
    raise SystemExit(main())
