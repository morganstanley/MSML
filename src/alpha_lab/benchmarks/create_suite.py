"""Create a benchmark suite: persisted workspaces plus a registry DB indexing them.

A suite is a self-contained directory of the form::

    <output_dir>/
        suite.db
        workspaces/
            <id_1>/{data/, config.json, benchmark_manifest.json}
            <id_2>/...

Workspaces are materialized by iterating a :class:`StructuralCausalGenerator`;
``suite.db`` is built from the resulting per-workspace configs and manifests.
"""

from __future__ import annotations

import argparse
import getpass
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from alpha_lab.benchmarks.manifest import BENCHMARK_MANIFEST_NAME
from alpha_lab.benchmarks.registry.seed import insert_benchmark_row
from alpha_lab.benchmarks.registry.store import connect_registry, ensure_schema


LOGGER = logging.getLogger(__name__)

SUITE_DB_NAME = "suite.db"
WORKSPACES_SUBDIR = "workspaces"
SUITE_CONFIGS_FILE = Path(__file__).parent / "configs" / "suites.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a benchmark suite (workspaces + registry DB)."
    )
    parser.add_argument(
        "--suite",
        required=True,
        help=f"Suite identifier defined in {SUITE_CONFIGS_FILE.relative_to(Path(__file__).parent)}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write suite.db and workspaces/ into.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing workspaces and suite.db.",
    )
    parser.add_argument(
        "--owner",
        default=None,
        help="Owner of this suite (defaults to current user).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Materialize the suite and write its registry DB."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    suite_config = _load_suite_config(args.suite)
    output_dir = args.output_dir.resolve()
    workspaces_dir = output_dir / WORKSPACES_SUBDIR
    suite_db = output_dir / SUITE_DB_NAME

    if suite_db.exists() and not args.overwrite:
        LOGGER.error(
            "Suite DB already exists: %s. Use --overwrite to replace.", suite_db
        )
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    workspaces_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Materializing workspaces under %s", workspaces_dir)
    from alpha_lab.benchmarks.generators.structural_causal import StructuralCausalGenerator
    generator = StructuralCausalGenerator(
        workspace_root=workspaces_dir,
        overwrite=args.overwrite,
        config_overrides=suite_config["config_overrides"],
        **suite_config["generator_kwargs"],
    )
    for workspace in generator:
        LOGGER.info("[materialized] %s", workspace)

    now = datetime.now(timezone.utc).isoformat()
    creator = getpass.getuser()
    owner = args.owner or creator

    if suite_db.exists():
        suite_db.unlink()
    LOGGER.info("Building suite DB at %s", suite_db)
    _build_suite_db(workspaces_dir, suite_db, created_at=now, creator=creator, owner=owner)

    LOGGER.info("Suite written to %s", output_dir)
    return 0


def _load_suite_config(suite_path: str) -> dict[str, Any]:
    """Load a suite from ``suites.yaml`` using a slash-separated path.

    Args:
        suite_path: Slash-separated identifier, e.g. ``scm_classification/smoke_test``.
    """
    document = yaml.safe_load(SUITE_CONFIGS_FILE.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"{SUITE_CONFIGS_FILE} must be a YAML mapping")

    suites = document.get("suites", {})
    parts = [p for p in suite_path.split("/") if p]
    node: Any = suites
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            available = _list_suites(suites)
            raise ValueError(
                f"Unknown suite {suite_path!r}. Available: "
                f"{', '.join(available) or '<none>'} (see {SUITE_CONFIGS_FILE})."
            )
        node = node[part]

    if not isinstance(node, dict):
        raise ValueError(f"Suite {suite_path!r} must be a mapping")
    for key in ("generator_kwargs", "config_overrides"):
        if not isinstance(node.get(key, {}), dict):
            raise ValueError(f"Suite {suite_path!r}: '{key}' must be a mapping")
    node.setdefault("generator_kwargs", {})
    node.setdefault("config_overrides", {})
    return node


def _list_suites(node: dict[str, Any], prefix: str = "") -> list[str]:
    """Return all non-private leaf suite paths under ``node``."""
    result = []
    for key, value in node.items():
        if key.startswith("_"):
            continue
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            if "generator_kwargs" in value or "config_overrides" in value:
                result.append(path)
            else:
                result.extend(_list_suites(value, path))
    return sorted(result)


def _build_suite_db(
    workspaces_dir: Path,
    suite_db: Path,
    *,
    created_at: str,
    creator: str,
    owner: str,
) -> None:
    """Insert one ``Benchmark`` row per materialized workspace into ``suite_db``."""
    conn = connect_registry(suite_db)
    try:
        ensure_schema(conn)
        for workspace in sorted(workspaces_dir.iterdir()):
            if not workspace.is_dir():
                continue
            insert_benchmark_row(
                conn,
                _row_from_workspace(
                    workspace,
                    created_at=created_at,
                    creator=creator,
                    owner=owner,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _row_from_workspace(
    workspace: Path,
    *,
    created_at: str,
    creator: str,
    owner: str,
) -> dict[str, Any]:
    """Build a registry row dict from a materialized workspace's config + manifest."""
    config = json.loads((workspace / "config.json").read_text())
    manifest = json.loads((workspace / BENCHMARK_MANIFEST_NAME).read_text())
    bench = manifest.get("benchmark", {})
    return {
        "id": workspace.name,
        "name": bench.get("name", workspace.name),
        "data_path": config["data_path"],
        "description": config["description"],
        "target": config.get("target", ""),
        "domain": config.get("domain", ""),
        "provider": config["provider"],
        "model": config["model"],
        "reasoning_effort": config["reasoning_effort"],
        "shell_timeout": config["shell_timeout"],
        "tool_output_max_chars": config["tool_output_max_chars"],
        "pipeline_json": json.dumps(config["pipeline"]),
        "adapter_path": None,
        "seed_path": None,
        "enabled": 1,
        "notes": bench.get("notes", ""),
        "created_at": created_at,
        "updated_at": created_at,
        "creator": creator,
        "owner": owner,
    }


if __name__ == "__main__":
    raise SystemExit(main())
