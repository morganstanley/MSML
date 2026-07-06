"""Create a benchmark suite: persisted workspaces plus a registry DB indexing them.

A suite is a self-contained directory of the form::

    <output_dir>/
        suite.db
        workspaces/
            <id_1>/{data/, config.json, benchmark_manifest.json}
            <id_2>/...

Workspaces are materialized by iterating the generator declared by the suite
entry in ``benchmarks/suites.yaml``; ``suite.db`` is built from the resulting
per-workspace configs and manifests.
"""

from __future__ import annotations

import argparse
import getpass
import json
import logging
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from alpha_lab.benchmarks.generators.base import WorkspaceGenerator
from alpha_lab.benchmarks.manifest import BENCHMARK_MANIFEST_NAME
from alpha_lab.benchmarks.registry.store import (
    connect_registry,
    ensure_schema,
    insert_benchmark_row,
)
from alpha_lab.utils import resolve_import


LOGGER = logging.getLogger(__name__)

SUITE_DB_NAME = "suite.db"
WORKSPACES_SUBDIR = "workspaces"


def _suite_configs_resource() -> Any:
    """Return the package resource handle for ``suites.yaml``."""
    return files("alpha_lab.benchmarks").joinpath("suites.yaml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a benchmark suite (workspaces + registry DB)."
    )
    parser.add_argument(
        "--suite",
        required=True,
        help=(
            "Suite identifier. Either a slash-separated key into the bundled "
            "suites.yaml (e.g. 'gp_blackbox/smoke_test'), or "
            "'<path>:<key>' to load 'key' from an external YAML file at 'path'."
        ),
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
    generator_cls = resolve_import(
        suite_config["generator"], types=WorkspaceGenerator,
    )
    generator = generator_cls(
        workspace_root=workspaces_dir,
        overwrite=args.overwrite,
        config_overrides=suite_config["config_overrides"],
        **suite_config["generator_kwargs"],
    )
    # The generator yields zero-arg factories per WorkspaceGenerator's
    # contract (see base.py); call each one to actually materialize the
    # workspace on disk. Tolerate older shapes that yield Paths directly.
    for factory in generator:
        workspace = factory() if callable(factory) else factory
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
    """Load a suite from a YAML file using a slash-separated key path.

    Args:
        suite_path: Either ``<group>/<tier>`` (resolved against the bundled
            ``suites.yaml``) or ``<path>:<key>`` (loads from an external YAML
            file at ``<path>`` and resolves ``<key>`` within it).
    """
    if ":" in suite_path:
        yaml_path, _, key_path = suite_path.partition(":")
        source: Any = Path(yaml_path)
        if not source.is_file():
            raise FileNotFoundError(f"Suite YAML not found: {source}")
        document = yaml.safe_load(source.read_text())
    else:
        source = _suite_configs_resource()
        document = yaml.safe_load(source.read_text())
        key_path = suite_path
    if not isinstance(document, dict):
        raise ValueError(f"{source} must be a YAML mapping")

    # Suite groups live at the top level of the document (no ``suites:``
    # wrapper). Anchor-only entries (keys starting with ``_``) are skipped
    # when listing available suites.
    parts = [p for p in key_path.split("/") if p]
    node: Any = document
    inherited_generator: str | None = None
    for i, part in enumerate(parts):
        if not isinstance(node, dict) or part not in node:
            available = _list_suites(document)
            raise ValueError(
                f"Unknown suite {suite_path!r}. Available: "
                f"{', '.join(available) or '<none>'} (see {source})."
            )
        next_node = node[part]
        # Inherit ``generator`` from intermediate (group) nodes so each
        # leaf tier doesn't have to repeat it. The leaf itself may still
        # override.
        if i < len(parts) - 1 and isinstance(next_node, dict):
            ancestor_gen = next_node.get("generator")
            if isinstance(ancestor_gen, str):
                inherited_generator = ancestor_gen
        node = next_node

    if not isinstance(node, dict):
        raise ValueError(f"Suite {suite_path!r} must be a mapping")
    if "generator" not in node and inherited_generator is not None:
        node = {**node, "generator": inherited_generator}
    if "generator" not in node:
        raise ValueError(
            f"Suite {suite_path!r}: missing required 'generator' field "
            f"(declare on the suite group, e.g. "
            f"'alpha_lab.benchmarks.generators.gp_regression:GPRegressionGenerator')."
        )
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


def cli(argv: list[str] | None = None) -> int:
    """Simplified CLI: ``--suite ID --dest PATH [--overwrite] [--owner NAME]``."""
    parser = argparse.ArgumentParser(
        description="Materialize a benchmark suite (workspaces + registry DB)."
    )
    parser.add_argument(
        "--suite", required=True,
        help=(
            "Suite identifier. Either '<group>/<tier>' (bundled suites.yaml) "
            "or '<path>:<key>' (external YAML file)."
        ),
    )
    parser.add_argument(
        "--dest", type=Path, required=True,
        help="Destination suite directory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--owner", default=None)
    args = parser.parse_args(argv)

    inner = ["--suite", args.suite, "--output-dir", str(args.dest)]
    if args.overwrite:
        inner.append("--overwrite")
    if args.owner is not None:
        inner += ["--owner", args.owner]
    return main(inner)


if __name__ == "__main__":
    raise SystemExit(main())
