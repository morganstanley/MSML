"""Run import-resolved benchmark generators with import-resolved runners."""

from __future__ import annotations

import argparse
import contextlib
import getpass
import json
import logging
import os
import shlex
import socket
import sys
import tempfile
from contextlib import AbstractContextManager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alpha_lab.benchmarks.agents import AgentConfig
from alpha_lab.benchmarks.manifest import RUN_MANIFEST_NAME
from alpha_lab.benchmarks.paths import git_commit
from alpha_lab.benchmarks.registry.store import connect_registry
from alpha_lab.utils import resolve_import


LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Alpha Lab benchmark workspaces.")
    parser.add_argument("--generator", required=True, help="Import path for workspace generator.")
    parser.add_argument("--generator-kwargs", default="{}", help="JSON object passed to generator.")
    parser.add_argument("--runner", required=True, help="Import path for runner.")
    parser.add_argument("--runner-kwargs", default="{}", help="JSON object passed to runner.")
    parser.add_argument(
        "--agent-config",
        default=None,
        help="JSON object with AgentConfig fields (e.g. provider, model).",
    )
    workspace_group = parser.add_mutually_exclusive_group(required=True)
    workspace_group.add_argument(
        "--workspace-root",
        type=Path,
        default=None,
        help=(
            "Persistent directory under which each workspace is materialized. "
            "Workspaces remain after the run completes."
        ),
    )
    workspace_group.add_argument(
        "--temporary-workspaces",
        action="store_true",
        help=(
            "Create workspaces inside a tempfile.TemporaryDirectory that is "
            "auto-deleted on exit. Mutually exclusive with --workspace-root."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--config-overrides",
        default="{}",
        help="JSON object deep-merged into each workspace's TaskConfig.",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    if args.num_workers < 1:
        LOGGER.error("--num-workers must be at least 1")
        return 2

    generator_kwargs = _json_object(args.generator_kwargs, "--generator-kwargs")
    runner_kwargs = _json_object(args.runner_kwargs, "--runner-kwargs")
    config_overrides = _json_object(args.config_overrides, "--config-overrides")
    agent_config = _agent_config(args.agent_config)

    generator_factory = resolve_import(args.generator)
    runner_factory = resolve_import(args.runner)
    if not callable(generator_factory):
        LOGGER.error("--generator %r resolved to a non-callable", args.generator)
        return 2
    if not callable(runner_factory):
        LOGGER.error("--runner %r resolved to a non-callable", args.runner)
        return 2
    runner = runner_factory(**runner_kwargs)

    workspace_root_cm: AbstractContextManager[str | Path]
    if args.temporary_workspaces:
        workspace_root_cm = tempfile.TemporaryDirectory(prefix="alpha-bench-")
    else:
        workspace_parent = args.workspace_root.resolve()
        workspace_parent.mkdir(parents=True, exist_ok=True)
        workspace_root_cm = contextlib.nullcontext(workspace_parent)

    with workspace_root_cm as workspace_root:
        temporary_workspace_root = Path(workspace_root)
        generator = generator_factory(
            workspace_root=temporary_workspace_root,
            overwrite=args.overwrite,
            agent_config=agent_config,
            config_overrides=config_overrides,
            **generator_kwargs,
        )
        effective_argv = (
            sys.argv
            if argv is None
            else ["python", "-m", "alpha_lab.benchmarks.scripts.run_benchmarks", *argv]
        )
        _write_run_manifest(
            temporary_workspace_root,
            argv=effective_argv,
            args=args,
            temporary_workspace_root=temporary_workspace_root,
            generator_kwargs=generator_kwargs,
            runner_kwargs=runner_kwargs,
            agent_config=agent_config,
        )
        exit_codes = runner.run_many(generator, num_workers=args.num_workers)

    return 0 if all(code == 0 for code in exit_codes) else 1


def _json_object(value: str, flag: str) -> dict[str, Any]:
    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError(f"{flag} must decode to a JSON object")
    return data


def _agent_config(value: str | None) -> AgentConfig | None:
    """Parse ``--agent-config`` JSON into an :class:`AgentConfig`."""
    if value is None:
        return None
    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("--agent-config must decode to a JSON object")
    return AgentConfig(**data)


def _write_run_manifest(
    run_root: Path,
    *,
    argv: list[str],
    args: argparse.Namespace,
    temporary_workspace_root: Path,
    generator_kwargs: dict[str, Any],
    runner_kwargs: dict[str, Any],
    agent_config: AgentConfig | None,
) -> None:
    """Write the per-invocation run manifest."""
    manifest = {
        "command": shlex.join(argv),
        "argv": argv,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "git_commit": git_commit(),
        "temporary_workspace_root": str(temporary_workspace_root),
        "workspace_parent": (
            str(args.workspace_root.resolve()) if args.workspace_root else None
        ),
        "overwrite": args.overwrite,
        "num_workers": args.num_workers,
        "generator": {
            "import": args.generator,
            "kwargs": generator_kwargs,
        },
        "runner": {
            "import": args.runner,
            "kwargs": runner_kwargs,
        },
        "agent": {
            "config": asdict(agent_config) if agent_config is not None else None,
        },
    }
    (run_root / RUN_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")


_RUNNER_REGISTRY: dict[str, str] = {
    "local": "alpha_lab.benchmarks.runners:LocalRunner",
    "mlflow": "alpha_lab.benchmarks.runners:MLflowRunner",
}


def _resolve_filter(db_path: Path, filter_values: list[str] | None) -> list[str] | None:
    """Translate ``--filter`` values to a list of benchmark ids.

    If ``filter_values`` is a single integer N, query ``db_path`` and return
    the first N ids in registry-insertion order. Otherwise the values are
    treated as literal ids. Returns ``None`` when no filter is set.
    """
    if not filter_values:
        return None
    if len(filter_values) == 1:
        try:
            n = int(filter_values[0])
        except ValueError:
            n = None
        if n is not None:
            if n < 1:
                raise ValueError(f"--filter integer must be >= 1, got {n}")
            conn = connect_registry(db_path)
            try:
                rows = conn.execute(
                    "SELECT id FROM benchmarks ORDER BY rowid"
                ).fetchmany(n)
            finally:
                conn.close()
            ids = [r["id"] for r in rows]
            if len(ids) < n:
                LOGGER.warning(
                    "--filter requested %d ids but only %d available in %s",
                    n, len(ids), db_path,
                )
            return ids
    return list(filter_values)


def _expand_flags(flags_json: str | None) -> list[str]:
    """Expand a JSON ``{flag: value}`` dict into argv form.

    Keys convert from ``snake_case`` to ``--kebab-case``. Booleans produce
    a bare flag when True (omitted when False). Dict/list values are
    re-serialized as JSON.
    """
    if not flags_json:
        return []
    try:
        obj = json.loads(flags_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--flags must be valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("--flags must decode to a JSON object")

    argv: list[str] = []
    for key, value in obj.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif isinstance(value, (dict, list)):
            argv += [flag, json.dumps(value)]
        elif value is None:
            continue
        else:
            argv += [flag, str(value)]
    return argv


def cli(argv: list[str] | None = None) -> int:
    """Simplified CLI: ``--db <suite.db> [--save DIR] [--filter ...] [--flags JSON]``."""
    parser = argparse.ArgumentParser(
        description="Run a benchmark suite from a suite.db file."
    )
    parser.add_argument("--db", type=Path, required=True, help="Path to suite.db.")
    parser.add_argument(
        "--save", type=Path, default=None, metavar="DIR",
        help="Persist materialized workspaces under DIR. Absent: tempdir.",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--runner", choices=sorted(_RUNNER_REGISTRY), default="local",
    )
    parser.add_argument(
        "--filter", nargs="+", default=None, metavar="ID_OR_N",
        help="One or more ids, OR a single int N → first N entries.",
    )
    parser.add_argument(
        "--flags", type=str, default=None, metavar="JSON",
        help="JSON dict of additional flags forwarded to the underlying main().",
    )
    args = parser.parse_args(argv)

    db_path = args.db.resolve()
    if not db_path.is_file():
        LOGGER.error("--db file not found: %s", db_path)
        return 2

    gen_kwargs: dict[str, Any] = {"registry": str(db_path)}
    try:
        ids = _resolve_filter(db_path, args.filter)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 2
    if ids is not None:
        gen_kwargs["benchmark_ids"] = ids

    inner: list[str] = [
        "--generator", "alpha_lab.benchmarks.generators.database:RegistryGenerator",
        "--generator-kwargs", json.dumps(gen_kwargs),
        "--runner", _RUNNER_REGISTRY[args.runner],
        "--num-workers", str(args.num_workers),
    ]
    if args.save is not None:
        inner += ["--workspace-root", str(args.save.resolve())]
    else:
        inner += ["--temporary-workspaces"]

    try:
        inner += _expand_flags(args.flags)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 2

    return main(inner)


if __name__ == "__main__":
    raise SystemExit(main())
