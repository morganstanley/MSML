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
from importlib import import_module
from pathlib import Path
from typing import Any

from alpha_lab.benchmarks.agents import AgentConfig
from alpha_lab.benchmarks.manifest import RUN_MANIFEST_NAME
from alpha_lab.benchmarks.paths import git_commit


LOGGER = logging.getLogger(__name__)


def resolve_import(
    import_path: str,
    types: type | tuple[type, ...] | None = None,
) -> Any:
    module_name, sep, object_name = import_path.partition(":")
    if not sep:
        module_name, _, object_name = import_path.rpartition(".")
    if not module_name or not object_name:
        raise ValueError(
            f"Import path must be 'module:object' or 'module.object': {import_path!r}"
        )

    module = import_module(module_name)
    if not hasattr(module, object_name):
        raise AttributeError(
            f"Module {module_name!r} does not have attribute {object_name!r}."
        )

    obj = getattr(module, object_name)
    if types is None:
        return obj
    if isinstance(obj, type) and issubclass(obj, types):
        return obj
    if not isinstance(obj, type) and isinstance(obj, types):
        return obj
    raise TypeError(f"{module_name}.{object_name} does not satisfy {types}.")


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
    parser.add_argument(
        "--workspace-root",
        type=Path,
        help="Optional parent directory for temporary benchmark workspaces.",
    )
    parser.add_argument(
        "--persistent-root",
        action="store_true",
        help=(
            "Use --workspace-root directly instead of a tempdir under it. "
            "Workspaces persist after the run; requires --workspace-root."
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

    workspace_parent = args.workspace_root.resolve() if args.workspace_root else None
    if args.persistent_root and workspace_parent is None:
        LOGGER.error("--persistent-root requires --workspace-root")
        return 2
    if workspace_parent is not None:
        workspace_parent.mkdir(parents=True, exist_ok=True)

    if args.persistent_root:
        runner_kwargs.setdefault("persist", False)

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
    if args.persistent_root:
        workspace_root_cm = contextlib.nullcontext(workspace_parent)
    else:
        workspace_root_cm = tempfile.TemporaryDirectory(
            prefix="alpha-bench-", dir=workspace_parent
        )

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
            else ["python", "-m", "alpha_lab.benchmarks.run_benchmarks", *argv]
        )
        _write_run_manifest(
            _run_manifest_root(runner),
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


def _run_manifest_root(runner: Any) -> Path:
    if not hasattr(runner, "output_root"):
        raise AttributeError("Runner must expose output_root so run metadata is persistent.")
    return Path(runner.output_root)


if __name__ == "__main__":
    raise SystemExit(main())
