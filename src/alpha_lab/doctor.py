"""Lightweight environment diagnostics for Alpha Lab.

The module avoids importing the main application stack at startup. The MLflow SDK
is loaded only when the live logging check runs.
"""

from __future__ import annotations


import importlib.metadata
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import click

_MIN_PYTHON = (3, 11)
_EXPECTED_MLFLOW_VERSION = "3.14.0"
_MLFLOW_REQUIRED_ENV = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_CLIENT_CERT_PATH",
    "MLFLOW_TRACKING_SERVER_CERT_PATH",
)
_CORE_REQUIREMENT = re.compile(
    r"^\s*pydantic[-_]core\s*==\s*([^;,\s]+)", re.IGNORECASE
)


@dataclass(frozen=True)
class CheckResult:
    """One diagnostic result."""

    name: str
    status: str  # pass | warn | fail
    detail: str


def check_python() -> CheckResult:
    """Verify the running interpreter meets the project minimum."""
    current = sys.version_info[:3]
    detail = f"Python {'.'.join(map(str, current))}"
    if current < _MIN_PYTHON:
        return CheckResult(
            "python",
            "fail",
            f"{detail}; Alpha Lab requires Python {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}+",
        )
    return CheckResult("python", "pass", detail)


def _required_pydantic_core(requirements: Sequence[str] | None) -> str | None:
    for requirement in requirements or ():
        match = _CORE_REQUIREMENT.match(requirement)
        if match:
            return match.group(1)
    return None


def check_pydantic() -> CheckResult:
    """Check Pydantic's installed core against its declared exact requirement.

    Metadata inspection avoids importing Pydantic, whose import itself fails when
    the two distributions are incompatible.
    """
    try:
        pydantic_version = importlib.metadata.version("pydantic")
        core_version = importlib.metadata.version("pydantic-core")
        required_core = _required_pydantic_core(
            importlib.metadata.requires("pydantic")
        )
    except importlib.metadata.PackageNotFoundError as error:
        return CheckResult(
            "pydantic",
            "fail",
            f"Missing required distribution: {error.name}",
        )

    if required_core is None:
        return CheckResult(
            "pydantic",
            "warn",
            f"pydantic {pydantic_version}; could not determine its core requirement",
        )
    if core_version != required_core:
        return CheckResult(
            "pydantic",
            "fail",
            f"pydantic {pydantic_version} requires pydantic-core {required_core}, "
            f"but {core_version} is installed",
        )
    return CheckResult(
        "pydantic",
        "pass",
        f"pydantic {pydantic_version} with pydantic-core {core_version}",
    )


def _mlflow_experiment_name(workspace: str | None) -> str | None:
    if name := os.getenv("MLFLOW_EXPERIMENT_NAME"):
        return name
    if workspace:
        try:
            return Path(workspace).expanduser().resolve().name or "workspace"
        except (OSError, RuntimeError):
            return None
    return None


def check_mlflow_config(workspace: str | None) -> CheckResult:
    """Verify the SDK, endpoint, mTLS files, and experiment selection."""
    try:
        version = importlib.metadata.version("mlflow")
    except importlib.metadata.PackageNotFoundError:
        return CheckResult("mlflow_config", "fail", "mlflow is not installed")
    if version != _EXPECTED_MLFLOW_VERSION:
        return CheckResult(
            "mlflow_config",
            "fail",
            f"mlflow {version} is installed; Alpha Lab pins {_EXPECTED_MLFLOW_VERSION}",
        )

    missing = [name for name in _MLFLOW_REQUIRED_ENV if not os.getenv(name)]
    if missing:
        return CheckResult(
            "mlflow_config",
            "fail",
            "missing environment variables: " + ", ".join(missing),
        )

    unreadable = []
    for name in _MLFLOW_REQUIRED_ENV[1:]:
        path = Path(os.environ[name]).expanduser()
        if not path.is_file() or not os.access(path, os.R_OK):
            unreadable.append(name)
    if unreadable:
        return CheckResult(
            "mlflow_config",
            "fail",
            "certificate files are missing or unreadable: " + ", ".join(unreadable),
        )

    experiment = os.getenv("MLFLOW_EXPERIMENT_ID") or _mlflow_experiment_name(workspace)
    if not experiment:
        return CheckResult(
            "mlflow_config",
            "fail",
            "set MLFLOW_EXPERIMENT_NAME or MLFLOW_EXPERIMENT_ID, or pass --workspace",
        )
    return CheckResult(
        "mlflow_config",
        "pass",
        f"mlflow {version}; tracking URI, mTLS files, and experiment are configured",
    )


def check_mlflow_logging(workspace: str | None) -> CheckResult:
    """Write, read back, and delete a temporary metric through MLflow."""
    try:
        import mlflow

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        if set_workspace := getattr(mlflow, "set_workspace", None):
            set_workspace(os.getenv("USER") or None)
        client = mlflow.MlflowClient()

        if experiment_id := os.getenv("MLFLOW_EXPERIMENT_ID"):
            experiment = client.get_experiment(experiment_id)
            if experiment is None:
                return CheckResult(
                    "mlflow_logging",
                    "fail",
                    f"MLflow experiment ID {experiment_id!r} was not found",
                )
        else:
            experiment_name = _mlflow_experiment_name(workspace)
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = client.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
    except Exception as error:
        return CheckResult(
            "mlflow_logging",
            "fail",
            f"MLflow setup or connection failed: {error}",
        )

    run_id: str | None = None
    cleanup_errors: list[str] = []
    try:
        run_name = f"alpha-lab-doctor-{uuid.uuid4().hex[:12]}"
        run = client.create_run(
            experiment_id=experiment_id,
            tags={"mlflow.runName": run_name, "alpha_lab.run_kind": "doctor"},
        )
        run_id = run.info.run_id
        client.log_metric(run_id, "alpha_lab.doctor", 1.0)
        recorded = client.get_run(run_id).data.metrics.get("alpha_lab.doctor")
        if recorded != 1.0:
            return CheckResult(
                "mlflow_logging",
                "fail",
                f"temporary metric read-back returned {recorded!r} instead of 1.0",
            )
    except Exception as error:
        return CheckResult(
            "mlflow_logging", "fail", f"temporary metric logging failed: {error}"
        )
    finally:
        if run_id is not None:
            try:
                client.set_terminated(run_id, status="FINISHED")
            except Exception as error:
                cleanup_errors.append(f"terminate: {error}")
            try:
                client.delete_run(run_id)
            except Exception as error:
                cleanup_errors.append(f"delete: {error}")

    if cleanup_errors:
        return CheckResult(
            "mlflow_logging",
            "warn",
            "metric logging passed, but temporary run cleanup failed: "
            + "; ".join(cleanup_errors),
        )
    return CheckResult(
        "mlflow_logging",
        "pass",
        "temporary metric was logged, read back, and its run was deleted",
    )


def check_fts5() -> CheckResult:
    """Verify that the stdlib SQLite build supports FTS5."""
    try:
        with sqlite3.connect(":memory:") as connection:
            connection.execute(
                "CREATE VIRTUAL TABLE doctor_fts USING fts5(content)"
            )
    except sqlite3.Error as error:
        return CheckResult("sqlite_fts5", "fail", f"FTS5 unavailable: {error}")
    return CheckResult(
        "sqlite_fts5", "pass", f"SQLite {sqlite3.sqlite_version} with FTS5"
    )


def check_executable(name: str, *, required: bool) -> CheckResult:
    """Check whether an external executable is available on PATH."""
    path = shutil.which(name)
    if path:
        return CheckResult(name, "pass", path)
    status = "fail" if required else "warn"
    suffix = "required" if required else "optional; agents will run in-process"
    return CheckResult(name, status, f"not found on PATH ({suffix})")


def check_workspace(workspace: str) -> CheckResult:
    """Verify that a workspace, or its nearest existing parent, is writable."""
    try:
        requested = Path(workspace).expanduser().resolve()
    except (OSError, RuntimeError) as error:
        return CheckResult("workspace", "fail", f"invalid path: {error}")
    target = requested
    while not target.exists() and target != target.parent:
        target = target.parent
    if not target.is_dir():
        return CheckResult(
            "workspace", "fail", f"nearest existing path is not a directory: {target}"
        )
    try:
        fd, probe = tempfile.mkstemp(prefix=".alpha_lab_doctor_", dir=target)
        os.close(fd)
        Path(probe).unlink(missing_ok=True)
    except OSError as error:
        return CheckResult("workspace", "fail", f"not writable: {target}: {error}")
    if requested.exists() and not requested.is_dir():
        return CheckResult("workspace", "fail", f"not a directory: {requested}")
    detail = f"writable: {requested if requested.exists() else target}"
    if not requested.exists():
        detail += f" (parent for requested workspace {requested})"
    return CheckResult("workspace", "pass", detail)


def run_checks(
    workspace: str | None = None, *, skip_network: bool = False
) -> list[CheckResult]:
    """Run the core diagnostic set."""
    mlflow_config = check_mlflow_config(workspace)
    results = [
        check_python(),
        check_pydantic(),
        check_fts5(),
        check_executable("git", required=True),
        check_executable("bwrap", required=False),
        mlflow_config,
    ]
    if mlflow_config.status == "fail":
        results.append(
            CheckResult(
                "mlflow_logging", "fail", "not attempted because configuration failed"
            )
        )
    elif skip_network:
        results.append(
            CheckResult("mlflow_logging", "warn", "skipped by --skip-network")
        )
    else:
        results.append(check_mlflow_logging(workspace))
    if workspace is not None:
        results.append(check_workspace(workspace))
    return results


@click.command()
@click.option(
    "--workspace",
    type=str,
    help="Optional workspace path whose writability should be checked.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit machine-readable JSON.",
)
@click.option(
    "--skip-network",
    is_flag=True,
    help="Validate MLflow configuration without writing a temporary metric.",
)
def main(workspace: str | None, json_output: bool, skip_network: bool) -> None:
    """Check core Alpha Lab runtime prerequisites."""
    results = run_checks(workspace, skip_network=skip_network)
    if json_output:
        click.echo(json.dumps([asdict(result) for result in results], indent=2))
    else:
        for result in results:
            click.echo(f"[{result.status.upper()}] {result.name}: {result.detail}")

    if any(result.status == "fail" for result in results):
        raise click.exceptions.Exit(1)


if __name__ == "__main__":
    main()
