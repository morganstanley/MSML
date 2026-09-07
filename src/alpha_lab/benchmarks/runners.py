"""Benchmark workspace runners."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from alpha_lab.benchmarks.manifest import update_manifest_run
from alpha_lab.benchmarks.paths import find_repo_root


LOGGER = logging.getLogger(__name__)


class LocalRunner:
    """Run each benchmark workspace through the local Alpha Lab runner script."""

    def __init__(
        self,
        *,
        output_root: str | Path,
        script: str | Path | None = None,
        prepare_only: bool = False,
        cwd: str | Path | None = None,
        python: str = sys.executable,
        overwrite: bool = False,
        persist: bool = True,
    ) -> None:
        self.output_root = Path(output_root).resolve()
        self.script = Path(script).resolve() if script is not None else find_repo_root() / "run.py"
        self.prepare_only = prepare_only
        self.cwd = Path(cwd).resolve() if cwd is not None else find_repo_root()
        self.python = python
        self.overwrite = overwrite
        self.persist = persist
        self.output_root.mkdir(parents=True, exist_ok=True)

    def command(self, workspace: Path) -> list[str]:
        return [
            self.python,
            str(self.script),
            "--config",
            str(workspace / "config.json"),
            "--workspace",
            str(workspace),
        ]

    def run(self, workspace: Path) -> int:
        command = self.command(workspace)
        if self.prepare_only:
            update_manifest_run(
                workspace,
                {"status": "prepared", "command": command, "exit_code": None},
            )
            self._persist_workspace(workspace)
            LOGGER.info("[prepared] %s", workspace)
            return 0

        update_manifest_run(
            workspace,
            {"status": "running", "command": command, "exit_code": None},
        )
        LOGGER.info("[running] %s", workspace)
        proc = subprocess.run(command, cwd=self.cwd)
        status = "completed" if proc.returncode == 0 else "failed"
        update_manifest_run(
            workspace,
            {"status": status, "command": command, "exit_code": proc.returncode},
        )
        if proc.returncode == 0:
            self._persist_workspace(workspace)
        LOGGER.info("[%s] %s: exit_code=%s", status, workspace, proc.returncode)
        return proc.returncode

    def run_many(self, workspaces: Iterable[Path], *, num_workers: int = 1) -> list[int]:
        """Run each workspace; return exit codes in input order."""
        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            return list(executor.map(self.run, workspaces))

    def _persist_workspace(self, workspace: Path) -> None:
        if not self.persist:
            return
        destination = self.output_root / workspace.name
        if destination.exists():
            if not self.overwrite:
                raise FileExistsError(
                    f"Benchmark output already exists: {destination}. "
                    "Set runner overwrite=true to replace it."
                )
            shutil.rmtree(destination)
        shutil.copytree(workspace, destination, symlinks=True)
        self._rewrite_config_paths(workspace, destination)

    @staticmethod
    def _rewrite_config_paths(workspace: Path, destination: Path) -> None:
        """Update ``config.json`` data_path after copying to a new location."""
        config_path = destination / "config.json"
        if not config_path.is_file():
            return
        config = json.loads(config_path.read_text())
        old = Path(config.get("data_path", ""))
        if old.is_absolute() and old.is_relative_to(workspace):
            config["data_path"] = str(destination / old.relative_to(workspace))
            config_path.write_text(json.dumps(config, indent=2) + "\n")


class MLflowRunner(LocalRunner):
    """LocalRunner variant that logs each workspace run to MLflow.

    Creates a Suite Run to group all pipeline runs in the benchmark.
    Each workspace is invoked with ``--mlflow --run-id <run_id>``; after it
    completes, the pipeline's MLflow Run is re-parented under the Suite Run.

    Requires: ``ALPHALAB_MLFLOW=1``, ``MLFLOW_TRACKING_URI``, and either
    ``MLFLOW_EXPERIMENT_NAME`` or ``MLFLOW_EXPERIMENT_ID`` in the environment.
    Falls back to plain :class:`LocalRunner` behaviour when MLflow is not active.
    """

    def __init__(
        self,
        *,
        suite_run_name: str | None = None,
        run_id_prefix: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.suite_run_name = suite_run_name
        self.run_id_prefix = run_id_prefix

    def _workspace_run_id(self, workspace: Path) -> str:
        if self.run_id_prefix:
            return f"{self.run_id_prefix}{workspace.name}"
        return workspace.name

    def command(self, workspace: Path) -> list[str]:
        return super().command(workspace) + [
            "--mlflow",
            "--run-id", self._workspace_run_id(workspace),
        ]

    def run_many(self, workspaces: Iterable[Path], *, num_workers: int = 1) -> list[int]:
        from alpha_lab import mlflow_logger as _ml
        if not _ml.is_active():
            LOGGER.warning(
                "MLflowRunner: MLflow not active "
                "(ALPHALAB_MLFLOW / MLFLOW_TRACKING_URI not set); "
                "falling back to LocalRunner."
            )
            return super().run_many(workspaces, num_workers=num_workers)

        _ml.configure_sdk()
        workspace_list = list(workspaces)

        suite_uuid = self._get_or_create_suite_run()
        exit_codes = super().run_many(workspace_list, num_workers=num_workers)

        if suite_uuid:
            for ws in workspace_list:
                try:
                    self._reparent_run(ws, suite_uuid)
                except Exception as e:
                    LOGGER.warning("Failed to reparent %s under suite run: %s", ws.name, e)

        return exit_codes

    def _get_or_create_suite_run(self) -> str | None:
        from alpha_lab.mlflow_logger import _resolve_experiment_id, _get_or_create_run
        from datetime import datetime, timezone
        try:
            import mlflow
        except ImportError:
            return None

        suite_name = self.suite_run_name or (
            f"suite-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
        experiment_id = _resolve_experiment_id()
        if experiment_id is None:
            return None

        info = _get_or_create_run(experiment_id, suite_name)
        if info is None:
            return None

        suite_uuid, _ = info
        try:
            client = mlflow.MlflowClient()
            client.set_tag(suite_uuid, "alpha_lab.run_kind", "suite")
            client.update_run(suite_uuid, status="RUNNING")
        except Exception as e:
            LOGGER.debug("Suite run tag/status update failed: %s", e)
        LOGGER.info("MLflow suite run: %s (%s)", suite_name, suite_uuid)
        return suite_uuid

    def _reparent_run(self, workspace: Path, suite_uuid: str) -> None:
        from alpha_lab.mlflow_logger import TRACE_INFO_FILENAME, _resolve_experiment_id
        try:
            import mlflow
        except ImportError:
            return

        trace_path = workspace / TRACE_INFO_FILENAME
        if not trace_path.is_file():
            LOGGER.debug("No trace_info.json in %s — skipping reparent", workspace)
            return

        info = json.loads(trace_path.read_text())
        run_name = info.get("run_id")
        if not run_name:
            return

        experiment_id = _resolve_experiment_id()
        if not experiment_id:
            return

        client = mlflow.MlflowClient()
        safe_name = run_name.replace("'", "\\'")
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"attributes.run_name = '{safe_name}'",
            max_results=1,
        )
        if not runs:
            LOGGER.debug("No MLflow run found for run_name=%r in workspace %s", run_name, workspace.name)
            return

        pipeline_uuid = runs[0].info.run_id
        client.set_tag(pipeline_uuid, "mlflow.parentRunId", suite_uuid)
        client.set_tag(pipeline_uuid, "alpha_lab.suite_run_id", suite_uuid)
        LOGGER.info("Re-parented %s (%s) → suite %s", run_name, pipeline_uuid, suite_uuid)
