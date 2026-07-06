"""Benchmark workspace runners."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from alpha_lab.benchmarks.manifest import update_manifest_run
from alpha_lab.benchmarks.paths import find_repo_root
from alpha_lab.benchmarks.validators import (
    Check,
    ValidationResult,
    validate_workspace,
)

LOGGER = logging.getLogger(__name__)


class LocalRunner:
    """Run each benchmark workspace through the local Alpha Lab runner script."""

    def __init__(
        self,
        *,
        script: str | Path | None = None,
        prepare_only: bool = False,
        cwd: str | Path | None = None,
        python: str = sys.executable,
    ) -> None:
        self.script = Path(script).resolve() if script is not None else find_repo_root() / "run.py"
        self.prepare_only = prepare_only
        self.cwd = Path(cwd).resolve() if cwd is not None else find_repo_root()
        self.python = python

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
            LOGGER.info("[prepared] %s", workspace)
            return 0

        update_manifest_run(
            workspace,
            {"status": "running", "command": command, "exit_code": None},
        )
        LOGGER.info("[running] %s", workspace)
        proc = subprocess.run(command, cwd=self.cwd)

        # Validate workspace artifacts. A clean exit code is necessary but not
        # sufficient — the validator catches silent regressions (skipped phases,
        # empty learnings.md, no completed experiments, [TOOL ERROR] in logs).
        #
        # The validator does filesystem IO and JSON/sqlite parsing on artifacts
        # the pipeline just wrote; if anything in there is truncated or has
        # unexpected permissions, a raise here would leave the manifest stuck
        # in ``running`` and lose all the partial validation detail. Treat any
        # validator exception as a validation failure (with the exception text
        # in the manifest) so the run is recorded and surfaced cleanly.
        try:
            validation = validate_workspace(workspace)
        except Exception as exc:  # noqa: BLE001 — surface, don't crash
            LOGGER.exception("validate_workspace raised in %s", workspace)
            validation = ValidationResult(
                ok=False,
                checks=[Check(
                    name="validator did not raise",
                    ok=False,
                    detail=f"{type(exc).__name__}: {exc}",
                )],
            )
        ok = proc.returncode == 0 and validation.ok
        status = "completed" if ok else "failed"
        update_manifest_run(
            workspace,
            {
                "status": status,
                "command": command,
                "exit_code": proc.returncode,
                "validation": validation.to_dict(),
            },
        )
        LOGGER.info(
            "[%s] %s: exit_code=%s validation_ok=%s",
            status, workspace, proc.returncode, validation.ok,
        )
        if not validation.ok:
            failed = [c.name for c in validation.checks if not c.ok]
            LOGGER.warning("validation failed in %s: %s", workspace, failed)
        # Surface validation failure as a non-zero exit code so callers
        # (e.g. run_benchmarks.main) can't mistake "ran cleanly but skipped a
        # phase" for success.  Preserve the original exit code when possible.
        if proc.returncode != 0:
            return proc.returncode
        return 0 if validation.ok else 1

    def run_many(self, generator: Iterable[Callable[[], Path]], *, num_workers: int = 1) -> list[int]:
        """Run each workspace; return exit codes in input order."""
        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            return list(executor.map(lambda f: self.run(f()), generator))


class MLflowRunner(LocalRunner):
    """LocalRunner that wraps a benchmark suite in one MLflow Suite Run.

    Inherits LocalRunner's bootstrap + parallel-subprocess plumbing. On top:

    1. At ``run_many`` entry, creates a top-level Suite Run in the MLflow
       experiment resolved from ``MLFLOW_EXPERIMENT_NAME`` (or
       ``MLFLOW_EXPERIMENT_ID``). Tags it ``alpha_lab.run_kind = "suite"``.
    2. Injects ``ALPHALAB_MLFLOW=1`` + the experiment env vars into every
       child subprocess, so each child's ``pipeline_run`` lands in the same
       experiment as the Suite Run.
    3. After each child exits, looks up the child's pipeline Run UUID by
       ``trace_info.json`` and sets three tags on it (mlflow.parentRunId,
       alpha_lab.parent_run_id, alpha_lab.parent_run_name) so the pipeline
       Run nests under the Suite Run in the UI.
    4. At suite end, logs aggregated counts and terminates the Suite Run.

    Falls back to ``LocalRunner.run_many`` when ``MLFLOW_TRACKING_URI`` is
    unset (so the runner degrades gracefully if MLflow isn't configured).
    """

    def __init__(
        self,
        *,
        experiment_name: str | None = None,
        suite_name: str | None = None,
        **localrunner_kwargs,
    ) -> None:
        super().__init__(**localrunner_kwargs)
        self.experiment_name = experiment_name or os.environ.get(
            "MLFLOW_EXPERIMENT_NAME", "alpha-lab-benchmarks",
        )
        self.suite_name = suite_name or _default_suite_name(self.experiment_name)

        self._experiment_id: str | None = None
        self._suite_run_uuid: str | None = None
        self._results_lock = threading.Lock()
        self._eval_rows: list[dict] = []

    def run(self, workspace: Path) -> int:
        """LocalRunner.run with MLflow env injection."""
        command = self.command(workspace)
        if self.prepare_only:
            return super().run(workspace)

        update_manifest_run(
            workspace,
            {"status": "running", "command": command, "exit_code": None},
        )
        LOGGER.info("[running] %s", workspace)
        env = os.environ.copy()
        env["ALPHALAB_MLFLOW"] = "1"
        env["MLFLOW_EXPERIMENT_NAME"] = self.experiment_name
        if self._experiment_id:
            env["MLFLOW_EXPERIMENT_ID"] = self._experiment_id
        proc = subprocess.run(command, cwd=self.cwd, env=env)
        status = "completed" if proc.returncode == 0 else "failed"
        update_manifest_run(
            workspace,
            {"status": status, "command": command, "exit_code": proc.returncode},
        )
        LOGGER.info("[%s] %s: exit_code=%s", status, workspace, proc.returncode)
        return proc.returncode

    def run_many(
        self,
        generator: Iterable[Callable[[], Path]],
        *,
        num_workers: int = 1,
    ) -> list[int]:
        """Set up Suite Run, run children, aggregate, terminate."""
        # Lazy imports so non-MLflow installs aren't forced to ship mlflow.
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            LOGGER.warning(
                "MLFLOW_TRACKING_URI not set; MLflowRunner degrades to LocalRunner"
            )
            return super().run_many(generator, num_workers=num_workers)

        os.environ["ALPHALAB_MLFLOW"] = "1"

        try:
            from alpha_lab import mlflow_logger
            mlflow_logger.configure_sdk()
            import mlflow as _mlflow
        except ImportError as e:
            LOGGER.warning(
                "MLflow SDK not available (%s); degrading to LocalRunner", e,
            )
            return super().run_many(generator, num_workers=num_workers)

        client = _mlflow.MlflowClient()
        # Resolve experiment id (creating by name if needed).
        try:
            exp = client.get_experiment_by_name(self.experiment_name)
            if exp is None:
                self._experiment_id = client.create_experiment(self.experiment_name)
            else:
                self._experiment_id = exp.experiment_id
        except Exception as e:
            LOGGER.warning(
                "Could not resolve MLflow experiment %r: %s; degrading to LocalRunner",
                self.experiment_name, e,
            )
            return super().run_many(generator, num_workers=num_workers)

        # Create the Suite Run.
        try:
            user = os.getenv("USER", "unknown")
            run = client.create_run(
                experiment_id=self._experiment_id,
                run_name=self.suite_name,
                tags={
                    "alpha_lab.suite": self.suite_name,
                    "alpha_lab.run_kind": "suite",
                    "mlflow.user": user,
                },
            )
            self._suite_run_uuid = run.info.run_id
        except Exception as e:
            LOGGER.warning("Could not create Suite Run: %s; degrading to LocalRunner", e)
            return super().run_many(generator, num_workers=num_workers)

        LOGGER.info(
            "Suite Run: %s (experiment_id=%s, run_id=%s)",
            self.suite_name, self._experiment_id, self._suite_run_uuid,
        )

        def _run_one(factory: Callable[[], Path]) -> int:
            workspace = factory()
            exit_code = self.run(workspace)
            try:
                self._on_workspace_done(workspace, exit_code, client)
            except Exception as e:
                LOGGER.warning("on_workspace_done failed for %s: %s", workspace, e)
            return exit_code

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                exit_codes = list(executor.map(_run_one, generator))
        finally:
            try:
                self._aggregate_to_suite_run(client)
            except Exception as e:
                LOGGER.warning("Suite-level aggregation failed: %s", e)
            try:
                client.set_terminated(self._suite_run_uuid, status="FINISHED")
            except Exception as e:
                LOGGER.warning("Suite Run termination failed: %s", e)

        return exit_codes

    def _on_workspace_done(
        self, workspace: Path, exit_code: int, client: object,
    ) -> None:
        """Re-parent the child pipeline Run under the Suite Run via tags."""
        pipeline_uuid = self._resolve_pipeline_run_uuid(workspace, client)
        if pipeline_uuid and self._suite_run_uuid:
            for key, value in (
                ("mlflow.parentRunId", self._suite_run_uuid),
                ("alpha_lab.parent_run_id", self._suite_run_uuid),
                ("alpha_lab.parent_run_name", self.suite_name),
            ):
                try:
                    client.set_tag(pipeline_uuid, key, value)
                except Exception as e:
                    LOGGER.debug(
                        "set_tag %s on %s failed: %s", key, pipeline_uuid, e,
                    )
            try:
                update_manifest_run(
                    workspace,
                    {
                        "status": "completed" if exit_code == 0 else "failed",
                        "exit_code": exit_code,
                        "pipeline_mlflow_run_uuid": pipeline_uuid,
                    },
                )
            except Exception as e:
                LOGGER.debug("Manifest update for %s failed: %s", workspace, e)

        with self._results_lock:
            self._eval_rows.append({
                "benchmark_name": workspace.name,
                "pipeline_run_uuid": pipeline_uuid,
                "exit_code": exit_code,
            })

    def _resolve_pipeline_run_uuid(self, workspace: Path, client: object) -> str | None:
        """Map a workspace path → its alpha-lab pipeline Run UUID in MLflow."""
        trace_info_path = workspace / "trace_info.json"
        if not trace_info_path.is_file():
            LOGGER.debug("No trace_info.json in %s", workspace)
            return None
        try:
            ti = json.loads(trace_info_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            LOGGER.warning("Failed to read %s: %s", trace_info_path, e)
            return None
        run_id = ti.get("run_id")
        if not run_id:
            return None
        try:
            runs = client.search_runs(
                experiment_ids=[self._experiment_id],
                filter_string=f"attributes.run_name = '{run_id}'",
                max_results=1,
            )
        except Exception as e:
            LOGGER.warning("search_runs for %r failed: %s", run_id, e)
            return None
        if not runs:
            return None
        return runs[0].info.run_id

    def _aggregate_to_suite_run(self, client: object) -> None:
        with self._results_lock:
            rows = list(self._eval_rows)
        if not rows or not self._suite_run_uuid:
            return
        metrics = {
            "alpha_lab.suite.benchmarks_total": float(len(rows)),
            "alpha_lab.suite.benchmarks_completed": float(
                sum(1 for r in rows if r["exit_code"] == 0)
            ),
        }
        for k, v in metrics.items():
            try:
                client.log_metric(self._suite_run_uuid, k, v)
            except Exception as e:
                LOGGER.debug("log_metric %s on suite run failed: %s", k, e)
        LOGGER.info("Suite aggregates: %s", metrics)


def _default_suite_name(experiment_name: str) -> str:
    """Auto suite name: ``<suite>-<YYYYMMDD-HHMMSS>`` (UTC)."""
    from datetime import datetime, timezone

    suite = experiment_name.rstrip("/").rsplit("/", 1)[-1] or "alpha-lab-bench"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{suite}-{timestamp}"
