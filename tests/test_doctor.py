"""Tests for the lightweight Alpha Lab environment doctor."""

from __future__ import annotations

import importlib.metadata
import json
import sys
from types import SimpleNamespace

from click.testing import CliRunner

from alpha_lab import doctor


def test_required_pydantic_core_parses_hyphen_and_marker() -> None:
    requirements = [
        "annotated-types>=0.6",
        'pydantic-core==2.46.4; python_version >= "3.9"',
    ]
    assert doctor._required_pydantic_core(requirements) == "2.46.4"


def test_check_pydantic_reports_compatible_pair(monkeypatch) -> None:
    versions = {"pydantic": "2.13.4", "pydantic-core": "2.46.4"}
    monkeypatch.setattr(
        doctor.importlib.metadata,
        "version",
        lambda name: versions[name],
    )
    monkeypatch.setattr(
        doctor.importlib.metadata,
        "requires",
        lambda _name: ["pydantic_core==2.46.4"],
    )

    result = doctor.check_pydantic()

    assert result.status == "pass"
    assert "2.46.4" in result.detail


def test_check_pydantic_reports_incompatible_pair(monkeypatch) -> None:
    versions = {"pydantic": "2.13.4", "pydantic-core": "2.47.0"}
    monkeypatch.setattr(
        doctor.importlib.metadata,
        "version",
        lambda name: versions[name],
    )
    monkeypatch.setattr(
        doctor.importlib.metadata,
        "requires",
        lambda _name: ["pydantic-core==2.46.4"],
    )

    result = doctor.check_pydantic()

    assert result.status == "fail"
    assert "requires pydantic-core 2.46.4" in result.detail
    assert "2.47.0 is installed" in result.detail


def test_check_pydantic_reports_missing_distribution(monkeypatch) -> None:
    def missing(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError("pydantic-core")

    monkeypatch.setattr(doctor.importlib.metadata, "version", missing)

    result = doctor.check_pydantic()

    assert result.status == "fail"
    assert "Missing required distribution" in result.detail


def test_mlflow_config_reports_missing_environment(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor.importlib.metadata,
        "version",
        lambda name: "3.14.0" if name == "mlflow" else "unused",
    )
    for name in doctor._MLFLOW_REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)

    result = doctor.check_mlflow_config("/tmp/workspace")

    assert result.status == "fail"
    assert "MLFLOW_TRACKING_URI" in result.detail


def test_mlflow_config_accepts_readable_certificates(monkeypatch, tmp_path) -> None:
    client_cert = tmp_path / "client.pem"
    server_cert = tmp_path / "server.pem"
    client_cert.write_text("client")
    server_cert.write_text("server")
    monkeypatch.setattr(
        doctor.importlib.metadata,
        "version",
        lambda name: "3.14.0" if name == "mlflow" else "unused",
    )
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example")
    monkeypatch.setenv("MLFLOW_TRACKING_CLIENT_CERT_PATH", str(client_cert))
    monkeypatch.setenv("MLFLOW_TRACKING_SERVER_CERT_PATH", str(server_cert))
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)

    result = doctor.check_mlflow_config(str(tmp_path / "my-workspace"))

    assert result.status == "pass"
    assert "mTLS files" in result.detail


def test_mlflow_logging_writes_reads_and_deletes_metric(monkeypatch) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.deleted: list[str] = []
            self.terminated: list[str] = []
            self.metrics: dict[str, float] = {}

        def get_experiment_by_name(self, _name):
            return SimpleNamespace(experiment_id="experiment-1")

        def create_run(self, *, experiment_id, tags):
            assert experiment_id == "experiment-1"
            assert tags["alpha_lab.run_kind"] == "doctor"
            return SimpleNamespace(info=SimpleNamespace(run_id="run-1"))

        def log_metric(self, run_id, key, value):
            assert run_id == "run-1"
            self.metrics[key] = value

        def get_run(self, run_id):
            assert run_id == "run-1"
            return SimpleNamespace(data=SimpleNamespace(metrics=self.metrics))

        def set_terminated(self, run_id, *, status):
            assert status == "FINISHED"
            self.terminated.append(run_id)

        def delete_run(self, run_id):
            self.deleted.append(run_id)

    client = FakeClient()
    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: None,
        set_workspace=lambda user: None,
        MlflowClient=lambda: client,
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "doctor-test")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)

    result = doctor.check_mlflow_logging("/tmp/workspace")

    assert result.status == "pass"
    assert client.metrics["alpha_lab.doctor"] == 1.0
    assert client.terminated == ["run-1"]
    assert client.deleted == ["run-1"]


def test_check_fts5_passes_with_runtime_sqlite() -> None:
    assert doctor.check_fts5().status == "pass"


def test_executable_required_and_optional(monkeypatch) -> None:
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: None)

    assert doctor.check_executable("git", required=True).status == "fail"
    assert doctor.check_executable("bwrap", required=False).status == "warn"


def test_workspace_existing_and_missing_parent(tmp_path) -> None:
    assert doctor.check_workspace(str(tmp_path)).status == "pass"
    assert doctor.check_workspace(str(tmp_path / "new" / "workspace")).status == "pass"


def test_workspace_file_fails(tmp_path) -> None:
    path = tmp_path / "not-a-directory"
    path.write_text("x")

    result = doctor.check_workspace(str(path))

    assert result.status == "fail"
    assert "not a directory" in result.detail


def test_workspace_path_resolution_failure_is_reported(monkeypatch) -> None:
    def fail_expanduser(_path):
        raise RuntimeError("unknown user")

    monkeypatch.setattr(doctor.Path, "expanduser", fail_expanduser)

    result = doctor.check_workspace("~unknownuser/workspace")

    assert result.status == "fail"
    assert "invalid path" in result.detail
    assert "unknown user" in result.detail


def test_main_json_returns_failure_for_required_check(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor,
        "run_checks",
        lambda _workspace=None, **_kwargs: [
            doctor.CheckResult("python", "pass", "ok"),
            doctor.CheckResult("git", "fail", "missing"),
        ],
    )

    result = CliRunner().invoke(doctor.main, ["--json"])
    payload = json.loads(result.output)

    assert result.exit_code == 1
    assert payload[1]["name"] == "git"
    assert payload[1]["status"] == "fail"


def test_main_text_returns_success_with_warning(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor,
        "run_checks",
        lambda _workspace=None, **_kwargs: [
            doctor.CheckResult("python", "pass", "ok"),
            doctor.CheckResult("bwrap", "warn", "optional"),
        ],
    )

    result = CliRunner().invoke(doctor.main)

    assert result.exit_code == 0
    assert "[PASS] python" in result.output
    assert "[WARN] bwrap" in result.output
