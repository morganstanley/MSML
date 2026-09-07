"""Regression coverage for live on-prem authentication in agent sandboxes."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import types
from pathlib import Path

import pytest

from alpha_lab.providers.utils import auth
from alpha_lab.sandboxing import sandbox


def test_live_scv_failure_retains_traceback(monkeypatch, caplog) -> None:
    """A failed live refresh must expose its real cause, not just return ``False``."""

    class FailingCredential:
        @staticmethod
        def get_token(_scope: str) -> None:
            raise FileNotFoundError("missing test CA bundle")

    setup_module = types.ModuleType(f"{auth.__package__}.scalar_2_sample_setup")
    setup_module.ping_credential = FailingCredential()
    setup_module.ai_dev_platform_ets_dev_scope = "test-scope"
    monkeypatch.setitem(sys.modules, setup_module.__name__, setup_module)

    with caplog.at_level(logging.WARNING, logger=auth.logger.name):
        assert auth._try_live_scv() is False

    record = next(
        record for record in caplog.records if record.name == auth.logger.name
    )
    assert record.exc_info is not None
    assert record.exc_info[0] is FileNotFoundError
    assert "missing test CA bundle" in record.getMessage()


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not available")
def test_scv_setup_imports_in_sandbox_without_printing_token() -> None:
    """The SCV setup must load with the sandbox's filesystem view and reveal no token."""

    repo_root = Path(__file__).resolve().parents[1]
    module = "alpha_lab.providers.utils.scalar_2_sample_setup"
    child_env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(repo_root / "src"),
    }

    proc = sandbox._popen(
        [sys.executable, "-c", f"import {module}"],
        ro_paths=[],
        rw_paths=[],
        blocked_paths=[],
        chdir=str(repo_root),
        project_ro_root=str(repo_root),
        needs_gpu=False,
        env=child_env,
    )
    stdout, stderr = proc.communicate(timeout=60)

    assert proc.returncode == 0, stderr
    assert "Token:" not in stdout
    assert "Token:" not in stderr
