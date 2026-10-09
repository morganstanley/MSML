from __future__ import annotations

from pathlib import Path
import os
import subprocess

import pytest

from tools import setup_ttft_repos


def test_gigatoken_build_prefers_nightly_over_conda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/conda/bin")
    monkeypatch.setenv("RUSTUP_TOOLCHAIN", "stable")
    monkeypatch.setattr(setup_ttft_repos.shutil, "which", lambda _: "/rustup")

    def resolve(*command, **kwargs):
        assert command == ("/rustup", "which", "--toolchain", "nightly", "cargo")
        assert kwargs == {"cwd": tmp_path, "capture": True}
        return subprocess.CompletedProcess(command, 0, b"/nightly/bin/cargo\n")

    monkeypatch.setattr(setup_ttft_repos, "run", resolve)
    env = setup_ttft_repos.build_environment(setup_ttft_repos.PROJECTS[1], tmp_path)
    assert env["PATH"] == "/nightly/bin" + os.pathsep + "/conda/bin"
    assert env["RUSTUP_TOOLCHAIN"] == "nightly"
    assert os.environ["RUSTUP_TOOLCHAIN"] == "stable"


def test_tiktoken_keeps_existing_build_environment(tmp_path: Path) -> None:
    assert setup_ttft_repos.build_environment(
        setup_ttft_repos.PROJECTS[0], tmp_path,
    ) == dict(os.environ)


@pytest.mark.parametrize("custom_home", [False, True])
def test_rustup_is_found_outside_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, custom_home: bool,
) -> None:
    monkeypatch.setattr(setup_ttft_repos.shutil, "which", lambda _: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cargo_home = tmp_path / ("custom-cargo" if custom_home else ".cargo")
    if custom_home:
        monkeypatch.setenv("CARGO_HOME", str(cargo_home))
    else:
        monkeypatch.delenv("CARGO_HOME", raising=False)
    executable = cargo_home / "bin" / ("rustup.exe" if os.name == "nt" else "rustup")
    executable.parent.mkdir(parents=True)
    executable.touch()
    executable.chmod(0o755)

    def resolve(*command, **kwargs):
        assert command[0] == str(executable)
        return subprocess.CompletedProcess(command, 0, b"/nightly/bin/cargo\n")

    monkeypatch.setattr(setup_ttft_repos, "run", resolve)
    env = setup_ttft_repos.build_environment(setup_ttft_repos.PROJECTS[1], tmp_path)
    assert env["RUSTUP_TOOLCHAIN"] == "nightly"


def test_missing_nightly_has_install_instruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(setup_ttft_repos.shutil, "which", lambda _: "/rustup")

    def missing(*command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(setup_ttft_repos, "run", missing)
    with pytest.raises(RuntimeError, match="rustup toolchain install nightly"):
        setup_ttft_repos.build_environment(setup_ttft_repos.PROJECTS[1], tmp_path)


def test_explicit_ttft_root_does_not_require_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_configuration() -> str:
        raise AssertionError("explicit --root should take precedence")

    monkeypatch.setattr(
        setup_ttft_repos,
        "ttft_repos_dir",
        missing_configuration,
    )
    assert setup_ttft_repos.resolve_root(tmp_path) == tmp_path.resolve()


def test_configured_ttft_root_is_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        setup_ttft_repos,
        "ttft_repos_dir",
        lambda: str(tmp_path),
    )
    assert setup_ttft_repos.resolve_root(None) == tmp_path.resolve()


@pytest.mark.parametrize(
    "relative",
    (Path(), Path("build/ttft")),
)
def test_ttft_root_must_be_outside_hiriluk(
    relative: Path,
) -> None:
    with pytest.raises(RuntimeError, match="outside Hiriluk's Cargo workspace"):
        setup_ttft_repos.resolve_root(setup_ttft_repos.REPO_DIR / relative)
