#!/usr/bin/env python3
"""Prepare isolated, instrumented tiktoken and Gigatoken TTFT checkouts."""

from __future__ import annotations

import argparse
import importlib.machinery
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from tools.config import ttft_repos_dir


PATCH_DIR = REPO_DIR / "benchmarks/external/patches"


@dataclass(frozen=True)
class Project:
    name: str
    url: str
    base_commit: str
    checkout_name: str
    patch_name: str
    extension_prefix: str
    probe: str
    rust_toolchain: str | None = None


PROJECTS = (
    Project(
        name="tiktoken",
        url="https://github.com/openai/tiktoken.git",
        base_commit="97e49cbadd500b5cc9dbb51a486f0b42e6701bee",
        checkout_name="tiktoken-ttft",
        patch_name="tiktoken-0.12.0-ttft.patch",
        extension_prefix="tiktoken/_tiktoken",
        probe=(
            "import tiktoken; "
            "enc=tiktoken.get_encoding('r50k_base'); "
            "assert hasattr(enc._core_bpe, "
            "'encode_to_tiktoken_buffer_profiled')"
        ),
    ),
    Project(
        name="gigatoken",
        rust_toolchain="nightly",
        url="https://github.com/marcelroed/gigatoken.git",
        base_commit="34a1599f0c0ae7d7cd0d1c530e6522320158b360",
        checkout_name="gigatoken-ttft",
        patch_name="gigatoken-0.10.0-ttft.patch",
        extension_prefix="gigatoken/gigatoken_rs",
        probe=(
            "import gigatoken; "
            "from gigatoken.gigatoken_rs import BPETokenizer; "
            "assert hasattr(BPETokenizer, 'encode_profiled'); "
            "assert hasattr(BPETokenizer, 'encode_files_profiled')"
        ),
    ),
)


def run(
    *command: str,
    cwd: Path | None = None,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    print(f"+ {shlex.join(command)}", flush=True)
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        env=env,
    )


def clone_at_base(project: Project, root: Path) -> Path:
    target = root / project.checkout_name
    if target.exists():
        if not (target / ".git").is_dir():
            raise RuntimeError(f"{target} exists but is not a Git checkout")
        return target

    with tempfile.TemporaryDirectory(prefix=f".{project.name}-", dir=root) as tmp:
        candidate = Path(tmp) / project.checkout_name
        run("git", "clone", "--filter=blob:none", project.url, str(candidate))
        run("git", "checkout", "--detach", project.base_commit, cwd=candidate)
        candidate.replace(target)
    return target


def apply_profile_patch(project: Project, checkout: Path) -> None:
    head = run("git", "rev-parse", "HEAD", cwd=checkout, capture=True)
    if head.stdout.decode().strip() != project.base_commit:
        raise RuntimeError(
            f"{checkout} is not at required base commit {project.base_commit}"
        )

    staged = subprocess.run(
        ("git", "diff", "--cached", "--quiet"),
        cwd=checkout,
        check=False,
    )
    if staged.returncode != 0:
        raise RuntimeError(f"{checkout} contains staged changes")

    patch_path = PATCH_DIR / project.patch_name
    expected = patch_path.read_bytes()
    current = run(
        "git",
        "diff",
        "--no-color",
        "--binary",
        "--no-ext-diff",
        cwd=checkout,
        capture=True,
    ).stdout
    if not current:
        run("git", "apply", "--check", str(patch_path), cwd=checkout)
        run("git", "apply", str(patch_path), cwd=checkout)
        current = run(
            "git",
            "diff",
            "--no-color",
            "--binary",
            "--no-ext-diff",
            cwd=checkout,
            capture=True,
        ).stdout
    if current != expected:
        raise RuntimeError(
            f"{checkout} has tracked changes other than {patch_path.name}"
        )
    run("git", "diff", "--check", cwd=checkout)


def extract_extension(project: Project, wheel: Path, checkout: Path) -> Path:
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    with zipfile.ZipFile(wheel) as archive:
        matches = [
            name
            for name in archive.namelist()
            if name.startswith(project.extension_prefix)
            and name.endswith(suffixes)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one {project.extension_prefix} extension in {wheel}; "
                f"found {matches}"
            )
        member = matches[0]
        relative = PurePosixPath(member)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe wheel member: {member}")
        destination = checkout.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        for suffix in suffixes:
            for old in destination.parent.glob(
                f"{PurePosixPath(project.extension_prefix).name}*{suffix}"
            ):
                old.unlink()
        destination.write_bytes(archive.read(member))
    return destination


def build_environment(project: Project, checkout: Path) -> dict[str, str]:
    env = os.environ.copy()
    if project.rust_toolchain is None:
        return env

    rustup = shutil.which("rustup")
    if rustup is None:
        cargo_home = Path(os.environ.get("CARGO_HOME", str(Path.home() / ".cargo"))).expanduser()
        candidate = cargo_home / "bin" / ("rustup.exe" if os.name == "nt" else "rustup")
        if candidate.is_file() and os.access(candidate, os.X_OK):
            rustup = str(candidate)
    if rustup is None:
        raise RuntimeError(
            f"{project.name} requires Rust {project.rust_toolchain}; "
            "install rustup and make it available on PATH"
        )
    try:
        result = run(
            rustup, "which", "--toolchain", project.rust_toolchain, "cargo",
            cwd=checkout, capture=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"{project.name} requires Rust {project.rust_toolchain}; run "
            f"`rustup toolchain install {project.rust_toolchain}` and retry"
        ) from error
    # Conda can put standalone stable cargo/rustc binaries ahead of rustup.
    # Select the actual toolchain binaries, not just RUSTUP_TOOLCHAIN.
    toolchain_bin = Path(result.stdout.decode().strip()).parent
    env["PATH"] = str(toolchain_bin) + os.pathsep + env.get("PATH", "")
    env["RUSTUP_TOOLCHAIN"] = project.rust_toolchain
    return env


def build_in_place(project: Project, checkout: Path) -> Path:
    build_env = build_environment(project, checkout)
    with tempfile.TemporaryDirectory(prefix=f"{project.name}-wheel-") as tmp:
        wheel_dir = Path(tmp)
        run(
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            str(checkout),
            cwd=checkout,
            env=build_env,
        )
        wheels = list(wheel_dir.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one wheel for {project.name}: {wheels}")
        extension = extract_extension(project, wheels[0], checkout)

    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(checkout)
        if not old_pythonpath
        else str(checkout) + os.pathsep + old_pythonpath
    )
    run(sys.executable, "-c", project.probe, env=env)
    return extension


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clone the exact upstream tokenizer releases, apply Hiriluk's "
            "opt-in TTFT patches, and build isolated native extensions."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        help=(
            "checkout directory (default: TTFT_REPOS_DIR from the environment "
            "or paths.env)"
        ),
    )
    return parser.parse_args()


def resolve_root(override: Path | None) -> Path:
    configured = override if override is not None else Path(ttft_repos_dir())
    root = configured.expanduser().resolve()
    if root == REPO_DIR or REPO_DIR in root.parents:
        raise RuntimeError(
            f"TTFT repository root must be outside Hiriluk's Cargo workspace: {root}"
        )
    return root


def main() -> None:
    args = parse_args()
    root = resolve_root(args.root)
    # Check prerequisites before spending time building either reference.
    for project in PROJECTS:
        build_environment(project, REPO_DIR)
    root.mkdir(parents=True, exist_ok=True)

    checkouts: dict[str, Path] = {}
    for project in PROJECTS:
        print(f"\n=== {project.name} TTFT profiler ===")
        checkout = clone_at_base(project, root)
        apply_profile_patch(project, checkout)
        extension = build_in_place(project, checkout)
        checkouts[project.name] = checkout
        print(f"built {extension}")

    print("\nTTFT profilers are ready.")
    print(f"TIKTOKEN_TTFT_REPO={checkouts['tiktoken']}")
    print(f"GIGATOKEN_TTFT_REPO={checkouts['gigatoken']}")
    print("Inspect the instrumentation with:")
    print(f"  git -C {shlex.quote(str(checkouts['tiktoken']))} diff")
    print(f"  git -C {shlex.quote(str(checkouts['gigatoken']))} diff")


if __name__ == "__main__":
    main()
