"""Filesystem defaults and git introspection for benchmark tooling."""

import functools
import subprocess
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
# Default location for registered benchmark suites. Override as needed.
DEFAULT_SUITE_DIR = Path("/path/to/benchmark-suites")


@functools.lru_cache(maxsize=1)
def find_repo_root() -> Path:
    """Find the source checkout root when running from an editable tree."""
    for parent in (PACKAGE_DIR, *PACKAGE_DIR.parents):
        if (parent / "pyproject.toml").exists() and (parent / "run.py").exists():
            return parent
    return Path.cwd()


def git_commit() -> str | None:
    """Return ``git rev-parse HEAD`` for the repo root, or ``None`` if unavailable."""
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=find_repo_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()
