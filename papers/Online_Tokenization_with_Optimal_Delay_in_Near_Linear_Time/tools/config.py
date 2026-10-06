"""Configurable root directory for benchmark corpora.

Mirrors `src/benchmark_utils/config.rs` so Rust and Python data tools agree on
where corpora are read and written.

Precedence: an exported environment variable wins; otherwise the value in
`<repo>/paths.env` is used. Configuration fails if neither source defines the
requested key. Edit `paths.env` to repoint both the Rust and Python sides at
once.

`DATA_DIR` is the root for all corpus source and derived files.
`TTFT_REPOS_DIR` holds the instrumented tiktoken and Gigatoken checkouts.
"""

import os
from pathlib import Path

_PATHS_ENV = Path(__file__).resolve().parent.parent / 'paths.env'


def _from_paths_env(key: str) -> str | None:
    """Look `key` up in the repo-root paths.env (KEY=VALUE lines)."""
    try:
        for line in _PATHS_ENV.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            k, sep, v = line.partition('=')
            if sep and k.strip() == key:
                return v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return None


def _configured(key: str) -> str:
    value = os.environ.get(key) or _from_paths_env(key)
    if value is None:
        raise RuntimeError(
            f'{key} is not configured: export it or add it to {_PATHS_ENV}'
        )
    return value


def data_dir() -> str:
    """Root directory for input corpora."""
    return _configured('DATA_DIR').rstrip('/')


def ttft_repos_dir() -> str:
    """Root directory for instrumented external TTFT repositories."""
    return _configured('TTFT_REPOS_DIR').rstrip('/')
