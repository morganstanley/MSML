"""On-prem authentication for alpha-lab LLM clients.

Provides a 3-tier bearer-token fallback, on-prem detection, and the MS CA
bundle used for TLS pinning:

  1. Live SCV/PingFed (needs Kerberos credentials entitled to read the SCV signing key)
  2. Cached token (run ``scripts/auth_setup.sh``, then run the pipeline as any account)
  3. Off-prem (set USE_OFFPREM=1 and OPENAI_API_KEY)
"""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from pathlib import Path

logger = logging.getLogger("alpha_lab.providers.utils.auth")

# ---------------------------------------------------------------------------
# Token cache (written by prefetch_token.py, read here)
# ---------------------------------------------------------------------------
# The shared default lives under /var/tmp, which is host-local: a token cached on
# one machine is invisible on every other one. It is also typically owned by
# whichever account prefetched first, so a second account often cannot write it.
# ``TOKEN_FILEPATH`` overrides the location for exactly those cases; the writer
# (prefetch_token.py) and this reader resolve it through the same helpers so the
# two ends cannot drift apart.
_DEFAULT_TOKEN_CACHE_FILE = Path("/var/tmp/svcacct/.token_cache.json")
_TOKEN_FILEPATH_ENV = "TOKEN_FILEPATH"


def token_cache_candidates() -> list[Path]:
    """Cache locations to consult, most specific first.

    ``TOKEN_FILEPATH`` (when set) comes first so an operator can redirect the
    cache to a path they own; the shared default is still consulted after it, so
    setting the override never hides a token someone else already cached.
    """
    candidates: list[Path] = []
    override = os.environ.get(_TOKEN_FILEPATH_ENV, "").strip()
    if override:
        candidates.append(Path(override).expanduser())
    if _DEFAULT_TOKEN_CACHE_FILE not in candidates:
        candidates.append(_DEFAULT_TOKEN_CACHE_FILE)
    return candidates


def token_cache_path() -> Path:
    """The location a freshly fetched token should be written to."""
    return token_cache_candidates()[0]


def _read_token_cache(path: Path) -> tuple[str, float] | None:
    """Return ``(token, expires_at)`` from ``path``, or None if unusable.

    Unreadable (missing, malformed, or permission-denied) caches are treated the
    same as absent so the caller can fall through to the next candidate.
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except PermissionError:
        logger.warning(
            "Token cache %s exists but is not readable by this account "
            "(owner must chmod it, or set %s to a path you own).",
            path,
            _TOKEN_FILEPATH_ENV,
        )
        return None
    except (json.JSONDecodeError, OSError):
        return None

    token = data.get("bearer_token")
    expires_at = data.get("expires_at", 0)
    if not token:
        return None
    return token, expires_at

# Module-level token cache
_cached_token: str | None = None
_token_expires_at: float = 0.0

# Auth constants
_CLIENT_ID = "ops-dev.00000000-0000-0000-0000-000000000000"
_REGISTRATION_ID = "uid:svcacct:svcacct-1"
_ASSERTION_AUDIENCE = "https://auth-dev.example.com"
_SCOPE = "urn:api:ops-dev.00000000-0000-0000-0000-000000000000/.app"


def _get_ca_cert_bundle() -> str:
    if platform.system().casefold() == "windows":
        return "\\\\fileshare\\pki\\root-ca.crt"
    return "/etc/pki/ca-trust/certs/internal-ca-chain.crt"


def _try_cached_token() -> bool:
    """Try to load a valid bearer token from the on-disk cache.

    Returns True (and sets module-level _cached_token / _token_expires_at)
    if a non-expired cached token was found.
    """
    global _cached_token, _token_expires_at
    expired: list[Path] = []
    for path in token_cache_candidates():
        entry = _read_token_cache(path)
        if entry is None:
            continue
        token, expires_at = entry
        if time.time() >= expires_at:
            expired.append(path)
            continue

        _cached_token = token
        _token_expires_at = expires_at
        remaining = int(expires_at - time.time())
        logger.info(
            "Using cached auth token from %s (expires in %dm %ds)",
            path,
            remaining // 60,
            remaining % 60,
        )
        return True

    if expired:
        logger.warning(
            "Cached token expired (%s). Run scripts/auth_setup.sh to refresh.",
            ", ".join(str(p) for p in expired),
        )
    return False


def _try_live_scv() -> bool:
    """Try to get a token via the live SCV/PingFed chain (requires proid Kerberos).

    Returns True and sets _cached_token / _token_expires_at on success.
    """
    global _cached_token, _token_expires_at
    try:
        from .scalar_2_sample_setup import ping_credential, ai_dev_platform_ets_dev_scope
        token_response = ping_credential.get_token(ai_dev_platform_ets_dev_scope)
        _cached_token = token_response.token
        expires_in = getattr(token_response, "expires_in", 3600) or 3600
        _token_expires_at = time.time() + expires_in - 60  # 60s safety buffer
        logger.info("Got live token via SCV/PingFed (expires in %ds)", expires_in)
        return True
    except Exception as exc:
        logger.warning(
            "Live SCV auth failed (%s): %s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return False


def _get_bearer_token() -> str:
    """Get a valid bearer token using 3-tier fallback:

    1. In-memory cache (fastest, already fetched this session)
    2. On-disk cache (.token_cache.json, written by prefetch_token.py)
    3. Live SCV/PingFed chain (requires proid Kerberos)

    Raises RuntimeError if all three fail.
    """
    global _cached_token, _token_expires_at
    now = time.time()

    # Tier 1: in-memory cache
    if _cached_token and now < _token_expires_at:
        return _cached_token

    # Tier 2: on-disk cache
    if _try_cached_token():
        return _cached_token

    # Tier 3: live SCV
    if _try_live_scv():
        return _cached_token

    checked = ", ".join(str(p) for p in token_cache_candidates())
    raise RuntimeError(
        "No valid auth token available.\n"
        f"Checked cache locations: {checked}\n"
        "Either:\n"
        "  1. Prefetch a token: ./scripts/auth_setup.sh (needs Kerberos credentials\n"
        "     entitled to read the SCV signing key; set TOKEN_FILEPATH first if the\n"
        "     default cache directory is owned by another account)\n"
        "  2. Run as the proid, which always carries that entitlement:\n"
        "     suu -tr '<ticket>' <proid> && python run.py ...\n"
        "  3. Use off-prem: export USE_OFFPREM=1 OPENAI_API_KEY=your-key"
    )


def _is_on_prem() -> bool:
    """Check if on-prem libraries are available or a cached token exists."""
    try:
        from ms.directory import LDAPConnection  # noqa: F401
        return True
    except ImportError:
        pass
    # Also count as on-prem if we have a valid cached token
    for path in token_cache_candidates():
        entry = _read_token_cache(path)
        if entry is not None and time.time() < entry[1]:
            return True

    return False


# Exported for backward compatibility — used by server.py, cli.py, run.py
ON_PREM_AVAILABLE: bool = _is_on_prem()
