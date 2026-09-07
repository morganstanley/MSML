#!/usr/bin/env python
"""Prefetch and cache a bearer token for Alpha-Lab.

Caches a bearer token that can then be used by any account able to read the
cache file, so the pipeline itself needs no Kerberos credentials.

Fetching requires Kerberos credentials entitled to read the SCV signing key.
The proid always carries that entitlement; a personal account may too, in which
case no ``suu`` is needed.

Usage:
    suu -tr "<ticket>" <proid>    # only if your own account lacks SCV access
    alpha-lab-prefetch-token
    exit
    # Now run the pipeline as your personal account

The cache location defaults to a shared path under /var/tmp (host-local, and
often owned by whoever prefetched first). Set ``TOKEN_FILEPATH`` to redirect it
to a path this account owns.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import httpx
from aidevplatformaccess.core import (
    PingFedAssertionTokenCredential,
    PingFedGateway,
    PingFedTokenManager,
    ScvJsonWebKeyProvider,
)
from aidevplatformaccess.scv import ScvClient
from scvlib import SecureCredentialsVault

from alpha_lab.providers.utils.auth import token_cache_path
from alpha_lab.utils import atomic_write

# Auth constants (same as scalar_2_sample_setup.py)
CLIENT_ID = "ops-dev.00000000-0000-0000-0000-000000000000"
REGISTRATION_ID = "uid:svcacct:svcacct-1"
ASSERTION_AUDIENCE = "https://auth-dev.example.com"
PINGFED_URL = "https://auth-dev.example.com/as/token.oauth2"
SCV_NAMESPACE = "AUTH/PINGFEDERATE-OIDC/OPS-DEV"
SCV_KEY = f"oidc/ops-dev/{REGISTRATION_ID}/credentials"
SCV_URL = "https://credential-vault.example.com"
SCOPE = "urn:api:ops-dev.00000000-0000-0000-0000-000000000000/.app"


def _get_ca_cert_bundle() -> str:
    import platform
    if platform.system().casefold() == "windows":
        return "\\\\fileshare\\pki\\root-ca.crt"
    return "/etc/pki/ca-trust/certs/internal-ca-chain.crt"


def _fetch_token() -> tuple[str, int]:
    """Run the full SCV -> PingFed chain and return (token, expires_in)."""
    ca_cert = _get_ca_cert_bundle()
    pingfed_http_client = httpx.Client(verify=ca_cert)
    pingfed_gateway = PingFedGateway(PINGFED_URL, pingfed_http_client, None)

    scv = SecureCredentialsVault(SCV_URL)
    scv_client = ScvClient(scv)
    key_provider = ScvJsonWebKeyProvider(scv_client, SCV_NAMESPACE, SCV_KEY)
    token_manager = PingFedTokenManager(key_provider, pingfed_gateway)

    credential = PingFedAssertionTokenCredential(
        token_manager, CLIENT_ID, ASSERTION_AUDIENCE
    )

    token_response = credential.get_token(SCOPE)
    token = token_response.token
    expires_in = getattr(token_response, "expires_in", 3600) or 3600
    return token, expires_in


def main() -> None:
    print("Fetching bearer token via SCV/PingFed ...")
    try:
        token, expires_in = _fetch_token()
    except Exception as exc:
        print(f"Failed to fetch token: {exc}", file=sys.stderr)
        print(
            "Make sure you are running as the proid (suu -tr \"<ticket>\" <proid>).",
            file=sys.stderr,
        )
        sys.exit(1)

    now = time.time()
    expires_at = now + expires_in

    cache = {
        "bearer_token": token,
        "expires_at": expires_at,
        "scope": SCOPE,
        "fetched_by": os.getenv("USER", "unknown"),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }

    cache_file = token_cache_path()

    # atomic_write needs the parent dir to exist; create it (fresh host may lack it).
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    # The dir may have been created by another account under a restrictive umask,
    # in which case neither the write below nor a later read would work. Widening
    # it is best-effort: it fails when we don't own it, which the write reports.
    try:
        cache_file.parent.chmod(0o755)
    except OSError:
        pass

    try:
        atomic_write(cache_file, json.dumps(cache, indent=2) + "\n")
    except PermissionError as exc:
        owner = "another account"
        try:
            import pwd
            owner = pwd.getpwuid(cache_file.parent.stat().st_uid).pw_name
        except (KeyError, OSError, ImportError):
            pass
        print(f"Cannot write the token cache: {exc}", file=sys.stderr)
        print(
            f"{cache_file.parent} is owned by {owner} and is not writable by "
            f"{os.getenv('USER', 'this account')}.\n"
            "Either have that account run 'chmod 777' on it, or point the cache at a\n"
            "path you own and export the same value wherever the pipeline runs:\n"
            f"  export TOKEN_FILEPATH=$HOME/.alpha-lab/token_cache.json",
            file=sys.stderr,
        )
        sys.exit(1)

    # Make cache readable by group/other so another account can read it
    try:
        cache_file.chmod(0o664)
    except OSError:
        pass  # already permissive enough, or not ours to chmod

    expiry_time = datetime.fromtimestamp(expires_at).strftime("%Y-%m-%d %H:%M:%S")
    print(f"Token cached to {cache_file}")
    print(f"Expires in {expires_in}s (at {expiry_time})")
    print(f"Fetched by: {os.getenv('USER', 'unknown')}")


if __name__ == "__main__":
    main()
