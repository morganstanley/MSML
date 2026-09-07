#!/usr/bin/env python
"""Prefetch and cache a bearer token for Alpha-Lab.

Run this as the proid (after suu) to cache a token that can then be used
by any account to run the pipeline without Kerberos credentials.

Usage:
    suu -tr "<ticket>" <proid>
    python prefetch_token.py
    exit
    # Now run the pipeline as your personal account
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add src to path (one level up from scripts/)
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, src_path)

# Cache file lives in the repo root (one level up from scripts/)
CACHE_FILE = Path(__file__).resolve().parent.parent / ".token_cache.json"

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
    import httpx
    from scvlib import SecureCredentialsVault
    from aidevplatformaccess.core import (
        PingFedAssertionTokenCredential,
        PingFedGateway,
        PingFedTokenManager,
        ScvJsonWebKeyProvider,
    )
    from aidevplatformaccess.scv import ScvClient

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

    CACHE_FILE.write_text(json.dumps(cache, indent=2) + "\n")
    # Make cache readable by group so personal account can read it
    try:
        CACHE_FILE.chmod(0o664)
    except PermissionError:
        pass  # already permissive enough

    expiry_time = datetime.fromtimestamp(expires_at).strftime("%Y-%m-%d %H:%M:%S")
    print(f"Token cached to {CACHE_FILE}")
    print(f"Expires in {expires_in}s (at {expiry_time})")
    print(f"Fetched by: {os.getenv('USER', 'unknown')}")


if __name__ == "__main__":
    main()
