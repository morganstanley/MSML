"""LLM client construction for alpha-lab.

Builds the concrete SDK clients — OpenAI (sync/async), native Anthropic, boto3
Bedrock, xAI/grok, and local vLLM endpoints — using the credential/detection
helpers from :mod:`alpha_lab.providers.utils.auth`. On-prem vs off-prem is
auto-detected; set ``USE_OFFPREM=1`` to force off-prem mode.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx2
from openai import AsyncOpenAI, OpenAI

from alpha_lab.providers.utils.auth import (
    _get_bearer_token,
    _get_ca_cert_bundle,
    _is_on_prem,
)

logger = logging.getLogger("alpha_lab.providers.utils.clients")

_AIGW_OPENAI_URL = "https://ai-gateway-dev.example.com/openai/v1"
# Native Anthropic Messages API on the gateway (SDK appends ``/v1/messages``).
_AIGW_ANTHROPIC_URL = "https://ai-gateway-dev.example.com/anthropic"
# xAI / grok gateway path — OpenAI-API-compatible (/responses), same bearer auth.
_AIGW_XAI_URL = "https://ai-gateway-dev.example.com/xai/v1"
_AIGW_BEDROCK_ENDPOINT = "https://ai-gateway-dev.example.com/aws"


def _resolve_userid() -> str:
    """Return the canonical LDAP userid for the ``X-Ms-Assert-Username`` header.

    Starts from ``$USER`` and resolves it to the canonical userid via LDAP,
    falling back to the raw ``$USER`` (or ``"unknown"``) when the lookup fails
    (e.g. off-prem / no Kerberos ticket).
    """
    userid = os.getenv("USER", "unknown")
    try:
        from ms.directory import LDAPConnection, FWD_PROD_HOST
        conn = LDAPConnection(host=FWD_PROD_HOST, kerberos=True)
        person = conn.getProdID(userid) or conn.getUser(userid)
        userid = person.userid
    except Exception:
        # LDAP lookup may fail without Kerberos — fall back to USER env var
        logger.warning("LDAP lookup failed, using USER env var: %s", userid)
    return userid


def _on_prem_openai_kwargs(base_url: str = _AIGW_OPENAI_URL) -> dict[str, Any]:
    """Common constructor kwargs for on-prem OpenAI clients (sync or async).

    Callers supply ``http_client`` (the sync/async choice) and may override
    ``base_url`` (e.g. the xAI gateway path); everything else — the per-request
    bearer auth header, user header, timeout — is shared.
    """
    return {
        "base_url": base_url,
        "api_key": "Dummy",  # real auth is the Authorization header injected per-request
        "default_headers": {"X-Ms-Assert-Username": _resolve_userid()},
        "timeout": 360000,
    }


def _build_on_prem_client(base_url: str, log_label: str) -> OpenAI:
    """Build a sync on-prem gateway OpenAI client.

    Validates the bearer token up-front and re-injects a fresh one on every
    request, so refreshed tokens from token_refresh.sh are picked up without a
    restart.
    """
    _get_bearer_token()  # validate we can get a token before building the client

    def _inject_token(request):
        request.headers["Authorization"] = f"Bearer {_get_bearer_token()}"

    http_client = httpx2.Client(
        verify=_get_ca_cert_bundle(),
        event_hooks={"request": [_inject_token]},
    )
    logger.info("%s client using AI Gateway endpoint %s", log_label, base_url)
    return OpenAI(http_client=http_client, **_on_prem_openai_kwargs(base_url))


async def _inject_token_async(request: httpx2.Request) -> None:
    """Async httpx request hook: same as the sync variant, awaitable."""
    request.headers["Authorization"] = f"Bearer {_get_bearer_token()}"


def get_openai_client(
    api_key: str | None = None, *, is_async: bool = False
) -> OpenAI | AsyncOpenAI:
    """Return a configured OpenAI client (gateway on-prem, or public off-prem).

    On-prem uses token-based gateway auth (``api_key`` ignored); set
    ``USE_OFFPREM=1`` to force the public OpenAI API with ``api_key`` /
    ``OPENAI_API_KEY``. Also serves as the web_search-proxy client for the
    non-OpenAI providers. Set ``is_async=True`` for the async client the
    PydanticAI path needs — the auth plumbing is identical, only the client
    class and the request hook differ.
    """
    use_offprem = os.environ.get("USE_OFFPREM", "").lower() in ("1", "true", "yes")
    if _is_on_prem() and not use_offprem:
        if is_async:
            _get_bearer_token()
            http_client = httpx2.AsyncClient(
                verify=_get_ca_cert_bundle(),
                event_hooks={"request": [_inject_token_async]},
            )
            return AsyncOpenAI(http_client=http_client, **_on_prem_openai_kwargs())
        return _build_on_prem_client(_AIGW_OPENAI_URL, "OpenAI")
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL")  # None -> default
    client_cls = AsyncOpenAI if is_async else OpenAI
    return client_cls(api_key=key, base_url=base_url)


def get_anthropic_client(api_key: str | None = None) -> Any:
    """Return a configured Anthropic client (native Messages API).

    On-prem it targets the MS AI Gateway's Anthropic route with per-request
    bearer-token auth; off-prem it builds a standard Anthropic client from
    ``ANTHROPIC_API_KEY``. The ``anthropic`` SDK is imported lazily so this
    module imports even when it isn't installed.
    """
    from anthropic import Anthropic

    use_offprem = os.environ.get("USE_OFFPREM", "").lower() in ("1", "true", "yes")

    if _is_on_prem() and not use_offprem:
        _get_bearer_token()  # validate we can get a token before building the client
        ca_cert = _get_ca_cert_bundle()

        def _inject_token(request):
            request.headers["Authorization"] = f"Bearer {_get_bearer_token()}"

        http_client = httpx2.Client(
            verify=ca_cert,
            event_hooks={"request": [_inject_token]},
        )
        base_url = os.environ.get("ANTHROPIC_BASE_URL", _AIGW_ANTHROPIC_URL)
        logger.info("Anthropic client using MS AI Gateway endpoint %s", base_url)
        return Anthropic(
            base_url=base_url,
            api_key="Dummy",  # real auth is the per-request Authorization header
            http_client=http_client,
            default_headers={"X-Ms-Assert-Username": _resolve_userid()},
            timeout=360000,
        )
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    base_url = os.environ.get("ANTHROPIC_BASE_URL")  # None -> default
    return Anthropic(api_key=key, base_url=base_url)


def get_bedrock_client():
    """Create a boto3 Bedrock client with Bearer token auth via MS AI Gateway."""
    import boto3

    aigw_endpoint = os.environ.get("BEDROCK_BASE_URL", _AIGW_BEDROCK_ENDPOINT)
    use_offprem = os.environ.get("USE_OFFPREM", "").lower() in ("1", "true", "yes")

    if _is_on_prem() and not use_offprem:
        ca_cert = _get_ca_cert_bundle()

        from botocore.config import Config
        bedrock_config = Config(
            read_timeout=300,      # 5 min — Opus generates long responses
            connect_timeout=30,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )

        logger.info("Bedrock client using MS AI Gateway endpoint %s", aigw_endpoint)
        client = boto3.client(
            service_name="bedrock-runtime",
            endpoint_url=aigw_endpoint,
            aws_access_key_id="test",
            aws_secret_access_key="test",
            region_name="us-east-1",
            verify=ca_cert,
            config=bedrock_config,
        )

        def add_bearer_header(request, **kwargs):
            # Re-fetch token on each request (uses in-memory cache, refreshes if expired)
            request.headers["Authorization"] = f"Bearer {_get_bearer_token()}"

        client.meta.events.register("before-send.*.*", add_bearer_header)
        return client
    # Off-prem: assume standard AWS credentials
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )


def get_grok_client(api_key: str | None = None) -> OpenAI:
    """Return an OpenAI SDK client pointed at the xAI / grok gateway path (/xai/v1).

    Same auth as :func:`get_openai_client` — only the base URL differs. Off-prem
    uses ``XAI_API_KEY`` (or ``OPENAI_API_KEY``) against the public x.ai endpoint.
    """
    use_offprem = os.environ.get("USE_OFFPREM", "").lower() in ("1", "true", "yes")
    if _is_on_prem() and not use_offprem:
        return _build_on_prem_client(_AIGW_XAI_URL, "grok")
    key = api_key or os.environ.get("XAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("XAI_BASE_URL") or "https://api.x.ai/v1"
    return OpenAI(api_key=key, base_url=base_url)


def get_local_client(base_url: str) -> OpenAI:
    """Return an OpenAI SDK client pointed at a lab/local Chat Completions endpoint.

    Suitable for internally-hosted models that speak OpenAI-compatible
    ``/v1/chat/completions`` but don't require auth (gpu-host-1, gpu-host-2, Kimi, GLM,
    etc.). ``base_url`` is required and normalized to end with ``/v1``. These
    endpoints are reached directly (no proxy, no CA bundle / bearer token).
    """
    resolved = base_url.rstrip("/")
    if not resolved.endswith("/v1"):
        resolved += "/v1"
    logger.info("local client base_url=%r", resolved)
    return OpenAI(base_url=resolved, api_key="dummy", timeout=600)
