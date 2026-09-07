"""Centralized client and provider factory for alpha-lab.

Every module that needs an LLM provider should call ``get_provider()``
instead of constructing one directly.  This makes it easy to swap between
OpenAI and Bedrock (or any future provider) by changing config.

Authentication supports three modes:
  1. Live SCV/PingFed (requires proid Kerberos — automatic when running as proid)
  2. Cached token (run ``scripts/auth_setup.sh`` as proid, then run pipeline as personal account)
  3. Off-prem (set USE_OFFPREM=1 and OPENAI_API_KEY)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from runcmp.llm.provider import Provider

logger = logging.getLogger("runcmp.llm.client")

# ---------------------------------------------------------------------------
# Token cache (written by prefetch_token.py, read here)
# ---------------------------------------------------------------------------
_TOKEN_CACHE_FILE = Path(__file__).resolve().parent.parent.parent / ".token_cache.json"

# Module-level token cache
_cached_token: str | None = None
_token_expires_at: float = 0.0

# Auth constants
_CLIENT_ID = "ops-dev.00000000-0000-0000-0000-000000000000"
_REGISTRATION_ID = "uid:svcacct:svcacct-1"
_ASSERTION_AUDIENCE = "https://auth-dev.example.com"
_SCOPE = "urn:api:ops-dev.00000000-0000-0000-0000-000000000000/.app"
_AIGW_OPENAI_URL = "https://ai-gateway-dev.example.com/openai/v1"
_AIGW_BEDROCK_ENDPOINT = "https://ai-gateway-dev.example.com/aws"
_AIGW_ANTHROPIC_URL = "https://ai-gateway-dev.example.com/anthropic"
# xAI / grok path on the gateway. Same bearer-token auth as the OpenAI
# path; the upstream API is OpenAI-compatible for chat.completions and
# responses (verified via scripts/try_grok.py).
_AIGW_XAI_URL = "https://ai-gateway-dev.example.com/xai/v1"

# Default lab endpoints. gpu-host-2 serves moonshotai/Kimi-K2.6; gpu-host-1 serves
# zai-org/GLM-5.1. Both are OpenAI-compatible vLLM servers on the internal
# network (plain HTTP, no auth). Override via KIMI_BASE_URL / GLM_BASE_URL.
_DEFAULT_KIMI_URL = "http://gpu-host-2.example.com:8000/v1"
_DEFAULT_GLM_URL = "http://gpu-host-1.example.com:8000/v1"


def _get_ca_cert_bundle() -> str:
    import platform
    if platform.system().casefold() == "windows":
        return "\\\\fileshare\\pki\\root-ca.crt"
    return "/etc/pki/ca-trust/certs/internal-ca-chain.crt"


def _try_cached_token() -> bool:
    """Try to load a valid bearer token from the on-disk cache.

    Returns True (and sets module-level _cached_token / _token_expires_at)
    if a non-expired cached token was found.
    """
    global _cached_token, _token_expires_at
    try:
        data = json.loads(_TOKEN_CACHE_FILE.read_text())
    except FileNotFoundError:
        return False
    except (json.JSONDecodeError, OSError):
        return False

    expires_at = data.get("expires_at", 0)
    token = data.get("bearer_token")
    if not token or time.time() >= expires_at:
        logger.warning("Cached token expired. Run scripts/auth_setup.sh as proid to refresh.")
        return False

    _cached_token = token
    _token_expires_at = expires_at
    remaining = int(expires_at - time.time())
    logger.info("Using cached auth token (expires in %dm %ds)", remaining // 60, remaining % 60)
    return True


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
        logger.debug("Live SCV auth failed: %s", exc)
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

    raise RuntimeError(
        "No valid auth token available.\n"
        "Either:\n"
        "  1. Run as proid: suu -tr '<ticket>' <proid> && python run.py ...\n"
        "  2. Prefetch a token: suu -tr '<ticket>' <proid>, then ./scripts/auth_setup.sh, then exit\n"
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
    try:
        data = json.loads(_TOKEN_CACHE_FILE.read_text())
        if data.get("bearer_token") and time.time() < data.get("expires_at", 0):
            return True
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return False


# Exported for backward compatibility — used by server.py, cli.py, run.py
ON_PREM_AVAILABLE: bool = _is_on_prem()


def get_client(api_key: str | None = None) -> OpenAI:
    """Return a configured OpenAI client.

    Parameters
    ----------
    api_key : str, optional
        Explicit API key.  Falls back to ``OPENAI_API_KEY`` env var.
        Ignored when using on-prem (uses token-based auth instead).

    On-prem mode is used automatically when the required MS libraries
    are available. To force off-prem mode, set USE_OFFPREM=1 env var.
    """
    use_offprem = os.environ.get("USE_OFFPREM", "").lower() in ("1", "true", "yes")

    if _is_on_prem() and not use_offprem:
        import httpx

        # Validate we can get a token before building the client
        _get_bearer_token()

        ca_cert = _get_ca_cert_bundle()

        # Re-inject token on every request so refreshed tokens from
        # token_refresh.sh are picked up without restarting.
        def _inject_token(request):
            request.headers["Authorization"] = f"Bearer {_get_bearer_token()}"

        http_client = httpx.Client(
            verify=ca_cert,
            event_hooks={"request": [_inject_token]},
        )

        # Get LDAP userid for header
        userid = os.getenv("USER", "unknown")
        try:
            from ms.directory import LDAPConnection, FWD_PROD_HOST
            conn = LDAPConnection(host=FWD_PROD_HOST, kerberos=True)
            person = conn.getProdID(userid) or conn.getUser(userid)
            userid = person.userid
        except Exception:
            # LDAP lookup may fail without Kerberos — fall back to USER env var
            logger.debug("LDAP lookup failed, using USER env var: %s", userid)

        return OpenAI(
            base_url=_AIGW_OPENAI_URL,
            api_key="Dummy",
            http_client=http_client,
            default_headers={"X-Ms-Assert-Username": userid},
            timeout=600,
        )
    else:
        # Off-prem: standard OpenAI API
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL")  # None -> default
        return OpenAI(api_key=key, base_url=base_url)


def get_anthropic_client(api_key: str | None = None) -> Any:
    """Return a client for the native Anthropic Messages API.

    On-prem this targets the gateway's Anthropic route with per-request bearer
    injection (so refreshed tokens are picked up without a restart), matching
    ``get_client``. Off-prem it builds a standard Anthropic client from
    ``ANTHROPIC_API_KEY``. The SDK is imported lazily so this module still
    imports where it is absent.
    """
    from anthropic import Anthropic

    use_offprem = os.environ.get("USE_OFFPREM", "").lower() in ("1", "true", "yes")
    if _is_on_prem() and not use_offprem:
        import httpx

        _get_bearer_token()          # fail early if auth is unavailable
        ca_cert = _get_ca_cert_bundle()

        def _inject_token(request):
            request.headers["Authorization"] = f"Bearer {_get_bearer_token()}"

        http_client = httpx.Client(
            verify=ca_cert, event_hooks={"request": [_inject_token]},
        )
        base_url = os.environ.get("ANTHROPIC_BASE_URL", _AIGW_ANTHROPIC_URL)
        logger.info("Anthropic client using gateway endpoint %s", base_url)
        return Anthropic(api_key="gateway", base_url=base_url,
                         http_client=http_client)

    return Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""))


def get_grok_client(api_key: str | None = None) -> OpenAI:
    """Return an OpenAI SDK client pointed at the xAI / grok gateway path.

    Same auth pattern as ``get_client()`` — Bearer token from
    ``.token_cache.json`` (or live SCV/PingFed), MS CA bundle for verify,
    ``X-Ms-Assert-Username`` header. Only the base URL differs.

    The xAI gateway implements OpenAI-compatible ``/chat/completions`` and
    ``/responses`` endpoints (verified via ``scripts/try_grok.py``). The
    Responses API tool-type literal differs (xAI wants ``web_search``, not
    OpenAI's ``web_search_preview``) but per system policy web search always
    flows through OpenAI regardless of the configured provider — so
    ``GrokProvider`` translates the built-in literal into a function tool that
    dispatches to ``_proxy_web_search`` using a real OpenAI client. The grok
    client itself never sees ``web_search_preview``.

    Reasoning-effort tiers Grok-4.3 accepts (verified empirically):
    ``minimal`` / ``low`` / ``medium`` / ``high``. ``max`` is rejected
    with HTTP 400.
    """
    use_offprem = os.environ.get("USE_OFFPREM", "").lower() in ("1", "true", "yes")

    if _is_on_prem() and not use_offprem:
        import httpx

        _get_bearer_token()
        ca_cert = _get_ca_cert_bundle()

        def _inject_token(request):
            request.headers["Authorization"] = f"Bearer {_get_bearer_token()}"

        http_client = httpx.Client(
            verify=ca_cert,
            event_hooks={"request": [_inject_token]},
        )

        userid = os.getenv("USER", "unknown")
        try:
            from ms.directory import LDAPConnection, FWD_PROD_HOST
            conn = LDAPConnection(host=FWD_PROD_HOST, kerberos=True)
            person = conn.getProdID(userid) or conn.getUser(userid)
            userid = person.userid
        except Exception:
            logger.debug("LDAP lookup failed, using USER env var: %s", userid)

        return OpenAI(
            base_url=_AIGW_XAI_URL,
            api_key="Dummy",
            http_client=http_client,
            default_headers={"X-Ms-Assert-Username": userid},
            timeout=600,
        )
    else:
        # Off-prem: the standard xAI API uses an x.ai-issued key + URL.
        key = api_key or os.environ.get("XAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("XAI_BASE_URL") or "https://api.x.ai/v1"
        return OpenAI(api_key=key, base_url=base_url)


def get_lab_client(base_url: str | None = None) -> OpenAI:
    """Return an OpenAI SDK client pointed at a lab/local Chat Completions endpoint.

    Suitable for internally-hosted models that speak OpenAI-compatible
    ``/v1/chat/completions`` but don't require auth (gpu-host-1, gpu-host-2, Kimi, GLM,
    etc.). The base URL is resolved in this order:

    1. ``base_url`` argument
    2. ``KIMI_BASE_URL`` env var
    3. ``_DEFAULT_KIMI_URL`` compile-time default

    These endpoints are reached directly (no proxy) — they live inside the
    internal network and are plain HTTP, so no CA bundle or bearer token is
    needed.
    """
    resolved = base_url or os.environ.get("KIMI_BASE_URL") or _DEFAULT_KIMI_URL
    # Strip trailing slash so URL joins work predictably
    resolved = resolved.rstrip("/")
    if not resolved.endswith("/v1"):
        resolved += "/v1"
    return OpenAI(
        base_url=resolved,
        api_key="dummy",  # lab servers ignore the key; OpenAI SDK requires non-empty
        timeout=600,
    )


def get_bedrock_client():
    """Create a boto3 Bedrock client with Bearer token auth via MS AI Gateway."""
    import boto3

    aigw_endpoint = os.environ.get("AIGW_BEDROCK_ENDPOINT", _AIGW_BEDROCK_ENDPOINT)
    use_offprem = os.environ.get("USE_OFFPREM", "").lower() in ("1", "true", "yes")

    if _is_on_prem() and not use_offprem:
        ca_cert = _get_ca_cert_bundle()

        from botocore.config import Config
        bedrock_config = Config(
            read_timeout=300,      # 5 min — Opus generates long responses
            connect_timeout=30,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )

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
    else:
        # Off-prem: assume standard AWS credentials
        return boto3.client(
            service_name="bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )


def get_provider(
    provider_name: str = "openai",
    api_key: str | None = None,
) -> Provider:
    """Return a configured Provider instance.

    Parameters
    ----------
    provider_name : str
        One of ``"openai"``, ``"anthropic"``, ``"bedrock"``, ``"grok"``, ``"kimi"``, or ``"glm"``.
    api_key : str, optional
        Explicit API key for OpenAI / xAI.  Ignored for Bedrock.
    """
    if provider_name == "openai":
        from runcmp.llm.provider_openai import OpenAIProvider
        return OpenAIProvider(get_client(api_key))
    elif provider_name == "anthropic":
        # Claude over the native Messages API: no schema translation, no
        # per-model thinking routing, and the full usage object (including
        # cache counters) is preserved verbatim in Response.usage_raw.
        # Web search still proxies through OpenAI, as for every provider here.
        from runcmp.llm.provider_anthropic import AnthropicProvider
        return AnthropicProvider(
            client=get_anthropic_client(api_key),
            openai_client=get_client(api_key),
        )
    elif provider_name == "grok":
        # xAI's gateway path is OpenAI-API-compatible for chat / responses, but
        # web search has to be proxied through a real OpenAI client (xAI's
        # responses endpoint rejects ``web_search_preview``, and system policy
        # is that web search always flows through OpenAI). GrokProvider holds
        # both clients: ``_client`` (grok) for the conversation, ``openai_client``
        # (OpenAI) exposed to the agent loop for the web_search proxy.
        from runcmp.llm.provider_grok import GrokProvider
        return GrokProvider(
            grok_client=get_grok_client(api_key),
            openai_client_for_proxy=get_client(api_key),
        )
    elif provider_name == "kimi":
        # Lab / locally-hosted Chat Completions endpoint (gpu-host-2, Kimi-K2.6).
        # Base URL: KIMI_BASE_URL env var or the compiled-in default.
        # Web search proxied through OpenAI (lab servers don't support it).
        # thinking_style="kimi": graded thinking_budget from reasoning_effort.
        from runcmp.llm.provider_chat import ChatProvider
        return ChatProvider(
            client=get_lab_client(),
            openai_client_for_proxy=get_client(api_key),
            thinking_style="kimi",
        )
    elif provider_name == "glm":
        # Lab Chat Completions endpoint (gpu-host-1, zai-org/GLM-5.1, vLLM).
        # Base URL: GLM_BASE_URL env var or the gpu-host-1 default.
        # GLM is a reasoning model whose thinking is on-by-default and binary
        # (toggled via chat_template_kwargs.enable_thinking — verified against
        # the live endpoint); thinking_style="glm" wires that dialect and
        # disables thinking for bounded utility completions (summarization).
        # GLM-5.1 needs the text-tool workaround (native tools degenerate it);
        # GLM-5.2 handles native tool calls, so GLM_NATIVE_TOOLS=1 selects the
        # native tools= path. Default off keeps GLM-5.1 working unchanged.
        from runcmp.llm.provider_chat import ChatProvider
        native = os.environ.get("GLM_NATIVE_TOOLS", "").lower() in ("1", "true", "yes")
        # GLM-5.2 is text-only: route image turns to a vision-capable opus
        # (like web search proxies via OpenAI). Built ONLY for GLM-5.2; None
        # for GLM-5.1 and everyone else, so their paths are unchanged.
        # GLM_VISION_PROVIDER selects the transport: "anthropic" (default,
        # the native Messages gateway), "bedrock" (legacy Converse path), or
        # "none" (no opus fallback — image turns go straight to the gpt-4o
        # proxy). Measured reason for the anthropic default: on 2026-07-31
        # three Bedrock 503s made an all-GLM run silently answer whole turns
        # with gpt-4o.
        vision_provider = None
        vision_dialect = os.environ.get(
            "GLM_VISION_PROVIDER", "anthropic").strip().lower()
        if native and vision_dialect not in ("", "none"):
            try:
                if vision_dialect == "anthropic":
                    from runcmp.llm.provider_anthropic import AnthropicProvider
                    vision_provider = AnthropicProvider(
                        client=get_anthropic_client(api_key),
                        openai_client=get_client(api_key),
                    )
                elif vision_dialect == "bedrock":
                    from runcmp.llm.provider_bedrock import BedrockProvider
                    vision_provider = BedrockProvider(
                        bedrock_client=get_bedrock_client(),
                        openai_client=get_client(api_key),
                    )
                else:
                    logger.warning(
                        "Unknown GLM_VISION_PROVIDER=%r (expected 'anthropic', "
                        "'bedrock', or 'none'); image turns will use the "
                        "gpt-4o proxy instead", vision_dialect,
                    )
            except Exception as exc:
                logger.warning(
                    "GLM-5.2 opus vision fallback via %s unavailable (%s); "
                    "image turns will use the gpt-4o proxy instead",
                    vision_dialect, exc,
                )
        return ChatProvider(
            client=get_lab_client(os.environ.get("GLM_BASE_URL") or _DEFAULT_GLM_URL),
            openai_client_for_proxy=get_client(api_key),
            thinking_style="glm",
            glm_native_tools=native,
            vision_provider=vision_provider,
            vision_model=os.environ.get("GLM_VISION_MODEL", "claude-opus-4-8"),
            vision_dialect=vision_dialect,
        )
    elif provider_name == "mlrllm":
        # Shared lab LiteLLM gateway (local-llm.example.com) hosting kimi-k3,
        # deepseek-v4-flash, gemma-4-31b (and glm-5.2). OpenAI Chat
        # Completions with native tool calls for all hosted models (verified
        # live 2026-08-04); per-model thinking dialect via
        # thinking_style="mlrllm". Web search proxies through OpenAI as for
        # every provider here.
        from runcmp.llm.provider_chat import ChatProvider
        return ChatProvider(
            client=get_lab_client(
                os.environ.get("MLRLLM_BASE_URL") or "http://local-llm.example.com/v1"),
            openai_client_for_proxy=get_client(api_key),
            thinking_style="mlrllm",
            glm_native_tools=True,
        )
    elif provider_name == "bedrock":
        from runcmp.llm.provider_bedrock import BedrockProvider
        return BedrockProvider(
            bedrock_client=get_bedrock_client(),
            openai_client=get_client(api_key),
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider_name!r}. "
            f"Use 'openai', 'grok', 'kimi', 'glm', 'mlrllm', or 'bedrock'."
        )
