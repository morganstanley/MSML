"""Centralized client and provider factory for alpha-lab.

Every module that needs an LLM provider should call ``get_provider()``
instead of constructing one directly.  This makes it easy to swap between
OpenAI and Anthropic (or any future provider) by changing config.

Authentication:
  - OpenAI: Set OPENAI_API_KEY env var (or pass api_key directly)
  - Anthropic: Set ANTHROPIC_API_KEY env var
"""

from __future__ import annotations

import logging
import os

from openai import OpenAI

from alpha_lab.provider import Provider

logger = logging.getLogger("alpha_lab.client")


def get_client(api_key: str | None = None) -> OpenAI:
    """Return a configured OpenAI client.

    Parameters
    ----------
    api_key : str, optional
        Explicit API key.  Falls back to ``OPENAI_API_KEY`` env var.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL")  # None -> default
    return OpenAI(api_key=key, base_url=base_url)


def get_anthropic_client():
    """Create an Anthropic client using ANTHROPIC_API_KEY."""
    import anthropic
    return anthropic.Anthropic()


def get_provider(
    provider_name: str = "openai",
    api_key: str | None = None,
) -> Provider:
    """Return a configured Provider instance.

    Parameters
    ----------
    provider_name : str
        One of "openai" or "anthropic".
    api_key : str, optional
        Explicit API key for OpenAI.  Ignored for Anthropic.
    """
    if provider_name == "openai":
        from alpha_lab.provider_openai import OpenAIProvider
        return OpenAIProvider(get_client(api_key))
    elif provider_name in ("anthropic", "bedrock"):
        from alpha_lab.provider_anthropic import AnthropicProvider
        return AnthropicProvider(
            anthropic_client=get_anthropic_client(),
            openai_client=get_client(api_key),
        )
    else:
        raise ValueError(f"Unknown provider: {provider_name!r}. Use 'openai' or 'anthropic'.")
