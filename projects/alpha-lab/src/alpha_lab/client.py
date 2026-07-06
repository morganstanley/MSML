"""Centralized client and provider factory for alpha-lab.

Every module that needs an LLM provider should call ``get_provider()``
instead of constructing one directly.

Authentication uses the standard OpenAI API: set ``OPENAI_API_KEY`` (and
optionally ``OPENAI_BASE_URL`` to point at a compatible endpoint).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from openai import AsyncOpenAI, OpenAI
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider as _PAI_OpenAIProvider

from alpha_lab.provider import Provider

logger = logging.getLogger("alpha_lab.client")


def get_client(api_key: str | None = None) -> OpenAI:
    """Return a configured OpenAI client.

    Parameters
    ----------
    api_key : str, optional
        Explicit API key.  Falls back to the ``OPENAI_API_KEY`` env var.
        ``OPENAI_BASE_URL`` may be set to target an OpenAI-compatible endpoint.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL")  # None -> default
    return OpenAI(api_key=key, base_url=base_url)


def get_async_client(api_key: str | None = None) -> AsyncOpenAI:
    """Return a configured async OpenAI client for PydanticAI agents.

    Mirrors :func:`get_client` with an async transport.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL")  # None -> default
    return AsyncOpenAI(api_key=key, base_url=base_url)


def get_provider(
    provider_name: str = "openai",
    api_key: str | None = None,
) -> Provider:
    """Return a configured Provider instance.

    Parameters
    ----------
    provider_name : str
        Currently only ``"openai"`` is supported.
    api_key : str, optional
        Explicit API key for OpenAI.
    """
    if provider_name == "openai":
        from alpha_lab.provider_openai import OpenAIProvider
        return OpenAIProvider(get_client(api_key))
    raise ValueError(f"Unknown provider: {provider_name!r}. Use 'openai'.")


def get_pydantic_ai_model(
    provider: str,
    model_name: str,
    api_key: str | None = None,
) -> Model:
    """Construct a PydanticAI Model."""
    if provider == "openai":
        return OpenAIResponsesModel(
            model_name=model_name,
            provider=_PAI_OpenAIProvider(openai_client=get_async_client(api_key)),
        )
    msg = f"Unknown provider: {provider!r}. Use 'openai'."
    raise ValueError(msg)
