"""LLM connectivity layer for alpha-lab.

Consolidates authentication, client construction, the provider factory, and the
per-backend translation/adjustment logic. Callers should import from this
package rather than the individual submodules.
"""

from __future__ import annotations

from types import MappingProxyType

from alpha_lab.providers.anthropic import AnthropicProvider
from alpha_lab.providers.bedrock import BedrockProvider
from alpha_lab.providers.grok import GrokProvider
from alpha_lab.providers.local import LocalProvider
from alpha_lab.providers.openai import OpenAIProvider
from alpha_lab.providers.types import Provider, Response, StreamEvent, ToolCall
from alpha_lab.providers.utils.auth import ON_PREM_AVAILABLE
from alpha_lab.providers.utils.clients import (
    get_bedrock_client,
    get_grok_client,
    get_local_client,
    get_openai_client,
)

PROVIDER_CLASSES = MappingProxyType({
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "grok": GrokProvider,
    "bedrock": BedrockProvider,
    "local": LocalProvider,
})


def get_provider(
    provider_name: str = "openai",
    api_key: str | None = None,
    model: str = "",
    model_tags: list[str | list[str]] | None = None,
) -> Provider:
    """Return a configured Provider for ``provider_name``.

    ``model_tags`` is forwarded only when set; it is a ``local``-only concern
    (see ``LocalProvider.from_config``), so the other backends' ``from_config``
    signatures are never touched. ``TaskConfig`` already rejects ``model_tags``
    with a non-local provider.
    """
    try:
        provider_class = PROVIDER_CLASSES[provider_name]
    except KeyError:
        raise ValueError(
            f"Unknown provider: {provider_name!r}. Use one of {sorted(PROVIDER_CLASSES)}."
        ) from None
    extra = {"model_tags": model_tags} if model_tags else {}
    return provider_class.from_config(api_key=api_key, model=model, **extra)


__all__ = [
    "AnthropicProvider",
    "BedrockProvider",
    "GrokProvider",
    "LocalProvider",
    "ON_PREM_AVAILABLE",
    "OpenAIProvider",
    "Provider",
    "PROVIDER_CLASSES",
    "Response",
    "StreamEvent",
    "ToolCall",
    "get_bedrock_client",
    "get_grok_client",
    "get_local_client",
    "get_openai_client",
    "get_provider",
]
