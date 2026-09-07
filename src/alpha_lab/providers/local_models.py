"""Per-model behavior table for the ``local`` provider.

A ``local`` deployment behind the LiteLLM proxy isn't a single model — a tag can
match a pool of them, and each may want different request-shaping (thinking
dialect, native vs text tools, a vision fallback, sampling caps). This module
holds that mapping: ``model_key -> LocalModelBehavior``, loaded from a
user-maintained JSON (path in ``LOCAL_MODEL_CONFIG``).

``LocalProvider`` selects a model per request and calls :func:`behavior_for` to
shape the call. When a model isn't in the table (or no table is configured),
:func:`behavior_for` falls back to :func:`sniff_dialect` + env defaults, so the
provider behaves exactly as before without any JSON.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError


def sniff_dialect(model: str) -> str:
    """Derive the local dialect (``"glm"``/``"kimi"``) from the model name.

    Substring match (case-insensitive) so both the short names (``"glm"``,
    ``"kimi"``) and the full ids (``"zai-org/GLM-5.1"``, ``"openai/glm-5.2"``)
    resolve.
    """
    m = (model or "").lower()
    if "glm" in m:
        return "glm"
    if "kimi" in m:
        return "kimi"
    raise ValueError(
        f"Cannot determine local dialect from model {model!r}. "
        f"The model name must contain 'glm' or 'kimi'."
    )


class LocalModelBehavior(BaseModel):
    """How to shape a chat-completion for one local model.

    ``thinking_style`` (``"glm"``/``"kimi"``) drives thinking control and the GLM
    text-tool workaround. ``glm_native_tools`` picks native tool-calling over that
    workaround (GLM-5.2). ``vision_provider`` names an image-fallback path
    (``"bedrock"`` → opus via ``vision_model``; ``None`` → none). ``temperature`` /
    ``max_tokens`` override the GLM sampling caps (``None`` → the provider's defaults).

    ``extra="forbid"`` so an unknown key in the user-authored JSON (a typo like
    ``"visoin"``) is a loud error rather than silently ignored; fields are typed
    so bad values fail too.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    thinking_style: Literal["glm", "kimi"]
    glm_native_tools: bool = False
    vision_provider: Literal["bedrock"] | None = None
    vision_model: str = "claude-opus-4-7"
    temperature: float | None = None
    max_tokens: int | None = None


def _behavior_from_mapping(model: str, spec: dict) -> LocalModelBehavior:
    """Validate one JSON entry into a ``LocalModelBehavior``.

    ``thinking_style`` may be omitted — it's then sniffed from the model key.
    Unknown keys / bad values raise ``pydantic.ValidationError``.
    """
    spec.setdefault("thinking_style", sniff_dialect(model))
    return LocalModelBehavior(**spec)


def load_behaviors(path: str | None) -> dict[str, LocalModelBehavior]:
    """Load the ``model_key -> LocalModelBehavior`` table from ``path`` (JSON).

    ``None``/empty → an empty table (every model then uses the sniff fallback).
    Raises a clear ``ValueError`` (naming the file and model) on a malformed
    entry — unknown key, wrong type, or bad enum value.
    """
    if not path:
        return {}
    resolved = str(Path(path).expanduser())
    with open(resolved) as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{resolved}: model-behavior JSON must be an object of model -> settings")
    table: dict[str, LocalModelBehavior] = {}
    for model, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"{resolved}: settings for model {model!r} must be an object")
        try:
            table[model] = _behavior_from_mapping(model, spec)
        except ValidationError as e:
            raise ValueError(f"{resolved}: invalid settings for model {model!r}: {e}") from e
    return table


def behavior_for(model: str, table: dict[str, LocalModelBehavior]) -> LocalModelBehavior:
    """Return the behavior for ``model``: table entry, else a sniff-based default.

    The fallback mirrors the legacy build-time logic — dialect sniffed from the
    name, ``glm_native_tools`` from ``GLM_NATIVE_TOOLS``, an opus vision fallback
    for GLM-5.2 (native + glm), ``vision_model`` from ``GLM_VISION_MODEL``.
    """
    if model in table:
        return table[model]

    style = sniff_dialect(model)
    native = os.environ.get("GLM_NATIVE_TOOLS", "").lower() in ("1", "true", "yes")
    vision_provider = "bedrock" if (style == "glm" and native) else None
    return LocalModelBehavior(
        thinking_style=style,
        glm_native_tools=native,
        vision_provider=vision_provider,
        vision_model=os.environ.get("GLM_VISION_MODEL", "claude-opus-4-7"),
    )
