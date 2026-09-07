"""Helpers for talking to a LiteLLM proxy's admin API.

``models_by_tags`` lists the model names a LiteLLM gateway exposes, optionally
filtered by the tags attached to each deployment's ``litellm_params``.

Named ``litellm_proxy`` rather than ``litellm``: a module named ``litellm.py``
would shadow the real ``litellm`` pip package on import.

Do not run this file by path (``python src/alpha_lab/providers/litellm_proxy.py``):
that puts ``providers/`` on ``sys.path[0]``, where the sibling ``types.py`` shadows
the standard-library ``types`` module and crashes the import chain. Import it
(``from alpha_lab.providers.litellm_proxy import models_by_tags``), run it as a
module (``python -m alpha_lab.providers.litellm_proxy``), or use the demo at
``examples/self-hosted/litellm_proxy.py``.
"""

from __future__ import annotations
import logging
import requests


logger = logging.getLogger("alpha_lab.providers.litellm_proxy")


def query_litellm_model_info(baseurl: str, timeout: int = 10) -> list[dict]:
    logger.info("Querying LiteLLM proxy %s for model info", baseurl)
    resp = requests.get(
        f"{baseurl}/v1/model/info",
        timeout=timeout,
    )
    resp.raise_for_status()
    models = resp.json()["data"]
    logger.info("LiteLLM proxy %s returned %d models", baseurl, len(models))
    return models


def get_available_model_tags(baseurl: str, timeout: int = 10) -> set[str]:
    models = query_litellm_model_info(baseurl, timeout)
    tags = set()
    for m in models:
        tags.update(m.get("litellm_params", {}).get("tags") or [])
    return tags


def validate_model_tags(available_model_tags: set[str], model_clauses: list[str | list[str]]) -> None:
    unknown_tags = set()
    for clause in model_clauses:
        if isinstance(clause, str):
            if clause not in available_model_tags:
                unknown_tags.add(clause)

        elif isinstance(clause, list):
            for tag in clause:
                if tag not in available_model_tags:
                    unknown_tags.add(tag)
        else:
            raise ValueError(
                f"Invalid clause type: {type(clause)}. "
                f"Expected str or list of str."
            )
    if unknown_tags:
        raise ValueError(
            f"Unknown model_tags: {unknown_tags}. "
            f"Available tags: {available_model_tags}"
        )


def models_by_tags(
    baseurl: str,
    spec: list[str | list[str]] | None = None,
    timeout: int = 10,
) -> list[str]:
    """Return the proxy's model names, optionally filtered by tag clauses.

    ``spec`` is a list of clauses matched against each model's tags: a bare
    string is a one-tag clause; a list is an AND of its tags. A model matches
    when it satisfies *any* clause (OR across clauses). ``None``/empty returns
    every model name.
    """
    models = query_litellm_model_info(baseurl, timeout)
    if not spec:
        return [m['model_info']['key'] for m in models]

    clauses = [frozenset([c] if isinstance(c, str) else c) for c in spec]
    return [
        m['model_info']['key']
        for m in models
        if any(
            clause <= set(m.get("litellm_params", {}).get("tags") or [])
            for clause in clauses
        )
    ]
