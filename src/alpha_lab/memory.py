"""Persistent, git-backed memory for Alpha Lab.

Layout under the Alpha Lab run memory root (``{workspace}/.alpha_lab/memory``):

- ``records/<key>.json`` — canonical, committed Pydantic ``Memory`` records.
- ``index.db`` — derived, gitignored SQLite metadata + FTS5 index.
- ``embeddings/`` — derived, gitignored memory-mapped embedding store.

The records are the single source of truth; the index and embeddings are rebuilt
from them by :class:`~alpha_lab.databases.ModelDB`, which also owns the git commit
and write locking. ``MemoryStore`` adds memory-specific policy (normalization,
similarity-dedup) and the ranked search API. Keys are store-assigned (the filename),
never a model field; reads/iteration surface ``(key, Memory)`` pairs.
"""

import getpass
import logging
import re
from collections.abc import Iterator, Sequence
from enum import StrEnum, auto
from itertools import islice
from pathlib import Path
from sys import maxsize
from typing import Annotated, Any, Self

from pydantic import Field, field_validator

from alpha_lab import deps
from alpha_lab.constants import MemoryKind
from alpha_lab.databases import ModelDB
from alpha_lab.git import GitRepository
from alpha_lab.models import SQLite, SQLiteModel
from alpha_lab.utils import get_timestamp

logger = logging.getLogger("alpha_lab.memory")


class SearchMode(StrEnum):
    """How :meth:`MemoryStore.search` ranks a query.

    ``EMBEDDED`` sorts by vector similarity against the embedded query.
    ``FULLTEXT`` sorts by bm25 score against the query as OR-joined keywords:
    each whitespace-separated keyword is quoted before it reaches FTS5, so
    punctuation is searched rather than parsed as syntax.
    ``RAW_FTS5`` sorts by bm25 score against the query as an FTS5 expression,
    which keeps operators, prefix terms and column filters available and
    rejects malformed syntax.
    ``RECENCY`` sorts by last update time and rejects a query.
    """

    EMBEDDED = auto()
    FULLTEXT = auto()
    RAW_FTS5 = auto()
    RECENCY = auto()


def _normalize_tags(value: Sequence[str] | str | None) -> tuple[str, ...]:
    """Lowercase, underscore-join, and dedupe (order-preserving) tag values."""
    if not value:
        return ()
    items = [value] if isinstance(value, str) else list(value)
    out: list[str] = []
    for raw in items:
        tag = re.sub(r"[^a-z0-9_\-]", "", re.sub(r"\s+", "_", str(raw).strip().lower()))
        if tag and tag not in out:
            out.append(tag)
    return tuple(out)


def _normalize_sources(value: Sequence[str] | str | None) -> tuple[str, ...]:
    """Strip, dedupe, and sort sources references."""
    if not value:
        return ()
    items = [value] if isinstance(value, str) else list(value)
    return tuple(sorted({s.strip() for s in (str(i) for i in items) if s.strip()}))


class Memory(SQLiteModel):
    """A single persistent memory record (key is store-assigned, not a field)."""
    kind: Annotated[MemoryKind, SQLite]
    summary: Annotated[str, SQLite(embedded=True, fulltext=True)]
    content: Annotated[str, SQLite(embedded=True, fulltext=True)]
    tags: Annotated[tuple[str, ...], SQLite] = ()
    agent: Annotated[str | None, SQLite] = None
    owner: Annotated[str, SQLite] = Field(default_factory=getpass.getuser)
    run_id: Annotated[str | None, SQLite] = None
    frozen: Annotated[bool, SQLite] = False
    sources: Annotated[tuple[str, ...], SQLite] = ()
    created_at: Annotated[str | None, SQLite] = None
    updated_at: Annotated[str | None, SQLite] = None

    def model_post_init(self, __context: Any) -> None:
        now = get_timestamp()
        if self.created_at is None:
            self.created_at = now if self.updated_at is None else min(now, self.updated_at)

        if self.updated_at is None:
            self.updated_at = min(now, self.created_at)

        if self.created_at > self.updated_at:
            msg = f"Creation time {self.created_at} must precede update time {self.updated_at}"
            raise ValueError(msg)

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, value: Any) -> tuple[str, ...]:
        return _normalize_tags(value)

    @field_validator("sources", mode="before")
    @classmethod
    def _normalize_sources(cls, value: Any) -> tuple[str, ...]:
        return _normalize_sources(value)

    def update(self, **fields: Any) -> Self:
        """Return an updated copy, unioning ``tags``/``sources`` with existing.

        Scalar fields are replaced; ``tags`` and ``sources`` union with the
        current values. History is the git log, so nothing is versioned here.

        Args:
            **fields: Field values to change (unknown or frozen fields are rejected).

        Returns:
            A new validated copy with the changes applied.

        Raises:
            ValueError: If the memory is frozen, or a field is unknown or frozen.
        """
        if self.frozen:
            raise ValueError("memory is frozen and cannot be edited")
        unknown = set(fields) - set(type(self).model_fields)
        if unknown:
            raise ValueError(f"unknown field(s): {', '.join(sorted(unknown))}")
        for name in fields:
            if type(self).model_fields[name].frozen:
                raise ValueError(f"cannot edit frozen field {name!r}")

        for seq in ("tags", "sources"):
            if seq in fields:
                incoming = fields[seq]
                incoming = [incoming] if isinstance(incoming, str) else list(incoming)
                fields[seq] = (*getattr(self, seq), *incoming)

        return type(self).model_validate({**self.model_dump(), **fields})

    def render(self, title: str | None = None) -> str:
        """Render this memory as a markdown block for agent-facing display.

        Args:
            title: Heading for the block — callers pass the store-assigned key in,
                e.g. ``f"Memory #{key}: {summary}"``; defaults to the summary.

        Returns:
            A markdown string: title, optional metadata line, then the content.
        """
        lines = [f"# {title or self.summary}", ""]
        meta: list[str] = []
        if self.kind:
            meta.append(f"kind={self.kind}")
        if self.agent:
            meta.append(f"agent={self.agent}")
        if self.run_id:
            meta.append(f"run_id={self.run_id}")
        if self.tags:
            meta.append(f"tags={', '.join(self.tags)}")
        if self.sources:
            meta.append(f"sources={', '.join(self.sources)}")
        if meta:
            lines.extend(["Metadata: " + "; ".join(meta), ""])
        lines.append(self.content)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


class MemoryStore:
    """Read/write persistent memories under a memory root.

    A thin, memory-specific facade over a :class:`~alpha_lab.databases.ModelDB`,
    which owns records, the SQLite index, embeddings, git, locking, and
    similarity-dedup. Normalization lives on :class:`Memory`.
    """

    def __init__(self, db: ModelDB[Memory]) -> None:
        self._db = db

    @property
    def repo(self) -> GitRepository | None:
        """The GitRepository backing this memory store, if any."""
        return self._db.repo

    @property
    def root(self) -> Path:
        """The memory root."""
        return self._db.root

    @property
    def model_type(self) -> type[Memory]:
        return self._db.model_type

    def close(self) -> None:
        """Release the embedding client, if semantic memory is configured.

        Pure delegation: the embedding store owns its client lifecycle
        (it never constructs a client just to close one, and a later use
        lazily rebuilds).
        """
        embedding = self._db.embedding
        if embedding is not None:
            embedding.close()

    def get(self, key: int) -> Memory | None:
        """Load one memory by key.

        Args:
            key: Store-assigned memory key.

        Returns:
            The memory, or ``None`` if no memory exists at ``key``.
        """
        return self._db.get(key, None)

    def add(self, memory: Memory, *, max_similarity: float | None = None, commit: bool | None = None) -> int:
        """Store a new memory and return its store-assigned key.

        Args:
            memory: The memory to store.
            max_similarity: If set, reject the write when the nearest existing
                memory's cosine similarity exceeds it (``None`` disables the check).
            commit: Whether to git-commit; ``None`` commits when the repo can.

        Returns:
            The store-assigned key of the new memory.

        Raises:
            ValueError: If ``max_similarity`` is set and the memory is too similar
                to an existing one.
            EmbeddingError: If producing the memory's embedding fails (aborts the write).
        """
        key = self._db.add(
            memory,
            message=f"store memory: {memory.summary[:60]}",
            commit=commit,
            max_similarity=max_similarity,
        )
        logger.info("Stored memory #%d: %s", key, memory.summary[:80])
        return key

    def search(
        self,
        query: str | None = None,
        *,
        mode: SearchMode | None = None,
        include: dict[str, Any] | None = None,
        exclude: dict[str, Any] | None = None,
        start: int = 0,
        limit: int = 10,
    ) -> list[tuple[int, Memory]]:
        """Retrieve memories by embedding similarity, full text, or recency.

        Args:
            query: Search text. Omitted/empty selects recency ordering.
            mode: A :class:`SearchMode`; ``None`` picks ``EMBEDDED`` when embeddings
                exist and a query is given, else ``FULLTEXT``, else ``RECENCY``.
            include: Metadata filters that must all match (see :meth:`ModelDB.filter`).
            exclude: Metadata filters that must not match.
            start: Number of leading results to skip.
            limit: Maximum number of results to return.

        Returns:
            ``(key, Memory)`` pairs in the mode's ranking order.

        Raises:
            ValueError: If ``start``/``limit`` are negative, ``mode`` is not a
                :class:`SearchMode`, a query is given for recency mode, or a
                non-recency mode is requested without a query.
            EmbeddingError: If embedding the query fails.
        """
        if start < 0 or limit < 0:
            raise ValueError("start and limit must be non-negative")

        mode = None if mode is None else SearchMode(mode)
        query = query.strip() if query is not None else None
        if query:
            if mode is None:
                mode = (
                    SearchMode.EMBEDDED
                    if self._db.embedding is not None
                    else SearchMode.FULLTEXT
                )
            elif mode is SearchMode.RECENCY:
                raise ValueError("recency search does not accept a query")
        else:
            if mode is None:
                mode = SearchMode.RECENCY
            if mode is not SearchMode.RECENCY:
                msg = f"{mode} search requires a query"
                raise ValueError(msg)

        if mode is SearchMode.RECENCY:
            keys = self._db.filter(include=include, exclude=exclude) if (include or exclude) else self._db.keys()
            results = []
            for key in keys:
                val = self._db.get(key, None)
                if val is not None:
                    results.append((key, val))
            results.sort(key=lambda x: (x[1].updated_at or "", x[0]), reverse=True)
            return results[start:start + limit]

        # Only narrow to a candidate key set when a filter is actually given; otherwise
        # search the whole store directly (avoids enumerating every key as a candidate).
        keys = self._db.filter(include=include, exclude=exclude) if (include or exclude) else None
        if mode is SearchMode.EMBEDDED:
            keys = (key for key, _ in self._db.embedding_search(query, keys))
        else:
            keys = (
                key
                for key, _ in self._db.fulltext_search(
                    query, keys, raw_fts5=mode is SearchMode.RAW_FTS5
                )
            )

        results: list[tuple[int, Memory]] = []
        for key in islice(keys, start, None):
            val = self._db.get(key, None)
            if val is not None:
                results.append((key, val))
                if len(results) >= limit:
                    break

        return results

    def update(self, key: int, **fields: Any) -> Memory:
        """Apply field changes to memory ``key`` and persist the result.

        Args:
            key: Key of the memory to update.
            **fields: Field values to change (``tags``/``sources`` union with existing).

        Returns:
            The updated memory.

        Raises:
            KeyError: If no memory exists at ``key``.
            ValueError: If the memory or a targeted field is frozen, or a field is unknown.
        """
        memory = self._db.get(key, None)
        if memory is None:
            raise KeyError(key)
        updated = memory.update(updated_at=get_timestamp(), **fields)
        self._db.set(key, updated, message=f"update memory #{key}")
        return updated

    def __iter__(self) -> Iterator[tuple[int, Memory]]:
        """Iterate all live ``(key, Memory)`` pairs in key order."""
        yield from self._db.items()


def remember_workspace_file(
    path: str | Path,
    summary: str,
    kind: str | MemoryKind,
    agent: str | None = None,
    run_id: str | None = None,
    commit: bool | None = None,
    **kwargs: Any,
) -> int:
    """Store a workspace file's content as a memory in the run's memory store.

    The path resolves under ``deps.workspace`` and must stay within it. An exact
    duplicate (same source, summary, and content) returns the existing memory's key
    instead of storing a copy.

    Args:
        path: Workspace-relative (or in-workspace absolute) path to the file.
        summary: One-line summary for the memory.
        kind: The memory kind.
        agent: Optional agent name to attribute the memory to.
        run_id: Owning run id; defaults to ``deps.run_id``.
        commit: Whether to git-commit; ``None`` commits when the repo can.
        **kwargs: Extra ``Memory`` fields (e.g. ``tags``).

    Returns:
        The store-assigned key (existing key on an exact duplicate).

    Raises:
        ValueError: If the path escapes the workspace, or the summary or file
            content is empty.
        EmbeddingError: If producing the memory's embedding fails.
    """
    # Resolve under the workspace and reject anything that escapes it (e.g. "../").
    workspace = Path(deps.workspace).resolve()
    path = Path(path)
    path = (workspace / path if not path.is_absolute() else path).resolve()
    if not path.is_relative_to(workspace):
        msg = f"path must be within the workspace: {path}"
        raise ValueError(msg)

    # Normalize and validate the summary
    summary = summary.strip()
    if not summary:
        msg = "Normalized summary must be a non-empty string."
        raise ValueError(msg)

    # Load, normalize, and validate the content
    store = deps.memory_store
    source = path.relative_to(workspace).as_posix()
    content = path.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        msg = "Normalized file content must be a non-empty string."
        raise ValueError(msg)

    # Check for exact duplicates from (workspace-relative) identical sources
    for memory_id, memory in store.search(include={"sources": source}, limit=maxsize):
        if memory.summary == summary and memory.content == content:
            return memory_id

    # Create and add the memory
    memory = store.model_type(
        summary=summary,
        content=content,
        kind=kind,
        agent=agent,
        run_id=deps.run_id if run_id is None else run_id,
        sources=[source],
        **kwargs,
    )
    return store.add(memory, commit=commit)

