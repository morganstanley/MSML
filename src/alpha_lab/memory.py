"""Persistent memory system for alpha-lab.

Filesystem-based memory store that allows agents across phases to
persist and search knowledge.  Stored under ``{workspace}/.memory/``.
"""

from __future__ import annotations

import fcntl
import json
import logging
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger("alpha_lab.memory")


@dataclass
class MemoryEntry:
    """A single memory record."""

    id: int
    tags: list[str]
    summary: str
    created_at: str
    file: str  # relative to entries/


class MemoryStore:
    """Read/write persistent memories in ``{workspace}/.memory/``."""

    def __init__(self, workspace: str) -> None:
        self.workspace = workspace
        self._base = Path(workspace) / ".memory"
        self._entries_dir = self._base / "entries"
        self._index_path = self._base / "index.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, content: str, tags: list[str], summary: str) -> int:
        """Store a new memory entry. Returns the entry ID."""
        self._ensure_dirs()

        with self._lock():
            index = self._load_index()
            entries = index.get("entries", [])
            next_id = max((e["id"] for e in entries), default=0) + 1

            slug = self._slugify(summary)
            filename = f"{next_id:03d}_{slug}.md"

            entry = MemoryEntry(
                id=next_id,
                tags=tags,
                summary=summary,
                created_at=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                file=filename,
            )

            # Write content file
            (self._entries_dir / filename).write_text(content)

            # Update index
            entries.append(asdict(entry))
            index["entries"] = entries
            self._write_index(index)

        logger.info("Stored memory #%d: %s", next_id, summary[:80])
        return next_id

    def search(
        self,
        query: str,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Search memories by keyword in summary + tags.

        Scores entries by how many query words appear in their
        summary + tag text.  Filtered by tags if provided.
        """
        index = self._load_index()
        entries = index.get("entries", [])
        if not entries:
            return []

        query_words = query.lower().split()
        if not query_words:
            return self.list_recent(limit)

        scored: list[tuple[int, int, MemoryEntry]] = []
        for raw in entries:
            entry = self._to_entry(raw)

            # Tag filter
            if tags:
                if not any(t in entry.tags for t in tags):
                    continue

            # Score: count query word hits in summary + tags
            haystack = (entry.summary + " " + " ".join(entry.tags)).lower()
            score = sum(1 for w in query_words if w in haystack)
            if score > 0:
                scored.append((score, entry.id, entry))

        # Sort by score desc, then recency desc
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [entry for _, _, entry in scored[:limit]]

    def read(self, memory_id: int) -> str:
        """Read the full content of a memory entry by ID."""
        index = self._load_index()
        for raw in index.get("entries", []):
            if raw["id"] == memory_id:
                filepath = self._entries_dir / raw["file"]
                if filepath.exists():
                    content = filepath.read_text()
                    # Cap at 10K chars to avoid blowing up context
                    if len(content) > 10_000:
                        return content[:10_000] + "\n[...truncated]"
                    return content
                return f"[ERROR] Memory file not found: {raw['file']}"
        return f"[ERROR] Memory #{memory_id} not found."

    def list_recent(self, limit: int = 20) -> list[MemoryEntry]:
        """List most recent memories."""
        index = self._load_index()
        entries = index.get("entries", [])
        recent = entries[-limit:] if len(entries) > limit else entries
        return [self._to_entry(e) for e in reversed(recent)]

    def list_by_tag(self, tag: str) -> list[MemoryEntry]:
        """List all memories with a given tag."""
        index = self._load_index()
        return [
            self._to_entry(e)
            for e in index.get("entries", [])
            if tag in e.get("tags", [])
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        self._base.mkdir(parents=True, exist_ok=True)
        self._entries_dir.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> dict[str, Any]:
        if not self._index_path.exists():
            return {"version": 1, "entries": []}
        try:
            return json.loads(self._index_path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt memory index, starting fresh")
            return {"version": 1, "entries": []}

    def _write_index(self, index: dict[str, Any]) -> None:
        self._index_path.write_text(json.dumps(index, indent=2))

    @contextmanager
    def _lock(self) -> Iterator[None]:
        """Acquire an exclusive file lock for index writes."""
        self._ensure_dirs()
        lock_path = self._base / ".index.lock"
        fd = open(lock_path, "w")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
                fd.close()
            except OSError:
                pass

    @staticmethod
    def _slugify(text: str, max_len: int = 40) -> str:
        """Convert text to a filename-safe slug."""
        slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
        return slug[:max_len] if slug else "entry"

    @staticmethod
    def _to_entry(raw: dict[str, Any]) -> MemoryEntry:
        return MemoryEntry(
            id=raw["id"],
            tags=raw.get("tags", []),
            summary=raw.get("summary", ""),
            created_at=raw.get("created_at", ""),
            file=raw.get("file", ""),
        )
