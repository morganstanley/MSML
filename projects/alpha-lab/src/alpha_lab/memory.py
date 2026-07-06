"""Persistent memory system for Alpha Lab.

This is intentionally lightweight memory infrastructure: workspace-local only,
no embeddings, no background workers, no external services. The design keeps
Alpha Lab's existing tool API intact while improving retrieval quality with a
minimal-dependency SQLite index.

Storage layout under ``{workspace}/.memory/``:
- ``entries/*.md``: human-inspectable, self-describing memory files
- ``index.json``: legacy-compatible metadata index
- ``memory.db``: lightweight SQLite metadata/content index for better search
"""

from __future__ import annotations

import fcntl
import json
import logging
import re
import sqlite3
import threading
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger("alpha_lab.memory")

_INDEX_VERSION = 2
_MAX_READ_CHARS = 10_000
_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: weakref.WeakValueDictionary[str, threading.RLock] = weakref.WeakValueDictionary()
_INIT_STATES_GUARD = threading.Lock()
_INIT_STATES: OrderedDict[str, "_InitializationState"] = OrderedDict()
_MAX_INIT_STATES = 256

MEMORY_KINDS = frozenset({
    "finding",
    "decision",
    "failure",
    "result",
    "hypothesis",
    "constraint",
    "reference",
})

_KIND_ALIASES = {
    "bug": "failure",
    "error": "failure",
    "exception": "failure",
    "fail": "failure",
    "failed": "failure",
    "issue": "failure",
    "problem": "failure",
    "review_failure": "failure",
    "test_failure": "failure",
    "choice": "decision",
    "conclusion": "decision",
    "rule": "decision",
    "insight": "finding",
    "observation": "finding",
    "fact": "finding",
    "experiment": "result",
    "experiment_result": "result",
    "debrief": "result",
    "metric": "result",
    "outcome": "result",
    "idea": "hypothesis",
    "proposal": "hypothesis",
    "requirement": "constraint",
    "limit": "constraint",
    "howto": "reference",
    "how_to": "reference",
    "knowledge": "reference",
    "note": "reference",
    "runbook": "reference",
    "topic": "reference",
}


@dataclass
class MemoryEntry:
    """A single memory record."""

    id: int
    tags: list[str]
    summary: str
    created_at: str
    file: str  # relative to entries/
    kind: str | None = None
    phase: str | None = None
    agent: str | None = None
    run_id: str | None = None
    source_path: str | None = None


@dataclass(frozen=True)
class _InitializationState:
    """Process-local bootstrap state for one memory directory."""

    signature: tuple[Any, ...]
    sqlite_ready: bool
    fts_ready: bool


@dataclass
class _SearchCandidate:
    """Internal search candidate with lightweight rank-fusion signals."""

    row: sqlite3.Row
    fts_rank: int | None = None
    lexical_rank: int | None = None

    @property
    def fusion_score(self) -> float:
        # Reciprocal-rank fusion: a small, stable signal that rewards candidates
        # found by multiple retrieval strategies without overwhelming textual
        # relevance. Use the common RRF k=60 constant and equal source weights;
        # these are not corpus-tuned values.
        score = 0.0
        if self.fts_rank is not None:
            score += 1.0 / (60.0 + self.fts_rank)
        if self.lexical_rank is not None:
            score += 1.0 / (60.0 + self.lexical_rank)
        return score


class MemoryStore:
    """Read/write persistent memories in ``{workspace}/.memory/``.

    The store preserves the original filesystem layout for transparency and
    backwards compatibility, while optionally indexing content into SQLite for
    higher-quality search. If SQLite/FTS is unavailable, it gracefully falls
    back to legacy metadata-only search.
    """

    def __init__(self, workspace: str) -> None:
        self.workspace = workspace
        self._base = Path(workspace) / ".memory"
        self._entries_dir = self._base / "entries"
        self._index_path = self._base / "index.json"
        self._db_path = self._base / "memory.db"
        self._cache_key = str(self._base.resolve())
        self._thread_lock = self._thread_lock_for_path(self._base)
        self._initialized = False
        self._sqlite_ready = False
        self._fts_ready = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        tags: list[str] | str | None,
        summary: str,
        *,
        kind: str | None = None,
        phase: str | None = None,
        agent: str | None = None,
        run_id: str | None = None,
        source_path: str | None = None,
    ) -> int:
        """Store a new memory entry and return its ID."""
        self._ensure_initialized()

        tags = self._normalize_tags(tags)
        summary = (summary or "").strip()
        content = content or ""
        kind = self._normalize_kind(kind)
        phase = self._normalize_optional(phase)
        agent = self._normalize_optional(agent)
        run_id = self._normalize_optional(run_id)
        source_path = self._normalize_optional(source_path)

        with self._lock():
            # _ensure_initialized() already bootstrapped this process; only redo
            # the heavy rebuild/sync if the SQLite index never came up or its file
            # disappeared. Avoids re-running DDL + legacy sync on every store.
            if not (self._sqlite_ready and self._db_path.exists()):
                self._bootstrap_locked()
            next_id = self._next_id_locked()
            slug = self._slugify(summary or f"memory_{next_id}")
            filename = f"{next_id:03d}_{slug}.md"

            entry = MemoryEntry(
                id=next_id,
                tags=tags,
                summary=summary,
                created_at=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                file=filename,
                kind=kind,
                phase=phase,
                agent=agent,
                run_id=run_id,
                source_path=source_path,
            )

            self._write_text_atomic(
                self._entries_dir / filename,
                self._serialize_entry_file(entry, content),
            )
            self._upsert_sqlite_entry_locked(entry, content)
            self._upsert_index_entry_locked(entry)
            self._remember_initialized_state(self._canonical_signature())

        logger.info("Stored memory #%d: %s", next_id, summary[:80])
        return next_id

    def search(
        self,
        query: str,
        tags: list[str] | None = None,
        limit: int = 10,
        *,
        kind: str | None = None,
        phase: str | None = None,
    ) -> list[MemoryEntry]:
        """Search memories using SQLite full-text search when available."""
        self._ensure_initialized()

        limit = max(int(limit), 1)
        tags = self._normalize_tags(tags or [])
        kind = self._normalize_kind(kind)
        phase = self._normalize_optional(phase)
        query = (query or "").strip()

        if not query:
            recent = self.list_recent(limit=max(limit * 3, limit))
            return self._filter_entries(recent, tags=tags, kind=kind, phase=phase)[:limit]

        try:
            candidates = self._candidate_rows(query, limit=limit, kind=kind, phase=phase)
            scored: list[tuple[float, int, MemoryEntry]] = []
            query_words = self._query_words(query)
            for candidate in candidates:
                row = candidate.row
                entry = self._entry_from_row(row)
                if not self._matches_filters(entry, tags=tags, kind=kind, phase=phase):
                    continue
                score = self._score_row(row, query, query_words, fusion_score=candidate.fusion_score)
                if score > 0:
                    scored.append((score, entry.id, entry))
            if scored:
                scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
                return [entry for _, _, entry in scored[:limit]]
        except sqlite3.Error as e:
            logger.warning("SQLite memory search failed, using fallback: %s", e)

        return self._legacy_search(query, tags=tags, limit=limit, kind=kind, phase=phase)

    def read(self, memory_id: int) -> str:
        """Read the full content of a memory entry by ID."""
        self._ensure_initialized()
        content = self._read_content(memory_id)
        if content is None:
            return f"[ERROR] Memory #{memory_id} not found."
        return self._truncate_content(content)

    def list_recent(self, limit: int = 20) -> list[MemoryEntry]:
        """List most recent memories."""
        self._ensure_initialized()
        limit = max(int(limit), 1)

        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM memories ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            if rows:
                return [self._entry_from_row(row) for row in rows]
        except sqlite3.Error as e:
            logger.warning("SQLite list_recent failed, using index fallback: %s", e)

        index = self._load_index()
        entries = index.get("entries", [])
        recent = entries[-limit:] if len(entries) > limit else entries
        return [self._to_entry(e) for e in reversed(recent)]

    def _all_index_entries(self) -> list[MemoryEntry]:
        """All entries from the canonical index, newest first (no row cap)."""
        index = self._load_index()
        return [self._to_entry(e) for e in reversed(index.get("entries", []))]

    def list_by_tag(self, tag: str) -> list[MemoryEntry]:
        """List all memories with a given tag."""
        self._ensure_initialized()
        tag = self._normalize_tag(tag)
        if not tag:
            return []

        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE tag_blob LIKE ? ESCAPE '\\' ORDER BY id DESC",
                    (self._like_pattern(f"|{tag}|"),),
                ).fetchall()
            if rows:
                return [self._entry_from_row(row) for row in rows]
        except sqlite3.Error as e:
            logger.warning("SQLite list_by_tag failed, using index fallback: %s", e)

        index = self._load_index()
        return [
            self._to_entry(e)
            for e in reversed(index.get("entries", []))
            if tag in self._normalize_tags(e.get("tags", []))
        ]

    def _read_content(self, memory_id: int) -> str | None:
        row = self._get_row_by_id(memory_id)
        if row is not None:
            file_name = row["file"]
            path = self._entries_dir / file_name
            if path.exists():
                _, content = self._load_entry_from_file(path, fallback_entry=self._entry_from_row(row))
                return content
            content = row["content"] or ""
            if content:
                return content
            return None

        index = self._load_index()
        for raw in index.get("entries", []):
            if raw.get("id") == memory_id:
                filepath = self._entries_dir / raw.get("file", "")
                if filepath.exists():
                    _, content = self._load_entry_from_file(filepath, fallback_entry=self._to_entry(raw))
                    return content
                return None
        return None

    # ------------------------------------------------------------------
    # Internal bootstrap / storage helpers
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        signature = self._canonical_signature()
        cached = self._cached_initialization_state(signature)
        if cached is not None:
            self._apply_initialization_state(cached)
            return

        with self._lock():
            if self._initialized:
                return
            signature = self._canonical_signature()
            cached = self._cached_initialization_state(signature)
            if cached is not None:
                self._apply_initialization_state(cached)
                return
            self._bootstrap_locked()
            self._initialized = True
            self._remember_initialized_state(self._canonical_signature())

    def _bootstrap_locked(self) -> None:
        self._ensure_dirs()
        self._init_sqlite_locked()
        self._rebuild_index_from_entries_locked()
        self._sync_legacy_index_to_sqlite_locked()

    def _ensure_dirs(self) -> None:
        self._base.mkdir(parents=True, exist_ok=True)
        self._entries_dir.mkdir(parents=True, exist_ok=True)

    def _canonical_signature(self) -> tuple[Any, ...]:
        """Cheap fingerprint of portable memory files used to skip re-bootstrap.

        The canonical artifacts are still the source of truth. New store
        instances can reuse a prior process-local bootstrap only when this
        signature is unchanged; edits to entry markdown or index.json force a
        refresh from disk.
        """
        entries: list[tuple[str, int, int]] = []
        try:
            paths = sorted(self._entries_dir.glob("*.md"))
        except OSError:
            paths = []
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append((path.name, stat.st_mtime_ns, stat.st_size))
        return (self._file_signature(self._index_path), tuple(entries))

    @staticmethod
    def _file_signature(path: Path) -> tuple[bool, int, int]:
        try:
            stat = path.stat()
        except OSError:
            return (False, 0, 0)
        return (True, stat.st_mtime_ns, stat.st_size)

    def _cached_initialization_state(self, signature: tuple[Any, ...]) -> _InitializationState | None:
        with _INIT_STATES_GUARD:
            cached = _INIT_STATES.get(self._cache_key)
            if cached is not None:
                _INIT_STATES.move_to_end(self._cache_key)
        if cached is None or cached.signature != signature:
            return None
        if cached.sqlite_ready and not self._db_path.exists():
            return None
        return cached

    def _apply_initialization_state(self, state: _InitializationState) -> None:
        self._sqlite_ready = state.sqlite_ready and self._db_path.exists()
        self._fts_ready = state.fts_ready and self._sqlite_ready
        self._initialized = True

    def _remember_initialized_state(self, signature: tuple[Any, ...]) -> None:
        state = _InitializationState(
            signature=signature,
            sqlite_ready=self._sqlite_ready and self._db_path.exists(),
            fts_ready=self._fts_ready and self._db_path.exists(),
        )
        with _INIT_STATES_GUARD:
            _INIT_STATES[self._cache_key] = state
            _INIT_STATES.move_to_end(self._cache_key)
            while len(_INIT_STATES) > _MAX_INIT_STATES:
                _INIT_STATES.popitem(last=False)

    def _load_index(self) -> dict[str, Any]:
        if not self._index_path.exists():
            return {"version": _INDEX_VERSION, "entries": []}
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8", errors="replace"))
            if not isinstance(data, dict):
                raise ValueError("memory index must be a JSON object")
            data.setdefault("version", _INDEX_VERSION)
            data.setdefault("entries", [])
            return data
        except (json.JSONDecodeError, OSError, ValueError):
            logger.warning("Corrupt memory index, starting fresh")
            return {"version": _INDEX_VERSION, "entries": []}

    def _write_index(self, index: dict[str, Any]) -> None:
        index["version"] = _INDEX_VERSION
        # index.json is a canonical, portable artifact: write to a temp file and
        # atomically replace so an interrupted write can't corrupt/erase memory.
        self._write_text_atomic(self._index_path, json.dumps(index, indent=2))

    @staticmethod
    def _write_text_atomic(path: Path, text: str) -> None:
        """Write text via a same-directory temp file and atomic replace."""
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)

    def _upsert_index_entry_locked(self, entry: MemoryEntry) -> None:
        index = self._load_index()
        entries = [e for e in index.get("entries", []) if e.get("id") != entry.id]
        entries.append(asdict(entry))
        entries.sort(key=lambda raw: raw.get("id", 0))
        index["entries"] = entries
        self._write_index(index)

    def _init_sqlite_locked(self) -> None:
        try:
            with self._connect() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY,
                        summary TEXT NOT NULL,
                        content TEXT NOT NULL,
                        tags_json TEXT NOT NULL,
                        tags_text TEXT NOT NULL,
                        tag_blob TEXT NOT NULL,
                        kind TEXT,
                        phase TEXT,
                        agent TEXT,
                        run_id TEXT,
                        source_path TEXT,
                        created_at TEXT NOT NULL,
                        file TEXT NOT NULL
                    )
                    """
                )
                # Drop stale indexes from earlier cache versions. The SQLite DB
                # is rebuildable; avoid retaining indexes that add write overhead
                # but are not used by the current query patterns.
                conn.execute("DROP INDEX IF EXISTS idx_memories_created_at")
                conn.execute("DROP INDEX IF EXISTS idx_memories_tag_blob")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_kind_phase ON memories(kind, phase)"
                )
                try:
                    conn.execute(
                        """
                        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                        USING fts5(summary, content, tags_text)
                        """
                    )
                    self._fts_ready = True
                except sqlite3.OperationalError:
                    self._fts_ready = False
                self._sqlite_ready = True
        except sqlite3.Error as e:
            logger.warning("SQLite memory backend unavailable, falling back to index-only mode: %s", e)
            self._sqlite_ready = False
            self._fts_ready = False

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _sync_legacy_index_to_sqlite_locked(self) -> None:
        if not self._sqlite_ready:
            return

        index = self._load_index()
        entries = index.get("entries", [])
        valid_ids: set[int] = set()

        with self._connect() as conn:
            for raw in entries:
                memory_id = int(raw.get("id", 0) or 0)
                if memory_id <= 0:
                    continue
                valid_ids.add(memory_id)
                entry = self._to_entry(raw)
                content_path = self._entries_dir / entry.file
                if content_path.exists():
                    parsed_entry, content = self._load_entry_from_file(
                        content_path,
                        fallback_entry=entry,
                    )
                    entry = parsed_entry
                else:
                    content = ""
                self._upsert_sqlite_entry_locked(entry, content, conn=conn)
            self._prune_sqlite_rows_locked(valid_ids, conn=conn)

    def _prune_sqlite_rows_locked(self, valid_ids: set[int], *, conn: sqlite3.Connection) -> None:
        if not self._sqlite_ready:
            return
        try:
            if valid_ids:
                placeholders = ",".join("?" for _ in valid_ids)
                stale_rows = conn.execute(
                    f"SELECT id FROM memories WHERE id NOT IN ({placeholders})",
                    sorted(valid_ids),
                ).fetchall()
                conn.execute(
                    f"DELETE FROM memories WHERE id NOT IN ({placeholders})",
                    sorted(valid_ids),
                )
            else:
                stale_rows = conn.execute("SELECT id FROM memories").fetchall()
                conn.execute("DELETE FROM memories")
            if self._fts_ready:
                for row in stale_rows:
                    conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (int(row["id"]),))
        except sqlite3.Error as e:
            logger.warning("Failed to prune stale SQLite memory rows: %s", e)

    def _rebuild_index_from_entries_locked(self) -> None:
        entries_by_id: dict[int, dict[str, Any]] = {}

        index = self._load_index()
        for raw in index.get("entries", []):
            memory_id = int(raw.get("id", 0) or 0)
            if memory_id > 0:
                entries_by_id[memory_id] = dict(raw)

        for path in sorted(self._entries_dir.glob("*.md")):
            fallback_entry = None
            match = re.match(r"^(\d+)_", path.name)
            if match:
                memory_id = int(match.group(1))
                raw = entries_by_id.get(memory_id)
                if raw is not None:
                    fallback_entry = self._to_entry(raw)
            entry, _ = self._load_entry_from_file(path, fallback_entry=fallback_entry)
            if entry.id <= 0:
                continue
            entries_by_id[entry.id] = asdict(entry)

        if entries_by_id:
            rebuilt = {
                "version": _INDEX_VERSION,
                "entries": [entries_by_id[k] for k in sorted(entries_by_id)],
            }
            self._write_index(rebuilt)

    def _next_id_locked(self) -> int:
        if self._sqlite_ready:
            with self._connect() as conn:
                row = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 AS next_id FROM memories").fetchone()
                if row is not None:
                    return int(row["next_id"])

        index = self._load_index()
        entries = index.get("entries", [])
        return max((int(e.get("id", 0)) for e in entries), default=0) + 1

    def _upsert_sqlite_entry_locked(
        self,
        entry: MemoryEntry,
        content: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        if not self._sqlite_ready:
            return

        owns_conn = conn is None
        if conn is None:
            conn = self._connect()

        tags_json = json.dumps(entry.tags)
        tags_text = " ".join(entry.tags)
        if entry.kind:
            tags_text += f" {entry.kind}"
        if entry.phase:
            tags_text += f" {entry.phase}"
        tag_blob = self._tag_blob(entry.tags)

        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories (
                    id, summary, content, tags_json, tags_text, tag_blob,
                    kind, phase, agent, run_id, source_path, created_at, file
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.summary,
                    content,
                    tags_json,
                    tags_text.strip(),
                    tag_blob,
                    entry.kind,
                    entry.phase,
                    entry.agent,
                    entry.run_id,
                    entry.source_path,
                    entry.created_at,
                    entry.file,
                ),
            )
            if owns_conn:
                conn.commit()

            if self._fts_ready:
                try:
                    conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (entry.id,))
                    conn.execute(
                        "INSERT INTO memories_fts(rowid, summary, content, tags_text) VALUES (?, ?, ?, ?)",
                        (entry.id, entry.summary, content, tags_text.strip()),
                    )
                    if owns_conn:
                        conn.commit()
                except sqlite3.Error as e:
                    logger.warning("SQLite memory FTS update failed; metadata/content row kept: %s", e)
                    if owns_conn:
                        conn.rollback()
                    self._fts_ready = False
        finally:
            if owns_conn:
                conn.close()

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------

    def _candidate_rows(
        self,
        query: str,
        *,
        limit: int,
        kind: str | None,
        phase: str | None,
    ) -> list[_SearchCandidate]:
        if not self._sqlite_ready:
            return []

        oversample = max(limit * 8, 25)
        filters = []
        params: list[Any] = []
        if kind:
            kind_values = self._kind_filter_values(kind)
            placeholders = ", ".join("?" for _ in kind_values)
            filters.append(f"m.kind IN ({placeholders})")
            params.extend(kind_values)
        if phase:
            filters.append("m.phase = ?")
            params.append(phase)
        where_sql = ""
        if filters:
            where_sql = " AND " + " AND ".join(filters)

        candidates: dict[int, _SearchCandidate] = {}

        def add_rows(rows: list[sqlite3.Row], *, source: str) -> None:
            for rank, row in enumerate(rows, start=1):
                memory_id = int(row["id"])
                candidate = candidates.get(memory_id)
                if candidate is None:
                    candidate = _SearchCandidate(row=row)
                    candidates[memory_id] = candidate
                elif source == "fts" and candidate.row["ft_rank"] is None:
                    # Prefer the FTS row shape when available so _score_row can
                    # still use bm25. Metadata/content columns are identical.
                    candidate.row = row
                if source == "fts" and candidate.fts_rank is None:
                    candidate.fts_rank = rank
                if source == "lexical" and candidate.lexical_rank is None:
                    candidate.lexical_rank = rank

        with self._connect() as conn:
            if self._fts_ready:
                fts_query = self._fts_query(query)
                if fts_query:
                    rows = conn.execute(
                        f"""
                        SELECT m.*, bm25(memories_fts, 5.0, 1.0, 2.0) AS ft_rank
                        FROM memories_fts
                        JOIN memories AS m ON memories_fts.rowid = m.id
                        WHERE memories_fts MATCH ?{where_sql}
                        ORDER BY ft_rank
                        LIMIT ?
                        """,
                        [fts_query, *params, oversample],
                    ).fetchall()
                    add_rows(rows, source="fts")

            # Always run a cheap lexical substring pass as well. SQLite FTS is
            # fast and useful, but tokenization can miss domain strings such as
            # CUDAOOM, run_42_fold3, or metric names where a substring is the
            # memorable part. Rank fusion below deduplicates rows that both
            # strategies find.
            query_sql = ""
            like_params: list[Any] = []
            word_clauses = []
            for word in self._query_words(query):
                word_clauses.append(
                    "(m.summary LIKE ? ESCAPE '\\' OR m.tags_text LIKE ? ESCAPE '\\' OR m.content LIKE ? ESCAPE '\\')"
                )
                pattern = self._like_pattern(word)
                like_params.extend([pattern, pattern, pattern])
            if word_clauses:
                query_sql = " AND (" + " OR ".join(word_clauses) + ")"
            else:
                return list(candidates.values())

            rows = conn.execute(
                f"SELECT m.*, NULL AS ft_rank FROM memories AS m WHERE 1=1{where_sql}{query_sql} ORDER BY id DESC LIMIT ?",
                [*params, *like_params, max(oversample, 200)],
            ).fetchall()
            add_rows(rows, source="lexical")

        return list(candidates.values())

    def _legacy_search(
        self,
        query: str,
        *,
        tags: list[str],
        limit: int,
        kind: str | None,
        phase: str | None,
    ) -> list[MemoryEntry]:
        entries = self._filter_entries(self._all_index_entries(), tags=tags, kind=kind, phase=phase)
        words = self._query_words(query)
        scored: list[tuple[int, int, MemoryEntry]] = []
        for entry in entries:
            haystack = " ".join(
                part
                for part in [entry.summary, " ".join(entry.tags), entry.kind or "", entry.phase or ""]
                if part
            ).lower()
            score = 0
            query_lower = query.lower()
            if query_lower in haystack:
                score += 100
            score += 10 * sum(1 for word in words if word in haystack)
            if score > 0:
                scored.append((score, entry.id, entry))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [entry for _, _, entry in scored[:limit]]

    def _score_row(
        self,
        row: sqlite3.Row,
        query: str,
        query_words: list[str],
        *,
        fusion_score: float = 0.0,
    ) -> float:
        summary = (row["summary"] or "").lower()
        content = (row["content"] or "").lower()
        tags_text = (row["tags_text"] or "").lower()
        query_lower = query.lower()
        score = 0.0

        query_exact_hits = sum(
            1 for haystack in (summary, tags_text, content)
            if query_lower and query_lower in haystack
        )
        word_hits = (
            sum(1 for word in query_words if word in summary)
            + sum(1 for word in query_words if word in tags_text)
            + sum(1 for word in query_words if word in content)
        )
        if query_exact_hits == 0 and word_hits == 0:
            return 0.0

        if query_lower in summary:
            score += 120.0
        if query_lower in tags_text:
            score += 80.0
        if query_lower in content:
            score += 40.0

        score += 12.0 * sum(1 for word in query_words if word in summary)
        score += 8.0 * sum(1 for word in query_words if word in tags_text)
        score += 4.0 * sum(1 for word in query_words if word in content)

        ft_rank = row["ft_rank"] if "ft_rank" in row.keys() else None
        if ft_rank is not None:
            try:
                score += 1.0 / (1.0 + abs(float(ft_rank)))
            except (TypeError, ValueError):
                pass

        # Small reciprocal-rank-fusion bonus for candidates recovered by FTS,
        # lexical substring search, or both.
        score += 20.0 * fusion_score

        # Mild recency preference without overwhelming semantic relevance.
        try:
            score += min(float(row["id"]), 1_000_000.0) / 1_000_000.0
        except (TypeError, ValueError):
            pass
        return score

    def _filter_entries(
        self,
        entries: list[MemoryEntry],
        *,
        tags: list[str],
        kind: str | None,
        phase: str | None,
    ) -> list[MemoryEntry]:
        return [
            entry for entry in entries
            if self._matches_filters(entry, tags=tags, kind=kind, phase=phase)
        ]

    def _matches_filters(
        self,
        entry: MemoryEntry,
        *,
        tags: list[str],
        kind: str | None,
        phase: str | None,
    ) -> bool:
        if tags and not any(tag in entry.tags for tag in tags):
            return False
        if kind and (entry.kind or "") != kind:
            return False
        if phase and (entry.phase or "") != phase:
            return False
        return True

    # ------------------------------------------------------------------
    # Row/index helpers
    # ------------------------------------------------------------------

    def _get_row_by_id(self, memory_id: int) -> sqlite3.Row | None:
        if not self._sqlite_ready:
            return None
        try:
            with self._connect() as conn:
                return conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        except sqlite3.Error as e:
            logger.warning("SQLite read lookup failed: %s", e)
            return None

    @staticmethod
    def _truncate_content(content: str) -> str:
        if len(content) > _MAX_READ_CHARS:
            return content[:_MAX_READ_CHARS] + "\n[...truncated]"
        return content

    @staticmethod
    def _serialize_entry_file(entry: MemoryEntry, content: str) -> str:
        metadata = {
            "id": entry.id,
            "summary": entry.summary,
            "created_at": entry.created_at,
            "tags": entry.tags,
            "file": entry.file,
            "kind": entry.kind,
            "phase": entry.phase,
            "agent": entry.agent,
            "run_id": entry.run_id,
            "source_path": entry.source_path,
        }
        metadata_json = json.dumps(metadata, indent=2, sort_keys=True)
        body = content.lstrip("\n")
        return f"---\n{metadata_json}\n---\n\n{body}"

    def _load_entry_from_file(
        self,
        path: Path,
        *,
        fallback_entry: MemoryEntry | None = None,
    ) -> tuple[MemoryEntry, str]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.warning("Failed to read memory entry %s: %s", path, e)
            if fallback_entry is not None:
                return fallback_entry, ""
            return self._entry_from_path_name(path), ""

        metadata, content = self._parse_entry_file(text)
        if metadata is not None:
            raw = dict(metadata)
            raw.setdefault("file", path.name)
            entry = self._to_entry(raw)
            if entry.id <= 0 and fallback_entry is not None:
                entry.id = fallback_entry.id
            if not entry.file:
                entry.file = path.name
            return entry, content

        if fallback_entry is not None:
            return fallback_entry, text

        return self._entry_from_path_name(path), text

    @staticmethod
    def _entry_from_path_name(path: Path) -> MemoryEntry:
        match = re.match(r"^(\d+)_", path.name)
        memory_id = int(match.group(1)) if match else 0
        summary = path.stem.split("_", 1)[1].replace("_", " ") if "_" in path.stem else path.stem
        return MemoryEntry(
            id=memory_id,
            tags=[],
            summary=summary,
            created_at="",
            file=path.name,
        )

    @staticmethod
    def _parse_entry_file(text: str) -> tuple[dict[str, Any] | None, str]:
        if not text.startswith("---\n"):
            return None, text
        marker = "\n---\n"
        end = text.find(marker, 4)
        if end == -1:
            return None, text
        metadata_text = text[4:end].strip()
        body = text[end + len(marker):].lstrip("\n")
        try:
            metadata = json.loads(metadata_text)
        except json.JSONDecodeError:
            return None, text
        if not isinstance(metadata, dict):
            return None, text
        return metadata, body

    @staticmethod
    def _query_words(query: str) -> list[str]:
        return [word for word in re.findall(r"[a-z0-9_]+", query.lower()) if word]

    def _fts_query(self, query: str) -> str:
        words = self._query_words(query)
        parts: list[str] = []
        phrase = " ".join(words).strip()
        if len(words) > 1 and phrase:
            parts.append(f'"{phrase}"')
        parts.extend(words)
        return " OR ".join(parts)

    @staticmethod
    def _normalize_tag(tag: str) -> str:
        tag = str(tag or "").strip().lower()
        tag = re.sub(r"\s+", "_", tag)
        tag = re.sub(r"[^a-z0-9_\-]", "", tag)
        return tag

    @classmethod
    def _normalize_tags(cls, tags: list[str] | str | None) -> list[str]:
        if not tags:
            return []
        tag_values = [tags] if isinstance(tags, str) else tags
        seen: set[str] = set()
        normalized: list[str] = []
        for tag in tag_values:
            clean = cls._normalize_tag(tag)
            if clean and clean not in seen:
                normalized.append(clean)
                seen.add(clean)
        return normalized

    @staticmethod
    def _normalize_optional(value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @classmethod
    def _normalize_kind(cls, value: str | None) -> str | None:
        clean = cls._normalize_optional(value)
        if clean is None:
            return None
        clean = cls._normalize_tag(clean)
        if not clean:
            return None
        if clean in MEMORY_KINDS:
            return clean
        return _KIND_ALIASES.get(clean, clean)

    @classmethod
    def _kind_filter_values(cls, value: str) -> list[str]:
        canonical = cls._normalize_kind(value)
        if canonical is None:
            return []
        values = {canonical}
        values.update(alias for alias, target in _KIND_ALIASES.items() if target == canonical)
        return sorted(values)

    @staticmethod
    def _tag_blob(tags: list[str]) -> str:
        return "|" + "|".join(tags) + "|" if tags else "|"

    @staticmethod
    def _escape_like(value: str) -> str:
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    @classmethod
    def _like_pattern(cls, value: str) -> str:
        return f"%{cls._escape_like(value)}%"

    @staticmethod
    def _slugify(text: str, max_len: int = 40) -> str:
        """Convert text to a filename-safe slug."""
        slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
        return slug[:max_len] if slug else "entry"

    @classmethod
    def _entry_from_row(cls, row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            tags=cls._decode_tags_json(row["tags_json"]),
            summary=row["summary"],
            created_at=row["created_at"],
            file=row["file"],
            kind=cls._normalize_kind(row["kind"]),
            phase=row["phase"],
            agent=row["agent"],
            run_id=row["run_id"],
            source_path=row["source_path"],
        )

    @classmethod
    def _decode_tags_json(cls, tags_json: str | None) -> list[str]:
        if not tags_json:
            return []
        try:
            tags = json.loads(tags_json)
        except (TypeError, json.JSONDecodeError):
            logger.warning("Invalid memory tags_json, treating tags as empty")
            return []
        if not isinstance(tags, list):
            logger.warning("Invalid memory tags_json type, treating tags as empty")
            return []
        return cls._normalize_tags(tags)

    @classmethod
    def _to_entry(cls, raw: dict[str, Any]) -> MemoryEntry:
        return MemoryEntry(
            id=int(raw.get("id", 0)),
            tags=cls._normalize_tags(raw.get("tags", [])),
            summary=raw.get("summary", ""),
            created_at=raw.get("created_at", ""),
            file=raw.get("file", ""),
            kind=cls._normalize_kind(raw.get("kind")),
            phase=cls._normalize_optional(raw.get("phase")),
            agent=cls._normalize_optional(raw.get("agent")),
            run_id=cls._normalize_optional(raw.get("run_id")),
            source_path=cls._normalize_optional(raw.get("source_path")),
        )

    @staticmethod
    def _thread_lock_for_path(path: Path) -> threading.RLock:
        key = str(path.resolve())
        with _THREAD_LOCKS_GUARD:
            lock = _THREAD_LOCKS.get(key)
            if lock is None:
                lock = threading.RLock()
                _THREAD_LOCKS[key] = lock
            return lock

    @contextmanager
    def _lock(self) -> Iterator[None]:
        """Acquire process-local and file locks for index/DB writes and migration."""
        self._ensure_dirs()
        lock_path = self._base / ".index.lock"
        with self._thread_lock:
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


def remember_text(
    workspace: str,
    content: str,
    *,
    tags: list[str] | str | None,
    summary: str,
    kind: str | None = None,
    phase: str | None = None,
    agent: str | None = None,
    run_id: str | None = None,
    source_path: str | None = None,
) -> int | None:
    """Store a memory unless an equivalent artifact-backed entry already exists."""
    content = (content or "").strip()
    if not content:
        return None

    store = MemoryStore(workspace)
    source_path = (source_path or "").strip() or None
    normalized_summary = (summary or "").strip()
    normalized_phase = store._normalize_optional(phase)
    normalized_kind = store._normalize_kind(kind)
    for entry in store.list_recent(limit=1000):
        if source_path is not None and entry.source_path != source_path:
            continue
        if source_path is None:
            if entry.summary != normalized_summary:
                continue
            if (entry.kind or "") != (normalized_kind or ""):
                continue
            if (entry.phase or "") != (normalized_phase or ""):
                continue
        existing = store._read_content(entry.id)
        if existing is not None and existing.strip() == content:
            return entry.id

    return store.store(
        content=content,
        tags=tags,
        summary=normalized_summary,
        kind=kind,
        phase=normalized_phase,
        agent=agent,
        run_id=run_id,
        source_path=source_path,
    )


def remember_workspace_file(
    workspace: str,
    source_path: str,
    *,
    tags: list[str] | str | None,
    summary: str,
    kind: str | None = None,
    phase: str | None = None,
    agent: str | None = None,
    run_id: str | None = None,
) -> int | None:
    """Store a workspace-relative file as memory if it exists and is non-empty."""
    source_path = (source_path or "").strip()
    if not source_path:
        return None

    workspace_root = Path(workspace).resolve()
    raw_path = Path(source_path)
    if raw_path.is_absolute():
        return None

    path = (workspace_root / raw_path).resolve()
    try:
        safe_source_path = path.relative_to(workspace_root).as_posix()
    except ValueError:
        return None

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not content.strip():
        return None
    return remember_text(
        workspace,
        content,
        tags=tags,
        summary=summary,
        kind=kind,
        phase=phase,
        agent=agent,
        run_id=run_id,
        source_path=safe_source_path,
    )
