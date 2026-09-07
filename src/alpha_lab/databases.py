"""Generic disk-backed store for Pydantic records.

Canonical records live as ``records/<key>{suffix}`` (the model's serialized form;
the key is store-assigned, never a model field). A derived SQLite index — built
with ``sqlite-utils`` — is rebuilt from them and backs the query surface
(``eq``/``contains``/``neq`` → ``select``, plus ``fulltext_search``). An optional
``EmbeddingStore`` adds ``embedding_search``. The record files are the source of
truth; the indexes are disposable.
"""

from __future__ import annotations

import fcntl
import json
import logging
import threading
from collections.abc import Iterable, Iterator
from contextlib import closing, contextmanager
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar
from weakref import WeakValueDictionary

import numpy as np
import sqlite_utils
from numpy.typing import NDArray

from alpha_lab.embeddings import EmbeddingError, EmbeddingStore
from alpha_lab.models.codecs import ModelCodec, get_model_codec
from alpha_lab.models.sqlite import SQLiteModel
from alpha_lab.utils import atomic_write

if TYPE_CHECKING:
    from alpha_lab.git import GitRepository

ModelT = TypeVar("ModelT", bound=SQLiteModel)

Fragment = tuple[str, list[Any]]
"""A SQL ``WHERE`` snippet plus its bound parameters."""

logger = logging.getLogger("alpha_lab.databases")

_THREAD_LOCKS: WeakValueDictionary[str, threading.RLock] = WeakValueDictionary()
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock_for_path(path: Path) -> threading.RLock:
    """One reentrant lock per resolved path, shared across stores on that path."""
    key = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _THREAD_LOCKS[key] = lock
        return lock


def logical_or(*fragments: Fragment) -> Fragment:
    """Combine fragments into a single ``(a OR b OR …)`` fragment."""
    return _combine("OR", fragments)


def logical_and(*fragments: Fragment) -> Fragment:
    """Combine fragments into a single ``(a AND b AND …)`` fragment."""
    return _combine("AND", fragments)


def _combine(operator: str, fragments: tuple[Fragment, ...]) -> Fragment:
    if not fragments:
        raise ValueError("need at least one fragment to combine")
    clauses = [clause for clause, _ in fragments]
    params = [p for _, fragment_params in fragments for p in fragment_params]
    return "(" + f" {operator} ".join(clauses) + ")", params


class ModelDB(Generic[ModelT]):
    """Disk-backed store of ``model_type`` records with a derived SQLite index
    and, optionally, a vector index for embedding search."""

    def __init__(
        self,
        root: Path,
        model_type: type[ModelT],
        codec: ModelCodec | None = None,
        embedding: str | EmbeddingStore | None = "text-embedding-3-large",
        repo: GitRepository | None = None,
        build: bool | Literal["force"] = False,
    ) -> None:
        """Bind to the store rooted at ``root``.

        Args:
            root: Directory holding ``records/``, ``index.db``, and ``embeddings/``.
            model_type: The ``SQLiteModel`` subclass stored here.
            codec: Record (de)serializer; defaults to the model's registered codec.
            embedding: An ``EmbeddingStore``, a model name to build one from (rooted
                at ``root/embeddings``), or ``None`` to disable embedding search.
            repo: Optional git repository backing the records for versioning/commits.
            build: Eagerly :meth:`build` the derived indexes at construction —
                ``True`` (re-derive + gap-fill) or ``"force"`` (full rebuild +
                re-embed). Defaults to ``False``: construction stays cheap and
                side-effect-free, deferring to the lazy first-use build. The run's
                writer store (``deps.memory_store``) opts into ``True``.
        """
        self._root = root
        self._model_type = model_type
        self._embedding = (
            EmbeddingStore(root=root / "embeddings", model=embedding)
            if isinstance(embedding, str)
            else embedding
        )
        self._codec = get_model_codec(model_type) if codec is None else codec
        self._repo = repo
        self._built = False
        self._record_dir = root / "records"
        self._index_path = root / "index.db"

        # A sqlite3 connection may only be used on the thread that created it, so
        # each thread gets its own index handle (see the ``index`` property).
        self._thread_local = threading.local()
        self._thread_lock = _thread_lock_for_path(root)

        # Open flock handle while the cross-process lock is held; ``None`` when
        # not held. Doubles as the reentrancy sentinel so a nested same-thread
        # ``_lock()`` (e.g. ``add`` -> ``set``) doesn't re-flock a second fd.
        self._process_lock: Any = None

        if build:
            self.build(index=build, embeddings=build)

    @contextmanager
    def _lock(self) -> Iterator[None]:
        """Process-local + cross-process (file) write lock, held during writes.

        Reentrant: the ``_thread_lock`` serializes threads, and ``_process_lock``
        tracks whether this process already holds the flock so a nested call
        (``add`` -> ``set``) doesn't try to re-acquire it on a second fd (which
        would self-deadlock). The lock file lives beside the root so a fresh clone
        into an empty root isn't blocked by a non-empty directory.
        """
        self._root.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self._root.parent / f"{self._root.name}.lock"
        with self._thread_lock:
            acquired = self._process_lock is None
            if acquired:
                self._process_lock = open(lock_path, "w")
                fcntl.flock(self._process_lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if acquired:
                    fcntl.flock(self._process_lock, fcntl.LOCK_UN)
                    self._process_lock.close()
                    self._process_lock = None

    # ------------------------------------------------------------------
    # Record store (source of truth; never touches the indexes)
    # ------------------------------------------------------------------

    def get(self, key: int, default: Any = MISSING) -> ModelT:
        """Load the record stored at ``key``.

        Args:
            key: Store-assigned record key.
            default: Returned when ``key`` is absent; if omitted, a missing key raises.

        Returns:
            The decoded record, or ``default`` when ``key`` is absent and one was given.

        Raises:
            KeyError: If ``key`` is absent and no ``default`` was provided.
        """
        path = self._path(key)
        if path.exists():
            return self._codec.decode(path.read_text(encoding="utf-8"))
        if default is MISSING:
            raise KeyError(key)
        return default

    def set(
        self,
        key: int,
        value: ModelT,
        message: str | None = None,
        commit: bool | None = None,
        embedding: NDArray[np.floating] | None = None,
    ) -> None:
        """Write ``value`` at ``key`` (record + derived index/vector).

        The embedding runs before the lock, so a failure never half-writes a record.

        Args:
            key: Record key to write (overwrites any existing record at ``key``).
            value: The record to store.
            message: Commit message; defaults to ``"<Model> #<key>"``.
            commit: Whether to git-commit; ``None`` commits when the repo can.
            embedding: A precomputed vector (e.g. from :meth:`add`, which already
                embedded for its similarity check) reused so it isn't computed
                twice; when ``None`` and the record has embeddable text, the vector
                is produced here.

        Raises:
            ValueError: If ``embedding`` is provided but this store has no EmbeddingStore.
            EmbeddingError: If producing the record's embedding fails.
            RuntimeError: If ``commit`` is requested but no committable repo is attached.
        """
        if commit and self._repo is None:
            msg = f"{type(self).__name__} instance must have a GitRepository attached when commit=True"
            raise RuntimeError(msg)
        if commit and not self._repo.can_commit:
            # Fail fast, before any write, so a doomed commit can't leave an uncommitted record.
            raise RuntimeError("Attached GitRepository is not configured to make commits")
        if embedding is not None and self._embedding is None:
            raise ValueError("embedding= provided but this store has no EmbeddingStore")

        vector, fingerprint = embedding, None
        if self._embedding is not None:
            text = value.get_embedded_text()
            if text is None:
                vector = None  # model has no embedded fields: store without a vector
            else:
                fingerprint = self._embedding.get_fingerprint(text)
                if vector is None:
                    vector = self._embedding.embed(text)

        with self._lock():
            self._record_dir.mkdir(parents=True, exist_ok=True)
            path = self._path(key)
            # Derived rows first, canonical record last: a crash mid-write leaves at
            # most an orphan index/vector row (skipped on read, overwritten on key
            # reuse), never a published record the index fails to cover.
            self._write(self.index, key, value)
            if vector is not None:
                self._embedding.write(key, fingerprint, vector)
            atomic_write(path, self._codec.encode(value))
            if commit is None:
                commit = self._repo is not None and self._repo.can_commit
            if commit:
                self._repo.add(path.relative_to(self._root))
                if self._repo.has_staged_changes():
                    self._repo.commit(message or f"{self._model_type.__name__} #{key}")

    def add(
        self,
        value: ModelT,
        message: str | None = None,
        commit: bool | None = None,
        max_similarity: float | None = None,
    ) -> int:
        """Store ``value`` under a fresh key and return it.

        Allocates the key inside the write lock so concurrent writers can't pick
        the same one. When ``max_similarity`` is set, the record is embedded once,
        rejected if its cosine similarity to an existing record exceeds it, and
        that same vector is reused for storage (no second embedding call).

        Args:
            value: The record to store.
            message: Commit message; defaults to ``"<Model> #<key>"``.
            commit: Whether to git-commit; ``None`` commits when the repo can.
            max_similarity: If set, reject the write when the nearest existing
                record's cosine similarity exceeds it (``None`` disables the check).

        Returns:
            The store-assigned key of the new record.

        Raises:
            EmbeddingError: If producing the record's embedding fails (the write is aborted).
            ValueError: If ``max_similarity`` is set and the record is too similar
                to an existing one.
            RuntimeError: If ``commit`` is requested but no committable repo is attached.
        """
        if commit and self._repo is None:
            msg = f"{type(self).__name__} instance must have a GitRepository attached when commit=True"
            raise RuntimeError(msg)

        vector: NDArray[np.floating] | None = None
        if self._embedding is not None and max_similarity is not None:
            try:
                vector = self.embed(value)
            except ValueError:
                vector = None  # no embeddable text: nothing to compare against

        with self._lock():
            keys = list(self.keys())
            if max_similarity is not None and vector is not None and keys:
                index, scores = self._embedding.compare(vector, keys)
                if scores.size:
                    best = int(scores.argmax())
                    neighbor, similarity = int(index[best]), float(scores[best])
                    if similarity > max_similarity:
                        msg = (
                            f"record too similar to #{neighbor} ({similarity=:.3f} > "
                            f"{max_similarity=}); pass a larger max_similarity to permit"
                        )
                        raise ValueError(msg)
            key = max(keys, default=0) + 1
            self.set(key, value, message, commit, embedding=vector)
        return key

    def keys(self) -> Iterator[int]:
        """Yield every stored key in ascending order."""
        if not self._record_dir.is_dir():
            return
        for path in sorted(self._record_dir.glob(f"*{self._codec.suffix}"), key=lambda p: int(p.stem)):
            yield int(path.stem)

    def values(self) -> Iterator[ModelT]:
        """Yield every stored record in key order."""
        for key in self.keys():
            yield self[key]

    def items(self) -> Iterator[tuple[int, ModelT]]:
        """Yield every ``(key, record)`` pair in key order."""
        for key in self.keys():
            yield key, self[key]

    def __getitem__(self, key: int) -> ModelT:
        return self.get(key)

    def __contains__(self, key: int) -> bool:
        return self._path(key).exists()

    def __iter__(self) -> Iterator[ModelT]:
        yield from self.values()

    @property
    def repo(self) -> GitRepository | None:
        return self._repo

    @property
    def root(self) -> Path:
        return self._root
    
    @property
    def embedding(self) -> EmbeddingStore | None:
        return self._embedding

    @property
    def model_type(self) -> type[ModelT]:
        return self._model_type

    # ------------------------------------------------------------------
    # Query surface (fragments -> select; fulltext/embedding are runners)
    # ------------------------------------------------------------------

    def eq(self, value: Any, fields: list[str] | None = None) -> Fragment:
        """Build a fragment matching ``value`` in scalar column(s).

        Args:
            value: The value to match.
            fields: Scalar columns to match against (OR'd); defaults to all scalar columns.

        Returns:
            A ``(clause, params)`` WHERE fragment.

        Raises:
            ValueError: If any named field is not a scalar column.
        """
        scalar = [n for n, c in self.model_type.sqlite_fields() if not c.fulltext and not c.multi_value]
        names = fields if fields is not None else scalar
        wrong = [n for n in names if n not in scalar]
        if wrong:
            msg = f"field(s) {wrong} are not scalar columns"
            raise ValueError(msg)
        return "(" + " OR ".join(f"{n} = ?" for n in names) + ")", [value] * len(names)

    def neq(self, value: Any, fields: list[str] | None = None) -> Fragment:
        """Negation of :meth:`eq`."""
        clause, params = self.eq(value, fields)
        return f"NOT {clause}", params

    def contains(self, value: Any, fields: list[str] | None = None) -> Fragment:
        """Build a fragment matching records whose set column(s) contain ``value``.

        Args:
            value: The membership value to look for.
            fields: Set (multi-value) columns to search (OR'd); defaults to all set columns.

        Returns:
            A ``(clause, params)`` WHERE fragment.

        Raises:
            ValueError: If any named field is not a set column.
        """
        sets = [n for n, c in self.model_type.sqlite_fields() if c.multi_value]
        names = fields if fields is not None else sets
        wrong = [n for n in names if n not in sets]
        if wrong:
            msg = f"field(s) {wrong} are not set columns"
            raise ValueError(msg)
        return (
            "(" + " OR ".join(f"{n} LIKE ?" for n in names) + ")",
            [f"%,{value},%"] * len(names),
        )

    def select(self, *fragments: Fragment) -> Iterator[int]:
        """Query keys of records matching all ``fragments``.

        Args:
            *fragments: WHERE fragments (from :meth:`eq`/:meth:`neq`/:meth:`contains`),
                combined with ``AND``; none means match all records.

        Yields:
            The key of each matching record.
        """
        sql = "SELECT key FROM records"
        params: list[Any] = []
        if fragments:
            sql += " WHERE " + " AND ".join(clause for clause, _ in fragments)
            params = [p for _, fragment_params in fragments for p in fragment_params]
        # Materialize before yielding: a lazily-consumed cursor keeps its read
        # transaction (SQLite's SHARED lock) open for as long as the caller
        # iterates — or indefinitely, if the iterator is abandoned mid-
        # consumption (top-k islice/break in memory.search) while its frame
        # stays alive. With searches running on every agent turn, that
        # coverage starves index writers into "database is locked" failures.
        yield from [row["key"] for row in self.index.query(sql, params)]

    def filter(
        self,
        include: dict[str, Any] | None = None,
        exclude: dict[str, Any] | None = None,
        start: int = 0,
        limit: int | None = None,
    ) -> list[int]:
        """Return keys matching ``include`` (AND) and not ``exclude``, paginated.

        Each ``{field: value(s)}`` entry becomes an ``eq`` (scalar column) or
        ``contains`` (sequence column) clause; multiple values OR within a field.
        Ranking is left to the ``*_search`` runners.

        Args:
            include: Filters that must all match (AND across fields, OR within a field).
            exclude: Filters that must not match (the whole group is negated).
            start: Number of leading keys to skip.
            limit: Maximum number of keys to return; ``None`` returns all remaining.

        Returns:
            The matching keys, in key order, after pagination.
        """
        sequences = {n for n, c in self.model_type.sqlite_fields() if c.multi_value}
        fragments: list[Fragment] = []
        for filters, negate in ((include, False), (exclude, True)):
            if not filters:
                continue
            field_fragments = []
            for field, value in filters.items():
                values = value if isinstance(value, (list, tuple)) else [value]
                verb = self.contains if field in sequences else self.eq
                field_fragments.append(logical_or(*(verb(v, [field]) for v in values)))
            clause, params = logical_and(*field_fragments)
            fragments.append((f"NOT {clause}", params) if negate else (clause, params))
        keys = list(self.select(*fragments))
        return keys[start:] if limit is None else keys[start:start + limit]

    def fulltext_search(
        self, text: str, keys: Iterable[int] | None = None, raw_fts5: bool = False
    ) -> Iterator[tuple[int, float]]:
        """Keyword-search records, most relevant first.

        Args:
            text: The query. Searched as OR-joined keywords unless ``raw_fts5``.
            keys: Candidate keys to restrict the search to; ``None`` searches all.
            raw_fts5: Pass ``text`` through as an FTS5 expression instead. FTS5
                then parses operators, prefixes and column filters — and rejects
                text it cannot parse.

        Yields:
            ``(key, score)`` pairs in relevance order (more relevant first).
        """
        if not raw_fts5:
            # FTS5 barewords admit only letters, digits and underscore, so any
            # other character is either syntax or a parse error. Quoting each
            # keyword removes that reading (embedded quotes are escaped by
            # doubling them, per the FTS5 string grammar), and joining with OR
            # lets bm25 rank partial matches instead of requiring every keyword
            # — which prose queries, especially the ones that fall back here
            # when embeddings are unavailable, would almost never satisfy.
            text = " OR ".join('"' + piece.replace('"', '""') + '"' for piece in text.split())

        where, where_args = None, None
        if keys is not None:
            # Pass candidates as a single JSON-array parameter (via json_each) rather
            # than one bound variable per key, which hits SQLite's variable limit.
            where = "key in (select value from json_each(:keys))"
            where_args = {"keys": json.dumps(list(keys))}
        # Materialized (list) so the FTS cursor always drains — see select().
        yield from [
            (row["key"], row["rank"])
            for row in self.index["records"].search(
                text, columns=["key"], where=where, where_args=where_args, include_rank=True
            )
        ]

    def embedding_search(
        self, text: str, keys: Iterable[int] | None = None
    ) -> Iterator[tuple[int, float]]:
        """Search records by embedding similarity, most relevant first.

        Args:
            text: The query string (embedded on the fly).
            keys: Candidate keys to restrict the search to; ``None`` searches all.

        Yields:
            ``(key, score)`` pairs in descending similarity order.

        Raises:
            RuntimeError: If this store has no ``EmbeddingStore``.
            EmbeddingError: If embedding the query fails.
        """
        if self._embedding is None:
            raise RuntimeError("embedding_search requires an EmbeddingStore")

        index = (
            None
            if keys is None
            else np.fromiter(keys, dtype=int)
            if isinstance(keys, Iterator)
            else np.asarray(keys)
        )
        if index is not None and index.size == 0:
            return

        index, scores = self._embedding.compare(text, index)
        for i in np.argsort(scores)[::-1]:
            yield int(index[i]), float(scores[i])

    def embed(self, value: ModelT) -> NDArray[np.floating]:
        """Embed ``value`` from the text of its embedded fields.

        Args:
            value: The record to embed.

        Returns:
            The embedding vector for the record's embedded text.

        Raises:
            RuntimeError: If this store has no ``EmbeddingStore``.
            ValueError: If ``value``'s model declares no embedded fields.
            EmbeddingError: If producing the embedding fails.
        """
        if self._embedding is None:
            raise RuntimeError("embed requires an EmbeddingStore")
        text = value.get_embedded_text()
        if text is None:
            raise ValueError("cannot embed a model with no embedded fields")
        return self._embedding.embed(text)

    # ------------------------------------------------------------------
    # Internals: derived indexes
    # ------------------------------------------------------------------

    def build(
        self,
        index: bool | Literal["force"] = True,
        embeddings: bool | Literal["force"] = True,
    ) -> None:
        """(Re)derive the disposable indexes from the canonical records.

        ``records/`` is the source of truth; ``index.db`` and the embedding pages
        are rebuilt from it. By default idempotent and additive: SQLite rows are
        upserted (never dropped, so a rebuild can't race a concurrent writer), and
        embeddings are gap-filled — a record is (re)embedded only when its vector
        is missing or its stored fingerprint no longer matches the record.

        Args:
            index: ``True`` upsert every record's SQLite/FTS row, ``"force"`` drop
                and recreate the table + FTS first (purges orphan rows, rebuilds
                FTS clean), ``False`` leave the SQLite index untouched.
            embeddings: ``True`` gap-fill missing/stale vectors, ``"force"``
                re-embed every record, ``False`` leave embeddings untouched.

        Serialized on ``_lock()`` (process + cross-process), so concurrent rebuilds
        can't corrupt the shared SQLite/vector pages. ``_lock()`` is reentrant, so a
        lazy build triggered from a write path (which already holds it) can't
        self-deadlock.
        """
        with self._lock():
            if self._repo is not None:
                self._repo.setup()
            self._record_dir.mkdir(parents=True, exist_ok=True)
            if index:
                with closing(sqlite_utils.Database(self._index_path)) as db:
                    db.execute("PRAGMA busy_timeout = 5000")  # wait out a concurrent writer
                    self._build_index(db, force=index == "force")
                self._built = True
            if embeddings and self._embedding is not None:
                self._build_embeddings(force=embeddings == "force")

    def _build_index(self, db: sqlite_utils.Database, force: bool = False) -> None:
        """Upsert every record's row; when ``force``, drop and recreate first.

        ``force`` drops the ``records`` table and its FTS companion before
        recreating them, clearing orphan rows and rebuilding the FTS index clean.
        """
        columns = list(self.model_type.sqlite_fields())
        if force:
            db["records_fts"].drop(ignore=True)
            db["records"].drop(ignore=True)
        if not db["records"].exists():
            schema = {"key": int, **{n: c.sqlite_type for n, c in columns}}
            db["records"].create(schema, pk="key")
            fulltext = [n for n, c in columns if c.fulltext]
            if fulltext:
                db["records"].enable_fts(fulltext, create_triggers=True)
        for key, value in self.items():
            self._write(db, key, value)

    def _build_embeddings(self, force: bool = False) -> None:
        """Gap-fill (or, when ``force``, recompute) embedding vectors from records."""
        if self._embedding is None:
            return
        for key, value in self.items():
            text = value.get_embedded_text()
            if text is None:
                continue  # model has no embedded fields
            fingerprint = self._embedding.get_fingerprint(text)
            if not force:
                existing = self._embedding.get(key)
                if existing is not None and float(existing[0]) == fingerprint:
                    continue  # present and current
            try:
                vector = self._embedding.embed(text)
            except EmbeddingError:  # transient failure: leave owed, retry on next build
                logger.warning("Embedding failed for record #%d; leaving vector owed", key, exc_info=True)
                continue
            self._embedding.write(key, fingerprint, vector)

    @property
    def index(self) -> sqlite_utils.Database:
        """The derived SQLite index — returned as a **per-thread** handle.

        A ``sqlite3`` connection may only be used on the thread that created it,
        and this store is shared across the agents' tool-execution threads. The
        index file is built once on first use (:meth:`build`); each thread then
        opens (and caches) its own connection to it here.
        """
        if not self._built:
            self.build()
        db = getattr(self._thread_local, "index", None)
        if db is None:
            db = sqlite_utils.Database(self._index_path)
            db.execute("PRAGMA busy_timeout = 5000")  # wait out a concurrent writer
            self._thread_local.index = db
        return db

    def _write(self, db: sqlite_utils.Database, key: int, value: ModelT) -> None:
        """Upsert ``value``'s scalar/multi-value columns into the SQLite index."""
        dumped = value.model_dump(mode="json")
        row: dict[str, Any] = {"key": key}
        for name, column in self.model_type.sqlite_fields():
            cell = dumped[name]
            row[name] = ("," + ",".join(cell) + "," if cell else ",") if column.multi_value else cell
        db["records"].upsert(row, pk="key")

    def _path(self, key: int) -> Path:
        return self._record_dir / f"{key}{self._codec.suffix}"
