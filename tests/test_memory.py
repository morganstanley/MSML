"""Tests for the ModelDB-backed memory layer."""

from __future__ import annotations

import sqlite3
import sys
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx2
import numpy as np
import pytest
from openai import APIConnectionError

from alpha_lab import deps
from alpha_lab.databases import ModelDB
from alpha_lab.embeddings import EmbeddingError, EmbeddingModel, EmbeddingStore
from alpha_lab.git import GitRepository, GitSpec
from alpha_lab.memory import (
    Memory,
    MemoryStore,
    remember_workspace_file,
)
from alpha_lab.utils import RetryTimer

_EMBEDDING = "text-embedding-3-large"
_REAL_EMBED = EmbeddingStore.embed


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    """A repo-less store (records + index + embeddings, no git commits)."""
    root = tmp_path / "ws" / ".memory"
    return MemoryStore(ModelDB(root=root, model_type=Memory, embedding=_EMBEDDING))


@pytest.fixture()
def git_store(tmp_path: Path) -> MemoryStore:
    """A git-backed store, so commit/gitignore behavior can be inspected."""
    root = tmp_path / "ws" / ".memory"
    repo = GitRepository(root=root, spec=GitSpec(user_name="t", user_email="t@example.com"))
    db = ModelDB(root=root, model_type=Memory, embedding=_EMBEDDING, repo=repo)
    return MemoryStore(db)


def _tracked(root: Path) -> set[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    ).stdout
    return set(out.split())


def _memory(content: str, *, summary: str | None = None, kind: str = "fact", **fields) -> Memory:
    return Memory(content=content, summary=summary or content, kind=kind, **fields)


class TestAddAndGet:
    def test_add_writes_record_file(self, store: MemoryStore) -> None:
        key = store.add(_memory("hello world", summary="greeting", tags=["intro"]))
        assert key == 1
        assert (store.root / "records" / "1.json").is_file()
        memory = store.get(1)
        assert memory is not None
        assert memory.summary == "greeting"
        assert memory.tags == ("intro",)
        assert memory.kind == "fact"

    def test_key_increments(self, store: MemoryStore) -> None:
        assert store.add(_memory("a")) == 1
        assert store.add(_memory("b")) == 2
        assert store.add(_memory("c")) == 3

    def test_get_missing_returns_none(self, store: MemoryStore) -> None:
        store.add(_memory("a"))
        assert store.get(999) is None

    def test_tags_recorded(self, store: MemoryStore) -> None:
        store.add(_memory("x", summary="s", tags=["t", "phase3"]))
        assert "phase3" in store.get(1).tags

    def test_owner_defaults_to_os_user(self, store: MemoryStore) -> None:
        store.add(_memory("x", summary="s"))
        assert store.get(1).owner

    def test_iter_yields_key_memory_pairs(self, store: MemoryStore) -> None:
        store.add(_memory("a"))
        store.add(_memory("b", kind="idea"))
        assert [(key, memory.summary) for key, memory in store] == [(1, "a"), (2, "b")]


class TestGitBacking:
    def test_git_spec_none_identity_stays_none(self) -> None:
        spec = GitSpec(user_name=None, user_email=None)
        assert spec.user_name is None
        assert spec.user_email is None

    def test_git_spec_empty_string_reads_git_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Result:
            def __init__(self, value: str) -> None:
                self.returncode = 0
                self.stdout = value

        def fake_run(args, **kwargs):
            return _Result("Name\n" if args[-1] == "user.name" else "email@example.com\n")

        monkeypatch.setattr("alpha_lab.git.subprocess.run", fake_run)
        spec = GitSpec(user_name="", user_email="")
        assert spec.user_name == "Name"
        assert spec.user_email == "email@example.com"

    def test_can_commit_uses_git_identity_resolution(self, tmp_path: Path) -> None:
        root = tmp_path / "repo"
        repo = GitRepository(root=root, spec=GitSpec(user_name="t", user_email="t@example.com"))
        repo.setup()
        assert repo.can_commit is True

    def test_can_commit_false_without_identity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Result:
            returncode = 1

        root = tmp_path / "repo"
        repo = GitRepository(root=root, spec=GitSpec(user_name=None, user_email=None))
        repo.init(exist_ok=True)
        monkeypatch.setattr("alpha_lab.git.subprocess.run", lambda *args, **kwargs: _Result())
        assert repo.can_commit is False

    def test_has_staged_changes_reflects_index_not_worktree(self, tmp_path: Path) -> None:
        root = tmp_path / "repo"
        repo = GitRepository(root=root, spec=GitSpec(user_name="t", user_email="t@example.com"))
        repo.setup()
        assert repo.has_staged_changes() is False
        (root / "f.txt").write_text("x")
        assert repo.has_staged_changes() is False  # untracked file is not staged
        repo.add("f.txt")
        assert repo.has_staged_changes() is True   # now it is

    def test_setup_does_not_raise_on_unstaged_only_changes(self, tmp_path: Path) -> None:
        """Regression: a tracked file edited but left unstaged must not make
        setup() attempt an empty ``git commit``, which fails and raises a
        RuntimeError, stalling every memory operation for the rest of a run.
        """
        root = tmp_path / "repo"
        repo = GitRepository(root=root, spec=GitSpec(user_name="t", user_email="t@example.com"))
        repo.setup()  # fresh repo -> initial commit, HEAD exists
        rec = root / "rec.json"
        rec.write_text("v1")
        repo.add("rec.json")
        repo.commit("add rec")
        rec.write_text("v2")  # modify WITHOUT staging
        assert repo.is_dirty() is True             # working tree is dirty ...
        assert repo.has_staged_changes() is False  # ... but nothing is staged
        repo.setup()  # must NOT raise (previously raised on the empty commit)
        # setup() left the unstaged edit uncommitted; it did not fabricate a commit
        assert repo.has_staged_changes() is False
        assert repo.is_dirty() is True

    def test_records_committed_indexes_ignored(self, git_store: MemoryStore) -> None:
        git_store.add(_memory("a"))
        tracked = _tracked(git_store.root)
        assert "records/1.json" in tracked
        assert ".gitignore" in tracked
        assert not any(t == "index.db" or t.startswith("embeddings/") for t in tracked)

    def test_add_creates_commit(self, git_store: MemoryStore) -> None:
        git_store.add(_memory("a"))
        git_store.add(_memory("b"))
        count = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=git_store.root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        # initial commit (gitignore) + two adds
        assert int(count) >= 3

    def test_idempotent_set_ignores_unstaged_changes(self, git_store: MemoryStore) -> None:
        git_store.add(_memory("a"))
        before = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=git_store.root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        ignore = git_store.root / ".gitignore"
        ignore.write_text(ignore.read_text() + "# unrelated unstaged edit\n")
        memory = git_store.get(1)
        git_store._db.set(1, memory)
        after = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=git_store.root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        assert after == before
        assert "# unrelated unstaged edit" in ignore.read_text()


class TestEmbeddings:
    def test_store_uses_default_model_when_none(self, tmp_path: Path) -> None:
        store = EmbeddingStore(tmp_path, model=None)
        assert store.model.name == _EMBEDDING


class TestSearch:
    def test_default_order_is_newest_first(self, store: MemoryStore) -> None:
        for i in range(3):
            store.add(_memory(f"c{i}", summary=f"s{i}"))
        assert [key for key, _ in store.search()] == [3, 2, 1]

    def test_keywords_match(self, store: MemoryStore) -> None:
        store.add(_memory("cuda out of memory", summary="oom"))
        store.add(_memory("sharpe ratio detrending", summary="sharpe"))
        assert [key for key, _ in store.search("cuda memory", mode="fulltext")] == [1]

    def test_fulltext_accepts_punctuation(self, store: MemoryStore) -> None:
        # FTS5 barewords admit only letters, digits and underscore, so raw agent
        # text used to reach MATCH as syntax: a colon reads as a column filter
        # ("no such column"), a hyphen as a negated one, and "[", "." or an
        # unbalanced quote as a syntax error. fulltext quotes each keyword instead.
        store.add(_memory("tuning the gpu batch size for low cardinality X 1"))
        for query in (
            "token:gpu",
            "low-cardinality tuning",
            "X[:,1] gpu",
            "0.75 batch",
            'say "hi" gpu',
            "a AND (b OR",
            "NEAR(gpu batch)",
        ):
            store.search(query, mode="fulltext")  # must not raise

    def test_fulltext_matches_through_separators(self, store: MemoryStore) -> None:
        # A quoted keyword is one phrase: its separators are dropped by the
        # tokenizer, so it matches the same tokens the record was indexed with.
        store.add(_memory("tuning the gpu batch size for low cardinality"))
        assert [key for key, _ in store.search("low-cardinality", mode="fulltext")] == [1]
        assert [key for key, _ in store.search("gpu batch", mode="fulltext")] == [1]

    def test_fulltext_matches_any_keyword(self, store: MemoryStore) -> None:
        # Keywords are OR-joined and bm25 ranks the matches, so a long query
        # returns its most relevant records rather than nothing. Requiring every
        # keyword would match neither record here.
        store.add(_memory("gpu tuning notes", summary="gpu"))
        store.add(_memory("batch size notes", summary="batch"))
        assert {key for key, _ in store.search("gpu batch", mode="fulltext")} == {1, 2}

    def test_fulltext_finds_stored_text_containing_a_quote(self, store: MemoryStore) -> None:
        # Quoting a keyword means escaping any quote inside it by doubling; this
        # covers the write-then-search round trip, not just a hostile query.
        store.add(_memory('the agent said "cuda oom" during the sweep'))
        assert [key for key, _ in store.search('"cuda', mode="fulltext")] == [1]
        assert [key for key, _ in store.search('said "cuda oom"', mode="fulltext")] == [1]

    def test_fulltext_handles_a_long_prose_query(self, store: MemoryStore) -> None:
        # The shape that actually reaches this path: a whole task description,
        # hyphenated and punctuated, where requiring every keyword would match
        # nothing at all.
        store.add(_memory("low-cardinality features hurt the gbdt baseline", summary="gbdt"))
        store.add(_memory("unrelated note about slurm queues", summary="slurm"))
        query = (
            "forecast the 20-day horizon for the exchange-rate panel using "
            "autoregressive baselines (naive, gbdt) at 0.95 coverage, noting "
            "low-cardinality features and any validation split leakage"
        )
        assert [key for key, _ in store.search(query, mode="fulltext")] == [1]

    def test_fulltext_ranks_more_matching_keywords_first(self, store: MemoryStore) -> None:
        store.add(_memory("slurm queue notes", summary="slurm"))
        store.add(_memory("gbdt baseline on the exchange rate panel", summary="gbdt"))
        found = [key for key, _ in store.search("gbdt exchange panel", mode="fulltext")]
        assert found[0] == 2  # matches three keywords; key 1 matches none
        assert 1 not in found

    def test_raw_fts5_parses_the_expression(self, store: MemoryStore) -> None:
        store.add(_memory("gpu tuning notes", summary="gpu"))
        store.add(_memory("batch size notes", summary="batch"))
        # A trailing "*" marks a prefix term for FTS5; fulltext quotes it, and the
        # tokenizer then drops it, leaving the keyword "tun" — which matches no
        # whole token. Column filters likewise only apply on the raw path.
        assert [key for key, _ in store.search("tun*", mode="raw_fts5")] == [1]
        assert [key for key, _ in store.search("tun*", mode="fulltext")] == []
        assert [key for key, _ in store.search("summary:gpu", mode="raw_fts5")] == [1]

    def test_raw_fts5_surfaces_syntax_errors(self, store: MemoryStore) -> None:
        store.add(_memory("gpu tuning notes"))
        with pytest.raises(sqlite3.OperationalError):
            store.search("token:gpu", mode="raw_fts5")

    def test_unknown_mode_rejected(self, store: MemoryStore) -> None:
        with pytest.raises(ValueError):
            store.search("gpu", mode="keyword")

    def test_fulltext_with_include_narrows_via_json_each(self, store: MemoryStore) -> None:
        # A non-empty candidate set routes full-text narrowing through the
        # json_each(:keys) branch; both rows match the query, so the assertion
        # proves the tag filter actually restricted the candidates (key 2 dropped).
        store.add(_memory("cuda out of memory", summary="oom", tags=["gpu"]))
        store.add(_memory("cuda kernel crash", summary="crash", tags=["cpu"]))
        assert [key for key, _ in store.search("cuda", mode="fulltext", include={"tags": "gpu"})] == [1]

    def test_target_ranks_all_candidates(self, store: MemoryStore) -> None:
        store.add(_memory("gpu vram exhausted", summary="vram"))
        store.add(_memory("unrelated note", summary="note"))
        assert {key for key, _ in store.search("gpu memory", mode="embedded")} == {1, 2}

    def test_default_uses_fulltext_without_embeddings(self, tmp_path: Path) -> None:
        store = MemoryStore(ModelDB(root=tmp_path / ".memory", model_type=Memory, embedding=None))
        store.add(_memory("cuda out of memory", summary="oom"))
        store.add(_memory("unrelated note", summary="note"))
        assert [key for key, _ in store.search("cuda memory")] == [1]

    def test_include_tag_filter(self, store: MemoryStore) -> None:
        store.add(_memory("a", tags=["gpu"]))
        store.add(_memory("b", tags=["cpu"]))
        assert [key for key, _ in store.search(include={"tags": "gpu"})] == [1]

    def test_exclude_kind_filter(self, store: MemoryStore) -> None:
        store.add(_memory("a"))
        store.add(_memory("b", kind="idea"))
        assert [key for key, _ in store.search(exclude={"kind": "fact"})] == [2]

    def test_bounds(self, store: MemoryStore) -> None:
        for i in range(5):
            store.add(_memory(f"c{i}", summary=f"s{i}"))
        # Unupdated memories sort newest-first; start/limit slices that order.
        assert [key for key, _ in store.search(start=1, limit=2)] == [4, 3]

    def test_ranking_sources_mutually_exclusive(self, store: MemoryStore) -> None:
        store.add(_memory("a"))
        with pytest.raises(ValueError, match="recency"):
            store.search("x", mode="recency")

    def test_query_modes_require_query(self, store: MemoryStore) -> None:
        with pytest.raises(ValueError, match="requires a query"):
            store.search(mode="fulltext")

    def test_negative_bounds_rejected(self, store: MemoryStore) -> None:
        with pytest.raises(ValueError):
            store.search(start=-1)

    def test_unknown_filter_field_rejected(self, store: MemoryStore) -> None:
        store.add(_memory("a"))
        with pytest.raises(ValueError, match="not scalar columns"):
            store.search(include={"bogus": "x"})


class TestUpdate:
    def test_update_unions_tags(self, store: MemoryStore) -> None:
        store.add(_memory("a", tags=["x"]))
        updated = store.update(1, tags=["y"])
        assert set(updated.tags) == {"x", "y"}
        assert set(store.get(1).tags) == {"x", "y"}

    def test_update_missing_key_raises(self, store: MemoryStore) -> None:
        with pytest.raises(KeyError):
            store.update(999, summary="x")


class TestDedupe:
    def test_rejects_too_similar(self, store: MemoryStore) -> None:
        store.add(_memory("identical content", summary="first"))
        # Measure the score the next write would produce, then forbid anything that close.
        candidate = _memory("identical content", summary="second")
        _, scores = store._db.embedding.compare(store._db.embed(candidate), [1])
        similarity = float(scores[0])
        with pytest.raises(ValueError, match="too similar"):
            store.add(candidate, max_similarity=similarity - 0.01)

    def test_default_allows_duplicates(self, store: MemoryStore) -> None:
        store.add(_memory("identical content", summary="first"))
        assert store.add(_memory("identical content", summary="second")) == 2


class TestMemoryModel:
    def _memory(self) -> Memory:
        return Memory(
            kind="fact", summary="s", content="c", tags=("a",), sources=("e1",),
            created_at="t", updated_at="t",
        )

    def test_update_unions_sequences(self) -> None:
        updated = self._memory().update(tags=["b"], sources=["e2"])
        assert set(updated.tags) == {"a", "b"}
        assert set(updated.sources) == {"e1", "e2"}

    def test_update_rejects_unknown_field(self) -> None:
        with pytest.raises(ValueError, match="unknown field"):
            self._memory().update(bogus=1)

    def test_update_rejects_frozen_memory(self) -> None:
        memory = Memory(kind="fact", summary="s", content="c", tags=(), frozen=True,
                        created_at="t", updated_at="t")
        with pytest.raises(ValueError, match="frozen"):
            memory.update(summary="x")

    def test_tags_normalized(self) -> None:
        memory = Memory(kind="fact", summary="s", content="c",
                        tags=["Hello World", "hello_world"], created_at="t", updated_at="t")
        assert memory.tags == ("hello_world",)

    def test_str_renders_header_and_content(self) -> None:
        text = str(self._memory())
        assert text.startswith("# s")
        assert text.endswith("c")


class TestRememberWorkspaceFile:
    """``remember_workspace_file`` ingests a workspace file into the run's
    ``deps.memory_store`` (the autouse default); paths resolve under ``deps.workspace``."""

    def _write(self, name: str, body: str) -> None:
        p = deps.workspace / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)

    def test_ingests_relative_file(self) -> None:
        self._write("note.txt", "file body")
        key = remember_workspace_file("note.txt", summary="file", kind="fact", tags=["f"])
        memory = deps.memory_store.get(key)
        assert memory.content == "file body"
        assert memory.summary == "file"
        assert memory.sources == ("note.txt",)
        assert memory.run_id == "test"

    def test_empty_content_raises(self) -> None:
        self._write("empty.txt", "   ")
        with pytest.raises(ValueError):
            remember_workspace_file("empty.txt", summary="x", kind="fact")

    def test_path_outside_workspace_raises(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside.txt"
        outside.write_text("body")
        with pytest.raises(ValueError):
            remember_workspace_file(outside, summary="x", kind="fact")

    def test_relative_traversal_raises(self) -> None:
        # A relative path that escapes the workspace via ".." must be rejected.
        with pytest.raises(ValueError):
            remember_workspace_file("../escape.txt", summary="x", kind="fact")

    def test_exact_duplicate_returns_existing_id(self) -> None:
        self._write("dup.txt", "same body")
        first = remember_workspace_file("dup.txt", summary="s", kind="fact")
        second = remember_workspace_file("dup.txt", summary="s", kind="fact")
        assert second == first
        assert len(list(deps.memory_store)) == 1


class TestThreadSafety:
    """The store is shared across the agents' tool-execution threads."""

    def test_index_is_usable_from_another_thread(self, tmp_path: Path) -> None:
        """Regression: querying the index from a second thread must not raise.

        A ``sqlite3`` connection is bound to its creating thread, so a single
        cached connection raised ``ProgrammingError: SQLite objects created in a
        thread can only be used in that same thread`` on every cross-thread read
        (94% of memory calls during a real multi-agent run). Each thread must get
        its own handle.
        """
        db = ModelDB(root=tmp_path / "m", model_type=Memory, embedding=None)
        key = db.add(_memory("cross-thread lookup", summary="s", kind="fact"))
        # Build and cache the index connection on THIS thread first.
        assert list(db.select()) == [key]

        found: list[int] = []
        errors: list[Exception] = []

        def worker() -> None:
            try:
                found.extend(db.select())
            except Exception as exc:  # capture so the assertion can surface it
                errors.append(exc)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=30)

        assert not thread.is_alive(), "worker thread hung (possible deadlock)"
        assert not errors, f"index unusable from a worker thread: {errors!r}"
        assert found == [key]


class TestKeyAllocation:
    """Key allocation happens under the write lock, so concurrent adds can't collide."""

    def test_concurrent_adds_get_distinct_keys(self, store: MemoryStore) -> None:
        n = 20
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                store.add(_memory(f"content {i}", summary=f"s{i}"))
            except Exception as exc:  # capture so the assertion can surface it
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert not any(t.is_alive() for t in threads), "an add() thread hung"
        assert not errors, f"add() raised concurrently: {errors!r}"
        keys = list(store._db.keys())
        assert len(keys) == n
        assert len(set(keys)) == n  # no overwrite / collision


class TestBuildAndEmbeddings:
    """build() re-derives disposable indexes; embeddings gap-fill by content hash."""

    def test_embed_failure_aborts_write(
        self, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Embedding runs before the write, so a provider failure aborts cleanly
        # with nothing persisted (no half-written record).
        def boom(self: EmbeddingStore, text: str) -> np.ndarray:
            raise RuntimeError("simulated rate limit")

        monkeypatch.setattr(EmbeddingStore, "embed", boom)
        with pytest.raises(RuntimeError, match="rate limit"):
            store.add(_memory("body", summary="s"))
        assert list(store._db.keys()) == []

    def test_build_gap_fills_seeded_record(self, store: MemoryStore) -> None:
        db = store._db
        store.add(_memory("first", summary="s1"))  # builds index + embeds key 1
        # Simulate a git-seeded record: a record file with no vector on disk.
        seeded = _memory("seeded body", summary="s2")
        (db.root / "records" / "2.json").write_text(db._codec.encode(seeded))
        assert db.embedding.get(2) is None
        db.build(index=True, embeddings=True)
        assert db.embedding.get(2) is not None  # gap-filled from the record

    def test_fill_skips_unchanged_records(
        self, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store.add(_memory("stable content", summary="s"))
        calls = {"n": 0}
        real = EmbeddingStore.embed

        def counting(self: EmbeddingStore, text: str) -> np.ndarray:
            calls["n"] += 1
            return real(self, text)

        monkeypatch.setattr(EmbeddingStore, "embed", counting)
        store._db.build(index=False, embeddings=True)
        assert calls["n"] == 0  # present + hash matches -> no embedding calls

    def test_fill_refreshes_stale_vector(self, store: MemoryStore) -> None:
        db = store._db
        key = store.add(_memory("original text", summary="s"))
        before = db.embedding.get(key).copy()
        # Rewrite the record content out-of-band (same key), as a checkout would.
        changed = _memory("completely different content now", summary="s")
        (db.root / "records" / f"{key}.json").write_text(db._codec.encode(changed))
        db.build(index=False, embeddings=True)
        after = db.embedding.get(key)
        assert after is not None
        assert float(after[0]) != float(before[0])  # content-hash tag changed

    def test_force_reembeds_every_record(
        self, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store.add(_memory("a", summary="a"))
        store.add(_memory("b", summary="b"))
        calls = {"n": 0}
        real = EmbeddingStore.embed

        def counting(self: EmbeddingStore, text: str) -> np.ndarray:
            calls["n"] += 1
            return real(self, text)

        monkeypatch.setattr(EmbeddingStore, "embed", counting)
        store._db.build(index=False, embeddings="force")
        assert calls["n"] == 2  # every record re-embedded, hashes ignored

    def test_build_index_picks_up_new_record_file(self, store: MemoryStore) -> None:
        db = store._db
        store.add(_memory("first", summary="s1"))  # builds the index (key 1)
        new = _memory("second", summary="s2")
        (db.root / "records" / "2.json").write_text(db._codec.encode(new))
        db.build(index=True, embeddings=False)  # re-derive SQLite from records/
        assert 2 in set(db.select())

    def test_build_index_force_purges_orphan_rows(self, store: MemoryStore) -> None:
        db = store._db
        store.add(_memory("a", summary="a"))  # key 1
        store.add(_memory("b", summary="b"))  # key 2
        (db.root / "records" / "2.json").unlink()  # delete record -> row now orphaned
        db.build(index=True, embeddings=False)  # additive: orphan row survives
        assert 2 in set(db.select())
        db.build(index="force", embeddings=False)  # drop + recreate: orphan gone
        assert set(db.select()) == {1}

    def test_init_build_flag_controls_eager_build(self, tmp_path: Path) -> None:
        lazy = ModelDB(root=tmp_path / "lazy", model_type=Memory, embedding=None, build=False)
        assert lazy._built is False  # deferred to first use
        eager = ModelDB(root=tmp_path / "eager", model_type=Memory, embedding=None, build=True)
        assert eager._built is True  # built at construction


def test_set_commit_on_noncommittable_repo_fails_before_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(GitRepository, "can_commit", property(lambda self: False))
    root = tmp_path / "ws" / ".memory"
    repo = GitRepository(root=root, spec=GitSpec(user_name="t", user_email="t@example.com"))
    db = ModelDB(root=root, model_type=Memory, embedding=None, repo=repo)
    with pytest.raises(RuntimeError, match="not configured to make commits"):
        db.set(1, _memory("body", summary="s"), commit=True)
    assert list(db.keys()) == []  # failed fast: no record written


def test_set_embedding_without_store_raises(tmp_path: Path) -> None:
    db = ModelDB(root=tmp_path / "m", model_type=Memory, embedding=None)
    # Passing embedding= to an embedding-less store is misuse and must fail loudly.
    with pytest.raises(ValueError, match="no EmbeddingStore"):
        db.set(1, _memory("body", summary="s"), embedding=np.zeros(3072))


def test_embedding_store_roundtrip_and_hash(tmp_path: Path) -> None:
    es = EmbeddingStore(root=tmp_path / "emb", model=_EMBEDDING)
    vec = es.embed("hello world")
    es.write(5, 123.0, vec)
    row = es.get(5)
    assert row is not None
    assert row.shape == (es.model.dim + 1,)
    assert row.dtype == np.float32
    assert np.load(es.root / "0.npy", mmap_mode="r").shape[0] == 1024
    assert float(row[0]) == 123.0
    assert np.allclose(row[1:], vec)
    assert es.get(6) is None  # absent row


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_embedding_store_uses_model_dtype(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dtype: type[np.float32] | type[np.float64],
) -> None:
    monkeypatch.setattr(EmbeddingStore, "embed", _REAL_EMBED)
    model = EmbeddingModel(dtype=dtype)
    store = EmbeddingStore(tmp_path / dtype.__name__, model=model, page_limit=2)
    monkeypatch.setattr(
        EmbeddingModel,
        "__call__",
        lambda self, text: np.ones(self.dim, dtype=np.float64),
    )

    vector = store.embed("hello world")
    fingerprint = store.get_fingerprint("hello world")
    store.write(0, fingerprint, vector)
    row = store.get(0)

    assert vector.dtype == np.dtype(dtype)
    assert float(dtype(fingerprint)) == fingerprint
    assert row is not None
    assert row.dtype == np.dtype(dtype)
    assert np.load(store.root / "0.npy", mmap_mode="r").dtype == np.dtype(dtype)


def test_embedding_model_rejects_unsupported_dtype() -> None:
    with pytest.raises(ValueError, match="dtype"):
        EmbeddingModel(dtype=np.float16)


def test_embedding_store_retries_transient_provider_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(EmbeddingStore, "embed", _REAL_EMBED)
    store = EmbeddingStore(
        tmp_path / "emb",
        model=None,
        retry_timer=RetryTimer(tries=3, delay=0.25, randomness=0),
    )
    expected = np.ones(2)
    transient = APIConnectionError(
        request=httpx2.Request("POST", "https://example.test/embeddings")
    )
    model = MagicMock(side_effect=[transient, transient, expected])
    monkeypatch.setattr(EmbeddingModel, "__call__", model)
    delays: list[float] = []
    monkeypatch.setattr("alpha_lab.utils.sleep", delays.append)

    result = store.embed("retry me")
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)
    assert model.call_count == 3
    assert delays == [0.25, 0.5]


def test_embedding_store_does_not_retry_deterministic_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(EmbeddingStore, "embed", _REAL_EMBED)
    store = EmbeddingStore(
        tmp_path / "emb",
        model=None,
        retry_timer=RetryTimer(tries=3, delay=0.25, randomness=0),
    )
    model = MagicMock(side_effect=ValueError("invalid model configuration"))
    monkeypatch.setattr(EmbeddingModel, "__call__", model)
    sleep = MagicMock()
    monkeypatch.setattr("alpha_lab.utils.sleep", sleep)

    with pytest.raises(EmbeddingError, match="invalid model configuration"):
        store.embed("do not retry me")

    model.assert_called_once()
    sleep.assert_not_called()


def test_embedding_store_context_correction_is_immediate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(EmbeddingStore, "embed", _REAL_EMBED)
    store = EmbeddingStore(
        tmp_path / "emb", model=None, retry_timer=RetryTimer(tries=1, delay=10)
    )
    expected = np.ones(2)
    inputs: list[str] = []

    def model(self: EmbeddingModel, text: str) -> np.ndarray:
        inputs.append(text)
        if len(inputs) == 1:
            raise ValueError("maximum context length exceeded")
        return expected

    monkeypatch.setattr(EmbeddingModel, "__call__", model)
    sleep = MagicMock()
    monkeypatch.setattr("alpha_lab.utils.sleep", sleep)

    result = store.embed("x" * 100)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)
    assert [len(text) for text in inputs] == [100, 85]
    sleep.assert_not_called()


def test_embedding_store_raises_after_retry_timer_exhaustion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(EmbeddingStore, "embed", _REAL_EMBED)
    store = EmbeddingStore(
        tmp_path / "emb",
        model=None,
        retry_timer=RetryTimer(tries=3, delay=0, randomness=0),
    )
    transient = APIConnectionError(
        request=httpx2.Request("POST", "https://example.test/embeddings")
    )
    model = MagicMock(side_effect=transient)
    monkeypatch.setattr(EmbeddingModel, "__call__", model)
    monkeypatch.setattr("alpha_lab.utils.sleep", MagicMock())

    with pytest.raises(EmbeddingError, match="after all attempts") as exc_info:
        store.embed("keep failing")

    assert model.call_count == 3
    assert exc_info.value.__cause__ is transient


def test_embedding_store_rejects_invalid_token_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="token_limit"):
        EmbeddingStore(tmp_path / "emb", model=None, token_limit=0)


class _FakeEmbeddings:
    """Stand-in for ``client.embeddings``, recording every request."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = [3.0, 4.0] if vector is None else vector
        self.inputs: list[str] = []

    def create(self, *, model: str, input: str):
        del model
        self.inputs.append(input)
        return SimpleNamespace(data=[SimpleNamespace(embedding=self.vector)])


class _FakeClient:
    """Stand-in for the OpenAI client, recording ``with_options`` kwargs."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self.embeddings = _FakeEmbeddings(vector)
        self.options: list[dict[str, object]] = []

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self


@pytest.fixture()
def fake_client() -> _FakeClient:
    return _FakeClient()


def _model_with(client: _FakeClient, **kwargs) -> EmbeddingModel:
    """An EmbeddingModel wired to ``client``, bypassing the cached property."""
    model = EmbeddingModel(**kwargs)
    model.__dict__["_client"] = client
    return model


class TestEmbeddingModel:
    """``EmbeddingModel.__call__`` — untouched by the autouse embedding stub,
    which patches ``EmbeddingStore.embed`` one layer above it."""

    @pytest.fixture(autouse=True)
    def _clear_vector_cache(self) -> None:
        # The raw-vector cache is class-level, so it outlives a single test.
        EmbeddingModel._embed.cache_clear()

    def test_repeat_text_is_served_from_the_cache(self, fake_client: _FakeClient) -> None:
        model = _model_with(fake_client)
        assert model("hello") == pytest.approx([0.6, 0.8])
        assert model("hello") == pytest.approx([0.6, 0.8])
        assert fake_client.embeddings.inputs == ["hello"]  # one request, not two

    def test_cache_is_shared_across_models_and_dtypes(self, fake_client: _FakeClient) -> None:
        # The key is (model name, text), so dtype and normalize are applied
        # afterwards to each caller's own copy rather than being cached.
        small = _model_with(fake_client)
        wide = _model_with(fake_client, dtype=np.float64)
        assert small("hello").dtype == np.dtype(np.float32)
        assert wide("hello").dtype == np.dtype(np.float64)
        assert wide("hello") == pytest.approx([0.6, 0.8])
        assert fake_client.embeddings.inputs == ["hello"]

    def test_cached_vector_survives_a_caller_mutating_its_copy(
        self, fake_client: _FakeClient
    ) -> None:
        model = _model_with(fake_client)
        first = model("hello")
        first *= 0.0  # in place, as normalization does
        assert model("hello") == pytest.approx([0.6, 0.8])

    def test_non_finite_response_raises(self) -> None:
        model = _model_with(_FakeClient([np.nan, 1.0]))
        with pytest.raises(EmbeddingError, match="non-finite"):
            model("bad")

    def test_zero_vector_is_returned_unscaled(self) -> None:
        model = _model_with(_FakeClient([0.0, 0.0]))
        assert model("zeros") == pytest.approx([0.0, 0.0])

    def test_sdk_retries_are_disabled_on_the_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # EmbeddingStore.embed owns the retry policy; leaving the SDK's enabled
        # would stack a second round of attempts and backoff underneath it.
        client = _FakeClient()
        monkeypatch.setattr(
            "alpha_lab.embeddings.get_openai_client", lambda *a, **k: client
        )
        EmbeddingModel()("hello")
        assert client.options == [{"max_retries": 0}]

    def test_concurrent_calls_for_one_text_make_few_requests(
        self, fake_client: _FakeClient
    ) -> None:
        # The cache lock guards lookup and insert, not the request, so racing
        # threads may duplicate a request but must agree on the result.
        model = _model_with(fake_client)
        results: list[np.ndarray] = []
        errors: list[Exception] = []

        def worker() -> None:
            try:
                results.append(model("shared text"))
            except Exception as exc:  # capture so the assertion can surface it
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert not any(t.is_alive() for t in threads), "an embed thread hung"
        assert not errors, f"embed raised concurrently: {errors!r}"
        assert len(results) == 12
        for vector in results:
            assert vector == pytest.approx([0.6, 0.8])
        assert len(fake_client.embeddings.inputs) < 12  # the cache did its job


class TestContention:
    """Reads, writes and the locks between them, across threads and processes."""

    def test_searches_and_writes_interleave_cleanly(self, store: MemoryStore) -> None:
        # Agents search on nearly every turn while other agents store, so reads
        # and writes overlap constantly on one store.
        store.add(_memory("seed record for the readers", summary="seed"))
        writes, errors = 12, []
        stop = threading.Event()

        def reader() -> None:
            try:
                while not stop.is_set():
                    store.search("record", mode="fulltext", limit=5)
                    store.search(mode="recency", limit=5)
            except Exception as exc:  # capture so the assertion can surface it
                errors.append(exc)

        def writer(i: int) -> None:
            try:
                store.add(_memory(f"written record {i}", summary=f"w{i}"))
            except Exception as exc:
                errors.append(exc)

        readers = [threading.Thread(target=reader, daemon=True) for _ in range(3)]
        for thread in readers:
            thread.start()
        writers = [threading.Thread(target=writer, args=(i,)) for i in range(writes)]
        for thread in writers:
            thread.start()
        for thread in writers:
            thread.join(timeout=60)
        stop.set()
        for thread in readers:
            thread.join(timeout=30)

        assert not any(t.is_alive() for t in writers), "a writer hung"
        assert not errors, f"concurrent search/add raised: {errors!r}"
        assert len(list(store._db.keys())) == writes + 1

    @staticmethod
    def _writer_is_blocked(store: MemoryStore) -> bool:
        """Whether an index write can get the write lock right now.

        A near-zero timeout makes the answer immediate rather than a 5s wait.
        """
        connection = sqlite3.connect(store._db._index_path, timeout=0.05)
        try:
            connection.execute("UPDATE records SET summary = 'w' WHERE key = 1")
            connection.commit()
            return False
        except sqlite3.OperationalError:
            return True
        finally:
            connection.close()

    def test_search_leaves_no_read_lock_behind(self, store: MemoryStore) -> None:
        """``search`` stops at ``limit``, abandoning its ranked-key generator.

        That generator holds an open cursor, but it is a local of ``search``, so
        it is finalized when the call returns and the read transaction ends with
        it. Writers must therefore not be blocked afterwards — the property the
        rest of the store's locking relies on.
        """
        for i in range(6):
            store.add(_memory(f"gpu tuning note {i}", summary=f"n{i}"))
        assert len(store.search("gpu", mode="fulltext", limit=1)) == 1
        assert not self._writer_is_blocked(store)

    def test_partly_consumed_iterators_hold_no_read_lock(self, store: MemoryStore) -> None:
        """Partly consumed iterators hold no read lock beyond the query itself.

        ``select`` and ``fulltext_search`` fetch their rows inside the call and
        yield from the materialized list, so even an iterator abandoned
        mid-consumption — in a long-lived frame, a paused generator, another
        thread — leaves no read transaction open. This replaces the previous
        contract ("held iterators are the caller's responsibility to consume"),
        which every future caller had to know about to avoid starving index
        writers into "database is locked" failures.
        """
        for i in range(6):
            store.add(_memory(f"gpu tuning note {i}", summary=f"n{i}"))

        partial = store._db.select()
        next(partial)  # one row consumed, iterator deliberately kept alive
        ranked = store._db.fulltext_search("gpu")
        next(ranked)   # same for the FTS path

        assert not self._writer_is_blocked(store)

    def test_contended_index_write_surfaces_and_publishes_nothing(
        self, store: MemoryStore
    ) -> None:
        # Derived rows are written before the canonical record, so a lock the
        # writer cannot get aborts the whole store call rather than publishing a
        # record the index does not cover.
        store.add(_memory("first", summary="s1"))
        store._db.index.execute("PRAGMA busy_timeout = 1")
        blocker = sqlite3.connect(store._db._index_path)
        blocker.execute("BEGIN EXCLUSIVE")
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                store.add(_memory("second", summary="s2"))
        finally:
            blocker.rollback()
            blocker.close()

        assert list(store._db.keys()) == [1]  # nothing half-written

    def test_separate_processes_get_distinct_keys(self, tmp_path: Path) -> None:
        # The flock exists to serialize writers across processes, which threads
        # alone never exercise.
        root = tmp_path / "ws" / ".memory"
        script = (
            "import sys;"
            "sys.path.insert(0, 'src');"
            "from alpha_lab.databases import ModelDB;"
            "from alpha_lab.memory import Memory;"
            "db = ModelDB(root=__import__('pathlib').Path(sys.argv[1]),"
            "             model_type=Memory, embedding=None);"
            "print(db.add(Memory(kind='fact', summary=sys.argv[2], content=sys.argv[2])))"
        )
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", script, str(root), f"from-{i}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for i in range(2)
        ]
        outs = [p.communicate(timeout=120) for p in procs]
        for (out, err), proc in zip(outs, procs):
            assert proc.returncode == 0, f"child failed: {err}"

        keys = sorted(int(out.strip()) for out, _ in outs)
        assert keys == [1, 2]  # allocated under the cross-process lock


@pytest.mark.slow
class TestScaleLimits:
    """The two places the store has a hard limit that small tests never reach."""

    def test_candidate_narrowing_past_the_sql_variable_limit(self, store: MemoryStore) -> None:
        # Candidates are passed as one JSON array through json_each precisely so
        # a large filtered set can't exhaust SQLite's bound-variable limit
        # (999 by default). This drives well past it.
        count = 1200
        for i in range(count):
            store.add(_memory(f"record {i} about gpu tuning", summary=f"s{i}", tags=["bulk"]))

        found = store.search("gpu", mode="fulltext", include={"tags": "bulk"}, limit=5)
        assert len(found) == 5
        assert len(store._db.filter(include={"tags": "bulk"})) == count

    def test_vectors_roll_over_to_a_second_page(self, store: MemoryStore) -> None:
        # Vector rows are paged at page_limit; crossing it must create the next
        # page and keep earlier rows readable.
        limit = store._db.embedding.page_limit
        for i in range(limit + 2):
            store.add(_memory(f"content {i}", summary=f"s{i}"))

        pages = sorted(p.name for p in (store.root / "embeddings").glob("*.npy"))
        assert pages == ["0.npy", "1.npy"]
        assert store._db.embedding.get(1) is not None          # first page
        assert store._db.embedding.get(limit + 1) is not None  # second page


class TestClose:
    """Each layer owns its own release: store → embedding store → model."""

    def test_noop_without_embedding(self, tmp_path: Path) -> None:
        store = MemoryStore(
            ModelDB(root=tmp_path / ".memory", model_type=Memory, embedding=None)
        )
        store.close()  # nothing to release; must not raise

    def test_never_builds_a_client_just_to_close_it(self, store: MemoryStore) -> None:
        model = store._db.embedding.model
        assert model.__dict__.get("_client") is None
        store.close()
        assert model.__dict__.get("_client") is None

    def test_releases_and_clears_cached_client(self, store: MemoryStore) -> None:
        # End to end through the real layers: MemoryStore.close ->
        # EmbeddingStore.close -> EmbeddingModel.close clears the cache.
        model = store._db.embedding.model
        client = MagicMock()
        model._client = client
        store.close()
        client.close.assert_called_once_with()
        # Cleared, not just closed: a later use lazily rebuilds rather than
        # hitting a closed client.
        assert model.__dict__.get("_client") is None

    def test_delegates_to_the_embedding_store(
        self, monkeypatch: pytest.MonkeyPatch, store: MemoryStore
    ) -> None:
        released: list[bool] = []
        monkeypatch.setattr(
            store._db.embedding, "close", lambda: released.append(True)
        )
        store.close()
        assert released == [True]
