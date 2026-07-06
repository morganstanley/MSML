"""Tests for the persistent memory system."""

from __future__ import annotations

import json
import time
import threading
from pathlib import Path

import pytest

from alpha_lab.memory import MemoryStore, remember_text, remember_workspace_file


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    """Create a MemoryStore with a temporary workspace."""
    ws = str(tmp_path / "workspace")
    Path(ws).mkdir()
    return MemoryStore(ws)


class TestMemoryStore:
    def test_store_creates_entry_and_file(self, store: MemoryStore) -> None:
        entry_id = store.store(
            content="XGBoost achieves Sharpe 1.2 on EUR/USD",
            tags=["experiment", "xgboost"],
            summary="XGBoost baseline Sharpe 1.2",
        )
        assert entry_id == 1
        files = list(store._entries_dir.glob("001_*.md"))
        assert len(files) == 1
        text = files[0].read_text()
        assert '"summary": "XGBoost baseline Sharpe 1.2"' in text
        assert "XGBoost achieves Sharpe 1.2 on EUR/USD" in text
        assert store._index_path.exists()
        assert store._db_path.exists()

    def test_store_accepts_null_tags(self, store: MemoryStore) -> None:
        entry_id = store.store(
            content="memory without tags",
            tags=None,
            summary="No tags",
        )

        assert entry_id == 1
        assert store.list_recent(limit=1)[0].tags == []

    def test_load_index_reads_utf8(self, store: MemoryStore) -> None:
        store._ensure_dirs()
        payload = {
            "version": 2,
            "entries": [
                {
                    "id": 1,
                    "tags": ["cafe"],
                    "summary": "Café index",
                    "created_at": "2026-06-04T00:00:00",
                    "file": "001_cafe.md",
                }
            ],
        }
        store._index_path.write_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        loaded = store._load_index()

        assert loaded["entries"][0]["summary"] == "Café index"

    def test_ensure_initialized_bootstraps_once_across_threads(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        store = MemoryStore(ws)
        calls = 0
        calls_lock = threading.Lock()
        start = threading.Barrier(8)

        def fake_bootstrap() -> None:
            nonlocal calls
            with calls_lock:
                calls += 1
            time.sleep(0.05)
            store._ensure_dirs()

        store._bootstrap_locked = fake_bootstrap  # type: ignore[method-assign]

        def worker() -> None:
            start.wait()
            store._ensure_initialized()

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert calls == 1

    def test_ensure_initialized_reuses_bootstrap_across_instances(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        calls = 0
        original_bootstrap = MemoryStore._bootstrap_locked

        def counting_bootstrap(self: MemoryStore) -> None:
            nonlocal calls
            calls += 1
            original_bootstrap(self)

        monkeypatch.setattr(MemoryStore, "_bootstrap_locked", counting_bootstrap)

        MemoryStore(ws).list_recent()
        MemoryStore(ws).list_recent()

        assert calls == 1

    def test_sqlite_init_drops_unused_legacy_indexes(self, store: MemoryStore) -> None:
        store.store(content="first", tags=["a"], summary="first")
        with store._connect() as conn:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_tag_blob ON memories(tag_blob)")
            conn.commit()

        with store._lock():
            store._init_sqlite_locked()

        with store._connect() as conn:
            names = {
                row["name"]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()
            }

        assert "idx_memories_created_at" not in names
        assert "idx_memories_tag_blob" not in names
        assert "idx_memories_kind_phase" in names

    def test_store_keeps_memory_row_when_fts_update_fails(self, store: MemoryStore) -> None:
        store.store(content="first", tags=["a"], summary="first")
        if not store._fts_ready:
            pytest.skip("SQLite FTS5 is unavailable in this Python build")
        with store._connect() as conn:
            conn.execute("DROP TABLE memories_fts")
            conn.commit()

        entry_id = store.store(content="second survives", tags=["b"], summary="second")

        assert entry_id == 2
        assert store._fts_ready is False
        recent = store.list_recent(limit=1)
        assert recent[0].id == 2
        assert recent[0].summary == "second"
        assert "second survives" in store.read(2)

    def test_store_creates_directory_lazily(self, store: MemoryStore) -> None:
        assert not store._base.exists()
        store.store(content="test", tags=["test"], summary="test entry")
        assert store._base.exists()
        assert store._entries_dir.exists()
        assert store._index_path.exists()
        assert store._db_path.exists()

    def test_atomic_text_write_replaces_without_temp_leftover(self, tmp_path: Path) -> None:
        path = tmp_path / "entry.md"
        MemoryStore._write_text_atomic(path, "first")
        MemoryStore._write_text_atomic(path, "café")

        assert path.read_text(encoding="utf-8") == "café"
        assert path.read_bytes() == "café".encode("utf-8")
        assert not path.with_name("entry.md.tmp").exists()

    def test_sqlite_cache_is_refreshed_from_canonical_entry_files(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        first_store = MemoryStore(ws)
        first_store.store(content="old cache token", tags=["sync"], summary="Sync entry")
        entry_path = next((ws / ".memory" / "entries").glob("001_*.md"))
        entry_path.write_text(
            entry_path.read_text(encoding="utf-8").replace("old cache token", "new canonical token"),
            encoding="utf-8",
        )

        refreshed_store = MemoryStore(ws)
        results = refreshed_store.search("new canonical token", limit=5)

        assert [entry.summary for entry in results] == ["Sync entry"]

    def test_sqlite_cache_prunes_rows_removed_from_canonical_artifacts(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        store = MemoryStore(ws)
        store.store(content="keep canonical token", tags=["sync"], summary="Keep entry")
        stale_id = store.store(content="stale_unique_token", tags=["sync"], summary="Stale entry")
        stale_path = next((ws / ".memory" / "entries").glob(f"{stale_id:03d}_*.md"))
        stale_path.unlink()
        index = store._load_index()
        index["entries"] = [entry for entry in index["entries"] if entry["id"] != stale_id]
        store._write_index(index)

        refreshed_store = MemoryStore(ws)

        assert [entry.summary for entry in refreshed_store.search("keep canonical token")] == ["Keep entry"]
        assert refreshed_store.search("stale_unique_token") == []

    def test_load_entry_from_file_replaces_invalid_utf8(self, store: MemoryStore) -> None:
        store._ensure_dirs()
        path = store._entries_dir / "001_bad.md"
        path.write_bytes(b"valid\xfftext")

        entry, content = store._load_entry_from_file(path)

        assert entry.id == 1
        assert content == "valid\ufffdtext"

    def test_remember_workspace_file_replaces_invalid_utf8(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        artifact = ws / "artifact.md"
        artifact.write_bytes(b"bad\xffartifact")

        memory_id = remember_workspace_file(
            str(ws),
            "artifact.md",
            tags=["artifact"],
            summary="Bad artifact",
            kind="finding",
        )

        assert memory_id == 1
        assert "bad\ufffdartifact" in MemoryStore(ws).read(1)

    def test_store_increments_ids(self, store: MemoryStore) -> None:
        id1 = store.store(content="first", tags=["a"], summary="first")
        id2 = store.store(content="second", tags=["b"], summary="second")
        id3 = store.store(content="third", tags=["c"], summary="third")
        assert id1 == 1
        assert id2 == 2
        assert id3 == 3

    def test_search_by_keyword(self, store: MemoryStore) -> None:
        store.store(content="details", tags=["data"], summary="EUR/USD has 12% null values")
        store.store(content="details", tags=["model"], summary="LSTM baseline results")
        store.store(content="details", tags=["data"], summary="GBP/USD null rate is 5%")

        results = store.search("null values")
        assert len(results) >= 2
        summaries = [r.summary for r in results]
        assert any("EUR/USD" in s for s in summaries)
        assert any("GBP/USD" in s for s in summaries)

    def test_search_by_content_uses_sqlite_index(self, store: MemoryStore) -> None:
        store.store(
            content="Walk-forward leakage discovered in fold construction.",
            tags=["analysis"],
            summary="Cross-validation issue",
        )

        results = store.search("leakage fold construction")
        assert len(results) == 1
        assert results[0].summary == "Cross-validation issue"

    def test_sqlite_fts5_index_is_queryable_when_available(self, store: MemoryStore) -> None:
        store.store(
            content="The uncommonftsneedle token is indexed in the full text table.",
            tags=["analysis"],
            summary="FTS smoke test",
        )
        if not store._fts_ready:
            pytest.skip("SQLite FTS5 is unavailable in this Python build")

        with store._connect() as conn:
            rows = conn.execute(
                "SELECT rowid, summary FROM memories_fts WHERE memories_fts MATCH ?",
                ("uncommonftsneedle",),
            ).fetchall()

        assert len(rows) == 1
        assert rows[0]["rowid"] == 1
        assert rows[0]["summary"] == "FTS smoke test"

    def test_search_uses_lexical_fallback_alongside_fts(self, store: MemoryStore) -> None:
        store.store(
            content="General OOM troubleshooting note.",
            tags=["gpu"],
            summary="OOM note",
        )
        store.store(
            content="The CUDAOOM signature appears when CUDA errors are concatenated in logs.",
            tags=["gpu", "logs"],
            summary="Encoded CUDA failure signature",
        )

        results = store.search("oom", limit=5)
        summaries = [entry.summary for entry in results]
        assert "OOM note" in summaries
        assert "Encoded CUDA failure signature" in summaries

    def test_search_deduplicates_hybrid_candidates(self, store: MemoryStore) -> None:
        store.store(
            content="Sobol initialization improved bayesian optimizer stability.",
            tags=["phase3", "optimizer"],
            summary="Bayesian optimizer Sobol result",
        )

        results = store.search("bayesian optimizer sobol", limit=10)
        assert [entry.id for entry in results] == [1]

    def test_search_by_tag_filter(self, store: MemoryStore) -> None:
        store.store(content="d1", tags=["data"], summary="data quality issue")
        store.store(content="d2", tags=["model"], summary="data from model run")

        results = store.search("data", tags=["data"])
        assert len(results) == 1
        assert results[0].tags == ["data"]

    def test_search_by_metadata_filter(self, store: MemoryStore) -> None:
        store.store(
            content="Real-data reality check failed due to leakage.",
            tags=["validation"],
            summary="Validation failure",
            kind="failure",
            phase="phase2",
            agent="tester",
        )
        store.store(
            content="Improved harness logging.",
            tags=["infra"],
            summary="Infra improvement",
            kind="decision",
            phase="phase3",
        )

        results = store.search("failure", kind="failure", phase="phase2")
        assert len(results) == 1
        assert results[0].kind == "failure"
        assert results[0].phase == "phase2"
        assert results[0].agent == "tester"

    def test_kind_aliases_are_normalized(self, store: MemoryStore) -> None:
        store.store(
            content="Run failed with a CUDA OOM.",
            tags=["gpu"],
            summary="OOM error",
            kind="error",
            phase="phase3",
        )
        store.store(
            content="Experiment reached the best metric so far.",
            tags=["metric"],
            summary="Best metric",
            kind="experiment_result",
            phase="phase3",
        )

        failures = store.search("cuda oom", kind="issue")
        assert len(failures) == 1
        assert failures[0].kind == "failure"

        results = store.search("best metric", kind="debrief")
        assert len(results) == 1
        assert results[0].kind == "result"

    def test_search_no_results(self, store: MemoryStore) -> None:
        store.store(content="something", tags=["a"], summary="unrelated entry")
        results = store.search("nonexistent_keyword_xyz")
        assert results == []

    def test_search_empty_store(self, store: MemoryStore) -> None:
        results = store.search("anything")
        assert results == []

    def test_read_entry(self, store: MemoryStore) -> None:
        store.store(
            content="Full detailed analysis of feature importance...",
            tags=["analysis"],
            summary="Feature importance analysis",
        )
        content = store.read(1)
        assert "Full detailed analysis" in content

    def test_read_nonexistent(self, store: MemoryStore) -> None:
        result = store.read(999)
        assert "[ERROR]" in result

    def test_read_truncates_large_content(self, store: MemoryStore) -> None:
        large_content = "x" * 15_000
        store.store(content=large_content, tags=["big"], summary="big entry")
        result = store.read(1)
        assert len(result) < 15_000
        assert "[...truncated]" in result

    def test_list_recent(self, store: MemoryStore) -> None:
        for i in range(5):
            store.store(content=f"entry {i}", tags=["test"], summary=f"entry {i}")
        recent = store.list_recent(limit=3)
        assert len(recent) == 3
        assert recent[0].id == 5
        assert recent[1].id == 4
        assert recent[2].id == 3

    def test_list_by_tag(self, store: MemoryStore) -> None:
        store.store(content="a", tags=["alpha", "beta"], summary="a")
        store.store(content="b", tags=["beta"], summary="b")
        store.store(content="c", tags=["gamma"], summary="c")

        beta_entries = store.list_by_tag("beta")
        assert len(beta_entries) == 2
        assert all("beta" in e.tags for e in beta_entries)

    def test_list_by_tag_treats_underscore_literally(self, store: MemoryStore) -> None:
        store.store(content="a", tags=["phase_1"], summary="phase one")
        store.store(content="b", tags=["phasex1"], summary="phase x one")

        entries = store.list_by_tag("phase_1")

        assert len(entries) == 1
        assert entries[0].summary == "phase one"

    def test_lexical_search_treats_underscore_literally(self, store: MemoryStore) -> None:
        store.store(content="run_42_fold3 failed with leakage", tags=["run"], summary="underscored run")
        store.store(content="runX42Xfold3 failed for another reason", tags=["run"], summary="wildcard-looking run")

        results = store.search("run_42_fold3", limit=10)
        summaries = [entry.summary for entry in results]

        assert "underscored run" in summaries
        assert "wildcard-looking run" not in summaries

    def test_corrupt_sqlite_tags_json_does_not_crash(self, store: MemoryStore) -> None:
        store.store(content="corrupt tags content", tags=["valid"], summary="corrupt tags")
        with store._connect() as conn:
            conn.execute("UPDATE memories SET tags_json = ? WHERE id = 1", ("not-json",))
            conn.commit()

        recent = store.list_recent(limit=1)
        assert len(recent) == 1
        assert recent[0].tags == []

        results = store.search("corrupt tags content")
        assert len(results) == 1
        assert results[0].tags == []

    def test_memory_stores_share_thread_lock_for_workspace(self, store: MemoryStore) -> None:
        other = MemoryStore(store.workspace)
        assert other._thread_lock is store._thread_lock

    def test_load_entry_from_file_handles_read_error(self, store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> None:
        path = store._entries_dir / "123_unreadable.md"
        fallback = store._entry_from_path_name(path)

        def fail_read_text(self: Path, *args, **kwargs) -> str:
            raise OSError("gone")

        monkeypatch.setattr(Path, "read_text", fail_read_text)

        entry, content = store._load_entry_from_file(path, fallback_entry=fallback)
        assert entry == fallback
        assert content == ""

        entry, content = store._load_entry_from_file(path)
        assert entry.id == 123
        assert entry.summary == "unreadable"
        assert content == ""

    def test_concurrent_stores(self, store: MemoryStore) -> None:
        """Multiple threads storing simultaneously should not corrupt the index."""
        errors: list[Exception] = []

        def store_entry(n: int) -> None:
            try:
                store.store(
                    content=f"concurrent entry {n}",
                    tags=["concurrent"],
                    summary=f"thread {n}",
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=store_entry, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent store errors: {errors}"
        recent = store.list_recent(limit=20)
        assert len(recent) == 10

    def test_portable_artifacts_survive_sqlite_rebuild(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        store = MemoryStore(ws)
        store.store(
            content="Portable memory should survive framework changes.",
            tags=["portability", "phase1"],
            summary="Portable memory",
            kind="finding",
            phase="phase1",
        )

        assert store._db_path.exists()
        store._db_path.unlink()

        reloaded = MemoryStore(ws)
        results = reloaded.search("framework changes")
        assert len(results) == 1
        assert results[0].summary == "Portable memory"
        assert results[0].kind == "finding"
        assert results[0].phase == "phase1"
        assert "framework changes" in reloaded.read(1)
        assert reloaded._db_path.exists()

    def test_portable_entries_survive_index_and_sqlite_loss(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        store = MemoryStore(ws)
        store.store(
            content="Keep the memory files portable across harnesses.",
            tags=["portability", "skills"],
            summary="Portable skill",
            kind="decision",
            phase="phase3",
            agent="strategist",
        )

        store._db_path.unlink()
        store._index_path.unlink()

        reloaded = MemoryStore(ws)
        results = reloaded.search("portable across harnesses")
        assert len(results) == 1
        assert results[0].summary == "Portable skill"
        assert results[0].kind == "decision"
        assert results[0].phase == "phase3"
        assert results[0].agent == "strategist"
        assert reloaded._index_path.exists()
        assert reloaded._db_path.exists()

    def test_migrates_legacy_index_to_sqlite(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        entries_dir = ws / ".memory" / "entries"
        entries_dir.mkdir(parents=True)
        (entries_dir / "001_legacy_note.md").write_text(
            "Legacy content about benchmark contamination and leakage."
        )
        (ws / ".memory" / "index.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "entries": [
                        {
                            "id": 1,
                            "tags": ["legacy", "analysis"],
                            "summary": "Legacy note",
                            "created_at": "2026-01-01T00:00:00",
                            "file": "001_legacy_note.md",
                        }
                    ],
                }
            )
        )

        store = MemoryStore(ws)
        results = store.search("benchmark contamination leakage")
        assert len(results) == 1
        assert results[0].summary == "Legacy note"
        assert store._db_path.exists()

    def test_metadata_round_trip(self, store: MemoryStore) -> None:
        store.store(
            content="Use walk-forward validation only.",
            tags=["validation", "phase1"],
            summary="Validation decision",
            kind="decision",
            phase="phase1",
            agent="strategist",
            run_id="run-123",
            source_path="data_report/baseline_profile.md",
        )

        entry = store.list_recent(limit=1)[0]
        assert entry.kind == "decision"
        assert entry.phase == "phase1"
        assert entry.agent == "strategist"
        assert entry.run_id == "run-123"
        assert entry.source_path == "data_report/baseline_profile.md"

    def test_remember_text_dedupes_with_normalized_summary_and_phase(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()

        first = remember_text(
            str(ws),
            "same content",
            tags=["finding"],
            summary=" Same summary ",
            kind="finding",
            phase=" phase1 ",
        )
        second = remember_text(
            str(ws),
            "same content",
            tags=["finding"],
            summary="Same summary",
            kind="finding",
            phase="phase1",
        )

        assert first == 1
        assert second == 1
        assert len(MemoryStore(ws).list_recent(limit=10)) == 1

    def test_remember_workspace_file_rejects_paths_outside_workspace(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        outside = tmp_path / "outside.md"
        outside.write_text("outside secret")

        assert remember_workspace_file(
            str(ws),
            "../outside.md",
            tags=["artifact"],
            summary="outside",
        ) is None
        assert remember_workspace_file(
            str(ws),
            str(outside),
            tags=["artifact"],
            summary="outside",
        ) is None
        assert not (ws / ".memory").exists()

    def test_remember_workspace_file_handles_read_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        target = ws / "artifact.md"
        target.write_text("content")
        original_read_text = Path.read_text

        def flaky_read_text(self: Path, *args, **kwargs) -> str:
            if self.resolve() == target.resolve():
                raise OSError("removed")
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", flaky_read_text)

        assert remember_workspace_file(
            str(ws),
            "artifact.md",
            tags=["artifact"],
            summary="artifact",
        ) is None
        assert not (ws / ".memory").exists()

    def test_remember_workspace_file_normalizes_safe_source_path(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        nested = ws / "reports"
        nested.mkdir(parents=True)
        (nested / "note.md").write_text("safe note")

        entry_id = remember_workspace_file(
            str(ws),
            "reports/../reports/note.md",
            tags=["artifact"],
            summary="safe note",
        )

        assert entry_id == 1
        entry = MemoryStore(ws).list_recent(limit=1)[0]
        assert entry.source_path == "reports/note.md"

    def test_remember_workspace_file_dedupes_identical_content(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "learnings.md").write_text("# Learnings\n- Keep it portable")

        first = remember_workspace_file(
            str(ws),
            "learnings.md",
            tags=["phase1", "learnings"],
            summary="Phase 1 learnings",
            kind="finding",
            phase="phase1",
            agent="phase1",
        )
        second = remember_workspace_file(
            str(ws),
            "learnings.md",
            tags=["phase1", "learnings"],
            summary="Phase 1 learnings",
            kind="finding",
            phase="phase1",
            agent="phase1",
        )

        assert first == 1
        assert second == 1
        store = MemoryStore(ws)
        assert len(store.list_recent(limit=10)) == 1

    def test_remember_workspace_file_stores_new_version_when_content_changes(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        path = ws / "learnings.md"
        path.write_text("# Learnings\n- First pass")

        first = remember_workspace_file(
            str(ws),
            "learnings.md",
            tags=["phase1", "learnings"],
            summary="Phase 1 learnings",
            kind="finding",
            phase="phase1",
            agent="phase1",
        )
        path.write_text("# Learnings\n- Second pass")
        second = remember_workspace_file(
            str(ws),
            "learnings.md",
            tags=["phase1", "learnings"],
            summary="Phase 1 learnings",
            kind="finding",
            phase="phase1",
            agent="phase1",
        )

        assert first == 1
        assert second == 2

    def test_normalize_tags_handles_falsey_values(self) -> None:
        assert MemoryStore._normalize_tags(None) == []
        assert MemoryStore._normalize_tags([]) == []

    def test_normalize_tags_accepts_single_string(self) -> None:
        assert MemoryStore._normalize_tags("Phase 1") == ["phase_1"]

    def test_slugify(self) -> None:
        assert MemoryStore._slugify("Hello World!") == "hello_world"
        assert MemoryStore._slugify("   ") == "entry"
        assert MemoryStore._slugify("a" * 100) == "a" * 40


class TestMemoryToolDispatch:
    """Test that tools.py dispatches memory tools correctly."""

    def test_memory_store_tool(self, tmp_path: Path) -> None:
        from alpha_lab.tools import execute_tool

        ws = str(tmp_path / "workspace")
        Path(ws).mkdir()
        result = execute_tool(
            name="memory_store",
            arguments={"content": "test content", "tags": ["t"], "summary": "test"},
            workspace=ws,
        )
        assert "Memory #1 stored" in result["output"]

    def test_memory_store_tool_accepts_metadata(self, tmp_path: Path) -> None:
        from alpha_lab.tools import execute_tool

        ws = str(tmp_path / "workspace")
        Path(ws).mkdir()
        execute_tool(
            name="memory_store",
            arguments={
                "content": "detailed finding",
                "tags": ["analysis", "phase2"],
                "summary": "finding",
                "kind": "finding",
                "phase": "phase2",
                "agent": "tester",
                "run_id": "abc",
                "source_path": "reports/out.md",
            },
            workspace=ws,
        )
        entry = MemoryStore(ws).list_recent(limit=1)[0]
        assert entry.kind == "finding"
        assert entry.phase == "phase2"
        assert entry.agent == "tester"
        assert entry.run_id == "abc"
        assert entry.source_path == "reports/out.md"

    def test_memory_search_tool(self, tmp_path: Path) -> None:
        from alpha_lab.tools import execute_tool

        ws = str(tmp_path / "workspace")
        Path(ws).mkdir()
        execute_tool(
            name="memory_store",
            arguments={"content": "data quality", "tags": ["data"], "summary": "null values found"},
            workspace=ws,
        )
        result = execute_tool(
            name="memory_search",
            arguments={"query": "null values"},
            workspace=ws,
        )
        assert "Found 1 memories" in result["output"]
        assert "null values found" in result["output"]

    def test_memory_search_tool_filters_metadata(self, tmp_path: Path) -> None:
        from alpha_lab.tools import execute_tool

        ws = str(tmp_path / "workspace")
        Path(ws).mkdir()
        execute_tool(
            name="memory_store",
            arguments={
                "content": "baseline failed on real data",
                "tags": ["validation"],
                "summary": "real-data failure",
                "kind": "failure",
                "phase": "phase2",
            },
            workspace=ws,
        )
        execute_tool(
            name="memory_store",
            arguments={
                "content": "revised prompt",
                "tags": ["prompting"],
                "summary": "prompt tweak",
                "kind": "decision",
                "phase": "phase1",
            },
            workspace=ws,
        )
        result = execute_tool(
            name="memory_search",
            arguments={"query": "failure", "kind": "failure", "phase": "phase2"},
            workspace=ws,
        )
        assert "Found 1 memories" in result["output"]
        assert "kind=failure" in result["output"]
        assert "phase=phase2" in result["output"]

    def test_memory_search_no_results(self, tmp_path: Path) -> None:
        from alpha_lab.tools import execute_tool

        ws = str(tmp_path / "workspace")
        Path(ws).mkdir()
        result = execute_tool(
            name="memory_search",
            arguments={"query": "nothing here"},
            workspace=ws,
        )
        assert "No matching memories" in result["output"]

    def test_memory_read_tool(self, tmp_path: Path) -> None:
        from alpha_lab.tools import execute_tool

        ws = str(tmp_path / "workspace")
        Path(ws).mkdir()
        execute_tool(
            name="memory_store",
            arguments={"content": "detailed findings", "tags": ["x"], "summary": "s"},
            workspace=ws,
        )
        result = execute_tool(
            name="memory_read",
            arguments={"memory_id": 1},
            workspace=ws,
        )
        assert "detailed findings" in result["output"]
