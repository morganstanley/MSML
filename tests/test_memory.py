"""Tests for the persistent memory system."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from alpha_lab.memory import MemoryStore


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
        # File should exist
        entries_dir = store._entries_dir
        assert entries_dir.exists()
        files = list(entries_dir.glob("001_*.md"))
        assert len(files) == 1
        assert "XGBoost" in files[0].read_text()

    def test_store_creates_directory_lazily(self, store: MemoryStore) -> None:
        assert not store._base.exists()
        store.store(content="test", tags=["test"], summary="test entry")
        assert store._base.exists()
        assert store._entries_dir.exists()
        assert store._index_path.exists()

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
        # Both null-related entries should match
        summaries = [r.summary for r in results]
        assert any("EUR/USD" in s for s in summaries)
        assert any("GBP/USD" in s for s in summaries)

    def test_search_by_tag_filter(self, store: MemoryStore) -> None:
        store.store(content="d1", tags=["data"], summary="data quality issue")
        store.store(content="d2", tags=["model"], summary="data from model run")

        results = store.search("data", tags=["data"])
        assert len(results) == 1
        assert results[0].tags == ["data"]

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
        # Most recent first
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

        # All 10 entries should be present
        recent = store.list_recent(limit=20)
        assert len(recent) == 10

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

    def test_memory_search_tool(self, tmp_path: Path) -> None:
        from alpha_lab.tools import execute_tool

        ws = str(tmp_path / "workspace")
        Path(ws).mkdir()
        # Store first
        execute_tool(
            name="memory_store",
            arguments={"content": "data quality", "tags": ["data"], "summary": "null values found"},
            workspace=ws,
        )
        # Search
        result = execute_tool(
            name="memory_search",
            arguments={"query": "null values"},
            workspace=ws,
        )
        assert "Found 1 memories" in result["output"]
        assert "null values found" in result["output"]

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
