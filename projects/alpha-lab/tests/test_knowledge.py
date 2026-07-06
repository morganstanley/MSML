from __future__ import annotations

from datetime import datetime as real_datetime
from pathlib import Path

import pytest

import alpha_lab.knowledge as knowledge
from alpha_lab.knowledge import TopicKnowledgeStore
from alpha_lab.memory import MemoryStore
from alpha_lab.memory_cli import main as memory_cli_main


def test_topic_save_writes_current_doc_and_indexes_memory(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    store = TopicKnowledgeStore(str(ws))

    memory_id = store.save_topic(
        "data_access.exchange_rates",
        "## Steps\nUse the managed CSV export.",
        title="Accessing exchange rate data",
        tags=["data_access"],
        owner="research-platform",
        last_verified="2026-06-04",
    )

    assert memory_id == 1
    topic_path = ws / ".memory" / "topics" / "data_access.exchange_rates.md"
    assert topic_path.exists()
    topic_text = topic_path.read_text()
    assert '"topic": "data_access.exchange_rates"' in topic_text
    assert "# Accessing exchange rate data" in topic_text
    assert "Use the managed CSV export" in topic_text

    entries = store.search_topics("managed CSV export")
    assert len(entries) == 1
    assert entries[0].kind == "reference"
    assert "topic" in entries[0].tags
    assert "topic_data_access_exchange_rates" in entries[0].tags
    assert entries[0].source_path == ".memory/topics/data_access.exchange_rates.md"


def test_topic_save_dedupes_unchanged_content_but_versions_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    store = TopicKnowledgeStore(str(ws))

    class FakeDatetime:
        calls = 0

        @classmethod
        def now(cls) -> real_datetime:
            cls.calls += 1
            return real_datetime(2026, 6, 4, 12, 0, cls.calls)

    monkeypatch.setattr(knowledge, "datetime", FakeDatetime)

    first = store.save_topic("auth.tokens", "Token comes from the provider.", title="Auth tokens")
    text_after_first = store.read_topic("auth.tokens")
    second = store.save_topic("auth.tokens", "Token comes from the provider.", title="Auth tokens")
    text_after_second = store.read_topic("auth.tokens")
    third = store.save_topic("auth.tokens", "Token comes from the provider; refresh when expired.", title="Auth tokens")

    assert first == 1
    assert second == 1
    assert third == 2
    assert text_after_second == text_after_first
    assert len(MemoryStore(str(ws)).list_by_tag("topic_auth_tokens")) == 2
    assert "refresh when expired" in store.read_topic("auth.tokens")


def test_topic_list_and_read(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    store = TopicKnowledgeStore(str(ws))
    store.save_topic("gpu.local_executor", "Set CUDA_VISIBLE_DEVICES.", owner="ml-platform")

    records = store.list_topics()

    assert len(records) == 1
    assert records[0].topic == "gpu.local_executor"
    assert records[0].owner == "ml-platform"
    assert "CUDA_VISIBLE_DEVICES" in store.read_topic("gpu/local executor")


def test_topic_reads_replace_invalid_utf8(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    topic_dir = ws / ".memory" / "topics"
    topic_dir.mkdir(parents=True)
    (topic_dir / "bad.topic.md").write_bytes(b"---\n{}\n---\n\nBad \xff topic")
    store = TopicKnowledgeStore(str(ws))

    records = store.list_topics()

    assert len(records) == 1
    assert records[0].topic == "bad.topic"
    assert "Bad \ufffd topic" in store.read_topic("bad.topic")


def test_reference_kind_aliases_support_topic_search(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    memory = MemoryStore(str(ws))
    memory.store("Runbook body", tags=["topic"], summary="Runbook", kind="runbook")

    results = memory.search("Runbook", kind="reference")

    assert len(results) == 1
    assert results[0].kind == "reference"


def test_memory_cli_topic_add_search_and_read(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    ws = tmp_path / "workspace"
    ws.mkdir()
    body_file = tmp_path / "topic.md"
    body_file.write_text("Use the approved data entitlement group.")

    assert memory_cli_main([
        "--workspace",
        str(ws),
        "topic",
        "add",
        "data_access.market_data",
        "--file",
        str(body_file),
        "--title",
        "Market data access",
        "--tag",
        "data_access",
    ]) == 0
    out = capsys.readouterr().out
    assert "Saved topic 'data_access.market_data' as memory #1" in out

    assert memory_cli_main([
        "--workspace",
        str(ws),
        "topic",
        "search",
        "entitlement group",
    ]) == 0
    out = capsys.readouterr().out
    assert "#1: Market data access" in out

    assert memory_cli_main([
        "--workspace",
        str(ws),
        "topic",
        "read",
        "data_access.market_data",
    ]) == 0
    out = capsys.readouterr().out
    assert "Use the approved data entitlement group" in out


def test_memory_cli_topic_add_replaces_invalid_utf8_file_bytes(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    ws = tmp_path / "workspace"
    ws.mkdir()
    body_file = tmp_path / "topic.md"
    body_file.write_bytes(b"Use managed export \xff only.")

    assert memory_cli_main([
        "--workspace",
        str(ws),
        "topic",
        "add",
        "data_access.invalid_utf8",
        "--file",
        str(body_file),
    ]) == 0

    capsys.readouterr()
    text = TopicKnowledgeStore(str(ws)).read_topic("data_access.invalid_utf8")
    assert "Use managed export \ufffd only." in text


def test_memory_cli_topic_add_reports_missing_file_cleanly(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()

    with pytest.raises(SystemExit, match="could not read"):
        memory_cli_main([
            "--workspace",
            str(ws),
            "topic",
            "add",
            "data_access.missing",
            "--file",
            str(tmp_path / "missing.md"),
        ])
