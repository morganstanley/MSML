from __future__ import annotations

import json
from pathlib import Path

from tools import get_data


def test_sampling_rules_have_expected_document_counts() -> None:
    totals = {
        "english": 156_289,
        "chinese": 230_792,
        "github": 334_662,
    }
    for name, total in totals.items():
        spec = get_data.DATASETS[name]
        assert sum(map(spec.tiny_keep, range(total))) == (
            spec.expected_tiny_documents
        )
        assert sum(map(spec.short_keep, range(total))) == (
            spec.expected_short_documents
        )


def test_build_outputs_generates_tiny_short_and_full_atomically(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.jsonl"
    rows = [
        {"text": "a\r\nb", "id": 0},
        {"text": "cc", "id": 1},
        {"text": "ddd", "id": 2},
        {"text": "eeee", "id": 3},
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    spec = get_data.DatasetSpec(
        name="test",
        source_url="https://example.invalid/source.jsonl",
        source_path="source.jsonl",
        source_format="jsonl",
        batch_path="test/tiny.jsonl",
        tiny_stream_path="test/stream_tiny.txt",
        short_stream_path="test/stream_short.txt",
        full_stream_path="test/stream.txt",
        tiny_keep=lambda index: index < 2,
        short_keep=lambda index: index % 2 == 0,
        tiny_sample_description="first two",
        short_sample_description="even indices",
        expected_source_bytes=source.stat().st_size,
        expected_source_sha256=None,
        expected_tiny_documents=2,
        expected_tiny_text_bytes=5,
        expected_short_documents=2,
    )
    monkeypatch.setattr(get_data, "SHORT_TARGET_BYTES", 7)
    monkeypatch.setattr(get_data, "SHORT_TARGET_TOLERANCE_BYTES", 0)

    stats = get_data.build_outputs(
        spec,
        source,
        tmp_path,
        parquet_batch_size=2,
    )

    assert stats == get_data.BuildStats(
        total_documents=4,
        tiny_documents=2,
        short_documents=2,
        full_text_bytes=12,
        tiny_text_bytes=5,
        short_text_bytes=6,
    )
    assert (tmp_path / spec.tiny_stream_path).read_text() == "a\nb\ncc"
    assert (tmp_path / spec.short_stream_path).read_text() == "a\nb\nddd"
    assert (tmp_path / spec.full_stream_path).read_text() == (
        "a\nb\ncc\nddd\neeee"
    )
    tiny_rows = [
        json.loads(line)
        for line in (tmp_path / spec.batch_path).read_text().splitlines()
    ]
    assert [row["text"] for row in tiny_rows] == ["a\nb", "cc"]

    get_data.write_manifest(spec, tmp_path, stats)
    assert get_data.outputs_current(spec, tmp_path)
    (tmp_path / spec.short_stream_path).write_text("stale", encoding="utf-8")
    assert not get_data.outputs_current(spec, tmp_path)
