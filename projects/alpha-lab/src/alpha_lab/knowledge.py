"""Curated topic knowledge built on top of the portable memory store.

Topic records are intended for shared institutional knowledge: data access
recipes, operational runbooks, gotchas, and other reusable notes that should be
useful to both humans and agents. The canonical topic document is a Markdown
file under ``{workspace}/.memory/topics/``; each save is also indexed as a normal
memory entry so existing memory search and prompt recall can find it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from alpha_lab.memory import MemoryEntry, MemoryStore, remember_text


@dataclass
class TopicRecord:
    """A curated institutional-knowledge topic."""

    topic: str
    title: str
    summary: str
    tags: list[str]
    owner: str | None
    last_verified: str | None
    sensitivity: str | None
    updated_at: str
    file: str


class TopicKnowledgeStore:
    """Manage curated topic records for a workspace."""

    def __init__(self, workspace: str) -> None:
        self.workspace = workspace
        self._workspace_path = Path(workspace)
        self._topics_dir = self._workspace_path / ".memory" / "topics"
        self._memory = MemoryStore(workspace)

    def save_topic(
        self,
        topic: str,
        content: str,
        *,
        title: str | None = None,
        summary: str | None = None,
        tags: list[str] | str | None = None,
        owner: str | None = None,
        last_verified: str | None = None,
        sensitivity: str | None = "internal",
    ) -> int | None:
        """Write a curated topic document and index it into memory.

        Re-saving identical content returns the existing memory ID. Re-saving
        changed content updates the current topic document and creates a new
        memory entry, preserving history while keeping ``.memory/topics`` as the
        latest human-readable version.
        """
        topic_slug = self.normalize_topic(topic)
        if not topic_slug:
            raise ValueError("topic must not be empty")
        content = (content or "").strip()
        if not content:
            raise ValueError("content must not be empty")

        title = (title or self._title_from_topic(topic_slug)).strip()
        summary = (summary or title).strip()
        normalized_tags = self._topic_tags(topic_slug, tags)
        record = TopicRecord(
            topic=topic_slug,
            title=title,
            summary=summary,
            tags=normalized_tags,
            owner=self._clean_optional(owner),
            last_verified=self._clean_optional(last_verified),
            sensitivity=self._clean_optional(sensitivity),
            updated_at=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            file=f"{topic_slug}.md",
        )

        self._topics_dir.mkdir(parents=True, exist_ok=True)
        topic_path = self._topics_dir / record.file
        source_path = str(topic_path.relative_to(self._workspace_path))
        existing_id = self._existing_topic_memory_id_if_unchanged(
            topic_path,
            source_path=source_path,
            proposed=record,
            proposed_content=content,
        )
        if existing_id is not None:
            return existing_id

        document = self._serialize_topic(record, content)
        MemoryStore._write_text_atomic(topic_path, document)

        return remember_text(
            self.workspace,
            document,
            tags=record.tags,
            summary=summary,
            kind="reference",
            phase=None,
            agent="topic_knowledge",
            source_path=source_path,
        )

    def list_topics(self) -> list[TopicRecord]:
        """List current curated topic documents, sorted by topic."""
        if not self._topics_dir.exists():
            return []
        records: list[TopicRecord] = []
        for path in sorted(self._topics_dir.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            record, _ = self._parse_topic_file(text, fallback_file=path.name)
            if record is not None:
                records.append(record)
        return records

    def read_topic(self, topic: str) -> str:
        """Read the current curated document for a topic."""
        topic_slug = self.normalize_topic(topic)
        if not topic_slug:
            return "[ERROR] topic must not be empty."
        path = self._topics_dir / f"{topic_slug}.md"
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return f"[ERROR] Topic '{topic_slug}' not found."

    def search_topics(self, query: str, *, limit: int = 10) -> list[MemoryEntry]:
        """Search indexed curated topic records."""
        return self._memory.search(query, tags=["topic"], kind="reference", limit=limit)

    @classmethod
    def normalize_topic(cls, topic: str | None) -> str:
        """Normalize a dotted/slashed topic name into a stable portable slug."""
        value = str(topic or "").strip().lower()
        value = re.sub(r"\s+", "_", value)
        value = re.sub(r"[^a-z0-9_.\-/-]", "", value)
        value = value.replace("/", ".")
        value = re.sub(r"[._-]+", lambda m: "." if "." in m.group(0) else m.group(0)[0], value)
        return value.strip("._-")

    @classmethod
    def topic_tag(cls, topic: str) -> str:
        """Return the stable tag used for a specific topic."""
        slug = cls.normalize_topic(topic)
        return "topic_" + re.sub(r"[^a-z0-9]+", "_", slug).strip("_")

    @classmethod
    def _topic_tags(cls, topic: str, tags: list[str] | str | None) -> list[str]:
        tag_values: list[str] = ["topic", cls.topic_tag(topic)]
        if isinstance(tags, str):
            tag_values.append(tags)
        elif tags:
            tag_values.extend(tags)
        return MemoryStore._normalize_tags(tag_values)

    @staticmethod
    def _clean_optional(value: str | None) -> str | None:
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    @staticmethod
    def _title_from_topic(topic: str) -> str:
        return topic.replace(".", " ").replace("_", " ").title()

    def _existing_topic_memory_id_if_unchanged(
        self,
        topic_path: Path,
        *,
        source_path: str,
        proposed: TopicRecord,
        proposed_content: str,
    ) -> int | None:
        try:
            text = topic_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        existing, body = self._parse_topic_file(text, fallback_file=topic_path.name)
        if existing is None:
            return None
        if not self._same_stable_topic(existing, body, proposed, proposed_content):
            return None
        for entry in self._memory.list_by_tag(self.topic_tag(proposed.topic)):
            if entry.source_path == source_path and entry.kind == "reference":
                return entry.id
        return None

    @classmethod
    def _same_stable_topic(
        cls,
        existing: TopicRecord,
        existing_body: str,
        proposed: TopicRecord,
        proposed_content: str,
    ) -> bool:
        return (
            existing.topic == proposed.topic
            and existing.title == proposed.title
            and existing.summary == proposed.summary
            and existing.tags == proposed.tags
            and existing.owner == proposed.owner
            and existing.last_verified == proposed.last_verified
            and existing.sensitivity == proposed.sensitivity
            and cls._content_from_topic_body(existing_body, existing.title) == proposed_content.strip()
        )

    @staticmethod
    def _content_from_topic_body(body: str, title: str) -> str:
        body = body.strip()
        heading = f"# {title}"
        if body == heading:
            return ""
        if body.startswith(heading + "\n"):
            return body[len(heading):].lstrip("\n").strip()
        return body

    @staticmethod
    def _serialize_topic(record: TopicRecord, content: str) -> str:
        metadata = {
            "topic": record.topic,
            "title": record.title,
            "summary": record.summary,
            "tags": record.tags,
            "owner": record.owner,
            "last_verified": record.last_verified,
            "sensitivity": record.sensitivity,
            "updated_at": record.updated_at,
        }
        metadata = {key: value for key, value in metadata.items() if value not in (None, [], "")}
        return f"---\n{json.dumps(metadata, indent=2)}\n---\n\n# {record.title}\n\n{content.strip()}\n"

    @classmethod
    def _parse_topic_file(cls, text: str, *, fallback_file: str) -> tuple[TopicRecord | None, str]:
        metadata: dict[str, Any] = {}
        body = text
        if text.startswith("---\n"):
            end = text.find("\n---\n", 4)
            if end != -1:
                metadata_text = text[4:end].strip()
                body = text[end + len("\n---\n"):].lstrip("\n")
                try:
                    parsed = json.loads(metadata_text)
                    if isinstance(parsed, dict):
                        metadata = parsed
                except json.JSONDecodeError:
                    metadata = {}

        topic = cls.normalize_topic(metadata.get("topic") or Path(fallback_file).stem)
        if not topic:
            return None, body
        tags = MemoryStore._normalize_tags(metadata.get("tags") or ["topic", cls.topic_tag(topic)])
        record = TopicRecord(
            topic=topic,
            title=str(metadata.get("title") or cls._title_from_topic(topic)),
            summary=str(metadata.get("summary") or metadata.get("title") or cls._title_from_topic(topic)),
            tags=tags,
            owner=cls._clean_optional(metadata.get("owner")),
            last_verified=cls._clean_optional(metadata.get("last_verified")),
            sensitivity=cls._clean_optional(metadata.get("sensitivity")),
            updated_at=str(metadata.get("updated_at") or ""),
            file=fallback_file,
        )
        return record, body


def today_iso() -> str:
    """Return today's date for CLI defaults and tests."""
    return date.today().isoformat()
