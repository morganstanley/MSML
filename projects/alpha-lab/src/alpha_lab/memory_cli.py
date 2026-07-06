"""Small CLI for workspace memory and curated topic knowledge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alpha_lab.knowledge import TopicKnowledgeStore, today_iso
from alpha_lab.memory import MemoryStore


def _read_content(args: argparse.Namespace) -> str:
    if args.file:
        try:
            return Path(args.file).read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise SystemExit(f"could not read {args.file}: {e}") from e
    if args.content:
        return args.content
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("content is required: pass --file, --content, or stdin")


def _print_entries(entries) -> None:  # type: ignore[no-untyped-def]
    if not entries:
        print("No matching memories found.")
        return
    for entry in entries:
        meta = []
        if entry.kind:
            meta.append(f"kind={entry.kind}")
        if entry.phase:
            meta.append(f"phase={entry.phase}")
        if entry.source_path:
            meta.append(f"source={entry.source_path}")
        tags = f" tags={','.join(entry.tags)}" if entry.tags else ""
        suffix = f" ({'; '.join(meta)})" if meta else ""
        print(f"#{entry.id}: {entry.summary}{suffix}{tags}")


def _cmd_topic_add(args: argparse.Namespace) -> int:
    store = TopicKnowledgeStore(args.workspace)
    memory_id = store.save_topic(
        args.topic,
        _read_content(args),
        title=args.title,
        summary=args.summary,
        tags=args.tag,
        owner=args.owner,
        last_verified=args.last_verified or (today_iso() if args.mark_verified else None),
        sensitivity=args.sensitivity,
    )
    print(f"Saved topic '{TopicKnowledgeStore.normalize_topic(args.topic)}' as memory #{memory_id}.")
    return 0


def _cmd_topic_list(args: argparse.Namespace) -> int:
    records = TopicKnowledgeStore(args.workspace).list_topics()
    if not records:
        print("No topic records found.")
        return 0
    for record in records:
        bits = [record.topic]
        if record.owner:
            bits.append(f"owner={record.owner}")
        if record.last_verified:
            bits.append(f"verified={record.last_verified}")
        print(f"- {record.title} ({'; '.join(bits)})")
    return 0


def _cmd_topic_read(args: argparse.Namespace) -> int:
    print(TopicKnowledgeStore(args.workspace).read_topic(args.topic))
    return 0


def _cmd_topic_search(args: argparse.Namespace) -> int:
    entries = TopicKnowledgeStore(args.workspace).search_topics(args.query, limit=args.limit)
    _print_entries(entries)
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    entries = MemoryStore(args.workspace).search(
        args.query,
        tags=args.tag,
        kind=args.kind,
        phase=args.phase,
        limit=args.limit,
    )
    _print_entries(entries)
    return 0


def _cmd_read(args: argparse.Namespace) -> int:
    print(MemoryStore(args.workspace).read(args.memory_id))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alpha-lab-memory",
        description="Manage lightweight workspace memory and curated topic knowledge.",
    )
    parser.add_argument("--workspace", default=".", help="Workspace path containing .memory/ (default: current directory).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    topic = subparsers.add_parser("topic", help="Manage curated institutional topic records.")
    topic_sub = topic.add_subparsers(dest="topic_command", required=True)

    topic_add = topic_sub.add_parser("add", help="Add or update a curated topic record.")
    topic_add.add_argument("topic", help="Stable topic name, e.g. data_access.exchange_rates")
    topic_add.add_argument("--file", help="Markdown file containing the topic body.")
    topic_add.add_argument("--content", help="Topic body text. If omitted, stdin is used when piped.")
    topic_add.add_argument("--title", help="Human-readable topic title.")
    topic_add.add_argument("--summary", help="One-line summary for search results.")
    topic_add.add_argument("--tag", action="append", default=[], help="Additional tag; may be repeated.")
    topic_add.add_argument("--owner", help="Owner or team responsible for keeping this current.")
    topic_add.add_argument("--last-verified", help="Date this information was last verified, YYYY-MM-DD.")
    topic_add.add_argument("--mark-verified", action="store_true", help="Set --last-verified to today's date.")
    topic_add.add_argument("--sensitivity", default="internal", help="Sensitivity label (default: internal).")
    topic_add.set_defaults(func=_cmd_topic_add)

    topic_list = topic_sub.add_parser("list", help="List current topic records.")
    topic_list.set_defaults(func=_cmd_topic_list)

    topic_read = topic_sub.add_parser("read", help="Read the current document for a topic.")
    topic_read.add_argument("topic")
    topic_read.set_defaults(func=_cmd_topic_read)

    topic_search = topic_sub.add_parser("search", help="Search curated topic records.")
    topic_search.add_argument("query")
    topic_search.add_argument("--limit", type=int, default=10)
    topic_search.set_defaults(func=_cmd_topic_search)

    search = subparsers.add_parser("search", help="Search all memory entries.")
    search.add_argument("query")
    search.add_argument("--tag", action="append", default=[])
    search.add_argument("--kind")
    search.add_argument("--phase")
    search.add_argument("--limit", type=int, default=10)
    search.set_defaults(func=_cmd_search)

    read = subparsers.add_parser("read", help="Read a memory entry by ID.")
    read.add_argument("memory_id", type=int)
    read.set_defaults(func=_cmd_read)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
