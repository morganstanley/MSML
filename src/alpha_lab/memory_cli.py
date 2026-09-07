"""Small CLI for workspace memory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alpha_lab.constants import MemoryKind
from alpha_lab.databases import ModelDB
from alpha_lab.memory import Memory, MemoryStore, SearchMode


def _store(workspace: str) -> MemoryStore:
    root = Path(workspace) / ".alpha_lab" / "memory"
    return MemoryStore(ModelDB(root=root, model_type=Memory))


def _print_entries(entries: list[tuple[int, Memory]]) -> None:
    if not entries:
        print("No matching memories found.")
        return
    for key, entry in entries:
        meta = []
        if entry.kind:
            meta.append(f"kind={entry.kind}")
        if entry.sources:
            meta.append(f"sources={','.join(entry.sources)}")
        tags = f" tags={','.join(entry.tags)}" if entry.tags else ""
        suffix = f" ({'; '.join(meta)})" if meta else ""
        print(f"#{key}: {entry.summary}{suffix}{tags}")


def _read_store_content(args: argparse.Namespace) -> str:
    if args.content is not None:
        return args.content
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("content is required: pass --content or pipe stdin")


def _cmd_store(args: argparse.Namespace) -> int:
    content = _read_store_content(args).strip()
    if not content:
        raise SystemExit("content must not be empty")
    try:
        fields = {
            "content": content,
            "summary": args.summary,
            "tags": args.tag,
            "kind": args.kind,
            "agent": "memory_cli",
            "sources": args.source,
            "frozen": args.frozen,
        }
        if args.owner:
            fields["owner"] = args.owner
        memory_id = _store(args.workspace).add(Memory(**fields))
    except (RuntimeError, ValueError) as e:
        raise SystemExit(str(e)) from e
    print(f"Memory #{memory_id} stored.")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    include = {
        k: v
        for k, v in {
            "tags": args.tag,
            "kind": args.kind,
            "agent": args.agent,
            "run_id": args.run_id,
            "sources": args.source,
            "owner": args.owner,
        }.items()
        if v
    }
    try:
        entries = _store(args.workspace).search(
            args.query,
            mode=args.mode,
            include=include,
            limit=args.limit,
        )
    except (RuntimeError, ValueError) as e:
        raise SystemExit(str(e)) from e
    _print_entries(entries)
    return 0


def _cmd_read(args: argparse.Namespace) -> int:
    try:
        memory = _store(args.workspace).get(args.memory_id)
    except (RuntimeError, ValueError) as e:
        raise SystemExit(str(e)) from e
    if memory is None:
        raise SystemExit(f"Memory #{args.memory_id} not found.")
    print(memory.render(f"Memory #{args.memory_id}: {memory.summary}"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alpha-lab-memory",
        description="Search lightweight workspace memory.",
    )
    parser.add_argument("--workspace", default=".", help="Workspace path containing .alpha_lab/memory/ (default: current directory).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    store = subparsers.add_parser("store", help="Store a manual memory entry.")
    store.add_argument("--content", help="Memory body text. If omitted, stdin is used when piped.")
    store.add_argument("--summary", required=True, help="One-line summary for search results.")
    store.add_argument("--kind", required=True, type=MemoryKind, help="Memory kind.")
    store.add_argument("--tag", action="append", required=True, help="Search tag; may be repeated.")
    store.add_argument("--owner", help="Memory owner. Defaults to the current OS user.")
    store.add_argument("--source", action="append", default=[], help="Source reference; may be repeated.")
    store.add_argument("--frozen", action="store_true", help="Prevent this memory record from being edited or removed later.")
    store.set_defaults(func=_cmd_store)

    search = subparsers.add_parser("search", help="Search all memory entries.")
    search.add_argument("query", nargs="?", help="Omit for recency ordering.")
    search.add_argument("--mode", type=SearchMode, choices=tuple(SearchMode))
    search.add_argument("--tag", action="append", default=[])
    search.add_argument("--kind", type=MemoryKind)
    search.add_argument("--agent")
    search.add_argument("--run-id")
    search.add_argument("--source", action="append", default=[])
    search.add_argument("--owner")
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
