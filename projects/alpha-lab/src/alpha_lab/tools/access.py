"""Derive an agent's workspace mount schema from its tools.

Each :class:`~alpha_lab.tools.tool_definition.ToolDefinition` declares a
``workspace_access`` mapping of workspace-relative path to ``effect`` (read-only /
read-write). :func:`build_minimal_workspace_access_schema_for_tools` unions those
footprints across the tools an agent is granted (its ``allowed-tools``) into the
minimal set of read-only and read-write paths the agent needs — the input a sandbox
runtime mounts.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from alpha_lab.tools.tool_definition import ToolDefinition, ToolEffect


@dataclass(frozen=True)
class WorkspaceAccess:
    """The minimal read-only and read-write workspace paths an agent needs.

    Each set holds both a symlink path and its resolved real target where the two
    differ, so a mount works whether referenced by its link or its real location.
    """

    ro: frozenset[Path]
    rw: frozenset[Path]


def path_covered_by(path: Path, ancestors: Iterable[Path]) -> bool:
    """True if *path* equals, or is nested under, any path in *ancestors*."""
    return any(path == ancestor or path.is_relative_to(ancestor) for ancestor in ancestors)


def _mount_closure(roots: set[Path]) -> set[Path]:
    """Expand *roots* to every path that must be mounted to make them usable.

    For each path we add the path itself and its resolved real target (so a symlink
    contributes both its link and its real location). Directories are walked without
    following symlinks; every symlink found inside is expanded the same way — all the
    way down the tree — so a directory pulls in the real targets of the symlinks it
    contains, and symlinked directories have their contents walked via the resolved
    target. ``mounts``/``walked`` membership guards against symlink cycles.
    """
    mounts: set[Path] = set()
    walked: set[Path] = set()
    pending = list(roots)
    while pending:
        path = pending.pop()
        if path in mounts:
            continue
        mounts.add(path)
        real = path.resolve()
        mounts.add(real)
        if real.is_dir() and real not in walked:
            walked.add(real)
            for parent, dir_names, file_names in os.walk(real, followlinks=False):
                for name in (*dir_names, *file_names):
                    entry = Path(parent) / name
                    if entry.is_symlink():
                        pending.append(entry)
    return mounts


def _resolve_path(workspace: Path, rel_path: str) -> Path:
    """Resolve a workspace-relative footprint path (``.`` / ``""`` is the root itself)."""
    return workspace if rel_path in (".", "") else workspace / rel_path


def build_minimal_workspace_access_schema_for_tools(
    tools: Iterable[ToolDefinition], workspace: Path | str
) -> WorkspaceAccess:
    """Build the workspace mount schema for an agent's granted *tools*.

    Each footprint is expanded to its mount closure — the path, its resolved real
    target, and every symlink (and target) reachable by walking it, all the way down
    — so symlinks anywhere in the tree are mounted at both their link and real
    locations. An ancestor path then subsumes its descendants within the same bucket,
    and a read-write path subsumes a read-only request for the same subtree (so a
    read-write workspace root collapses the whole set to ``rw={root}``).
    """
    workspace = Path(workspace)
    ro: set[Path] = set()
    rw: set[Path] = set()
    for tool in tools:
        for rel_path, effect in tool.workspace_access.items():
            bucket = rw if effect is ToolEffect.RW else ro
            bucket.add(_resolve_path(workspace, rel_path))

    rw_mounts = _mount_closure(rw)
    ro_mounts = _mount_closure(ro)
    minimal_rw = {path for path in rw_mounts if not path_covered_by(path, rw_mounts - {path})}
    minimal_ro = {
        path
        for path in ro_mounts
        if not path_covered_by(path, minimal_rw)
        and not path_covered_by(path, ro_mounts - {path})
    }
    return WorkspaceAccess(ro=frozenset(minimal_ro), rw=frozenset(minimal_rw))
