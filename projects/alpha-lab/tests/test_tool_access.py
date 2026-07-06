"""Tests for alpha_lab.tools.access — minimal workspace footprint derivation.

Focuses on symlink-closure behavior in
``build_minimal_workspace_access_schema_for_tools`` plus the subsumption /
edge-case rules. Tools declare a ``workspace_access`` mapping of workspace-relative
path to effect (``ro``/``rw``). Uses the real filesystem (``tmp_path``) and real
``ToolDefinition`` instances; no mocking is needed. ``tmp_path`` is resolved up front
so comparisons against ``.resolve()`` outputs are stable on platforms where the temp
dir is itself a symlink (e.g. macOS).
"""

from __future__ import annotations

import dataclasses

from pathlib import Path

import pytest

from alpha_lab.tools.access import (
    WorkspaceAccess,
    build_minimal_workspace_access_schema_for_tools,
    path_covered_by,
)
from alpha_lab.tools.tool_definition import ToolDefinition, ToolEffect


def make_tool(name: str, access: dict[str, ToolEffect] | None = None) -> ToolDefinition:
    """Build a ToolDefinition with the given {relative-path: effect} footprint."""
    return ToolDefinition(
        name=name,
        description=f"{name} description",
        parameters={},
        workspace_access=dict(access or {}),
    )


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """A resolved workspace directory so .resolve() comparisons are stable."""
    ws = tmp_path.resolve() / "workspace"
    ws.mkdir()
    return ws


# --------------------------------------------------------------------------
# path_covered_by
# --------------------------------------------------------------------------


def test_path_covered_by_equal_path_returns_true() -> None:
    """A path is covered by an identical path."""
    path = Path("/a/b")
    assert path_covered_by(path, [Path("/a/b")]) is True


def test_path_covered_by_descendant_returns_true() -> None:
    """A path is covered by one of its ancestors."""
    assert path_covered_by(Path("/a/b/c"), [Path("/a/b")]) is True


def test_path_covered_by_sibling_returns_false() -> None:
    """Sibling paths do not cover each other."""
    assert path_covered_by(Path("/a/c"), [Path("/a/b")]) is False


def test_path_covered_by_unrelated_returns_false() -> None:
    """Paths in unrelated subtrees do not cover each other."""
    assert path_covered_by(Path("/x/y"), [Path("/a/b")]) is False


def test_path_covered_by_empty_ancestors_returns_false() -> None:
    """With no ancestors to check against, nothing covers the path."""
    assert path_covered_by(Path("/a/b"), []) is False


# --------------------------------------------------------------------------
# Empty / no-access inputs
# --------------------------------------------------------------------------


def test_empty_tools_yields_empty_access(workspace: Path) -> None:
    """No tools granted -> both the read-only and read-write sets are empty."""
    access = build_minimal_workspace_access_schema_for_tools([], workspace)
    assert access.ro == frozenset()
    assert access.rw == frozenset()


def test_tool_with_no_workspace_access_contributes_nothing(workspace: Path) -> None:
    """A tool with an empty ``workspace_access`` mapping adds no paths."""
    tool = make_tool("noop")
    access = build_minimal_workspace_access_schema_for_tools([tool], workspace)
    assert access.ro == frozenset()
    assert access.rw == frozenset()


# --------------------------------------------------------------------------
# Path resolution / basic RO / RW
# --------------------------------------------------------------------------


def test_ro_path_resolves_under_workspace(workspace: Path) -> None:
    """A read-only footprint path resolves to ``<workspace>/<path>``."""
    (workspace / "adapter").mkdir()
    tool = make_tool("reader", {"adapter": ToolEffect.RO})
    access = build_minimal_workspace_access_schema_for_tools([tool], workspace)
    assert access.ro == frozenset({workspace / "adapter"})
    assert access.rw == frozenset()


def test_distinct_rw_paths_all_present_when_unnested(workspace: Path) -> None:
    """Sibling read-write paths are all kept (none is an ancestor of another).

    ``experiments.db``, ``.memory`` and ``playbook.md`` sit side by side under the
    workspace, so subsumption drops nothing.
    """
    tool = make_tool(
        "writer",
        {"experiments.db": ToolEffect.RW, ".memory": ToolEffect.RW, "playbook.md": ToolEffect.RW},
    )
    access = build_minimal_workspace_access_schema_for_tools([tool], workspace)
    assert access.rw == frozenset(
        {
            workspace / "experiments.db",
            workspace / ".memory",
            workspace / "playbook.md",
        }
    )
    assert access.ro == frozenset()


def test_multiple_tools_footprints_union(workspace: Path) -> None:
    """Footprints from several granted tools union into one access schema."""
    (workspace / "adapter").mkdir()
    reader = make_tool("reader", {"adapter": ToolEffect.RO})
    writer = make_tool("writer", {"experiments.db": ToolEffect.RW})
    access = build_minimal_workspace_access_schema_for_tools([reader, writer], workspace)
    assert access.ro == frozenset({workspace / "adapter"})
    assert access.rw == frozenset({workspace / "experiments.db"})


def test_per_path_effects_within_one_tool_bucket_separately(workspace: Path) -> None:
    """A single tool may declare different effects per path; each is bucketed by its
    own effect (read-only vs read-write), not by a single tool-wide effect."""
    (workspace / "adapter").mkdir()
    tool = make_tool("mixed", {"adapter": ToolEffect.RO, "experiments.db": ToolEffect.RW})
    access = build_minimal_workspace_access_schema_for_tools([tool], workspace)
    assert access.ro == frozenset({workspace / "adapter"})
    assert access.rw == frozenset({workspace / "experiments.db"})


def test_workspace_accepts_str_and_path_equivalently(workspace: Path) -> None:
    """The ``workspace`` argument may be a ``str`` or a ``Path`` with identical results."""
    (workspace / "adapter").mkdir()
    tool = make_tool("reader", {"adapter": ToolEffect.RO})
    from_path = build_minimal_workspace_access_schema_for_tools([tool], workspace)
    from_str = build_minimal_workspace_access_schema_for_tools([tool], str(workspace))
    assert from_path == from_str


def test_arbitrary_relative_paths_resolve_under_workspace(workspace: Path) -> None:
    """Any declared relative path (including nested ones) resolves verbatim under the
    workspace — there is no fixed/validated set of allowed path names."""
    tool = make_tool("custom", {"some_dir": ToolEffect.RO, "nested/path.txt": ToolEffect.RO})
    access = build_minimal_workspace_access_schema_for_tools([tool], workspace)
    assert access.ro == frozenset({workspace / "some_dir", workspace / "nested/path.txt"})
    assert access.rw == frozenset()


def test_subpath_footprint_does_not_pull_in_workspace_root(workspace: Path) -> None:
    """A footprint on a sub-path mounts only that sub-path, never the workspace root.

    The closure expands downward (path + target + reachable symlinks) and subsumption
    only drops paths, so an ancestor is never synthesized from a child footprint.
    """
    tool = make_tool("writer", {"some_dir": ToolEffect.RW})
    access = build_minimal_workspace_access_schema_for_tools([tool], workspace)
    assert access.rw == frozenset({workspace / "some_dir"})
    assert workspace not in access.rw
    assert workspace not in access.ro


# --------------------------------------------------------------------------
# Subsumption rules
# --------------------------------------------------------------------------


def test_rw_root_collapses_entire_set(workspace: Path) -> None:
    """A read-write footprint on ``.`` (the workspace root) subsumes every other
    footprint, collapsing the whole schema to ``rw={workspace}`` and ``ro={}``."""
    (workspace / "adapter").mkdir()
    root_writer = make_tool("root_writer", {".": ToolEffect.RW})
    sub_writer = make_tool("sub_writer", {"experiments.db": ToolEffect.RW})
    reader = make_tool("reader", {"adapter": ToolEffect.RO})
    access = build_minimal_workspace_access_schema_for_tools(
        [root_writer, sub_writer, reader], workspace
    )
    assert access.rw == frozenset({workspace})
    assert access.ro == frozenset()


def test_rw_subsumes_ro_for_same_subtree(workspace: Path) -> None:
    """When the same subtree is requested both read-only and read-write, the
    read-write grant wins and the read-only entry is dropped."""
    (workspace / "adapter").mkdir()
    reader = make_tool("reader", {"adapter": ToolEffect.RO})
    writer = make_tool("writer", {"adapter": ToolEffect.RW})
    access = build_minimal_workspace_access_schema_for_tools([reader, writer], workspace)
    assert access.rw == frozenset({workspace / "adapter"})
    assert access.ro == frozenset()


# --------------------------------------------------------------------------
# Symlink closure
# --------------------------------------------------------------------------


def test_plain_path_contributes_single_entry(workspace: Path) -> None:
    """A non-symlink directory contributes exactly one mount (link == resolved).

        <ws>/
        └── experiments/        (footprint, rw)
            └── note.txt
    """
    experiments = workspace / "experiments"
    experiments.mkdir()
    (experiments / "note.txt").write_text("data")
    tool = make_tool("writer", {"experiments": ToolEffect.RW})
    access = build_minimal_workspace_access_schema_for_tools([tool], workspace)
    assert access.rw == frozenset({experiments})


def test_dir_with_symlink_to_external_file_includes_target_prunes_link(
    workspace: Path, tmp_path: Path
) -> None:
    """A symlink inside the footprint dir, pointing at an external file, pulls the
    resolved target into the mount set; the in-tree link path is pruned because the
    mounted parent dir already covers it.

        <tmp>/
        ├── external_file.txt                       (resolved target — kept)
        └── workspace/  (= <ws>)
            └── experiments/                         (footprint, rw — mounted)
                └── link.txt  ──►  <tmp>/external_file.txt   (link pruned)
    """
    external_target = (tmp_path.resolve() / "external_file.txt")
    external_target.write_text("external")

    experiments = workspace / "experiments"
    experiments.mkdir()
    link = experiments / "link.txt"
    link.symlink_to(external_target)

    tool = make_tool("writer", {"experiments": ToolEffect.RW})
    access = build_minimal_workspace_access_schema_for_tools([tool], workspace)

    assert experiments in access.rw
    assert external_target in access.rw
    assert link not in access.rw


def test_dir_with_symlink_to_external_dir_walks_target_contents(
    workspace: Path, tmp_path: Path
) -> None:
    """A symlink to an external *directory* is walked all the way down: a symlink
    nested inside that external dir contributes its own resolved target too.

        <tmp>/
        ├── external_dir/                            (resolved target — kept, walked)
        │   └── nested_link.txt  ──►  <tmp>/nested_target.txt
        ├── nested_target.txt                        (reached via the walk — kept)
        └── workspace/
            └── experiments/                         (footprint, rw)
                └── ext  ──►  <tmp>/external_dir
    """
    external_dir = tmp_path.resolve() / "external_dir"
    external_dir.mkdir()
    nested_target = tmp_path.resolve() / "nested_target.txt"
    nested_target.write_text("nested")
    (external_dir / "nested_link.txt").symlink_to(nested_target)

    experiments = workspace / "experiments"
    experiments.mkdir()
    (experiments / "ext").symlink_to(external_dir, target_is_directory=True)

    tool = make_tool("writer", {"experiments": ToolEffect.RW})
    access = build_minimal_workspace_access_schema_for_tools([tool], workspace)

    assert experiments in access.rw
    assert external_dir in access.rw
    # The nested link's resolved target is reached by walking the external dir.
    assert nested_target in access.rw


def test_footprint_path_itself_is_symlink_returns_both_link_and_target(
    workspace: Path, tmp_path: Path
) -> None:
    """When the footprint path is *itself* a symlink, both the link path and its
    resolved target are returned (neither covers the other — they are siblings in the
    tree), and the linked dir is still walked down.

        <tmp>/
        ├── real_experiments/                        (resolved target — kept)
        │   └── deep_link.txt  ──►  <tmp>/deep_target.txt
        ├── deep_target.txt                          (reached via the walk — kept)
        └── workspace/
            └── experiments  ──►  <tmp>/real_experiments   (link path — also kept)
    """
    external_dir = tmp_path.resolve() / "real_experiments"
    external_dir.mkdir()
    nested_target = tmp_path.resolve() / "deep_target.txt"
    nested_target.write_text("deep")
    (external_dir / "deep_link.txt").symlink_to(nested_target)

    # The footprint path 'experiments' is itself a symlink to an external dir.
    experiments = workspace / "experiments"
    experiments.symlink_to(external_dir, target_is_directory=True)

    tool = make_tool("writer", {"experiments": ToolEffect.RW})
    access = build_minimal_workspace_access_schema_for_tools([tool], workspace)

    # Neither the link path nor the real target covers the other (siblings in
    # the tree), so both are returned; the linked dir is walked down.
    assert experiments in access.rw
    assert external_dir in access.rw
    assert nested_target in access.rw


def test_symlink_cycle_terminates(workspace: Path, tmp_path: Path) -> None:
    """A symlink cycle (a ⇄ b) is walked without infinite recursion; the closure
    guard terminates and both ends of the cycle are mounted.

        <tmp>/
        ├── dir_a/
        │   └── to_b  ──►  <tmp>/dir_b
        ├── dir_b/
        │   └── to_a  ──►  <tmp>/dir_a               (cycles back to dir_a)
        └── workspace/
            └── experiments  ──►  <tmp>/dir_a        (footprint, rw)
    """
    dir_a = tmp_path.resolve() / "dir_a"
    dir_b = tmp_path.resolve() / "dir_b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "to_b").symlink_to(dir_b, target_is_directory=True)
    (dir_b / "to_a").symlink_to(dir_a, target_is_directory=True)

    experiments = workspace / "experiments"
    experiments.symlink_to(dir_a, target_is_directory=True)

    tool = make_tool("writer", {"experiments": ToolEffect.RW})
    # The cycle guard must prevent infinite recursion; assert it simply returns.
    access = build_minimal_workspace_access_schema_for_tools([tool], workspace)
    assert isinstance(access, WorkspaceAccess)
    assert dir_a in access.rw
    assert dir_b in access.rw


# --------------------------------------------------------------------------
# WorkspaceAccess type guarantees
# --------------------------------------------------------------------------


def test_workspace_access_returns_frozensets(workspace: Path) -> None:
    """The returned ``ro`` and ``rw`` sets are ``frozenset`` instances."""
    (workspace / "adapter").mkdir()
    tool = make_tool("reader", {"adapter": ToolEffect.RO})
    access = build_minimal_workspace_access_schema_for_tools([tool], workspace)
    assert isinstance(access.ro, frozenset)
    assert isinstance(access.rw, frozenset)


def test_workspace_access_is_frozen() -> None:
    """``WorkspaceAccess`` is immutable — assigning to a field raises."""
    access = WorkspaceAccess(ro=frozenset(), rw=frozenset())
    with pytest.raises(dataclasses.FrozenInstanceError):
        access.ro = frozenset({Path("/x")})  # type: ignore[misc]
