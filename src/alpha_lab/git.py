"""Git repository helpers."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from dataclasses import InitVar, dataclass
from functools import cached_property
from pathlib import Path
from typing import Generic, Self, TypeVar
from warnings import warn

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

from alpha_lab.utils import NetworkPath

PathT = TypeVar("PathT", bound=str | Path)


@dataclass(frozen=True)
class GitPointer(Generic[PathT]):
    """Pointer for a Git repository with an optional reference.

    Attributes:
        src: Repository address.
        ref: Optional branch, tag, or commit-ish selector.
    """

    src: PathT | None = None
    ref: str | None = None
    validate_args: InitVar[bool] = True

    def __post_init__(self, validate_args: bool) -> None:
        if self.src is None:
            return

        # Coerce a string-typed local source to a Path.
        if isinstance(self.src, str) and not isinstance(self.src, NetworkPath):
            warn(f"Coercing {self.src!r} to a Path object", stacklevel=2)
            object.__setattr__(self, "src", Path(self.src))

        # Check for Path-typed remote sources
        if (
            validate_args
            and isinstance(self.src, Path)
            and isinstance(str(self.src), NetworkPath)
        ):
            msg = f"Path-typed src={self.src} cannot reference a remote."
            raise TypeError(msg)

    def exists(self) -> bool:
        """Whether the source exists: a local directory, or a reachable remote.

        Local sources are checked on disk; remotes are probed with
        ``git ls-remote`` (no clone). ``GIT_TERMINAL_PROMPT=0`` makes a private
        or missing remote fail fast rather than block on a credential prompt.
        """
        if self.is_local():
            return self.src is not None and self.src.is_dir()
        try:
            return subprocess.run(
                ["git", "ls-remote", str(self.src)],
                capture_output=True,
                check=False,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
                timeout=10,
            ).returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def is_local(self) -> bool:
        """Whether the instance points to a locally sourced repository."""
        return not isinstance(self.src, NetworkPath)


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class GitSpec:
    """Desired git settings Alpha Lab cares about for a repository.

    Attributes:
        remote: Optional remote address for the repository.
        upstream: Optional parent/source repository address.
        user_name: Optional git ``user.name`` for commits. Empty string (the
            default) means "resolve from the global git config"; ``None`` means
            explicitly unset; any other string is used verbatim.
        user_email: Optional git ``user.email`` for commits. Same semantics as
            ``user_name``.
    """

    remote: GitPointer[Path | str] | None = None
    upstream: GitPointer[Path | str] | None = None
    gitignore: Sequence[Path | str] = ()
    user_name: str | None = ""
    user_email: str | None = ""
    validate_args: InitVar[bool] = True

    def __post_init__(self, validate_args: bool) -> None:
        object.__setattr__(self, "gitignore", tuple(self.gitignore))
        for key in ("user_name", "user_email"):
            # "" means "not specified" -> fall back to the global git config;
            # None (explicit unset) and any real value are left as-is.
            if getattr(self, key) != "":
                continue
            result = subprocess.run(
                ["git", "config", "--global", "--get", key.replace("_", ".")],
                text=True,
                capture_output=True,
                check=False,
            )
            value = result.stdout.strip() if result.returncode == 0 else None
            object.__setattr__(self, key, value)
        if not validate_args:
            return

        if self.upstream and not self.upstream.exists():
            msg = f"Upstream {self.upstream!r} does not exist."
            raise ValueError(msg)


@dataclass(frozen=True)
class GitRepository:
    """Small wrapper around a git repository managed by a GitSpec."""
    root: Path
    spec: GitSpec
    head: InitVar[str | None] = None

    def __post_init__(self, head: str | None) -> None:
        object.__setattr__(self, "_prev_hash", head)

        dest = getattr(self.spec.remote, "src", None)
        if isinstance(dest, Path):
            dest = str(dest.resolve())

        object.__setattr__(
            self,
            "_origin",
            None if dest is None or dest == str(self.root.resolve()) else dest
        )

    def init(self, exist_ok: bool = False) -> Self:
        """Initialize a new git repository at ``root`` (no commits).

        This is just ``git init`` — it creates the repository but makes no
        commit, so there is no ``HEAD`` until the first commit is made.

        Args:
            exist_ok: If False, raise an error if a repository already exists at ``root``.

        Returns:
            This repository.
        """
        if self.is_initialized():
            msg = f"Cannot re-initialize the repository at {self.root!r}."
            raise RuntimeError(msg)

        if self.spec.upstream:
            msg = "Cannot initialize a repository with an upstream; use clone instead."
            raise RuntimeError(msg)

        self.root.mkdir(parents=True, exist_ok=exist_ok)
        self._run("init")
        return self

    def clone(self) -> Self:
        """Clone ``spec.upstream`` into ``root``.

        Returns:
            This repository.
        """
        upstream = self.spec.upstream
        if upstream is None:
            raise ValueError("clone requires an upstream repository")

        self.root.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", str(upstream.src), str(self.root)],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"git clone failed: {detail}")
        object.__setattr__(self, "_prev_hash", self.curr_hash())
        return self

    def setup(self) -> Self:
        """Ensure the repository exists and is on its spec's ref, then return it.

        Reuses an already-initialized repo (in-place / reopen), clones when the
        spec has an upstream, or initializes a fresh repo with a base commit; then
        checks out ``spec.upstream.ref`` if set, so the in-place case lands on it.

        Returns:
            This repository.
        """
        if not self.is_initialized():
            if self.spec.upstream is not None:
                self.clone()
            else:
                self.init(exist_ok=True)

        # Point the ``upstream`` (pull) and ``origin`` (push) remotes at the spec.
        remotes = self._run("remote").stdout.split()
        if self.spec.upstream:
            verb = "set-url" if "upstream" in remotes else "add"
            self._run("remote", verb, "upstream", str(self.spec.upstream.src))
            if self.spec.upstream.ref is not None:
                self.checkout(self.spec.upstream.ref)

        if self._origin:
            verb = "set-url" if "origin" in remotes else "add"
            self._run("remote", verb, "origin", self._origin)

        if self.spec.remote is not None and self.spec.remote.ref is not None:
            self.checkout(self.spec.remote.ref, create=True)

        ignore_path = self.root / ".gitignore"
        if ignore_path.is_file():
            with (self.root / ".gitignore").open("r") as f:
                ignored = f.read().splitlines()
        else:
            ignored = []

        with ignore_path.open("a") as f:
            for item in self.spec.gitignore:
                line = item if isinstance(item, str) else str(item.relative_to(self.root))
                if line not in ignored:
                    f.write(line + "\n")

        self.add(".gitignore")
        # Commit only when the index has changes after setup() stages .gitignore.
        # As with any `git commit`, this also includes changes already staged by
        # the caller. Guarding on is_dirty() would also fire for *unstaged*
        # working-tree changes, so `git commit` could run with nothing staged,
        # fail, and stall the memory store. has_staged_changes() reflects only
        # what a commit would record.
        if self.can_commit and self.has_staged_changes():
            self.commit("Initial commit")

        return self

    def curr_hash(self) -> str | None:
        """Return the current HEAD commit hash, or None if the repo has no commits yet."""
        result = self._run("rev-parse", "HEAD", check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    def is_stale(self) -> bool:
        """Whether HEAD has moved since it was last synced."""
        return self.prev_hash != self.curr_hash()

    def has_staged_changes(self) -> bool:
        """Whether the index holds changes staged for the next commit.

        Distinct from :meth:`is_dirty`, which is also true for *unstaged*
        working-tree changes; this reflects only what a ``commit`` would record.
        """
        return self._run("diff", "--cached", "--quiet", check=False).returncode != 0

    def is_dirty(self) -> bool:
        """Whether the working tree or index has uncommitted changes."""
        unstaged = self._run("diff", "--quiet", check=False).returncode != 0
        return unstaged or self.has_staged_changes()

    def is_initialized(self) -> bool:
        """Whether ``root`` is a usable git repository (initialized, with a commit).

        Requires a resolvable ``HEAD``, not merely a ``.git`` directory: the
        wrapper's operations need ``HEAD`` and ``init`` always writes a base
        commit. ``rev-parse`` is authoritative (it also fails outside a repo); the
        ``root`` guard just keeps it from running with a nonexistent ``cwd``.
        """
        if not self.root.is_dir():
            return False
        return self._run("rev-parse", "--verify", "HEAD", check=False).returncode == 0

    def add(
        self,
        *paths: str | Path,
        flags: str | list[str] | None = None,
    ) -> None:
        """Stage paths in the repository."""
        args = [
            "add",
            *([] if flags is None else [flags] if isinstance(flags, str) else flags)
        ]
        if paths:
            args.append("--")
            args.extend(str(path) for path in paths)
        self._run(*args)

    def commit(self, message: str) -> str:
        """Commit staged changes and update ``head``.

        Spec identity is applied by ``_run`` as per-command ``-c`` overrides, so
        nothing is written to the repository's persisted git config.
        """
        self._run("commit", "-m", message)
        object.__setattr__(self, "_prev_hash", self.curr_hash())
        return self.prev_hash

    def checkout(self, branch: str, create: bool = False) -> None:
        """Check out a branch or ref and update ``head``.

        With ``create``, the branch is created from the current ``HEAD`` when it
        does not already exist; otherwise it is simply checked out.
        """
        if create and self._run("rev-parse", "--verify", branch, check=False).returncode != 0:
            self._run("checkout", "-b", branch)
        else:
            self._run("checkout", branch)
        object.__setattr__(self, "_prev_hash", self.curr_hash())

    def pull(self) -> None:
        """Fast-forward ``root`` from the ``upstream`` remote and update ``head``."""
        self._run("pull", "--ff-only", "upstream", "HEAD")
        object.__setattr__(self, "_prev_hash", self.curr_hash())

    def push(self, strict: bool = False) -> None:
        """Push ``HEAD`` to the ``origin`` remote.

        When there is no distinct remote to push to (``spec.remote`` unset, its
        ``src`` is ``None``, or it resolves to ``root`` itself), ``strict=False``
        makes this a no-op while ``strict=True`` raises.
        """
        if self._origin is None:
            if not strict:
                return

            src = getattr(self.spec.remote, "src", None)
            tip = (
                "must be configured"
                if src is None
                else f"cannot shadow `root={self.root}`"
            )
            msg = f"`remote.src={src}` {tip} when calling push()"
            raise RuntimeError(msg)

        self._run("push", "-u", "origin", "HEAD")

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        identity: list[str] = []
        if self.spec.user_name is not None:
            identity += ["-c", f"user.name={self.spec.user_name}"]
        if self.spec.user_email is not None:
            identity += ["-c", f"user.email={self.spec.user_email}"]
        result = subprocess.run(
            ["git", *identity, *args],
            cwd=self.root,
            text=True,
            capture_output=True,
            check=False,
        )
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
        return result

    @property
    def prev_hash(self) -> str | None:
        """Return the last-synced HEAD, computing it once if needed (None until a commit)."""
        if self._prev_hash is None:
            object.__setattr__(self, "_prev_hash", self.curr_hash())
        return self._prev_hash

    @cached_property
    def can_commit(self) -> bool:
        identity: list[str] = []
        if self.spec.user_name is not None:
            identity += ["-c", f"user.name={self.spec.user_name}"]
        if self.spec.user_email is not None:
            identity += ["-c", f"user.email={self.spec.user_email}"]
        cwd = self.root if self.root.exists() else None
        author = subprocess.run(
            ["git", *identity, "var", "GIT_AUTHOR_IDENT"],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        committer = subprocess.run(
            ["git", *identity, "var", "GIT_COMMITTER_IDENT"],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        return author.returncode == 0 and committer.returncode == 0
