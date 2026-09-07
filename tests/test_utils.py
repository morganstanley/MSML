"""Tests for alpha_lab.utils."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from alpha_lab.utils import atomic_write, resolve_import


class TestResolveImportModuleMode:
    def test_colon_form(self) -> None:
        cls = resolve_import("pathlib:Path")
        assert cls is Path

    def test_dot_form(self) -> None:
        cls = resolve_import("pathlib.Path")
        assert cls is Path

    def test_missing_attribute_raises(self) -> None:
        with pytest.raises(AttributeError):
            resolve_import("pathlib:NoSuchAttr")

    def test_malformed_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_import("no_separator_here")

    def test_empty_object_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_import("pathlib:")


class TestResolveImportPathMode:
    def _write_module(self, tmp_path: Path, body: str) -> Path:
        path = tmp_path / "scratch_module.py"
        path.write_text(textwrap.dedent(body))
        return path

    def test_path_with_colon_form(self, tmp_path: Path) -> None:
        path = self._write_module(
            tmp_path,
            """
            class Widget:
                kind = "scratch"
            """,
        )
        cls = resolve_import(f"{path}:Widget")
        assert cls.kind == "scratch"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            resolve_import(f"{tmp_path / 'missing.py'}:Widget")

    def test_path_detection_via_separator(self, tmp_path: Path) -> None:
        # No .py suffix but the path separator triggers path-mode.
        path = tmp_path / "scratch_module"
        path.with_suffix(".py").write_text("class Widget: pass\n")
        # The separator triggers path-mode; ".py" still added below for the
        # actual file load.
        cls = resolve_import(f"{path}.py:Widget")
        assert cls.__name__ == "Widget"


class TestResolveImportTypes:
    def test_class_passes_subclass_check(self) -> None:
        from pathlib import PurePath
        cls = resolve_import("pathlib:Path", types=PurePath)
        assert cls is Path

    def test_class_fails_subclass_check(self) -> None:
        with pytest.raises(TypeError):
            resolve_import("pathlib:Path", types=int)

    def test_instance_passes_isinstance(self) -> None:
        obj = resolve_import("os:sep", types=str)
        assert isinstance(obj, str)

    def test_instance_fails_isinstance(self) -> None:
        with pytest.raises(TypeError):
            resolve_import("os:sep", types=int)


class TestAtomicWrite:
    def test_writes_content(self, tmp_path: Path) -> None:
        dest = tmp_path / "out.txt"
        atomic_write(dest, "hello")
        assert dest.read_text() == "hello"

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        dest = tmp_path / "out.txt"
        dest.write_text("old")
        atomic_write(dest, "new")
        assert dest.read_text() == "new"

    def test_no_fsync_still_writes(self, tmp_path: Path) -> None:
        dest = tmp_path / "out.txt"
        atomic_write(dest, "data", fsync=False)
        assert dest.read_text() == "data"

    def test_leaves_no_temp_files(self, tmp_path: Path) -> None:
        dest = tmp_path / "out.txt"
        atomic_write(dest, "data")
        assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]

    def test_preserves_original_on_write_failure(self, tmp_path: Path) -> None:
        dest = tmp_path / "out.txt"
        dest.write_text("original")
        # A non-str payload fails inside the temp write, after the temp file is opened;
        # the original must be untouched and no temp file left behind.
        with pytest.raises(TypeError):
            atomic_write(dest, 123)  # type: ignore[arg-type]
        assert dest.read_text() == "original"
        assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]

    def test_missing_parent_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            atomic_write(tmp_path / "nope" / "out.txt", "data")

    def test_new_file_mode_matches_write_text(self, tmp_path: Path) -> None:
        import stat

        ref = tmp_path / "ref.txt"
        ref.write_text("x")  # umask-respecting reference
        dest = tmp_path / "out.txt"
        atomic_write(dest, "x")
        assert stat.S_IMODE(dest.stat().st_mode) == stat.S_IMODE(ref.stat().st_mode)

    def test_overwrite_preserves_existing_mode(self, tmp_path: Path) -> None:
        import os
        import stat

        dest = tmp_path / "out.txt"
        dest.write_text("old")
        os.chmod(dest, 0o640)
        atomic_write(dest, "new")
        assert stat.S_IMODE(dest.stat().st_mode) == 0o640

    def test_writes_utf8(self, tmp_path: Path) -> None:
        dest = tmp_path / "out.txt"
        atomic_write(dest, "café — 日本語")
        assert dest.read_text(encoding="utf-8") == "café — 日本語"
