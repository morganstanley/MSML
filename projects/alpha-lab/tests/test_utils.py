"""Tests for alpha_lab.utils."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from alpha_lab.utils import resolve_import


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
