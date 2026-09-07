"""Tests for adapter_loader: manifest field normalization on load."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alpha_lab.adapter import PROMPT_KEYS
from alpha_lab.adapter_loader import load_adapter


def _write_adapter(tmp_path: Path, manifest: dict) -> Path:
    """Write a minimal loadable adapter directory."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "manifest.json").write_text(json.dumps(manifest))
    for key in PROMPT_KEYS:
        (adapter_dir / f"{key}.md").write_text(f"# {key}\n")
    return adapter_dir


class TestReviewFileNormalization:
    def test_framework_dir_prefix_is_stripped(self, tmp_path: Path) -> None:
        # Adapter-patching agents sometimes write the review path
        # workspace-relative; consumers join framework_dir + review_file,
        # so a "backtest/" prefix would resolve to backtest/backtest/review.md.
        adapter_dir = _write_adapter(tmp_path, {
            "domain_name": "time_series",
            "phase2_review_file": "backtest/review.md",
            "experiment": {"framework_dir": "backtest"},
        })
        adapter = load_adapter(adapter_dir)
        assert adapter.phase2_review_file == "review.md"

    def test_bare_filename_unchanged(self, tmp_path: Path) -> None:
        adapter_dir = _write_adapter(tmp_path, {
            "domain_name": "time_series",
            "phase2_review_file": "review.md",
            "experiment": {"framework_dir": "backtest"},
        })
        adapter = load_adapter(adapter_dir)
        assert adapter.phase2_review_file == "review.md"

    def test_unrelated_subdir_left_alone(self, tmp_path: Path) -> None:
        # Only the framework_dir prefix is stripped; a deliberate subdir
        # inside the framework dir keeps working.
        adapter_dir = _write_adapter(tmp_path, {
            "domain_name": "time_series",
            "phase2_review_file": "docs/review.md",
            "experiment": {"framework_dir": "backtest"},
        })
        adapter = load_adapter(adapter_dir)
        assert adapter.phase2_review_file == "docs/review.md"

    def test_similarly_named_dir_is_not_a_prefix(self, tmp_path: Path) -> None:
        # Path-segment comparison, not string startswith: "backtesting/..."
        # is not inside "backtest".
        adapter_dir = _write_adapter(tmp_path, {
            "domain_name": "time_series",
            "phase2_review_file": "backtesting/review.md",
            "experiment": {"framework_dir": "backtest"},
        })
        adapter = load_adapter(adapter_dir)
        assert adapter.phase2_review_file == "backtesting/review.md"

    def test_trailing_slash_framework_dir_still_stripped(self, tmp_path: Path) -> None:
        adapter_dir = _write_adapter(tmp_path, {
            "domain_name": "time_series",
            "phase2_review_file": "backtest/review.md",
            "experiment": {"framework_dir": "backtest/"},
        })
        adapter = load_adapter(adapter_dir)
        assert adapter.phase2_review_file == "review.md"

    def test_default_when_absent(self, tmp_path: Path) -> None:
        adapter_dir = _write_adapter(tmp_path, {"domain_name": "time_series"})
        adapter = load_adapter(adapter_dir)
        assert adapter.phase2_review_file == "review.md"

    def test_non_string_value_fails_the_load(self, tmp_path: Path) -> None:
        # A null/non-string value means the adapter-patching agent wrote a bad
        # manifest. Guessing "review.md" here could silently disagree with the
        # customized prompts, so the load fails with the manifest key named.
        for case, bad in enumerate((None, 7, ["review.md"], "")):
            base = tmp_path / f"case{case}"
            base.mkdir()
            adapter_dir = _write_adapter(base, {
                "domain_name": "time_series",
                "phase2_review_file": bad,
                "experiment": {"framework_dir": "backtest"},
            })
            with pytest.raises(ValueError, match="phase2_review_file"):
                load_adapter(adapter_dir)
