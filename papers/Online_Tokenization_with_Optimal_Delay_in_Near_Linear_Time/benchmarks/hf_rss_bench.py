#!/usr/bin/env python3
"""Measure RSS scaling for Hugging Face-compatible tokenizers."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.benchmark_suites import HF_SUITE
from tools.rss_bench import run_rss_benchmark


def main(argv: Sequence[str] | None = None) -> None:
    run_rss_benchmark(HF_SUITE, argv)


if __name__ == "__main__":
    main()
