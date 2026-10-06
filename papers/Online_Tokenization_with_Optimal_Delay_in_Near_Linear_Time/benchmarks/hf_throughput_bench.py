#!/usr/bin/env python3
"""Benchmark Hugging Face-compatible tokenizer throughput on paper corpora."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.throughput_bench import run_throughput_benchmark
from tools.benchmark_suites import HF_SUITE


def main(argv: Sequence[str] | None = None) -> None:
    run_throughput_benchmark(HF_SUITE, argv)


if __name__ == "__main__":
    main()
