"""Registry primitives for Alpha Lab benchmarks."""

from alpha_lab.benchmarks.registry.models import Benchmark
from alpha_lab.benchmarks.registry.store import (
    connect_registry,
    ensure_schema,
    load_benchmarks,
)

__all__ = [
    "Benchmark",
    "connect_registry",
    "ensure_schema",
    "load_benchmarks",
]
