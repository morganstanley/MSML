"""Benchmark workspace generators."""

from alpha_lab.benchmarks.generators.base import WorkspaceGenerator
from alpha_lab.benchmarks.generators.database import RegistryGenerator

__all__ = [
    "RegistryGenerator",
    "WorkspaceGenerator",
]

try:
    from alpha_lab.benchmarks.generators.structural_causal import StructuralCausalGenerator
    __all__ = [*__all__, "StructuralCausalGenerator"]
except ImportError:
    pass
