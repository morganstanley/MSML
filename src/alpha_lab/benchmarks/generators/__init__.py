"""Benchmark workspace generators."""

from alpha_lab.benchmarks.generators.base import WorkspaceGenerator
from alpha_lab.benchmarks.generators.database import RegistryGenerator
from alpha_lab.benchmarks.generators.gp_blackbox import GPBlackboxGenerator
from alpha_lab.benchmarks.generators.gp_regression import GPRegressionGenerator

__all__ = [
    "GPBlackboxGenerator",
    "GPRegressionGenerator",
    "RegistryGenerator",
    "WorkspaceGenerator",
]

try:
    from alpha_lab.benchmarks.generators.structural_causal import StructuralCausalGenerator
    __all__ = [*__all__, "StructuralCausalGenerator"]
except ImportError:
    pass
