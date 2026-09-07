"""Script entry points for Alpha Lab benchmark management."""

from alpha_lab.benchmarks.scripts.copy_workspace import copy_workspace
from alpha_lab.benchmarks.scripts.register_workspaces import register
from alpha_lab.benchmarks.scripts.remove_benchmark import remove_benchmarks

__all__ = [
    "copy_workspace",
    "register",
    "remove_benchmarks",
]
