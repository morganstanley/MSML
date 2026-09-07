"""GP-prior regression benchmark generator."""

from alpha_lab.benchmarks.generators.gp_regression.generator import GPRegressionGenerator
from alpha_lab.benchmarks.generators.gp_regression.matern_prior import MaternPrior

__all__ = ["GPRegressionGenerator", "MaternPrior"]
