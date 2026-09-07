"""Global optimization of MaternPrior draws via Sobol grid + L-BFGS-B."""

import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

from alpha_lab.benchmarks.generators.gp_regression.matern_prior import MaternPrior


def find_optimum(
    prior: MaternPrior,
    n_restarts: int = 64,
    n_sobol: int = 2**16,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """Find the global minimum of a MaternPrior draw via Sobol grid + L-BFGS-B."""
    d = len(prior.lengthscales)
    bounds = [(0.0, 1.0)] * d

    sampler = Sobol(d, scramble=True, seed=seed)
    X = sampler.random(n_sobol)
    y = prior.latent(X)
    top_idx = np.argsort(y)[:n_restarts]

    best_x, best_y = X[top_idx[0]], y[top_idx[0]]
    for idx in top_idx:
        result = minimize(
            fun=lambda x: prior.latent(x.reshape(1, -1)).item(),
            x0=X[idx],
            jac=lambda x: prior.gradient(x.reshape(1, -1)).ravel(),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if result.fun < best_y:
            best_x, best_y = result.x, result.fun

    return best_x, float(best_y)
