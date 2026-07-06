"""Tests for the Matern-5/2 random-Fourier-feature prior.

The prior must sample the true (rotationally-coupled) Matern-5/2 kernel, not a
separable product of per-axis 1D Materns. The distinction only appears for
n_dims > 1, so the covariance check runs in 2D. Error is the Monte-Carlo rate
of random Fourier features, O(1/sqrt(n_features)), so a large basis is used to
afford a tight tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha_lab.benchmarks.generators.gp_regression.matern_prior import MaternPrior


def _matern52_unit(r: np.ndarray, lengthscale: float) -> np.ndarray:
    """Unit-variance Matern-5/2 correlation at radius ``r``."""
    a = np.sqrt(5.0) * np.abs(r) / lengthscale
    return (1.0 + a + a**2 / 3.0) * np.exp(-a)


def test_matern_prior_covariance_matches_isotropic_matern52_2d() -> None:
    """Coeff-marginal RFF covariance matches the isotropic Matern-5/2 in 2D.

    Uses the exact marginal covariance over the N(0,1) feature coefficients,
    ``scale**2 * cos(proj) @ cos(proj).T``, so the only error source is the
    finite frequency draw. A separable product kernel (the prior bug) deviates
    from the isotropic form by ~0.05-0.07 here, well outside the tolerance.
    """
    n_features = 131_072
    lengthscale = 0.3
    signal_variance = 2.0
    tol = 0.02  # ~2x the worst case observed over many seeds at this n_features

    seed = 7  # fixed for determinism; max_rel_err is ~0.007 at this n_features
    point_rng = np.random.default_rng(0)
    points = point_rng.random((24, 2))

    prior = MaternPrior(
        n_features=n_features,
        lengthscales=np.full(2, lengthscale),
        signal_variance=signal_variance,
        seed=seed,
    )
    features = np.cos(points @ prior.weights.T + prior.phases)
    cov_empirical = prior.scale**2 * features @ features.T

    diff = points[:, None, :] - points[None, :, :]
    radius = np.sqrt((diff**2).sum(-1))
    cov_isotropic = signal_variance * _matern52_unit(radius, lengthscale)

    max_rel_err = np.abs(cov_empirical - cov_isotropic).max() / signal_variance
    assert max_rel_err < tol, (
        f"covariance deviates from isotropic Matern-5/2 "
        f"(max_rel_err={max_rel_err:.4f}, tol={tol}, seed={seed})"
    )


def test_lengthscales_generated_from_rel_lengthscale_and_n_dims() -> None:
    """rel_lengthscale + n_dims generate the isotropic rel*sqrt(n_dims) vector."""
    prior = MaternPrior(n_dims=4, rel_lengthscale=0.1, seed=1)
    np.testing.assert_allclose(prior.lengthscales, np.full(4, 0.2))


def test_explicit_lengthscales_supports_ard() -> None:
    """Explicit per-dimension lengthscales infer the dimension and feed weights."""
    prior = MaternPrior(lengthscales=[0.1, 0.5], seed=1)
    np.testing.assert_allclose(prior.lengthscales, [0.1, 0.5])
    assert prior.weights.shape[1] == 2


def test_lengthscales_and_rel_lengthscale_agreement_accepted() -> None:
    """Supplying both forms is allowed when they agree."""
    prior = MaternPrior(
        n_dims=4, rel_lengthscale=0.1, lengthscales=np.full(4, 0.2), seed=1
    )
    np.testing.assert_allclose(prior.lengthscales, np.full(4, 0.2))


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="neither"),
        pytest.param({"n_dims": 3}, id="n_dims_without_rel"),
        pytest.param({"rel_lengthscale": 0.1}, id="rel_without_n_dims"),
    ],
)
def test_underspecified_lengthscales_raise(kwargs: dict) -> None:
    """lengthscales is required unless both n_dims and rel_lengthscale are given."""
    with pytest.raises(ValueError, match="Provide lengthscales"):
        MaternPrior(seed=1, **kwargs)


def test_lengthscales_dimension_disagreement_raises() -> None:
    with pytest.raises(ValueError, match="disagrees with len"):
        MaternPrior(n_dims=3, lengthscales=np.full(4, 0.2), seed=1)


def test_lengthscales_value_disagreement_raises() -> None:
    with pytest.raises(ValueError, match="disagree with rel_lengthscale"):
        MaternPrior(rel_lengthscale=0.1, lengthscales=np.full(4, 0.99), seed=1)


def test_provided_weights_dimension_must_match_lengthscales() -> None:
    donor = MaternPrior(lengthscales=np.full(3, 0.2), seed=1)
    with pytest.raises(ValueError, match="weights dimension"):
        MaternPrior(
            lengthscales=np.full(4, 0.2),
            n_features=donor.n_features,
            shift=0.0,
            weights=donor.weights,
            phases=donor.phases,
            coeffs=donor.coeffs,
            seed=1,
        )


def test_reconstruction_from_stored_parameters_reproduces_function() -> None:
    """Round-tripping the stored latent_parameters reproduces the draw."""
    prior = MaternPrior(n_dims=4, rel_lengthscale=0.1, signal_variance=1.0, seed=1)
    state = {
        "n_dims": 4,
        "n_features": prior.n_features,
        "lengthscales": prior.lengthscales.tolist(),
        "signal_variance": 1.0,
        "shift": -1.23,
        "weights": prior.weights.tolist(),
        "phases": prior.phases.tolist(),
        "coeffs": prior.coeffs.tolist(),
    }
    rebuilt = MaternPrior(**state)
    x = np.array([[0.1, 0.2, 0.3, 0.4]])
    np.testing.assert_allclose(rebuilt.latent(x), prior.latent(x) - 1.23)


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param([0.2, 0.0], id="zero"),
        pytest.param([0.2, -0.1], id="negative"),
        pytest.param([0.2, np.nan], id="nan"),
        pytest.param([0.2, np.inf], id="inf"),
    ],
)
def test_non_positive_or_nonfinite_lengthscales_raise(bad: list) -> None:
    """lengthscales divide the frequencies, so they must be finite and positive."""
    with pytest.raises(ValueError, match="lengthscales must be"):
        MaternPrior(lengthscales=bad, seed=1)


def test_empty_lengthscales_raise() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        MaternPrior(lengthscales=[], seed=1)


def test_non_positive_rel_lengthscale_raises() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        MaternPrior(n_dims=3, rel_lengthscale=0.0, seed=1)


def test_non_2d_weights_raise() -> None:
    with pytest.raises(ValueError, match="weights must be a 2D array"):
        MaternPrior(
            lengthscales=np.full(2, 0.2),
            n_features=4,
            shift=0.0,
            weights=np.zeros(4),
            phases=np.zeros(4),
            coeffs=np.zeros(4),
            seed=1,
        )
