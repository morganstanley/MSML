"""Matern 5/2 GP prior via random Fourier features."""

from dataclasses import InitVar, dataclass, field

import numpy as np
from numpy.random import Generator


@dataclass
class MaternPrior:
    """Draw a function from a Matern 5/2 GP prior via random Fourier features.

    Per-dimension ``lengthscales`` (ARD) may be supplied directly, or generated
    from a scalar ``rel_lengthscale`` and ``n_dims`` via the isotropic mapping
    ``lengthscales = rel_lengthscale * sqrt(n_dims)``. ``n_dims`` and
    ``rel_lengthscale`` are init-only; the resolved ``lengthscales`` array is
    the retained source of truth (and encodes the dimension).
    """

    n_features: int = 1024
    signal_variance: float = 1.0
    noise_variance: float = 0.0
    seed: int = 42
    shift: float | None = None
    weights: np.ndarray | None = None
    phases: np.ndarray | None = None
    coeffs: np.ndarray | None = None
    lengthscales: np.ndarray | None = None
    n_dims: InitVar[int | None] = None
    rel_lengthscale: InitVar[float | None] = None
    scale: float = field(init=False, repr=False)
    rng: Generator = field(init=False, repr=False)

    def __post_init__(self, n_dims: int | None, rel_lengthscale: float | None) -> None:
        self.lengthscales = self._resolve_lengthscales(n_dims, rel_lengthscale)
        n = len(self.lengthscales)

        provided = [x is not None for x in (self.weights, self.phases, self.coeffs, self.shift)]
        if any(provided) and not all(provided):
            raise ValueError("weights, phases, coeffs, and shift must all be provided or all be None")
        if all(provided):
            self.weights = np.asarray(self.weights)
            self.phases = np.asarray(self.phases)
            self.coeffs = np.asarray(self.coeffs)
            if self.weights.ndim != 2:
                raise ValueError(
                    f"weights must be a 2D array, got {self.weights.ndim}D"
                )
            if self.weights.shape[1] != n:
                raise ValueError(
                    f"weights dimension ({self.weights.shape[1]}) does not match "
                    f"lengthscales ({n})"
                )
        else:
            self.shift = 0.0
            rng = np.random.default_rng(self.seed)
            # Matern-5/2 spectral density is a multivariate Student-t with
            # df = 2*nu = 5. Draw it as a Gaussian scale mixture, sharing the
            # chi-squared radial draw across all coordinates of each feature.
            # A per-coordinate standard_t draw would use an independent radial
            # scale per axis, yielding a separable product-of-1D-Materns kernel
            # rather than the true (rotationally-coupled) Matern-5/2. The
            # per-axis lengthscales rescale each frequency coordinate (ARD).
            df = 5
            z = rng.standard_normal(size=(self.n_features, n))
            radial = rng.chisquare(df, size=(self.n_features, 1))
            self.weights = z * np.sqrt(df / radial) / self.lengthscales
            self.phases = rng.uniform(0, 2 * np.pi, size=self.n_features)
            self.coeffs = rng.standard_normal(self.n_features)
        self.rng = np.random.default_rng(self.seed)
        self.scale = np.sqrt(2 * self.signal_variance / self.n_features)

    def _resolve_lengthscales(
        self, n_dims: int | None, rel_lengthscale: float | None
    ) -> np.ndarray:
        """Resolve per-dimension lengthscales and validate the inputs agree.

        ``lengthscales`` takes precedence when supplied; ``n_dims`` and
        ``rel_lengthscale`` are then optional but, if given, must be consistent
        (matching dimension and the ``rel_lengthscale * sqrt(n_dims)`` value).
        When ``lengthscales`` is absent, both ``n_dims`` and ``rel_lengthscale``
        are required and generate an isotropic lengthscale vector.

        Args:
            n_dims: Number of input dimensions (init-only).
            rel_lengthscale: Scalar lengthscale relative to the unit cube
                (init-only).

        Returns:
            1D ``float64`` array of strictly positive per-dimension lengthscales.

        Raises:
            ValueError: Inputs are insufficient, disagree, or are not a
                non-empty, finite, strictly positive lengthscale vector.
        """
        if self.lengthscales is not None:
            lengthscales = np.asarray(self.lengthscales, dtype=float)
            if lengthscales.ndim != 1:
                raise ValueError("lengthscales must be a 1D array")
            if n_dims is not None and len(lengthscales) != n_dims:
                raise ValueError(
                    f"n_dims ({n_dims}) disagrees with len(lengthscales) "
                    f"({len(lengthscales)})"
                )
            if rel_lengthscale is not None:
                n = n_dims if n_dims is not None else len(lengthscales)
                expected = rel_lengthscale * np.sqrt(n)
                if not np.allclose(lengthscales, expected):
                    raise ValueError(
                        f"lengthscales disagree with rel_lengthscale * sqrt(n_dims) "
                        f"(expected {expected})"
                    )
        elif n_dims is None or rel_lengthscale is None:
            raise ValueError(
                "Provide lengthscales, or both n_dims and rel_lengthscale"
            )
        else:
            lengthscales = np.full(n_dims, rel_lengthscale * np.sqrt(n_dims), dtype=float)

        # lengthscales divide the sampled frequencies; reject anything that
        # would produce a divide-by-zero or an ill-defined draw.
        if lengthscales.size == 0:
            raise ValueError("lengthscales must be non-empty")
        if not np.all(np.isfinite(lengthscales)):
            raise ValueError("lengthscales must be finite")
        if np.any(lengthscales <= 0):
            raise ValueError("lengthscales must be strictly positive")
        return lengthscales

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the drawn function at X (shape: (n, d) or (d,))."""
        y = self.latent(X)
        if self.noise_variance > 0:
            y = y + self.rng.normal(
                0, np.sqrt(self.noise_variance), size=y.shape
            )
        return y

    def latent(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the drawn function without observation noise."""
        projection = X @ self.weights.T + self.phases
        return np.cos(projection) @ (self.scale * self.coeffs) + self.shift

    def gradient(self, X: np.ndarray) -> np.ndarray:
        """Gradient of the noiseless function at X (shape: (n, d) or (d,))."""
        projection = X @ self.weights.T + self.phases
        return -(np.sin(projection) * (self.scale * self.coeffs)) @ self.weights
