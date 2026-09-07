"""Observation-noise functions for benchmark objectives.

Each noise function has the signature ``f(seed, variance, history, x) -> float``
and returns a single zero-mean Gaussian sample of standard deviation
``sqrt(variance)``. The function is deterministic in its inputs; randomness comes
entirely from a :class:`numpy.random.SeedSequence` mixed from ``seed`` and a
subset of ``(len(history), x, history-derived counts)`` chosen by the function.

``history`` is the list of prior observations as JSON-loaded ``dict``s with keys
``"x"``, ``"y"``, ``"f"``; the current call's ``x`` is not yet appended.

Two strategies are provided:

- :func:`iid_noise` indexes by ``(seed, N)`` where ``N = len(history)``.
  Each evaluation in a run draws the next iid sample; re-runs with the
  same call sequence reproduce the same trace.
- :func:`crn_noise` indexes by ``(seed, n_i, x)`` where ``n_i`` is the
  number of prior visits to ``x``. The k-th visit to a point is reproducible
  across runs and across solvers (common random numbers / variance reduction).
"""

import numpy as np


def iid_noise(
    seed: int,
    variance: float,
    history: list[dict],
    x: np.ndarray,
) -> float:
    """Sample iid Gaussian noise indexed by total call count.

    Args:
        seed: Root seed for the deterministic stream.
        variance: Noise variance; ``0.0`` yields ``0.0``.
        history: Prior observations as ``{"x", "y", "f"}`` dicts.
        x: Current query point (unused; accepted for signature uniformity).

    Returns:
        A single ``float`` drawn from ``N(0, variance)``.
    """
    del x
    ss = np.random.SeedSequence(entropy=[seed, len(history)])
    return float(np.random.default_rng(ss).normal(0.0, np.sqrt(variance)))


def crn_noise(
    seed: int,
    variance: float,
    history: list[dict],
    x: np.ndarray,
) -> float:
    """Sample common-random-numbers Gaussian noise indexed by (x, visit count).

    Args:
        seed: Root seed for the deterministic stream.
        variance: Noise variance; ``0.0`` yields ``0.0``.
        history: Prior observations as ``{"x", "y", "f"}`` dicts.
        x: 1D ``float64`` query point; the k-th visit reproduces the same draw.

    Returns:
        A single ``float`` drawn from ``N(0, variance)``.
    """
    n_i = sum(1 for entry in history if np.array_equal(entry["x"], x))
    x_ints = [int(b) for b in x.view(np.uint64)]
    ss = np.random.SeedSequence(entropy=[seed, n_i, *x_ints])
    return float(np.random.default_rng(ss).normal(0.0, np.sqrt(variance)))
