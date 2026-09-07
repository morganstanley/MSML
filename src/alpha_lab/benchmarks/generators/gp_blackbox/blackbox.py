"""Black-box optimization interface.

When copied into a workspace at ``{workspace}/harness/blackbox.py``, this
module is the public-facing harness imported by ``run_experiment.py`` via
``from harness import blackbox``. It exposes :func:`evaluate` and
:func:`smoke_test` to drive the strategy; the actual ground-truth oracle is
deferred to ``private/objective.py``.

All paths resolve from ``ALPHALAB_WORKSPACE``.
"""

import json
import os
import shutil
from importlib.resources import as_file, files
from pathlib import Path

import numpy as np

WORKSPACE = Path(os.environ["ALPHALAB_WORKSPACE"])


def _validate(x: np.ndarray, n_dims: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (n_dims,):
        raise ValueError(f"Expected shape ({n_dims},), got {x.shape}")
    if np.any(x < 0) or np.any(x > 1):
        raise ValueError("x must be in [0, 1]^d")
    return x


def smoke_test(strategy) -> None:
    """Call ``strategy`` and validate the proposed ``x`` without evaluating.

    Args:
        strategy: Callable ``(X, y) -> x`` proposing a next query.

    Raises:
        ValueError: ``strategy`` returns a malformed ``x``.
    """
    with (WORKSPACE / "public" / "problem_state.json").open() as f:
        state = json.load(f)
    X = np.array([h["x"] for h in state["history"]]).reshape(-1, state["n_dims"])
    y = np.array([h["y"] for h in state["history"]])
    _validate(strategy(X, y), state["n_dims"])


def create_runner() -> None:
    """Copy the package-side ``run_experiment.py`` into the current directory.

    Intended to be called from the experiment directory by the worker.
    """
    src = files("alpha_lab.benchmarks.generators.gp_blackbox") / "run_experiment.py"
    with as_file(src) as p:
        shutil.copy(p, "run_experiment.py")


def evaluate(strategy) -> float:
    """Call ``strategy``, evaluate the proposed ``x``, and write metrics.

    Args:
        strategy: Callable ``(X, y) -> x`` proposing a next query.

    Returns:
        The noisy observation produced by ``private.objective.evaluate``.
    """
    from private import objective

    with (WORKSPACE / "public" / "problem_state.json").open() as f:
        state = json.load(f)
    X = np.array([h["x"] for h in state["history"]]).reshape(-1, state["n_dims"])
    y = np.array([h["y"] for h in state["history"]])
    x = _validate(strategy(X, y), state["n_dims"])
    value = objective.evaluate(x)
    Path("results").mkdir(exist_ok=True)
    with Path("results/metrics.json").open("w") as f:
        json.dump({"y": value, "x": x.tolist()}, f)
    return value
