"""Private objective function. Hidden from workers by bwrap at runtime.

When copied into a workspace at ``{workspace}/private/objective.py``, this
module is imported by ``harness/blackbox.py`` via ``from private import
objective`` and exposes :func:`evaluate` as the ground-truth oracle. It reads
per-benchmark state from ``private/problem_state.json``, reconstructs the GP
prior, draws observation noise from the configured noise function, and appends
to the private and public state history.

All paths resolve from ``ALPHALAB_WORKSPACE``.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from alpha_lab.benchmarks.generators.gp_regression.matern_prior import MaternPrior
from alpha_lab.utils import resolve_import


WORKSPACE = Path(os.environ["ALPHALAB_WORKSPACE"])


def save_state(state: dict) -> None:
    """Persist the updated problem state to disk (private + public).

    Writes the full state to ``private/problem_state.json`` and a redacted
    projection to ``public/problem_state.json`` that strips ``latent_parameters``,
    ``noise_generator``, ``noise_parameters``, and the ``f`` field of each
    history entry.

    Args:
        state: Full private state, including the updated ``history``.
    """
    _atomic_write_json(WORKSPACE / "private" / "problem_state.json", state)

    public_state = {
        "n_dims": state["latent_parameters"]["n_dims"],
        "eval_limit": state["eval_limit"],
        "history": [{"x": h["x"], "y": h["y"]} for h in state["history"]],
    }
    _atomic_write_json(WORKSPACE / "public" / "problem_state.json", public_state)


def _atomic_write_json(path: Path, payload) -> None:
    """Write ``payload`` to ``path`` atomically via temp-file + replace.

    Guarantees the destination is either the previous contents or the new
    contents — never a partially-written file — if the process is interrupted.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def evaluate(x: np.ndarray) -> float:
    """Evaluate the latent function plus noise at ``x``, log, and return.

    Reads private problem state, reconstructs the GP prior, computes the
    noiseless ``f(x)``, draws noise from the configured noise function,
    appends private + public log entries, and returns the noisy observation.

    Args:
        x: 1D ``float64`` query point.

    Returns:
        The noisy observation ``y = f(x) + noise``.

    Raises:
        RuntimeError: Evaluation budget exhausted.
    """
    with (WORKSPACE / "private" / "problem_state.json").open() as f:
        state = json.load(f)

    history = state["history"]
    eval_limit = state["eval_limit"]
    if len(history) >= eval_limit:
        (WORKSPACE / "KILL_SIGNAL").write_text("")
        raise RuntimeError(
            f"Evaluation budget exhausted ({eval_limit} evaluations used)"
        )

    latent_fn = MaternPrior(**state["latent_parameters"])
    noise_fn = resolve_import(state["noise_generator"])
    f_val = latent_fn(x)
    y_val = f_val + noise_fn(x=x, history=history, **state["noise_parameters"])

    history.append({"x": x.tolist(), "y": y_val, "f": f_val})
    save_state(state)

    return y_val
