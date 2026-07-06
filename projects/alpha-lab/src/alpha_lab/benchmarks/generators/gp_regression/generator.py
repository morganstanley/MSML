"""GP-prior regression benchmark workspace generator."""

import hashlib
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import scipy.stats.qmc

from alpha_lab.benchmarks.generators.base import WorkspaceGenerator
from alpha_lab.benchmarks.generators.gp_regression.matern_prior import MaternPrior
from alpha_lab.benchmarks.manifest import write_benchmark_manifest


@dataclass(frozen=True, kw_only=True)
class GPRegressionGenerator(WorkspaceGenerator):
    """Materialize regression benchmarks from Matern 5/2 GP prior draws.

    Each workspace contains ``train_data.npz`` and ``test_data.npz`` with
    arrays ``X`` (float feature matrix) and ``y`` (float targets). The
    ground-truth noiseless function values are stored in ``ground_truth.npz``.
    """

    seed: int = 42
    count: int = 1
    n_dims: int = 5
    n_train: int = 200
    n_test: int = 100
    n_rff: int = 1024
    rel_lengthscale: float = 0.1
    signal_variance: float = 1.0
    noise_variance: float = 0.01
    input_range: tuple[float, float] = (0.0, 1.0)

    def bootstrap(self, seed: int) -> Path:
        """Generate one regression dataset and assemble its workspace."""
        workspace = Path(self.workspace_root) / f"seed_{seed}"
        if workspace.exists():
            if not self.overwrite:
                raise FileExistsError(
                    f"Workspace already exists: {workspace}. "
                    "Use --overwrite to replace it."
                )
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True)
        data_dir = workspace / "data"
        data_dir.mkdir()
        private_dir = workspace / "private"
        private_dir.mkdir()

        prior = MaternPrior(
            n_dims=self.n_dims,
            n_features=self.n_rff,
            rel_lengthscale=self.rel_lengthscale,
            signal_variance=self.signal_variance,
            noise_variance=self.noise_variance,
            seed=seed,
        )

        rng = np.random.default_rng(seed)
        lo, hi = self.input_range

        X_train = rng.uniform(lo, hi, size=(self.n_train, self.n_dims))
        sobol = scipy.stats.qmc.Sobol(d=self.n_dims, scramble=True, seed=seed)
        X_test = sobol.random(self.n_test) * (hi - lo) + lo

        y_train = prior(X_train)
        y_clean_train = prior.latent(X_train)
        y_test = prior.latent(X_test)

        train_path = data_dir / "train_data.npz"
        test_path = private_dir / "test_data.npz"
        truth_path = private_dir / "ground_truth.npz"
        np.savez(train_path, X=X_train, y=y_train)
        np.savez(test_path, X=X_test, y=y_test)
        np.savez(truth_path, y_train=y_clean_train)

        data = {
            "data_path": str(data_dir),
            "description": (
                f"Tabular regression: R^{self.n_dims} -> R. "
                f"{self.n_train} train samples. "
                "`{workspace}/data/train_data.npz` holds 'X' (float feature "
                "matrix) and 'y' (float targets)."
            ),
            "target": (
                "Predict continuous targets (y) from the feature matrix (X)."
            ),
            "domain": "tabular_regression",
            "pipeline": {"phases": ["phase1", "phase2", "phase3"]},
            "workspace_includes": ["private"],
        }
        config = self._write_workspace_config(workspace, data)

        write_benchmark_manifest(
            workspace,
            source={"kind": "generator", "seed": seed},
            benchmark={
                "id": workspace.name,
                "name": f"GP Regression seed={seed}",
                "notes": "Generated benchmark workspace; no registry row required.",
            },
            materialized={
                "data_path": str(data_dir),
                "train_path": str(train_path),
                "test_path": str(test_path),
                "ground_truth_path": str(truth_path),
                "data_source": "generated",
                "data_is_symlink": False,
                "adapter_source": None,
                "seed_source": None,
            },
            config=config,
            generator_info={
                "seed": seed,
                "n_train": self.n_train,
                "n_test": self.n_test,
                "n_dims": self.n_dims,
                "n_rff": self.n_rff,
                "rel_lengthscale": self.rel_lengthscale,
                "lengthscales": prior.lengthscales.tolist(),
                "signal_variance": self.signal_variance,
                "noise_variance": self.noise_variance,
                "input_range": list(self.input_range),
                "content_hash": _content_hash(train_path, test_path),
            },
        )
        self.validate(workspace)
        return workspace

    def __iter__(self) -> Iterator[Callable[[], Path]]:
        if self.count < 1:
            raise ValueError("count must be at least 1")
        for index in range(self.count):
            yield partial(self.bootstrap, seed=self.seed + index)


def _content_hash(*paths: Path) -> str:
    """Return a sha256 over the concatenated bytes of ``paths``."""
    h = hashlib.sha256()
    for path in paths:
        h.update(path.read_bytes())
    return h.hexdigest()
