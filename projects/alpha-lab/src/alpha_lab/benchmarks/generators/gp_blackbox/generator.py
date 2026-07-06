"""GP-prior black-box optimization benchmark workspace generator."""

import json
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from importlib.resources import as_file, files
from pathlib import Path

from alpha_lab.benchmarks.generators.base import WorkspaceGenerator
from alpha_lab.benchmarks.generators.gp_blackbox.find_optimum import find_optimum
from alpha_lab.benchmarks.generators.gp_regression.matern_prior import MaternPrior
from alpha_lab.benchmarks.manifest import write_benchmark_manifest


@dataclass(frozen=True, kw_only=True)
class GPBlackboxGenerator(WorkspaceGenerator):
    """Materialize black-box optimization benchmarks from Matern 5/2 GP draws.

    Each workspace contains:
    - ``harness/blackbox.py``: public interface (``evaluate``, ``smoke_test``),
      copied verbatim from the package.
    - ``private/objective.py``: ground-truth oracle, copied verbatim.
    - ``private/problem_state.json``: latent parameters, noise spec, and
      observation history (with noiseless ``f``).
    - ``public/problem_state.json``: ``n_dims``, ``eval_limit``, and the
      redacted observation history (no ``f``).
    """

    seed: int = 42
    count: int = 1
    n_dims: int = 2
    n_rff: int = 1024
    eval_limit: int = 100
    rel_lengthscale: float = 0.1
    signal_variance: float = 1.0
    noise_variance: float = 0.0
    noise_generator: str = "alpha_lab.benchmarks.generators.noise:crn_noise"
    noise_seed: int | None = None
    n_restarts: int = 64
    n_sobol: int = 2**16

    def bootstrap(self, seed: int) -> Path:
        """Generate one optimization problem and assemble its workspace."""
        workspace = Path(self.workspace_root) / f"seed_{seed}"
        if workspace.exists():
            if not self.overwrite:
                raise FileExistsError(
                    f"Workspace already exists: {workspace}. "
                    "Use --overwrite to replace it."
                )
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True)

        harness_dir = workspace / "harness"
        harness_dir.mkdir()
        private_dir = workspace / "private"
        private_dir.mkdir()
        public_dir = workspace / "public"
        public_dir.mkdir()

        prior = MaternPrior(
            n_dims=self.n_dims,
            n_features=self.n_rff,
            rel_lengthscale=self.rel_lengthscale,
            signal_variance=self.signal_variance,
            noise_variance=self.noise_variance,
            seed=seed,
        )

        optimizer, min_value = find_optimum(
            prior,
            n_restarts=self.n_restarts,
            n_sobol=self.n_sobol,
            seed=seed,
        )
        pkg = files("alpha_lab.benchmarks.generators.gp_blackbox")
        with as_file(pkg / "blackbox.py") as p:
            shutil.copy(p, harness_dir / "blackbox.py")
        with as_file(pkg / "objective.py") as p:
            shutil.copy(p, private_dir / "objective.py")

        noise_seed = self.noise_seed if self.noise_seed is not None else seed
        with (public_dir / "problem_state.json").open("w") as f:
            json.dump({
                "n_dims": self.n_dims,
                "eval_limit": self.eval_limit,
                "history": [],
            }, f)
        with (private_dir / "problem_state.json").open("w") as f:
            json.dump({
                "eval_limit": self.eval_limit,
                "history": [],
                "latent_parameters": {
                    "n_dims": self.n_dims,
                    "n_features": self.n_rff,
                    "lengthscales": prior.lengthscales.tolist(),
                    "signal_variance": self.signal_variance,
                    "shift": -min_value,
                    "weights": prior.weights.tolist(),
                    "phases": prior.phases.tolist(),
                    "coeffs": prior.coeffs.tolist(),
                },
                "noise_generator": self.noise_generator,
                "noise_parameters": {
                    "seed": noise_seed,
                    "variance": self.noise_variance,
                },
            }, f)

        data = {
            "data_path": str(public_dir),
            "description": (
                f"Black-box optimization: {self.n_dims}-dimensional objective "
                f"over [0, 1]^{self.n_dims}. "
                "Use blackbox.evaluate(strategy) to query the objective. "
                "Use blackbox.smoke_test(strategy) to validate your strategy."
            ),
            "target": (
                "Find the input x that minimizes the objective function."
            ),
            "domain": "blackbox",
            "pipeline": {"phases": ["phase3"]},
            "workspace_includes": ["private", "harness"],
        }
        config = self._write_workspace_config(workspace, data)

        write_benchmark_manifest(
            workspace,
            source={"kind": "generator", "seed": seed},
            benchmark={
                "id": workspace.name,
                "name": f"GP Blackbox seed={seed}",
                "notes": "Generated benchmark workspace; no registry row required.",
            },
            materialized={
                "data_path": str(public_dir),
                "data_source": "generated",
                "data_is_symlink": False,
                "adapter_source": None,
                "seed_source": None,
            },
            config=config,
            generator_info={
                "seed": seed,
                "n_dims": self.n_dims,
                "n_rff": self.n_rff,
                "eval_limit": self.eval_limit,
                "rel_lengthscale": self.rel_lengthscale,
                "lengthscales": prior.lengthscales.tolist(),
                "signal_variance": self.signal_variance,
                "noise_variance": self.noise_variance,
                "noise_generator": self.noise_generator,
                "noise_seed": noise_seed,
                "optimizer": optimizer.tolist(),
            },
        )
        self.validate(workspace)
        return workspace

    def __iter__(self) -> Iterator[Callable[[], Path]]:
        if self.count < 1:
            raise ValueError("count must be at least 1")
        for index in range(self.count):
            yield partial(self.bootstrap, seed=self.seed + index)
