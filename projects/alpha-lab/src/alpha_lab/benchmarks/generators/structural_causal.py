"""SCM-based synthetic benchmark workspace generator backed by ``tabicl``."""

import hashlib
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from tabicl.prior import PriorDataset

from alpha_lab.benchmarks.generators.base import WorkspaceGenerator
from alpha_lab.benchmarks.manifest import write_benchmark_manifest


@dataclass(frozen=True, kw_only=True)
class StructuralCausalGenerator(WorkspaceGenerator):
    """Materialize benchmarks from ``tabicl``'s SCM-derived prior.

    Tabicl's ``PriorDataset`` samples structural causal models with continuous
    targets and bins the targets into class labels via ``Reg2Cls``. Each
    workspace contains a ``data/`` directory with two NumPy archives
    (``train_data.npz`` and ``test_data.npz``); each holds ``X`` (float
    feature matrix) and ``y`` (integer class labels).
    """

    seed: int = 42
    count: int = 1
    min_features: int = 2
    max_features: int = 20
    max_classes: int = 4
    max_seq_len: int = 1024

    def bootstrap(self, seed: int) -> Path:
        """Generate one dataset and assemble its workspace."""
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

        X_train, y_train, X_test, y_test = _generate_dataset(
            seed=seed,
            min_features=self.min_features,
            max_features=self.max_features,
            max_classes=self.max_classes,
            max_seq_len=self.max_seq_len,
        )
        train_path = data_dir / "train_data.npz"
        test_path = private_dir / "test_data.npz"
        np.savez(train_path, X=X_train, y=y_train)
        np.savez(test_path, X=X_test, y=y_test)

        n_features = X_train.shape[1]
        n_classes = int(max(y_train.max(), y_test.max())) + 1
        data = {
            "data_path": str(data_dir),
            "description": (
                f"Tabular classification: "
                f"{X_train.shape[0]} train samples, {n_features} features, "
                f"{n_classes} classes. `{{workspace}}/data/train_data.npz` "
                "holds 'X' (float feature matrix) and 'y' (integer labels)."
            ),
            "target": (
                "Predict integer class labels (y) from the feature matrix (X)."
            ),
            "domain": "tabular_classification",
            "pipeline": {"phases": ["phase1", "phase2", "phase3"]},
            "workspace_includes": ["private"],
        }
        config = self._write_workspace_config(workspace, data)

        write_benchmark_manifest(
            workspace,
            source={"kind": "generator", "seed": seed},
            benchmark={
                "id": workspace.name,
                "name": f"Tabular seed={seed}",
                "notes": "Generated benchmark workspace; no registry row required.",
            },
            materialized={
                "data_path": str(data_dir),
                "train_path": str(train_path),
                "test_path": str(test_path),
                "data_source": "generated",
                "data_is_symlink": False,
                "adapter_source": None,
                "seed_source": None,
            },
            config=config,
            generator_info={
                "seed": seed,
                "n_train": int(X_train.shape[0]),
                "n_test": int(X_test.shape[0]),
                "n_features": n_features,
                "n_classes": n_classes,
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


def _generate_dataset(
    *,
    seed: int,
    min_features: int,
    max_features: int,
    max_classes: int,
    max_seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run tabicl's PriorDataset for one dataset and return train/test arrays."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    prior = PriorDataset(
        batch_size=1,
        min_features=min_features,
        max_features=max_features,
        max_classes=max_classes,
        max_seq_len=max_seq_len,
        n_jobs=1,
    )
    X, y, d, seq_lens, train_sizes = prior.get_batch()
    n = int(seq_lens[0])
    f = int(d[0])
    train_size = int(train_sizes[0])
    X_full = X[0, :n, :f].cpu().numpy().astype(float)
    y_full = y[0, :n].cpu().numpy().astype(int)
    return (
        X_full[:train_size],
        y_full[:train_size],
        X_full[train_size:],
        y_full[train_size:],
    )


def _content_hash(*paths: Path) -> str:
    """Return a sha256 over the concatenated bytes of ``paths`` (in order)."""
    h = hashlib.sha256()
    for path in paths:
        h.update(path.read_bytes())
    return h.hexdigest()
