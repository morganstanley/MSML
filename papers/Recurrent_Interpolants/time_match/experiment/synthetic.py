import argparse
import itertools
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
)
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset

from time_match.utils.evaluation import mmd, swd_vector, wasserstein
from time_match.utils.synthetic_data import AVAILABLE_DATASETS
from time_match.utils.toy_configs import get_data, get_diffuser

SIZES = ['XS', 'S', 'M', 'L', 'XL']
ALL_DATASETS = list(map('-'.join, itertools.product(AVAILABLE_DATASETS, SIZES)))

ALL_MODELS = [
    'ddpm-linear',
    'ddpm-cosine',
    'sgm',
    'blend',
    'fm',
    'si-trig-trig',
    'si-trig-linear',
    'si-trig-encdec',
    'si-quad-trig',
    'si-quad-linear',
    'si-quad-encdec',
    'si-sqrt-trig',
    'si-sqrt-linear',
    'si-sqrt-encdec',
    'si-zero-trig',
    'si-zero-linear',
    'si-zero-encdec',
]

ALL_NETS = [
    'ff-1',
    'ff-3',
    'ff-6',
    'resnet-1',
    'resnet-3',
    'resnet-6',
]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


class SyntheticDataModule(LightningDataModule):
    def __init__(
        self,
        source_name: Optional[str] = None,
        target_name: str = None,
        batch_size: int = 64,
        seed: int = 1,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.val_size = 10_000

        train_target = get_data(name=target_name, seed=seed)
        val_target = get_data(name=target_name, size=self.val_size, seed=seed)
        self.dim = train_target.shape[-1]

        if source_name is not None:
            train_source = get_data(name=target_name, seed=seed)
            val_source = get_data(name=target_name, size=self.val_size, seed=seed)
            trainsets = [train_source, train_target]
            valsets = [val_source, val_target]
        else:
            trainsets = [train_target]
            valsets = [val_target]

        trainsets = [torch.from_numpy(x).float() for x in trainsets]
        valsets = [torch.from_numpy(x).float() for x in valsets]

        self.trainset = TensorDataset(*trainsets)
        self.valset = TensorDataset(*valsets)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valset, batch_size=self.val_size, shuffle=False)


class SyntheticLightningModule(LightningModule):
    def __init__(
        self,
        dim: int,
        model_name: str,
        net_name: str,
        learning_rate: float = 1e-3,
        weight_decay: float = 0,
        source_name: Optional[str] = None,
    ):
        super().__init__()
        self.dim = dim
        self.source_name = source_name
        self.is_bridge = (source_name is not None)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        self.diffuser = get_diffuser(model_name, net_name, dim)

    def forward(self, batch, log_name):
        if self.is_bridge:
            context, target = batch
            loss = self.diffuser.get_loss(target=target, context=context).mean()
        else:
            loss = self.diffuser.get_loss(batch[0]).mean()

        self.log(log_name, loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.forward(batch, log_name='train_loss')

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if self.is_bridge:
            _, target = batch  # batch=(source,target)
        else:
            target = batch[0]  # batch=(target,)
        batch_size = target.shape[0]
        gen_samples = self.sample_traj(batch_size, num_traj_steps=1)[-1]  # sample from last step.

        scores = {
            'wasserstein': wasserstein(gen_samples, target),
            'mmd_multiscale': mmd(gen_samples, target, kernel='multiscale', device=target.device),
            'mmd_rbf': mmd(gen_samples, target, kernel='rbf', device=target.device),
            'swd': swd_vector(gen_samples, target, proj_per_repeat=1000, device=target.device),
        }
        self.log_dict(scores, prog_bar=True)

        loss = self.forward(batch, log_name='val_loss')
        return loss

    @torch.no_grad()
    def sample_traj(self, num_samples: int, num_traj_steps: int = 10, epsilon=None):
        if self.is_bridge:
            x_0 = get_data(self.source_name, seed=np.random.randint(1e6), size=num_samples)
            x_0 = torch.from_numpy(x_0).float().to(self.device)
            return self.diffuser.sample_traj(x_0, traj_steps=num_traj_steps, epsilon=epsilon)
        else:
            x = torch.randn(num_samples, self.dim).to(self.device)
            return self.diffuser.sample_traj(x, traj_steps=num_traj_steps)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.diffuser.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


def sanity_check_sampling(module, target_dim):
    samples = module.sample_traj(10, num_traj_steps=3).detach().cpu().numpy()
    assert samples.shape == (3, 10, target_dim)


def train(
    seed: int,
    target: str,
    source: str,
    model: str,
    net: str,
    device: int,
):
    set_seed(seed)
    name = f'{"gaussian" if source is None else source}_{target}_{model}_{net}_{seed}'
    out_dir = Path(__file__).resolve().parents[2] / 'results' / 'synthetic' / name

    if out_dir.exists():
        print(f'Path {out_dir} already exists. Skipping ...')
        return

    out_dir.mkdir(exist_ok=False, parents=True)

    print('Save output dir:', out_dir)

    dm = SyntheticDataModule(
        source_name=source,
        target_name=target,
        batch_size=10_000,
        seed=seed,
    )
    module = SyntheticLightningModule(
        dim=dm.dim,
        model_name=model,
        net_name=net,
        source_name=source,
    )

    trainer = Trainer(
        devices=[device],
        max_steps=20_000,
        val_check_interval=1_000,
        check_val_every_n_epoch=None,
        log_every_n_steps=1,
        enable_checkpointing=True,
        logger=CSVLogger(save_dir=out_dir, name='logs'),
        callbacks=[TQDMProgressBar()],
    )

    sanity_check_sampling(module, dm.dim)

    trainer.fit(module, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())

    samples = module.sample_traj(10_000, num_traj_steps=10).detach().cpu().numpy()
    with open(out_dir / 'samples.npy', 'wb') as f:
        np.save(f, samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--target', type=str, choices=ALL_DATASETS)
    parser.add_argument('--source', type=str, default=None, choices=AVAILABLE_DATASETS)
    parser.add_argument('--model', type=str, choices=ALL_MODELS)
    parser.add_argument('--net', type=str, choices=ALL_NETS)
    parser.add_argument('--device', type=int)
    args = parser.parse_args()

    train(args.seed, args.target, args.source, args.model, args.net, args.device)
