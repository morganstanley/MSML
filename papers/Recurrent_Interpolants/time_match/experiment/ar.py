import argparse
import random
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
)
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from statsmodels.tsa.ar_model import AutoReg
from torch.utils.data import DataLoader, TensorDataset

from time_match.utils.toy_configs import get_diffuser

AR_ALPHA = 0.9
AR_SEQ_LEN = 300

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
    'ff-1-wide',
    'resnet-3',
    'transformer-1',
    'transformer-3',
    'transformer-6',
]

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def get_ar1(num_samples: int) -> np.array:
    e = np.random.randn(AR_SEQ_LEN, num_samples)
    y = [np.zeros(num_samples)]

    for i in range(len(e) - 1):
        y.append(AR_ALPHA * y[i] + e[i])
    y = np.stack(y, 0).T
    return y


class SyntheticDataModule(LightningDataModule):
    def __init__(
        self,
        num_samples: int,
        batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        data = get_ar1(num_samples)
        self.trainset = TensorDataset(torch.from_numpy(data).float())

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)


class SyntheticLightningModule(LightningModule):
    def __init__(
        self,
        dim: int,
        model_name: str,
        net_name: str,
        learning_rate: float = 1e-3,
        weight_decay: float = 0,
    ):
        super().__init__()
        self.dim = dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        self.diffuser = get_diffuser(model_name, net_name, dim)

    def forward(self, batch, log_name):
        loss = self.diffuser.get_loss(batch[0]).mean()
        self.log(log_name, loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.forward(batch, log_name='train_loss')

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        gen_samples = self.sample_traj(1_000, num_traj_steps=1)[-1]
        gen_samples = gen_samples.detach().cpu().numpy()
        alphas = np.array([AutoReg(x, lags=1, trend='n').fit().params[0] for x in gen_samples])
        mae = np.abs(alphas - AR_ALPHA)
        mse =  np.square(alphas - AR_ALPHA)
        self.log_dict({
            'alpha_mae_mean': mae.mean(),
            'alpha_mae_std': mae.std(),
            'alpha_mse_mean': mse.mean(),
            'alpha_mse_std': mse.std(),
        }, prog_bar=True)

    @torch.no_grad()
    def sample_traj(self, num_samples: int, num_traj_steps: int = 10, epsilon=None) -> torch.Tensor:
        x = torch.randn(num_samples, self.dim).to(self.device)
        return self.diffuser.sample_traj(x, traj_steps=num_traj_steps)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.diffuser.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


def sanity_check_sampling(module: SyntheticLightningModule):
    samples = module.sample_traj(10, num_traj_steps=3).detach().cpu().numpy()
    assert samples.shape == (3, 10, AR_SEQ_LEN)


def train(
    seed: int,
    model: str,
    net: str,
    data_size: int,
    device: int,
):
    set_seed(seed)
    name = f'{model}_{net}_{data_size}_{seed}'
    out_dir = Path(__file__).resolve().parents[2] / 'results' / 'ar' / name

    if out_dir.exists():
        print(f'Path {out_dir} already exists. Skipping ...')
        return

    out_dir.mkdir(exist_ok=True, parents=True)

    print('Save output dir:', out_dir)

    dm = SyntheticDataModule(
        num_samples=data_size,
        batch_size=1_000,
    )
    module = SyntheticLightningModule(
        dim=AR_SEQ_LEN,
        model_name=model,
        net_name=net,
    )

    trainer = Trainer(
        devices=[device],
        max_steps=10_000,
        val_check_interval=200,
        limit_val_batches=1,
        check_val_every_n_epoch=None,
        log_every_n_steps=1,
        enable_checkpointing=True,
        logger=CSVLogger(save_dir=out_dir, name='logs'),
        callbacks=[TQDMProgressBar()],
    )

    sanity_check_sampling(module)

    trainer.fit(module, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.train_dataloader())

    module = module.to(torch.device('cuda', device))
    samples = module.sample_traj(10_000, num_traj_steps=10).detach().cpu().numpy()
    with open(out_dir / 'samples.npy', 'wb') as f:
        np.save(f, samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--data_size', type=int)
    parser.add_argument('--model', type=str, choices=ALL_MODELS)
    parser.add_argument('--net', type=str, choices=ALL_NETS)
    parser.add_argument('--device', type=int)
    args = parser.parse_args()
    train(args.seed, args.model, args.net, args.data_size, args.device)
