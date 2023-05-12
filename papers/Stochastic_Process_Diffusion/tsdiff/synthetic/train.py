import warnings
import argparse
import numpy as np
from pathlib import Path

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from tsdiff.synthetic.data import DataModule
from tsdiff.synthetic.diffusion_model import DiffusionModule
from tsdiff.synthetic.ode_model import ODEModule
from tsdiff.synthetic.nf_model import NFModule
from tsdiff.synthetic.sde_model import SDEModule

warnings.simplefilter(action='ignore', category=(np.VisibleDeprecationWarning))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_DIR = Path(__file__).parents[2].resolve() / 'data/samples'
SAMPLE_DIR.mkdir(exist_ok=True, parents=True)

def train(
    *,
    seed: int,
    dataset: str,
    diffusion: str,
    model: str,
    gp_sigma: float = None,
    ou_theta: float = None,
    beta_start: float = None,
    beta_end: float = None,
    batch_size: int = 256,
    hidden_dim: int = 128,
    predict_gaussian_noise: bool = True,
    beta_fn: str = 'linear',
    discrete_num_steps: int = 100,
    continuous_t1: float = 1,
    loss_weighting: str = 'exponential',
    learning_rate: float = 1e-3,
    weight_decay: float = 0,
    epochs: int = 100,
    patience: int = 20,
    return_model: bool = False,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    datamodule = DataModule(dataset, batch_size=batch_size)

    if diffusion is not None:
        if 'Continuous' in diffusion:
            beta_start, beta_end = 0.1, 20
        else:
            beta_start, beta_end = 1e-4, 20 / discrete_num_steps

    if model == 'ode':
        Module = ODEModule
    elif model == 'nf':
        Module = NFModule
    elif model == 'sde':
        Module = SDEModule
    else:
        Module = DiffusionModule

    # Load model
    module = Module(
        dim=datamodule.dim,
        data_mean=datamodule.x_mean,
        data_std=datamodule.x_std,
        max_t=datamodule.t_max,
        diffusion=diffusion,
        model=model,
        predict_gaussian_noise=predict_gaussian_noise,
        gp_sigma=gp_sigma,
        ou_theta=ou_theta,
        beta_fn=beta_fn,
        discrete_num_steps=discrete_num_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        continuous_t1=continuous_t1,
        loss_weighting=loss_weighting,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    # Train
    checkpointing = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, filename='best-checkpoint')
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=patience)
    trainer = Trainer(
        gpus=1,
        auto_select_gpus=True,
        max_epochs=epochs,
        log_every_n_steps=1,
        enable_checkpointing=True,
        callbacks=[early_stopping, checkpointing],
    )

    trainer.fit(module, train_dataloaders=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())

    # Load best model
    module = Module.load_from_checkpoint(checkpointing.best_model_path)

    # Evaluation
    metrics = trainer.test(module, datamodule.test_dataloader())

    # Generate samples
    if seed == 1:
        t = datamodule.trainset[:1000][1].to(device)
        samples = module.sample(t=t, use_ode=True)
        np.save(SAMPLE_DIR / f'{dataset}-{diffusion}-{model}-{gp_sigma or ou_theta}-{predict_gaussian_noise}', samples.detach().cpu().numpy())

    if return_model:
        return module, datamodule, trainer, metrics

    return metrics[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train forecasting model.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, choices=['cir', 'lorenz', 'ou', 'predator_prey', 'sine', 'sink'])
    parser.add_argument('--diffusion', type=str, choices=[
        'GaussianDiffusion', 'OUDiffusion', 'GPDiffusion',
        'ContinuousGaussianDiffusion', 'ContinuousOUDiffusion', 'ContinuousGPDiffusion',
    ])
    parser.add_argument('--model', type=str, choices=['feedforward', 'rnn', 'cnn', 'ode', 'transformer'])
    parser.add_argument('--gp_sigma', type=float, default=0.1)
    parser.add_argument('--ou_theta', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    metrics = train(**args.__dict__)

    for key, value in metrics.items():
        print(f'{key}:\t{value:.4f}')
