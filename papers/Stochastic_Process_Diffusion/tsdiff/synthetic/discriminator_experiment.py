import seml
import numpy as np
from sacred import Experiment
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from tsdiff.synthetic.train import train

DATA_DIR = Path(__file__).parents[2].resolve() / 'data/synthetic'
SAMPLE_DIR = Path(__file__).parents[2].resolve() / 'data/samples'

ex = Experiment()
seml.setup_logger(ex)

class Net(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.emb = nn.Linear(dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=4, batch_first=True),
            num_layers=4,
        )

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.emb(x)
        h = self.transformer(h)
        h = h.mean(dim=1)
        return self.proj(h).squeeze(-1)

class Model(LightningModule):
    def __init__(self, dim, hidden_dim, lr, weight_decay):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.loss = nn.BCELoss(reduction='mean')
        self.net = Net(dim, hidden_dim)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb, log_name='train_loss'):
        x, y = batch
        loss = self.loss(input=self.forward(x), target=y)
        self.log(log_name, loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        return self.training_step(batch, batch_nb, log_name='val_loss')

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(input=y_pred, target=y)
        accuracy = torch.sum((y_pred > 0.5).float() == y) / len(y)
        self.log("test_loss", loss)
        self.log("test_acc", accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

@ex.automain
def run(
    seed: int,
    dataset: str,
    model: str,
    diffusion: str,
    epochs: int,
    batch_size: int,
    gp_sigma: float = None,
    ou_theta: float = None,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # DATA
    filename = SAMPLE_DIR / f'{dataset}-{diffusion}-{model}-{gp_sigma or ou_theta}.npy'
    synthetic = np.load(filename)

    filename = DATA_DIR / f'{dataset}.npz'
    data = np.load(filename)['x'][:len(synthetic)]

    dim = data.shape[-1]

    all_data = torch.Tensor(np.concatenate([synthetic, data], 0))
    all_data = (all_data - all_data.mean()) / all_data.std().clamp(1e-4)
    all_labels = torch.cat([torch.ones(len(synthetic)), torch.zeros(len(data))], 0)

    ind = torch.randperm(len(all_data))
    all_data = all_data[ind]
    all_labels = all_labels[ind]

    ind1, ind2 = int(0.6 * len(all_data)), int(0.8 * len(all_data))

    trainloader = DataLoader(TensorDataset(all_data[:ind1], all_labels[:ind1]), batch_size=batch_size)
    valloader = DataLoader(TensorDataset(all_data[ind1:ind2], all_labels[ind1:ind2]), batch_size=batch_size)
    testloader = DataLoader(TensorDataset(all_data[ind2:], all_labels[ind2:]), batch_size=batch_size)

    # TRAINING
    model = Model(dim, hidden_dim=128, lr=1e-3, weight_decay=1e-5)

    checkpointing = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, filename='best-checkpoint')
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=20)
    trainer = Trainer(
        gpus=1,
        auto_select_gpus=True,
        max_epochs=epochs,
        log_every_n_steps=1,
        enable_checkpointing=True,
        callbacks=[early_stopping, checkpointing],
    )
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=testloader)

    # TESTING
    model = Model.load_from_checkpoint(checkpointing.best_model_path)
    metrics = trainer.test(model, testloader)

    return metrics[0]
