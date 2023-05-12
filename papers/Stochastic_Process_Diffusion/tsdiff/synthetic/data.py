import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from pathlib import Path

class DataModule(LightningDataModule):
    def __init__(self, name, batch_size: int, test_batch_size: int = None):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size or batch_size

        dataset = self._load_dataset()
        self.trainset, self.valset, self.testset = self._split_train_val_test(dataset)

    @property
    def dim(self):
        return self.trainset[0][0].shape[-1]

    @property
    def x_mean(self):
        return torch.cat([x[0] for x in self.trainset], 0).mean(0)

    @property
    def x_std(self):
        return torch.cat([x[0] for x in self.trainset], 0).std(0).clamp(1e-4)

    @property
    def t_max(self):
        return torch.cat([x[1] for x in self.trainset], 0).max().item()

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.test_batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.test_batch_size, shuffle=False)

    def _load_dataset(self):
        filepath = Path(__file__).parents[2] / f'data/synthetic/{self.name}.npz'
        data = np.load(filepath)
        dataset = TensorDataset(torch.Tensor(data['x']), torch.Tensor(data['t']))
        return dataset

    def _split_train_val_test(self, dataset):
        train_len, val_len = int(0.6 * len(dataset)), int(0.2 * len(dataset))
        return random_split(dataset, lengths=[train_len, val_len, len(dataset) - train_len - val_len])
