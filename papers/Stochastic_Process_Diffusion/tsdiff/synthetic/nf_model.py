from typing import Callable
from torchtyping import TensorType

import math
import torch
import torch.nn as nn
import torch.distributions as td
from pytorch_lightning import LightningModule
import stribor as st
from stribor.flows.cumsum import diff

import tsdiff
from tsdiff.utils import PositionalEncoding


class Cumsum(st.ElementwiseTransform):
    def __init__(self):
        super().__init__()

    def forward(self, x: TensorType[..., 'dim'], **kwargs) -> TensorType[..., 'dim']:
        y = x.cumsum(dim=-2)
        return y

    def inverse(self, y: TensorType[..., 'dim'], **kwargs) -> TensorType[..., 'dim']:
        x = y.diff(dim=-2, prepend=torch.zeros_like(y[...,:1,:]))
        return x

    def log_det_jacobian(
        self, x: TensorType[..., 'dim'], y: TensorType[..., 'dim'], **kwargs,
    ) -> TensorType[..., 1]:
        return torch.zeros_like(x[...,:1]).to(x)

    def log_diag_jacobian(
        self, x: TensorType[..., 'dim'], y: TensorType[..., 'dim'], **kwargs,
    ) -> TensorType[..., 'dim']:
        return torch.zeros_like(x).to(x)

class Diff(Cumsum):
    forward = Cumsum.inverse
    inverse = Cumsum.forward


class Model(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()

        transforms = [
            Cumsum()
        ]

        for i in range(12):
            transforms.append(
                st.Coupling(
                    st.Affine(
                        dim=dim,
                        latent_net=st.net.MLP(dim + hidden_dim, [hidden_dim] * 2, 2 * dim),
                    ),
                    mask='none' if dim == 1 else f'ordered_{i % 2}',
                )
            )

        base_dist = td.Independent(td.Normal(torch.zeros(dim).cuda(), torch.ones(dim).cuda()), 1)
        self.flow = st.NormalizingFlow(base_dist, transforms)
        self.t_enc = PositionalEncoding(hidden_dim - 1, max_value=1)

    def log_prob(
        self,
        x: TensorType['B', 'L', 'D'],
        t: TensorType['B', 'L', 1],
        **kwargs,
    ) -> TensorType['B', 'L', 'D']:
        t = torch.cat([t, self.t_enc(t)], -1)
        log_prob = self.flow.log_prob(x, latent=t)
        return log_prob

    def sample(self, t, **kwargs):
        t = torch.cat([t, self.t_enc(t)], -1)
        samples = self.flow.sample(num_samples=t.shape[:-1], latent=t)
        return samples

class NFModule(LightningModule):
    def __init__(
        self,
        # Data params
        dim: int,
        data_mean: torch.Tensor = None,
        data_std: torch.Tensor = None,
        max_t: float = None,
        # NN params
        hidden_dim: int = None,
        # Training params
        learning_rate: float = None,
        weight_decay: float = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dim = dim
        self.data_mean = data_mean
        self.data_std = data_std
        self.max_t = max_t

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.model = Model(
            dim=dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, batch, log_name=None):
        x, t = self._normalize_batch(batch)
        loss = -self.model.log_prob(x, t).mean()
        if log_name is not None:
            self.log(log_name, loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.forward(batch, 'train_loss')

    def validation_step(self, batch, batch_idx):
        return self.forward(batch, 'val_loss')

    def test_step(self, batch, batch_idx):
        x, t = self._normalize_batch(batch)
        log_prob = self.model.log_prob(x, t) / t.shape[-2]
        log_prob = log_prob - self.data_std.log().sum()
        self.log('test_log_prob', log_prob.mean())

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    @torch.no_grad()
    def sample(self, t, **kwargs):
        t = t / self.max_t
        self.model = self.model.to(t)
        samples = self.model.sample(t)
        return samples * self.data_std.to(t) + self.data_mean.to(t)

    def _normalize_batch(self, batch):
        x, t = batch
        x = (x - self.data_mean.to(x)) / self.data_std.to(x)
        t = t / self.max_t
        return x, t
