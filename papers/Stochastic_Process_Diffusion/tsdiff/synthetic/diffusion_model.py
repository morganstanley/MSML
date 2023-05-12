import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchtyping import TensorType

import tsdiff
from tsdiff.diffusion.beta_scheduler import get_beta_scheduler, get_loss_weighting
from tsdiff.utils import PositionalEncoding, FeedForward


class FeedForwardModel(nn.Module):
    def __init__(self, dim, hidden_dim, max_i, num_layers=3, **kwargs):
        super().__init__()
        self.t_enc = PositionalEncoding(hidden_dim, max_value=1)
        self.i_enc = PositionalEncoding(hidden_dim, max_value=max_i)
        self.input_proj = nn.Linear(dim, hidden_dim)
        self.net = FeedForward(3 * hidden_dim, [hidden_dim] * num_layers, dim)

    def forward(self, x, *, t, i, **kwargs):
        t = self.t_enc(t)
        i = self.i_enc(i)
        x = self.input_proj(x)
        x = torch.cat([x, t, i], -1)
        return self.net(x)

class RNNModel(nn.Module):
    def __init__(self, dim, hidden_dim, max_i, num_layers=2, bidirectional=True, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.directions = 2 if bidirectional else 1

        self.t_enc = PositionalEncoding(hidden_dim, max_value=1)
        self.i_enc = PositionalEncoding(hidden_dim, max_value=max_i)
        self.init_proj = FeedForward(hidden_dim, [], self.num_layers * self.directions * hidden_dim)
        self.input_proj = FeedForward(dim, [], hidden_dim)
        self.rnn = nn.GRU(
            3 * hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.output_proj = FeedForward(self.directions * hidden_dim, [], dim)

    def forward(
        self,
        x: TensorType['B', 'L', 'D'],
        *,
        t: TensorType['B', 'L', 1],
        i: TensorType['B', 'L', 1],
        **kwargs,
    ) -> TensorType['B', 'L', 'D']:
        shape = x.shape

        t = self.t_enc(t.view(-1, shape[-2], 1))
        i = self.i_enc(i.view(-1, shape[-2], 1))
        x = self.input_proj(x.view(-1, *shape[-2:]))

        init = self.init_proj(i[:,0])
        init = init.view(self.num_layers * self.directions, -1, self.hidden_dim)

        x = torch.cat([x, t, i], -1)

        y, _ = self.rnn(x, init)
        y = self.output_proj(y)
        y = y.view(*shape)

        return y


class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_size, residual_channels, dilation, padding_mode):
        super().__init__()
        self.step_projection = nn.Linear(hidden_size, residual_channels)
        self.time_projection = nn.Linear(hidden_size, residual_channels)

        self.x_step_proj = nn.Sequential(
            nn.Conv2d(residual_channels, residual_channels, kernel_size=1, padding='same', padding_mode=padding_mode),
            nn.LeakyReLU(0.4),
        )
        self.x_time_proj = nn.Sequential(
            nn.Conv2d(residual_channels, residual_channels, kernel_size=1, padding='same', padding_mode=padding_mode),
            nn.LeakyReLU(0.4),
        )

        self.latent_projection = nn.Conv2d(
            1, 2 * residual_channels, kernel_size=1, padding='same', padding_mode=padding_mode,
        )
        self.dilated_conv = nn.Conv2d(
            1 * residual_channels,
            2 * residual_channels,
            kernel_size=3,
            dilation=dilation,
            padding='same',
            padding_mode=padding_mode,
        )
        self.output_projection = nn.Conv2d(
            residual_channels, 2 * residual_channels, kernel_size=1, padding='same', padding_mode=padding_mode,
        )

    def forward(self, x, t=None, i=None):
        i = self.step_projection(i).transpose(-1, -2).unsqueeze(-1)

        y = x + i
        y = y + self.x_step_proj(y)

        t = self.time_projection(t).transpose(-1, -2).unsqueeze(-1)
        y = y + self.x_time_proj(y + t)

        y = self.dilated_conv(y)

        gate, filter = y.chunk(2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        y = F.leaky_relu(y, 0.4)

        residual, skip = y.chunk(2, dim=1)
        return (x + residual) / math.sqrt(2), skip

class CNNModel(nn.Module):
    def __init__(self, dim, hidden_dim, max_i, num_layers=8, residual_channels=8, padding_mode='circular'):
        super().__init__()

        self.input_projection = nn.Conv2d(1, residual_channels, kernel_size=1, padding='same', padding_mode=padding_mode)
        self.step_embedding = PositionalEncoding(hidden_dim, max_value=max_i)
        self.time_embedding = PositionalEncoding(hidden_dim, max_value=1)

        self.residual_layers = nn.ModuleList([
            ResidualBlock(dim, hidden_dim, residual_channels, dilation=2**(i % 2), padding_mode=padding_mode)
            for i in range(num_layers)
        ])

        self.skip_projection = nn.Conv2d(
            residual_channels, residual_channels, kernel_size=3, padding='same', padding_mode=padding_mode,
        )
        self.output_projection = nn.Conv2d(
            residual_channels, 1, kernel_size=3, padding='same', padding_mode=padding_mode,
        )

        self.time_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LeakyReLU(0.4),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.4),
        )

    def forward(self, x, t=None, i=None, **kwargs):
        shape = x.shape

        x = x.view(-1, *x.shape[-2:])
        t = t.view(-1, *t.shape[-2:])
        i = i.view(-1, *i.shape[-2:])

        x = x.unsqueeze(1)
        x = self.input_projection(x)
        x = F.leaky_relu(x, 0.4)

        i = self.step_embedding(i)
        t = self.time_proj(t)

        skip_agg = 0
        for layer in self.residual_layers:
            x, skip = layer(x, t=t, i=i)
            skip_agg = skip_agg + skip

        x = skip_agg / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.leaky_relu(x, 0.4)
        x = self.output_projection(x).squeeze(1)

        x = x.view(*shape)
        return x


class TransformerModel(nn.Module):
    def __init__(self, dim, hidden_dim, max_i, num_layers=8,
                 num_ref_points=10, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.t_enc = PositionalEncoding(hidden_dim, max_value=1)
        self.i_enc = PositionalEncoding(hidden_dim, max_value=max_i)

        self.input_proj = FeedForward(dim, [], hidden_dim)

        self.proj = FeedForward(3 * hidden_dim, [], hidden_dim, final_activation=nn.ReLU())

        self.enc_att = []
        self.i_proj = []
        for _ in range(num_layers):
            self.enc_att.append(nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True))
            self.i_proj.append(nn.Linear(3 * hidden_dim, hidden_dim))
        self.enc_att = nn.ModuleList(self.enc_att)
        self.i_proj = nn.ModuleList(self.i_proj)

        self.output_proj = FeedForward(hidden_dim, [], dim)

    def forward(
        self,
        x: TensorType['B', 'L', 'D'],
        *,
        t: TensorType['B', 'L', 1],
        i: TensorType['B', 'L', 1],
        **kwargs,
    ) -> TensorType['B', 'L', 'D']:
        shape = x.shape

        x = x.view(-1, *shape[-2:])
        t = t.view(-1, shape[-2], 1)
        i = i.view(-1, shape[-2], 1)

        x = self.input_proj(x)
        t = self.t_enc(t)
        i = self.i_enc(i)

        x = self.proj(torch.cat([x, t, i], -1))

        for att_layer, i_proj in zip(self.enc_att, self.i_proj):
            y, _ = att_layer(
                query=x,
                key=x,
                value=x,
            )
            x = x + torch.relu(y)

        x = self.output_proj(x)
        x = x.view(*shape)
        return x

class DiffusionModule(LightningModule):
    def __init__(
        self,
        # Data params
        dim: int,
        data_mean: torch.Tensor = None,
        data_std: torch.Tensor = None,
        max_t: float = None,
        # Diffusion params
        diffusion: str = None,
        gp_sigma: float = None,
        ou_theta: float = None,
        discrete_num_steps: int = None,
        predict_gaussian_noise: bool = None,
        continuous_t1: float = None,
        beta_fn: str = None,
        beta_start: float = None,
        beta_end: float = None,
        loss_weighting: str = None,
        # NN params
        model: str = None,
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

        self.diffusion = getattr(tsdiff.diffusion, diffusion)(
            dim=dim,
            beta_fn=get_beta_scheduler(beta_fn)(beta_start, beta_end),
            sigma=gp_sigma,
            theta=ou_theta,
            num_steps=discrete_num_steps,
            predict_gaussian_noise=predict_gaussian_noise,
            t1=continuous_t1,
            loss_weighting=get_loss_weighting(loss_weighting),
        )

        if model == 'rnn':
            model = RNNModel
        elif model == 'feedforward':
            model = FeedForwardModel
        elif model == 'cnn':
            model = CNNModel
        elif model == 'transformer':
            model = TransformerModel

        max_i = continuous_t1 if 'Continuous' in diffusion else discrete_num_steps

        self.model = model(
            dim=dim,
            hidden_dim=hidden_dim,
            max_i=max_i,
        )

    def forward(self, batch, log_name=None):
        x, t = self._normalize_batch(batch)
        loss = self.diffusion.get_loss(self.model, x, t=t).mean()
        if log_name is not None:
            self.log(log_name, loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.forward(batch, 'train_loss')

    def validation_step(self, batch, batch_idx):
        return self.forward(batch, 'val_loss')

    def test_step(self, batch, batch_idx):
        x, t = self._normalize_batch(batch)
        log_prob = self.diffusion.log_prob(self.model, x, t=t, num_samples=5)
        log_prob = log_prob - self.data_std.log().sum()
        self.log('test_log_prob', log_prob.mean())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def sample(self, t, **kwargs):
        t = t / self.max_t
        samples = self.diffusion.sample(
            self.model.to(t),
            num_samples=t.shape[:-1],
            t=t,
            device=t.device,
            **kwargs,
        )

        return samples * self.data_std.to(t) + self.data_mean.to(t)

    def _normalize_batch(self, batch):
        x, t = batch
        x = (x - self.data_mean.to(x)) / self.data_std.to(x)
        t = t / self.max_t
        return x, t
