from typing import Callable
from torchtyping import TensorType

import math
import torch
import torch.nn as nn
from torch import distributions
import torchsde
from pytorch_lightning import LightningModule

def _stable_division(a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign())
    return a / b

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        out, _ = self.gru(inp)
        out = self.lin(out)
        return out


class LatentSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, data_size, latent_size, context_size, hidden_size):
        super(LatentSDE, self).__init__()
        # Encoder.
        self.encoder = Encoder(input_size=data_size, hidden_size=hidden_size, output_size=context_size)
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)

        # Decoder.
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        self.h_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        # This needs to be an element-wise function for the SDE to satisfy diagonal noise.
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.Softplus(),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid()
                )
                for _ in range(latent_size)
            ]
        )
        self.projector = nn.Linear(latent_size, data_size)

        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        self._ctx = None

    def contextualize(self, ctx):
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).

    def f(self, t, y):
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        return self.f_net(torch.cat((y, ctx[i]), dim=1))

    def h(self, t, y):
        return self.h_net(y)

    def g(self, t, y):  # Diagonal diffusion.
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def forward(self, xs, ts, noise_std, adjoint=False, method="euler"):
        # Contextualization is only needed for posterior inference.
        xs = xs.transpose(0, 1)

        ctx = self.encoder(torch.flip(xs, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx))

        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)

        if adjoint:
            # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
            adjoint_params = (
                (ctx,) +
                tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(self.h_net.parameters())
            )
            zs, log_ratio = torchsde.sdeint_adjoint(
                self, z0, ts, adjoint_params=adjoint_params, dt=1e-2, logqp=True, method=method)
        else:
            zs, log_ratio = torchsde.sdeint(self, z0, ts, dt=1e-2, logqp=True, method=method)

        _xs = self.projector(zs)
        xs_dist = torch.distributions.Normal(loc=_xs, scale=noise_std)

        log_pxs = xs_dist.log_prob(xs).sum(dim=(0, 2)).mean(dim=0)

        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        logqp_path = log_ratio.sum(dim=0).mean(dim=0)
        return log_pxs, logqp0 + logqp_path

    @torch.no_grad()
    def sample(self, batch_size, ts, bm=None):
        eps = torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
        z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
        zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h'}, dt=1e-3, bm=bm)
        # Most of the times in ML, we don't sample the observation noise for visualization purposes.
        _xs = self.projector(zs)
        return _xs

class SDEModule(LightningModule):
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
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.data_mean = data_mean
        self.data_std = data_std
        self.max_t = max_t

        self.save_hyperparameters()

        self.latent_sde = LatentSDE(dim, hidden_dim, hidden_dim, hidden_dim)

    def forward(self, batch, log_name=None):
        x, t = self._normalize_batch(batch)
        t = t[0,:,0]
        log_prob, kl = self.latent_sde(x, t, noise_std=0.01)
        loss = -log_prob + kl
        loss = loss.mean()
        if log_name is not None:
            self.log(log_name, loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.forward(batch, 'train_loss')

    def validation_step(self, batch, batch_idx):
        return self.forward(batch, 'val_loss')

    def test_step(self, batch, batch_idx):
        log_prob = -self(batch) / batch[0].shape[-2]
        log_prob = log_prob - self.data_std.log().sum()
        self.log('test_log_prob', log_prob.mean())

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.latent_sde.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    @torch.no_grad()
    def sample(self, t, **kwargs):
        t = t / self.max_t
        t = t.cpu()

        eps = torch.randn(t.shape[0], 1).to(t)
        bm = torchsde.BrownianInterval(
            t0=t[0,0,0],
            t1=t[0,-1,0],
            size=(t.shape[0], 1),
            device=t.device,
            levy_area_approximation='space-time',
        )

        samples = self.latent_sde.sample_q(ts=t[0,:,0], batch_size=t.shape[0], eps=eps, bm=bm).squeeze()
        samples = samples.transpose(0, 1).unsqueeze(-1)
        return samples * self.data_std.to(t) + self.data_mean.to(t)

    def _normalize_batch(self, batch):
        x, t = batch
        x = (x - self.data_mean.to(x)) / self.data_std.to(x)
        t = t / self.max_t
        return x, t
